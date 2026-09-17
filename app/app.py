"""FaceMatch -- Gradio wrapper around the ArcFace bank/score workflow.

Build a reference "bank" from photos of one person, then score other photos
against it. Two tabs: Face Match (build a bank, score candidates, save
keepers into a dataset) and Dataset Builder (identity-gated harvesting from
your own photo/video collections). All scoring goes through facematch.core,
the single ArcFace implementation shared with the CLI and the tools/*.py
scripts.

This app used to exist as two separate apps (a two-tab one and a standalone
single-purpose one); this file is their merge. See README.md for what came
from which and why.
"""
import argparse
import csv
import datetime
import glob
import html
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import warnings

warnings.filterwarnings("ignore")

# facematch/ lives one directory up from this file; make it importable
# whether this script is run directly (`python app/app.py`), via run.ps1, or
# double-clicked -- none of those put the repo root on sys.path for us.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from PIL import Image, ImageDraw

import dataset_builder
from facematch import core

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
os.makedirs(RUNS, exist_ok=True)


# stdout gets buffered into oblivion when the app is launched detached, so log
# to a file and flush every record.
class _Flushing(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[_Flushing(os.path.join(HERE, "facematch.log"), encoding="utf8"), logging.StreamHandler()],
)
log = logging.getLogger("facematch")


def stamp():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


DEFAULT_PORT = 7861
DATASETS_BASE = os.path.join(HERE, "datasets")

_app = None
_lock = threading.Lock()


def get_analyzer():
    """Lazy singleton FaceAnalysis; CUDA first, CPU fallback (see
    facematch.core.analyzer -- this just adds the process-wide singleton)."""
    global _app
    with _lock:
        if _app is None:
            _app = core.analyzer(gpu=True)
            log.info("FaceMatch: analyzer ready on %s", core.provider_of(_app))
        return _app


# ---------------------------------------------------------------- native pickers
# The dialog has to run outside this process: tkinter is not thread-safe and
# gradio handlers run on worker threads, so an in-process Tk() intermittently
# deadlocks the whole app. A short-lived child process is boring and always
# works.
_PICKER_SRC = r"""
import sys
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)
if sys.argv[1] == "folder":
    p = filedialog.askdirectory(title="Choose a folder of images")
    picked = [p] if p else []
else:
    picked = list(filedialog.askopenfilenames(
        title="Choose image(s)",
        filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All files", "*.*")]))
root.destroy()
sys.stdout.buffer.write("\n".join(picked).encode("utf8"))
"""

CREATE_NO_WINDOW = 0x08000000


def _pythonw():
    """pythonw keeps a console window from flashing up behind the dialog."""
    cand = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
    return cand if os.path.exists(cand) else sys.executable


def _pick(mode):
    try:
        r = subprocess.run([_pythonw(), "-c", _PICKER_SRC, mode], capture_output=True, timeout=600,
                           creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0)
    except Exception as e:
        log.warning("picker failed (%s): %s", mode, e)
        return []
    if r.returncode != 0:
        log.warning("picker exit %s: %s", r.returncode, r.stderr.decode("utf8", "replace")[:400])
        return []
    out = r.stdout.decode("utf8", "replace").strip()
    return [p for p in out.split("\n") if p]


def picked_md(paths):
    if not paths:
        return "*No individual images picked.*"
    names = ", ".join("`" + os.path.basename(p) + "`" for p in paths[:8])
    more = " …+" + str(len(paths) - 8) + " more" if len(paths) > 8 else ""
    return "**" + str(len(paths)) + " image(s) picked:** " + names + more


def on_pick_folder():
    got = _pick("folder")
    if not got:
        return gr.update()  # cancelled -- leave whatever is in the box
    log.info("picked folder: %s", got[0])
    return got[0]


def on_pick_images(existing):
    got = _pick("files")
    if not got:
        return existing, picked_md(existing), gr.update(visible=bool(existing))
    merged = list(dict.fromkeys(list(existing or []) + got))
    log.info("picked %s image(s), %s total", len(got), len(merged))
    return merged, picked_md(merged), gr.update(visible=True)


def on_clear_picks():
    return [], picked_md([]), gr.update(visible=False)


def drive_roots():
    """Gradio 6 will only serve files from cwd or the system temp dir, so a
    preview of a folder on another drive dies with InvalidPathError. This
    app's entire job is reading image folders the user names, anywhere on
    the machine, so every local drive root goes on the allowlist. Safe only
    because the server binds 127.0.0.1 by default -- see main()'s guard
    against pairing this with --share or a LAN host."""
    if os.name != "nt":
        return ["/"]
    return [d + ":\\" for d in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if os.path.isdir(d + ":\\")]


# ---------------------------------------------------------------- file gathering
def gather(folder, uploads, recursive=False):
    """Folder path and/or uploaded files -> list of real image paths on
    disk. Both input styles are additive so a run can mix a folder with a
    few extra uploaded stragglers."""
    out = []
    if folder and os.path.isdir(folder):
        pat = "**/*" if recursive else "*"
        for p in sorted(glob.glob(os.path.join(folder, pat), recursive=recursive)):
            if os.path.splitext(p)[1].lower() in core.IMAGE_EXT:
                out.append(p)
    for u in uploads or []:
        p = u if isinstance(u, str) else getattr(u, "name", None)
        if p and os.path.splitext(p)[1].lower() in core.IMAGE_EXT:
            out.append(p)
    return out


# ---------------------------------------------------------------- result rendering
def label(score):
    if score is None:
        return "no face"
    if score >= 0.50:
        return "same person"
    if score >= 0.35:
        return "plausible"
    if score >= 0.28:
        return "weak"
    return "different person"


def verdict_class(score):
    if score is None:
        return "fm-none"
    if score >= 0.50:
        return "fm-same"
    if score >= 0.35:
        return "fm-plaus"
    if score >= 0.28:
        return "fm-weak"
    return "fm-diff"


def annotate(path, face_scores, max_side=560):
    """Draw every scored face (not just the best match) so a bystander
    getting painted with the same identity -- LoRA bleed -- shows up as a
    second colored box, not a number you have to notice."""
    im = Image.open(path).convert("RGB")
    d = ImageDraw.Draw(im)
    for fs in face_scores:
        x0, y0, x1, y1 = [int(v) for v in fs["bbox"]]
        s = fs["sim"]
        col = "#38b000" if s >= 0.50 else ("#f7b801" if s >= 0.35 else "#d90429")
        w = max(2, im.width // 400)
        d.rectangle([x0, y0, x1, y1], outline=col, width=w)
        tag = f"{s:.3f}"
        ty = max(0, y0 - 22)
        d.rectangle([x0, ty, x0 + 8 * len(tag) + 8, ty + 20], fill=col)
        d.text((x0 + 4, ty + 5), tag, fill="white")
    if max(im.size) > max_side:
        r = max_side / max(im.size)
        im = im.resize((int(im.width * r), int(im.height * r)), Image.LANCZOS)
    return im


def names_html(rows, ceiling):
    """File-name view: a responsive CSS grid, so the column count follows
    the window width. A 400-image run becomes a wall you can scan, not a
    mile-long column."""
    if not rows:
        return ""
    scored = sorted([r for r in rows if r[1] is not None], key=lambda r: -r[1])
    noface = [r for r in rows if r[1] is None]
    cells = []
    for base, best, n, extra, verdict in scored + noface:
        title = base + " — " + verdict
        score_txt = "no face" if best is None else f"{best:.3f}"
        bits = []
        if best is not None:
            bits.append(str(n) + (" face" if n == 1 else " faces"))
            if extra is not None:
                bits.append(f"2nd {extra:.3f}")
        cells.append(
            '<div class="fm-cell ' + verdict_class(best) + '" title="' + html.escape(title, quote=True) + '">'
            + '<div class="fm-name">' + html.escape(base) + '</div>'
            + '<div class="fm-meta"><span class="fm-score">' + score_txt + '</span>'
            + '<span>' + html.escape(" · ".join(bits)) + '</span></div></div>'
        )
    head = (
        '<div class="fm-head">' + str(len(scored)) + ' scored, best first'
        + (' · ' + str(len(noface)) + ' with no face' if noface else '')
        + ' · ceiling ' + f"{ceiling:.3f}" + '</div>'
    )
    return head + '<div class="fm-grid">' + "".join(cells) + '</div>'


PREVIEW_CAP = 300  # about browser memory, not CPU -- previews skip face detection


def plain_names_html(paths, note=""):
    """File-name grid for the input sections, where there are no scores yet."""
    if not paths:
        return ""
    cells = [
        '<div class="fm-cell fm-plain" title="' + html.escape(p, quote=True) + '">'
        + '<div class="fm-name">' + html.escape(os.path.basename(p)) + '</div></div>'
        for p in paths
    ]
    return ('<div class="fm-head">' + str(len(paths)) + ' image(s)' + note + '</div>'
            + '<div class="fm-grid">' + "".join(cells) + '</div>')


def preview_of(folder, picks, recursive=False):
    files = gather(folder, picks, recursive)
    shown = files[:PREVIEW_CAP]
    note = "" if len(files) <= PREVIEW_CAP else (" · showing the first " + str(PREVIEW_CAP))
    gal = [(p, os.path.basename(p)) for p in shown]
    return gal, plain_names_html(shown, note)


def refresh_ref_preview(folder, picks):
    return preview_of(folder, picks)


def refresh_cand_preview(folder, uploads, recursive):
    return preview_of(folder, uploads, recursive)


def toggle_view(v):
    """One rule for every Thumbnails/File-names switch in the app: Thumbnails
    shows the gallery, File names shows the HTML grid."""
    return gr.update(visible=v == "Thumbnails"), gr.update(visible=v == "File names")


# ---------------------------------------------------------------- bank + scoring
def build_bank(folder, uploads, cap, progress=gr.Progress()):
    paths = gather(folder, uploads)
    if not paths:
        return None, "**No reference images found.** Load a folder, pick images, or upload some."
    if cap and len(paths) > cap:
        paths = paths[: int(cap)]
    app = get_analyzer()
    try:
        mean, stats = core.bank_from_paths(app, paths)
    except ValueError:
        return None, "**No faces detected in any reference image.**"
    bank = {"ref": mean, **stats}
    msg = (
        f"**Bank built** from {stats['n']} of {stats['of']} usable image(s) "
        f"({stats['dropped']} outlier(s) dropped). Ceiling (mean self-similarity) "
        f"**{stats['self_mean']:.3f}** (min {stats['self_min']:.3f}). "
        f"Running on {core.provider_of(app)}."
    )
    log.info("bank built: %s", msg.replace("\n", " "))
    return bank, msg


def clear_bank():
    return None, ""


def score_candidates(folder, uploads, recursive, bank, progress=gr.Progress()):
    empty_sel = gr.update(choices=[], value=[])
    if not bank or "ref" not in bank:
        return "**No reference bank yet** -- build one in the section above first.", None, "", None, empty_sel, {}
    files = gather(folder, uploads, recursive)
    if not files:
        return "**No images to score.** Load a folder, upload files, or give a folder path.", None, "", None, empty_sel, {}

    app = get_analyzer()
    ref = bank["ref"]
    rows, gallery, path_map = [], [], {}
    for i, p in enumerate(files):
        progress((i + 1) / len(files), desc=f"scoring {i + 1}/{len(files)}")
        base = os.path.basename(p)
        path_map[base] = p
        face_scores = core.score_image(app, p, ref)
        if not face_scores:
            rows.append((base, None, 0, None, "no face"))
            continue
        sims = sorted((f["sim"] for f in face_scores), reverse=True)
        best, extra = sims[0], (sims[1] if len(sims) > 1 else None)
        rows.append((base, best, len(face_scores), extra, label(best)))
        gallery.append((annotate(p, face_scores), f"{base} — best {best:.3f}"
                        + (f", 2nd face {extra:.3f}" if extra is not None else "")))

    scored = [r for r in rows if r[1] is not None]
    md = [f"### {len(scored)} of {len(rows)} image(s) scored · reference ceiling **{bank['self_mean']:.3f}**"]
    noface = [r[0] for r in rows if r[1] is None]
    if noface:
        md.append(f"No face detected in {len(noface)}: " + ", ".join(f"`{b}`" for b in noface[:10])
                  + (" …" if len(noface) > 10 else ""))

    out_csv = os.path.join(RUNS, f"scores_{stamp()}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "best", "faces", "second_face", "verdict"])
        for base, best, n, extra, verdict in rows:
            w.writerow([base, "" if best is None else f"{best:.4f}", n,
                       "" if extra is None else f"{extra:.4f}", verdict])
    log.info("scored %s image(s) -> %s", len(rows), out_csv)

    keep_sel = gr.update(choices=[r[0] for r in rows], value=[])
    return "\n\n".join(md), gallery, names_html(rows, bank["self_mean"]), out_csv, keep_sel, path_map


# ---------------------------------------------------------------- dataset saving
def list_datasets():
    if not os.path.isdir(DATASETS_BASE):
        return []
    return sorted(d for d in os.listdir(DATASETS_BASE) if os.path.isdir(os.path.join(DATASETS_BASE, d)))


def create_dataset(name):
    name = (name or "").strip()
    if not name or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._ -]*", name):
        return gr.update(choices=list_datasets()), \
            "Dataset name must start with a letter/digit and use only letters, digits, spaces, . _ -"
    os.makedirs(os.path.join(DATASETS_BASE, name), exist_ok=True)
    return gr.update(choices=list_datasets(), value=name), f"Dataset **{name}** ready."


def add_to_dataset(selected, path_map, dataset):
    if not dataset:
        return "Pick a dataset first (create one if the dropdown is empty)."
    if not selected:
        return "No images selected."
    dest_dir = os.path.join(DATASETS_BASE, dataset)
    if not os.path.isdir(dest_dir):
        return f"Dataset folder {dataset} no longer exists — create it again."
    copied, missing = 0, []
    for name in selected:
        src = (path_map or {}).get(name)
        if not src or not os.path.isfile(src):
            missing.append(name)
            continue
        base, ext = os.path.splitext(name)
        dest = os.path.join(dest_dir, name)
        n = 0
        while os.path.exists(dest):
            n += 1
            dest = os.path.join(dest_dir, f"{base}_{n}{ext}")
        shutil.copy2(src, dest)
        copied += 1
    msg = f"Copied {copied} image(s) to **{dataset}**."
    if missing:
        msg += f" Not found (folder moved / upload expired?): {', '.join(missing)}"
    return msg


def builder_create_dataset(name):
    """Create a dataset and refresh the dropdowns on both tabs."""
    upd, _msg = create_dataset(name)
    return upd, upd


def builder_harvest(src, bank, framing, min_sim, min_face, min_sharp, fps, window, dedup, max_pv, auto_rot,
                    progress=gr.Progress()):
    empty = gr.update(choices=[], value=[])
    if not bank:
        return [], empty, [], "**Build a reference bank on the Face Match tab first.**"
    src = (src or "").strip()
    if not src:
        return [], empty, [], "**Enter a source folder, file, or glob to harvest from.**"
    if not (os.path.exists(src) or glob.glob(src, recursive=True)):
        return [], empty, [], f"**Nothing matched** `{src}` — check the path or glob."
    dataset_builder.cleanup_temp()
    try:
        cands, stats = dataset_builder.harvest(
            src, bank["ref"], get_analyzer(), core.INFER_LOCK,
            framing=framing, min_sim=min_sim, min_face=int(min_face), min_sharp=min_sharp,
            fps=fps, window=window, dedup=dedup, max_per_video=int(max_pv), auto_rotate=auto_rot,
            progress_cb=lambda d, t, m: progress(d / max(t, 1), desc=m),
        )
    except Exception as e:
        return [], empty, [], f"**Harvest failed:** {e}"
    gallery = [(c["path"], f'{c["sim"]:.3f} | {os.path.basename(c["source"])}') for c in cands]
    ids = [c["id"] for c in cands]
    return gallery, gr.update(choices=ids, value=[]), cands, stats


def builder_save(cands, selected, dataset, trigger, framing_cap):
    if not dataset:
        return "**Pick a target dataset (or create one) first.**"
    if not selected:
        return "**Tick at least one crop to keep.**"
    return dataset_builder.save_selected(cands, selected, os.path.join(DATASETS_BASE, dataset),
                                         (trigger or "").strip(), framing_cap)


CSS = """
.gradio-container{max-width:min(2100px,98vw)!important}

/* Thumbnail view: override gradio's fixed --grid-cols with an auto-fill grid so the
   column count follows the window instead of a hardcoded number. Class-based, so the
   reference, to-score and results galleries all lay out identically. */
.fm-gal .grid-container{
  grid-template-columns:repeat(auto-fill,minmax(170px,1fr))!important;
  grid-template-rows:none!important;
  grid-auto-rows:minmax(130px,auto)!important;
}

/* File-name view */
.fm-names .fm-head{font-size:12px;color:var(--body-text-color-subdued);margin:0 0 6px}
.fm-names .fm-grid{
  display:grid;
  grid-template-columns:repeat(auto-fill,minmax(225px,1fr));
  gap:6px;
}
.fm-names .fm-cell{
  border:1px solid var(--border-color-primary);border-left-width:5px;border-radius:6px;
  padding:5px 8px;background:var(--block-background-fill);overflow:hidden;
}
.fm-names .fm-name{
  font-family:var(--font-mono);font-size:11px;color:var(--body-text-color);
  overflow-wrap:anywhere;word-break:break-all;line-height:1.25;
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;
}
.fm-names .fm-meta{
  display:flex;justify-content:space-between;gap:8px;margin-top:1px;
  font-size:11px;color:var(--body-text-color-subdued);
}
.fm-names .fm-score{font-weight:700;font-size:13px}
.fm-names .fm-same{border-left-color:#38b000}
.fm-names .fm-same .fm-score{color:#38b000}
.fm-names .fm-plaus{border-left-color:#f7b801}
.fm-names .fm-plaus .fm-score{color:#c98f00}
.fm-names .fm-weak{border-left-color:#ff8500}
.fm-names .fm-weak .fm-score{color:#ff8500}
.fm-names .fm-diff{border-left-color:#d90429}
.fm-names .fm-diff .fm-score{color:#d90429}
.fm-names .fm-none{border-left-color:var(--border-color-primary);opacity:.6}
.fm-names .fm-plain{border-left-color:var(--border-color-accent,var(--border-color-primary))}
"""

with gr.Blocks(title="FaceMatch", css=CSS, theme=gr.themes.Soft()) as demo:
    bank_state = gr.State(None)
    path_map_state = gr.State({})
    ref_picked = gr.State([])
    builder_cands = gr.State([])

    with gr.Tabs():
        with gr.Tab("Face Match"):
            gr.Markdown(
                "# FaceMatch\n"
                "Builds an identity reference from photos you trust (the \"bank\"), then scores "
                "other photos against it to tell you how likely each one shows the same person. "
                "Every face in an image is scored, not just the best match -- that's how a "
                "character LoRA painting itself onto a bystander shows up.\n\n"
                "Scores are ArcFace cosine similarity, read against the bank's own ceiling "
                "(its mean self-similarity), never against 1.0."
            )

            with gr.Accordion("1 · Reference bank (who you are matching against)", open=True):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        with gr.Row():
                            load_folder_btn = gr.Button("📁 Load folder", variant="secondary")
                            load_images_btn = gr.Button("🖼️ Load image(s)", variant="secondary")
                        ref_folder = gr.Textbox(
                            label="Reference folder", placeholder=r"path\to\reference_photos",
                            info="Every image in the folder, combined with any picks/uploads below.")
                        ref_picked_md = gr.Markdown(picked_md([]))
                        clear_picks_btn = gr.Button("clear picked images", size="sm", visible=False)
                        ref_files = gr.File(label="…or upload images", file_count="multiple",
                                            file_types=["image"])
                        ref_cap = gr.Slider(5, 300, value=60, step=5, label="Cap reference images",
                                            info="60 is plenty; more just costs time.")
                        build_btn = gr.Button("Build bank", variant="primary")
                        clear_btn = gr.Button("Clear bank", size="sm")
                    with gr.Column(scale=2):
                        bank_info = gr.Markdown()
                ref_view = gr.Radio(["Thumbnails", "File names"], value="Thumbnails", label="Loaded references")
                ref_gallery = gr.Gallery(show_label=False, elem_classes=["fm-gal"], height=280, preview=False)
                ref_names = gr.HTML(elem_classes=["fm-names"], visible=False)

            with gr.Accordion("2 · Images to score", open=True):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        load_cand_folder_btn = gr.Button("📁 Load folder", variant="secondary")
                        cand_folder = gr.Textbox(label="Folder to score", placeholder=r"path\to\candidates")
                        cand_files = gr.File(label="…or upload images", file_count="multiple",
                                             file_types=["image"])
                        recursive = gr.Checkbox(False, label="Include subfolders")
                        score_btn = gr.Button("Score", variant="primary")
                    with gr.Column(scale=2):
                        score_info = gr.Markdown()
                        csv_out = gr.File(label="Download results as CSV", height=90)
                cand_view = gr.Radio(["Thumbnails", "File names"], value="Thumbnails", label="Loaded candidates")
                cand_gallery = gr.Gallery(show_label=False, elem_classes=["fm-gal"], height=280, preview=False)
                cand_names = gr.HTML(elem_classes=["fm-names"], visible=False)

            gr.Markdown("### Results")
            view = gr.Radio(["Thumbnails", "File names"], value="Thumbnails", label="Results view",
                            info="Every run fills both views, so switching is instant.")
            gallery = gr.Gallery(label="Scored faces", elem_classes=["fm-gal"], height=560, preview=False)
            names_out = gr.HTML(elem_classes=["fm-names"], visible=False)

            gr.Markdown("### Datasets\nKeep the good matches: create a named dataset, tick the "
                        "scored images you want, and add them. Files land in `datasets\\<name>\\`.")
            with gr.Row():
                ds_name = gr.Textbox(label="New dataset name", scale=2)
                ds_create = gr.Button("Create dataset", size="sm", scale=1)
                ds_pick = gr.Dropdown(choices=list_datasets(), label="Dataset", scale=2)
            keep_sel = gr.CheckboxGroup(label="Scored images to keep (populated after a scoring run)", choices=[])
            ds_add = gr.Button("Add selected to dataset", size="sm")
            ds_info = gr.Markdown()

        with gr.Tab("Dataset Builder"):
            gr.Markdown(
                "# Dataset Builder\n"
                "Harvest identity-gated face crops from your own photos and videos, review "
                "the candidates, then commit the keepers to a dataset. Flow: **harvest → "
                "review → save**.\n\n"
                "Uses the reference bank from the Face Match tab -- build it there first. "
                "Harvest runs on the app's GPU and is serialized with scoring."
            )

            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    gr.Markdown("### Source")
                    src_path = gr.Textbox(label="Source folder / file / glob (photos and videos)")
                    auto_rotate = gr.Checkbox(value=True, label="Auto-rotate (sideways phone video)")
                    with gr.Accordion("Video settings", open=False):
                        fps = gr.Slider(0.5, 10, value=2, step=0.5, label="Sample rate (frames per second)")
                        window = gr.Slider(0.5, 10, value=2, step=0.5, label="Window (seconds)")
                        dedup = gr.Slider(0.0, 1.0, value=0.95, step=0.01, label="Dedup similarity",
                                          info="skip near-duplicate frames; 0 disables")
                        max_per_video = gr.Slider(0, 200, value=0, step=1, label="Max crops per video (0 = no cap)")
                with gr.Column(scale=1):
                    gr.Markdown("### Framing & gates")
                    framing = gr.Radio(choices=list(dataset_builder.FRAMINGS.keys()), value="Portrait",
                                       label="Framing",
                                       info="how much context around the face — mix framings across your dataset")
                    min_sim = gr.Slider(0.0, 1.0, value=0.35, step=0.01, label="Identity gate (similarity vs bank)")
                    min_face = gr.Slider(60, 400, value=200, step=10, label="Min face size in source px")
                    min_sharp = gr.Slider(0, 200, value=45, step=5, label="Min sharpness (Laplacian var)")

            with gr.Row():
                harvest_btn = gr.Button("Harvest", variant="primary")
            builder_status = gr.Markdown()

            cand_gallery_b = gr.Gallery(label="Candidates (score in caption)", columns=6, height=420)

            builder_keep = gr.CheckboxGroup(label="Crops to keep", choices=[])
            with gr.Row():
                keep_all_btn = gr.Button("Select all", size="sm")
                keep_none_btn = gr.Button("Clear selection", size="sm")

            gr.Markdown("### Save")
            with gr.Row():
                bld_ds_pick = gr.Dropdown(choices=list_datasets(), label="Target dataset")
                bld_ds_new = gr.Textbox(label="…or new dataset name", scale=1)
                bld_ds_create = gr.Button("Create", size="sm")
            trigger_box = gr.Textbox(label="Trigger word for captions (blank = no .txt files)")
            framing_cap = gr.Checkbox(value=True, label="Append framing to caption")
            save_btn = gr.Button("Save selected to dataset", variant="primary")
            save_info = gr.Markdown()

    REF_PREVIEW = [ref_gallery, ref_names]
    CAND_PREVIEW = [cand_gallery, cand_names]

    load_folder_btn.click(on_pick_folder, None, ref_folder).then(
        refresh_ref_preview, [ref_folder, ref_picked], REF_PREVIEW)
    load_images_btn.click(on_pick_images, ref_picked, [ref_picked, ref_picked_md, clear_picks_btn]).then(
        refresh_ref_preview, [ref_folder, ref_picked], REF_PREVIEW)
    clear_picks_btn.click(on_clear_picks, None, [ref_picked, ref_picked_md, clear_picks_btn]).then(
        refresh_ref_preview, [ref_folder, ref_picked], REF_PREVIEW)
    # blur/submit rather than change: .change fires per keystroke and would re-scan
    # the folder on every character typed into the path.
    ref_folder.blur(refresh_ref_preview, [ref_folder, ref_picked], REF_PREVIEW)
    ref_folder.submit(refresh_ref_preview, [ref_folder, ref_picked], REF_PREVIEW)
    ref_files.change(refresh_ref_preview, [ref_folder, ref_picked], REF_PREVIEW)
    ref_view.change(toggle_view, ref_view, REF_PREVIEW)

    load_cand_folder_btn.click(on_pick_folder, None, cand_folder).then(
        refresh_cand_preview, [cand_folder, cand_files, recursive], CAND_PREVIEW)
    cand_folder.blur(refresh_cand_preview, [cand_folder, cand_files, recursive], CAND_PREVIEW)
    cand_folder.submit(refresh_cand_preview, [cand_folder, cand_files, recursive], CAND_PREVIEW)
    cand_files.change(refresh_cand_preview, [cand_folder, cand_files, recursive], CAND_PREVIEW)
    recursive.change(refresh_cand_preview, [cand_folder, cand_files, recursive], CAND_PREVIEW)
    cand_view.change(toggle_view, cand_view, CAND_PREVIEW)

    build_btn.click(build_bank, inputs=[ref_folder, ref_picked, ref_cap], outputs=[bank_state, bank_info])
    clear_btn.click(clear_bank, inputs=[], outputs=[bank_state, bank_info])
    score_btn.click(score_candidates, inputs=[cand_folder, cand_files, recursive, bank_state],
                    outputs=[score_info, gallery, names_out, csv_out, keep_sel, path_map_state])
    view.change(toggle_view, view, [gallery, names_out])

    ds_create.click(create_dataset, inputs=[ds_name], outputs=[ds_pick, ds_info])
    ds_add.click(add_to_dataset, inputs=[keep_sel, path_map_state, ds_pick], outputs=[ds_info])

    harvest_btn.click(
        builder_harvest,
        inputs=[src_path, bank_state, framing, min_sim, min_face, min_sharp, fps, window, dedup,
                max_per_video, auto_rotate],
        outputs=[cand_gallery_b, builder_keep, builder_cands, builder_status])
    save_btn.click(builder_save, inputs=[builder_cands, builder_keep, bld_ds_pick, trigger_box, framing_cap],
                   outputs=[save_info])
    bld_ds_create.click(builder_create_dataset, inputs=[bld_ds_new], outputs=[bld_ds_pick, ds_pick])
    keep_all_btn.click(lambda cs: gr.update(value=[c["id"] for c in cs]), inputs=[builder_cands],
                       outputs=[builder_keep])
    keep_none_btn.click(lambda: gr.update(value=[]), inputs=[], outputs=[builder_keep])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    roots = drive_roots()
    if args.share or args.host not in ("127.0.0.1", "localhost"):
        # allowlisting every local drive (needed for the folder pickers/paths) is
        # only defensible on loopback -- see drive_roots()
        raise SystemExit("refusing to serve every local drive off-loopback; "
                         "drop --share and keep --host 127.0.0.1")
    log.info("starting FaceMatch on %s:%s (allowed roots: %s)", args.host, args.port, ", ".join(roots))
    get_analyzer()  # warm up now so provider/CUDA errors show at launch, not on first click
    demo.queue().launch(server_name=args.host, server_port=args.port, share=False, inbrowser=False,
                        allowed_paths=roots)


if __name__ == "__main__":
    main()
