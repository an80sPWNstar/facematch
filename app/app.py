"""FaceMatch -- Gradio wrapper around the ArcFace bank/score workflow.

Build a reference bank from photos of one person, then score candidate
photos against it. Reuses the buffalo_l + torch-DLL GPU init from
tools/bank_score.py, with CPU fallback so the app always comes up.
"""
import csv
import os
import re
import shutil
import sys
import tempfile
import threading
import warnings

warnings.filterwarnings("ignore")

TORCH_CUDA_DLLS = os.environ.get("FACEMATCH_CUDA_DLLS", "")  # dir holding CUDA/cuDNN DLLs for onnxruntime-gpu
# DLL dir must be on PATH before insightface/onnxruntime import
if os.path.isdir(TORCH_CUDA_DLLS):
    os.add_dll_directory(TORCH_CUDA_DLLS)
    os.environ["PATH"] = TORCH_CUDA_DLLS + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("FACEMATCH_GPU", "0"))

import cv2
import numpy as np
import gradio as gr

OUTLIER_CUT = 0.35
DEFAULT_THRESHOLD = 0.35
DATASETS_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")

_app = None
_provider = None
_lock = threading.Lock()
_infer_lock = threading.Lock()  # serialize inference: insightface isn't documented thread-safe


def get_faces(app, img):
    with _infer_lock:
        return app.get(img)


def get_analyzer():
    """Lazy singleton FaceAnalysis; CUDA first, CPU fallback."""
    global _app, _provider
    with _lock:
        if _app is not None:
            return _app
        import insightface
        try:
            a = insightface.app.FaceAnalysis(name="buffalo_l",
                                             providers=["CUDAExecutionProvider"])
            a.prepare(ctx_id=0, det_size=(640, 640))
            _provider = "CUDA"
        except Exception as e:
            print(f"CUDA init failed ({e!r}); falling back to CPU", file=sys.stderr)
            a = insightface.app.FaceAnalysis(name="buffalo_l",
                                             providers=["CPUExecutionProvider"])
            a.prepare(ctx_id=-1, det_size=(640, 640))
            _provider = "CPU"
        print(f"FaceMatch: analyzer ready on {_provider}", flush=True)
        _app = a
        return _app


def imread(path):
    try:
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def build_bank(files):
    if not files:
        return None, "Upload at least one reference image first."
    app = get_analyzer()
    embs, skipped = [], []
    for f in files:
        path = f.name if hasattr(f, "name") else f
        img = imread(path)
        if img is None:
            skipped.append(os.path.basename(path))
            continue
        faces = get_faces(app, img)
        if not faces:
            skipped.append(os.path.basename(path) + " (no face)")
            continue
        # biggest face per reference image
        faces.sort(key=lambda fc: -(fc.bbox[2] - fc.bbox[0]) * (fc.bbox[3] - fc.bbox[1]))
        embs.append(faces[0].normed_embedding)
    if not embs:
        return None, "No usable faces found in the uploaded reference images."
    E = np.stack(embs)
    ref = E.mean(0)
    ref /= np.linalg.norm(ref)
    # outlier rejection: drop members that don't match the initial centroid,
    # then rebuild the centroid from the survivors
    keep = (E @ ref) >= OUTLIER_CUT
    dropped = int((~keep).sum())
    if keep.sum() and not keep.all():
        E = E[keep]
        ref = E.mean(0)
        ref /= np.linalg.norm(ref)
    self_sims = E @ ref
    ceiling = float(self_sims.mean())
    bank = {"ref": ref, "count": int(len(E)), "dropped": dropped, "ceiling": ceiling}
    msg = (f"**Bank built** — {bank['count']} member(s), {dropped} outlier(s) dropped, "
           f"ceiling (mean self-similarity) {ceiling:.3f}. Running on {_provider}.")
    if skipped:
        msg += f"\n\nSkipped: {', '.join(skipped)}"
    return bank, msg


def annotate(img, face, sim, threshold):
    out = img.copy()
    x1, y1, x2, y2 = (int(v) for v in face.bbox[:4])
    color = (0, 200, 0) if sim >= threshold else (0, 0, 220)  # BGR
    thick = max(2, img.shape[1] // 400)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
    label = f"{sim:.3f}"
    scale = max(0.6, img.shape[1] / 1200)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    ty = y1 - 8 if y1 - th - 12 > 0 else y2 + th + 8
    cv2.rectangle(out, (x1, ty - th - 6), (x1 + tw + 6, ty + 6), color, -1)
    cv2.putText(out, label, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), thick, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def score_candidates(files, bank, threshold):
    empty_sel = gr.update(choices=[], value=[])
    if bank is None or "ref" not in (bank or {}):
        return [], [], None, "No reference bank yet — build one in the section above first.", empty_sel, {}
    if not files:
        return [], [], None, "Upload at least one candidate image.", empty_sel, {}
    app = get_analyzer()
    ref = bank["ref"]
    rows, gallery, notes = [], [], []
    path_map = {}  # scored filename -> original upload temp path, for dataset copies
    for f in files:
        path = f.name if hasattr(f, "name") else f
        name = os.path.basename(path)
        img = imread(path)
        if img is None:
            notes.append(f"{name}: unreadable, skipped")
            continue
        path_map[name] = path
        faces = get_faces(app, img)
        if not faces:
            rows.append([name, "NOFACE", 0])
            gallery.append((cv2.cvtColor(img, cv2.COLOR_BGR2RGB), f"{name} — no face"))
            continue
        # best-matching face vs the bank, not the biggest face
        sims = [float(fc.normed_embedding @ ref) for fc in faces]
        best_i = int(np.argmax(sims))
        best = sims[best_i]
        rows.append([name, round(best, 3), len(faces)])
        gallery.append((annotate(img, faces[best_i], best, threshold),
                        f"{name} — {best:.3f}"))
    rows.sort(key=lambda r: -(r[1] if isinstance(r[1], float) else -1))
    csv_path = None
    if rows:
        # purge result CSVs older than a day so a long-running app doesn't
        # accumulate them in %TEMP%
        import glob as _glob, time as _time
        for old in _glob.glob(os.path.join(tempfile.gettempdir(), "facematch_*.csv")):
            try:
                if _time.time() - os.path.getmtime(old) > 86400:
                    os.remove(old)
            except OSError:
                pass
        fd, csv_path = tempfile.mkstemp(suffix=".csv", prefix="facematch_")
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "similarity", "faces_found"])
            w.writerows(rows)
    status = f"Scored {len(rows)} image(s) against a {bank['count']}-member bank."
    if notes:
        status += "\n\n" + "\n".join(notes)
    keep_sel = gr.update(choices=[r[0] for r in rows], value=[])
    return rows, gallery, csv_path, status, keep_sel, path_map


def clear_bank():
    return None, ""


def list_datasets():
    if not os.path.isdir(DATASETS_BASE):
        return []
    return sorted(d for d in os.listdir(DATASETS_BASE)
                  if os.path.isdir(os.path.join(DATASETS_BASE, d)))


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
        msg += f" Not found (upload expired?): {', '.join(missing)}"
    return msg


def toggle_view(choice):
    show_thumbs = choice == "Thumbnails"
    return gr.update(visible=not show_thumbs), gr.update(visible=show_thumbs)


def preview_files(files):
    if not files:
        return []
    return [f.name if hasattr(f, "name") else f for f in files]


with gr.Blocks(title="FaceMatch") as demo:
    gr.Markdown(
        "# FaceMatch\n"
        "Builds an identity reference from photos you trust (the \"bank\"), then scores "
        "other photos against it to tell you how likely each one shows the same person. "
        "Scores are ArcFace cosine similarity: the same person typically scores above "
        "0.4, a different person below 0.3."
    )
    bank_state = gr.State(None)
    path_map_state = gr.State({})

    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            gr.Markdown("## Reference bank")
            ref_view = gr.Radio(["Filenames", "Thumbnails"], value="Filenames",
                                label="Upload view", interactive=True)
            ref_files = gr.File(label="Reference images (photos of the person)",
                                file_count="multiple", file_types=["image"], height=200)
            ref_preview = gr.Gallery(label="Uploaded references", columns=5,
                                     height=300, visible=False)
            with gr.Row():
                build_btn = gr.Button("Build bank", variant="primary", size="sm")
                clear_btn = gr.Button("Clear bank", size="sm")
            bank_info = gr.Markdown()
        with gr.Column(scale=1):
            gr.Markdown("## Score candidates")
            cand_view = gr.Radio(["Filenames", "Thumbnails"], value="Filenames",
                                 label="Upload view", interactive=True)
            cand_files = gr.File(label="Candidate images to score",
                                 file_count="multiple", file_types=["image"], height=200)
            cand_preview = gr.Gallery(label="Uploaded candidates", columns=5,
                                      height=300, visible=False)
            threshold = gr.Slider(0.0, 1.0, value=DEFAULT_THRESHOLD, step=0.01,
                                  label="Match threshold (box turns green at or above this)")
            with gr.Row():
                score_btn = gr.Button("Score", variant="primary", size="sm")
            score_info = gr.Markdown()

    results = gr.Dataframe(headers=["file", "similarity", "faces_found"],
                           label="Results (sorted by similarity)",
                           interactive=False, max_height=300)
    csv_out = gr.File(label="Download results as CSV", height=90)
    gallery = gr.Gallery(label="Annotated images", columns=5, height=300)

    gr.Markdown("## Datasets\nKeep the good matches: create a named dataset, tick the "
                "scored images you want, and add them. Files land in "
                "`webapp\\datasets\\<name>\\`.")
    with gr.Row():
        ds_name = gr.Textbox(label="New dataset name", scale=2)
        ds_create = gr.Button("Create dataset", size="sm", scale=1)
        ds_pick = gr.Dropdown(choices=list_datasets(), label="Dataset", scale=2)
    keep_sel = gr.CheckboxGroup(label="Scored images to keep (populated after a scoring run)",
                                choices=[])
    ds_add = gr.Button("Add selected to dataset", size="sm")
    ds_info = gr.Markdown()

    build_btn.click(build_bank, inputs=[ref_files], outputs=[bank_state, bank_info])
    clear_btn.click(clear_bank, inputs=[], outputs=[bank_state, bank_info])
    score_btn.click(score_candidates, inputs=[cand_files, bank_state, threshold],
                    outputs=[results, gallery, csv_out, score_info, keep_sel, path_map_state])
    ref_view.change(toggle_view, inputs=[ref_view], outputs=[ref_files, ref_preview])
    cand_view.change(toggle_view, inputs=[cand_view], outputs=[cand_files, cand_preview])
    ref_files.change(preview_files, inputs=[ref_files], outputs=[ref_preview])
    cand_files.change(preview_files, inputs=[cand_files], outputs=[cand_preview])
    ds_create.click(create_dataset, inputs=[ds_name], outputs=[ds_pick, ds_info])
    ds_add.click(add_to_dataset, inputs=[keep_sel, path_map_state, ds_pick],
                 outputs=[ds_info])


if __name__ == "__main__":
    get_analyzer()  # warm up now so provider/CUDA errors show at launch
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)
