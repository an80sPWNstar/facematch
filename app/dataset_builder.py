"""dataset_builder.py -- backend for FaceMatch's "Dataset Builder" tab.

Identity-gated face harvesting from photos AND videos with adjustable crop
context ("framing"). Given the app's reference-bank centroid, it walks the
supplied sources (files, folders, globs), keeps only frames where THAT person's
face matches the bank and is large + sharp enough, and writes crops at the
chosen framing to a per-run temp dir for gallery review. Selected crops are
then copied into a training dataset with optional caption sidecars.

The harvest core is ported from tools/face_harvest.py -- same gates, same
rotation sweep, same video best-of-window/dedup logic. The differences here:
  - the FaceAnalysis instance is INJECTED (the app owns GPU/env setup) and
    every analyzer.get(img) is serialized through the app's infer_lock;
  - crops are written to temp for review rather than straight to a dataset;
  - "framing" replaces the raw --margin flag (see FRAMINGS).

Face selection is by BEST MATCH to the reference, never biggest face, so
frames with other people crop the right person or nothing.
"""
import glob
import os
import shutil
import tempfile
import time
import uuid

import cv2
import numpy as np

# Extension sets ported verbatim from face_harvest.py.
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".3gp", ".mts", ".wmv", ".mpg", ".mpeg"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Framing -> crop margin (side = margin * face size) + caption fragment.
# Margins match the head-and-shoulders(2.5)/full-body(7.0) feel of the CLI's
# --margin; captions are appended to the trigger word in save_selected.
FRAMINGS = {
    "Face (tight)": {"margin": 1.6, "caption": "close-up portrait"},
    "Portrait":     {"margin": 2.5, "caption": "portrait"},
    "Half body":    {"margin": 4.0, "caption": "half body shot"},
    "Full person":  {"margin": 7.0, "caption": "full body shot"},
}

# Rotation sweep: phones rotated mid-capture store sideways frames and ArcFace
# is not rotation-invariant, so with auto_rotate we test all four orientations.
ROTATIONS = [None, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180]

_TEMP_PREFIX = "facematch_builder_"


# ---------------------------------------------------------------------------
# Image IO / geometry (ported from face_harvest.py)
# ---------------------------------------------------------------------------

def load_image(p):
    """Read an image honoring EXIF orientation (cv2.imread ignores it) and
    tolerating non-ASCII paths (Windows)."""
    try:
        from PIL import Image, ImageOps
        im = Image.open(p)
        im = ImageOps.exif_transpose(im).convert("RGB")
        return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    except Exception:
        try:
            data = np.fromfile(p, dtype=np.uint8)
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            return None


def best_match(analyzer, infer_lock, img, ref):
    """Return (face, sim) for the detected face most similar to the reference,
    or (None, -1.0). Inference is serialized through infer_lock -- insightface
    is not documented thread-safe and the app runs a single analyzer."""
    with infer_lock:
        fs = analyzer.get(img)
    if not fs:
        return None, -1.0
    sims = [float(f.normed_embedding @ ref) for f in fs]
    i = int(np.argmax(sims))
    return fs[i], sims[i]


def sharpness(img, bbox):
    """Laplacian variance of the face region, resolution-normalized to 256px."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(x1, 0), max(y1, 0)
    face = img[y1:y2, x1:x2]
    if face.size == 0:
        return 0.0
    face = cv2.resize(face, (256, 256))
    return float(cv2.Laplacian(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def make_crop(img, bbox, margin):
    """Square crop centered on the face, side = margin * face size, shifted up
    a little to keep hair, clamped inside the frame.

    Ported as-is including the min(side, W, H) clamp: for large margins ("Full
    person", 7.0) the square is capped to the frame's short side. Kept
    deliberately -- training buckets handle aspect and consistency beats
    cleverness here."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = bbox
    side = margin * max(x2 - x1, y2 - y1)
    side = min(side, W, H)
    half = side / 2.0
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0 - 0.10 * side  # bias upward: include top of head
    cx = min(max(cx, half), W - half)
    cy = min(max(cy, half), H - half)
    a, b, s = int(round(cx - half)), int(round(cy - half)), int(round(side))
    return img[b:b + s, a:a + s]


def achieved_framing(img_shape, bbox, margin):
    """Framing label for what the crop ACTUALLY contains. The requested margin
    gets clamped at the frame edge (a close selfie can't yield a half-body
    crop), and a caption that claims more context than the pixels show would
    teach the wrong association -- so captions are derived from the achieved
    crop-to-face ratio, not the requested framing."""
    H, W = img_shape[:2]
    face = max(bbox[2] - bbox[0], bbox[3] - bbox[1], 1.0)
    side = min(margin * face, W, H)
    ratio = side / face
    if ratio < 2.05:
        return "Face (tight)"
    if ratio < 3.25:
        return "Portrait"
    if ratio < 5.5:
        return "Half body"
    return "Full person"


def sanitize(stem):
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in stem)[:60]


def _imwrite_unicode(path, img, ext, params=None):
    """cv2.imencode + tofile: the unicode-safe write pattern for Windows paths."""
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        return False
    buf.tofile(path)
    return True


# ---------------------------------------------------------------------------
# Input gathering
# ---------------------------------------------------------------------------

def _gather_inputs(sources):
    """Expand a source path (file, folder recursed, or glob) into sorted,
    deduped (videos, images) lists."""
    paths = []
    if os.path.isdir(sources):
        for root, _, files in os.walk(sources):
            paths += [os.path.join(root, f) for f in files]
    elif os.path.isfile(sources):
        paths.append(sources)
    else:
        paths += glob.glob(sources, recursive=True)
    vids, imgs = [], []
    for p in sorted(set(paths)):
        ext = os.path.splitext(p)[1].lower()
        if ext in VIDEO_EXT:
            vids.append(p)
        elif ext in IMAGE_EXT:
            imgs.append(p)
    return vids, imgs


# ---------------------------------------------------------------------------
# Gates + rotation sweep (ported from face_harvest.evaluate)
# ---------------------------------------------------------------------------

class _Stats:
    def __init__(self):
        self.sampled = self.no_match = self.small = self.blurry = self.dup = self.kept = 0

    def line(self):
        return (f"kept {self.kept} / {self.sampled} analyzed "
                f"(rejected: no-match {self.no_match}, small {self.small}, "
                f"blurry {self.blurry}, dup {self.dup})")


def _evaluate(analyzer, infer_lock, img, ref, min_sim, min_face, min_sharp, auto_rotate, st):
    """Run all gates on one frame. Returns (face, sim, sharp, oriented_frame) or
    None. With auto_rotate the frame is tested at 0/90/270/180 and argmax sim
    wins -- NO short-circuit: a sharp 4K face can score 0.6+ even stored
    sideways, so every orientation is scored before choosing. The returned
    frame is the orientation that matched -- crop from it, not the original."""
    st.sampled += 1
    best = None  # (sim, face, oriented_frame)
    for rot in (ROTATIONS if auto_rotate else [None]):
        frame = img if rot is None else cv2.rotate(img, rot)
        face, sim = best_match(analyzer, infer_lock, frame, ref)
        if face is not None and (best is None or sim > best[0]):
            best = (sim, face, frame)
        # no short-circuit (see docstring): all orientations scored, argmax wins
    if best is None or best[0] < min_sim:
        st.no_match += 1
        return None
    sim, face, frame = best
    x1, y1, x2, y2 = face.bbox
    if min(x2 - x1, y2 - y1) < min_face:
        st.small += 1
        return None
    sh = sharpness(frame, face.bbox)
    if sh < min_sharp:
        st.blurry += 1
        return None
    return face, sim, sh, frame


# ---------------------------------------------------------------------------
# Candidate emission
# ---------------------------------------------------------------------------

def _write_candidate(temp_dir, cand_id, crop, sim, sharp, source, time_s, framing):
    """Write full-quality PNG crop + a downscaled JPEG preview; return the dict."""
    crop_path = os.path.join(temp_dir, f"{cand_id}.png")
    prev_path = os.path.join(temp_dir, f"{cand_id}_prev.jpg")
    _imwrite_unicode(crop_path, crop, ".png")

    # Preview: max side 512, JPEG q85 (gallery only, never the training copy).
    h, w = crop.shape[:2]
    scale = 512.0 / max(h, w) if max(h, w) > 512 else 1.0
    prev = cv2.resize(crop, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_AREA) if scale < 1.0 else crop
    _imwrite_unicode(prev_path, prev, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 85])

    return {"id": cand_id, "path": prev_path, "crop_path": crop_path,
            "sim": float(sim), "sharp": float(sharp), "source": source,
            "time_s": time_s, "framing": framing}


def _harvest_image(analyzer, infer_lock, p, ref, margin, framing, temp_dir,
                   min_sim, min_face, min_sharp, auto_rotate, st, cand_id):
    img = load_image(p)
    if img is None:
        return None
    r = _evaluate(analyzer, infer_lock, img, ref, min_sim, min_face, min_sharp, auto_rotate, st)
    if r is None:
        return None
    face, sim, sh, oriented = r
    crop = make_crop(oriented, face.bbox, margin)
    st.kept += 1
    fr = achieved_framing(oriented.shape, face.bbox, margin)
    return _write_candidate(temp_dir, cand_id, crop, sim, sh, p, None, fr)


def _harvest_video(analyzer, infer_lock, p, ref, margin, framing, temp_dir,
                   min_sim, min_face, min_sharp, fps, window, dedup,
                   max_per_video, auto_rotate, st, next_id, progress_cb):
    """cap.grab()/retrieve() decimation to `fps`, best-of-window by SHARPNESS,
    embedding dedup vs already-kept crops, max_per_video cap, final flush.
    Ported from face_harvest.harvest_video. Emits candidate dicts via the
    _emit closure. `next_id` is a callable returning the next unique id."""
    cap = cv2.VideoCapture(p)
    if not cap.isOpened():
        return []
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(native / fps))
    out = []
    kept_embs = []
    best = None  # (sharp, sim, emb, crop, t)
    cur_win = -1
    vid_kept_start = st.kept

    def flush():
        nonlocal best
        if best is None:
            return
        sh, sim, emb, crop, t, fr = best
        best = None
        if dedup > 0 and kept_embs and max(float(emb @ k) for k in kept_embs) >= dedup:
            st.dup += 1
            return
        st.kept += 1
        kept_embs.append(emb)
        out.append(_write_candidate(temp_dir, next_id(), crop, sim, sh, p, t, fr))

    idx = 0
    while True:
        if not cap.grab():
            break
        if idx % step == 0:
            if max_per_video and (st.kept - vid_kept_start) >= max_per_video:
                break
            ok, frame = cap.retrieve()
            if ok and frame is not None:
                t = idx / native
                win = int(t // window)
                if win != cur_win:
                    flush()
                    cur_win = win
                    if progress_cb:
                        progress_cb(0, 0, f"{os.path.basename(p)} @ {t:.0f}s (kept {st.kept - vid_kept_start})")
                r = _evaluate(analyzer, infer_lock, frame, ref, min_sim, min_face,
                              min_sharp, auto_rotate, st)
                if r is not None:
                    face, sim, sh, oriented = r
                    if best is None or sh > best[0]:
                        best = (sh, sim, face.normed_embedding,
                                make_crop(oriented, face.bbox, margin), t,
                                achieved_framing(oriented.shape, face.bbox, margin))
        idx += 1
    flush()
    cap.release()
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def harvest(sources, ref_vec, analyzer, infer_lock, *, framing="Portrait",
            min_sim=0.35, min_face=200, min_sharp=45.0, fps=2.0, window=2.0,
            dedup=0.95, max_per_video=0, auto_rotate=True,
            progress_cb=None):
    """Harvest identity-gated crops from `sources` at the chosen `framing`.

    Returns (candidates: list[dict], stats_line: str). Each candidate carries a
    temp preview JPEG (`path`) and a full-quality crop PNG (`crop_path`); pass
    the selected `id`s to save_selected to copy them into a dataset.

    progress_cb(done, total, msg) is called once per photo and per sampled
    video window (videos report done/total as 0/0 since window count is
    unknown up front)."""
    margin = FRAMINGS.get(framing, FRAMINGS["Portrait"])["margin"]
    vids, imgs = _gather_inputs(sources)

    temp_dir = os.path.join(tempfile.gettempdir(),
                            f"{_TEMP_PREFIX}{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}")
    os.makedirs(temp_dir, exist_ok=True)

    st = _Stats()
    candidates = []
    total = len(imgs) + len(vids)
    done = 0

    # Unique, filesystem-safe, stable-within-run ids.
    _counter = {"n": 0}

    def next_id():
        _counter["n"] += 1
        return f"c{_counter['n']:05d}"

    for p in imgs:
        cand = _harvest_image(analyzer, infer_lock, p, ref_vec, margin, framing,
                              temp_dir, min_sim, min_face, min_sharp, auto_rotate,
                              st, next_id())
        if cand is not None:
            candidates.append(cand)
        done += 1
        if progress_cb:
            progress_cb(done, total, f"photo {os.path.basename(p)}")

    for p in vids:
        vid_cands = _harvest_video(analyzer, infer_lock, p, ref_vec, margin, framing,
                                   temp_dir, min_sim, min_face, min_sharp, fps, window,
                                   dedup, max_per_video, auto_rotate, st, next_id,
                                   progress_cb)
        candidates.extend(vid_cands)
        done += 1
        if progress_cb:
            progress_cb(done, total, f"video {os.path.basename(p)} (+{len(vid_cands)})")

    return candidates, st.line()


def save_selected(candidates, selected_ids, dataset_dir, trigger="",
                  framing_caption=True):
    """Copy each selected candidate's full-quality PNG into `dataset_dir` with a
    clean name `{sanitized_source_stem}_{tag}_s{sim:.3f}.png` (collision ->
    `_1`, `_2`, like app.py's add_to_dataset). When `trigger` is non-empty,
    write a sibling `.txt` caption: `"{trigger}, {framing caption}"` (or just
    `"{trigger}"` when framing_caption is False). Returns a markdown status."""
    selected = set(selected_ids or [])
    by_id = {c["id"]: c for c in candidates if c["id"] in selected}
    if not selected:
        return "No candidates selected."
    if not by_id:
        return "Selected candidates not found (harvest expired?)."
    os.makedirs(dataset_dir, exist_ok=True)

    copied, missing = 0, []
    for cid in selected_ids:
        c = by_id.get(cid)
        if c is None:
            continue
        src = c.get("crop_path")
        if not src or not os.path.isfile(src):
            missing.append(cid)
            continue
        stem = sanitize(os.path.splitext(os.path.basename(c["source"]))[0])
        # tag: video timestamp or "img" for photos, matching the CLI's flavor
        tag = f"t{c['time_s']:08.1f}" if c.get("time_s") is not None else "img"
        base = f"{stem}_{tag}_s{c['sim']:.3f}"
        dest = os.path.join(dataset_dir, base + ".png")
        n = 0
        while os.path.exists(dest):
            n += 1
            dest = os.path.join(dataset_dir, f"{base}_{n}.png")
        shutil.copy2(src, dest)
        copied += 1
        if trigger:
            caption = trigger
            if framing_caption:
                frag = FRAMINGS.get(c.get("framing", ""), {}).get("caption")
                if frag:
                    caption = f"{trigger}, {frag}"
            txt = os.path.splitext(dest)[0] + ".txt"
            with open(txt, "w", encoding="utf-8") as fh:
                fh.write(caption)

    msg = f"Copied {copied} crop(s) to **{dataset_dir}**."
    if trigger:
        msg += f" Captions written with trigger `{trigger}`."
    if missing:
        msg += f" Missing (temp expired?): {', '.join(missing)}"
    return msg


def cleanup_temp(max_age_s=86400):
    """Purge facematch_builder_* temp dirs older than max_age_s so long-running
    app sessions don't accumulate crop dirs in %TEMP%."""
    root = tempfile.gettempdir()
    now = time.time()
    for name in os.listdir(root):
        if not name.startswith(_TEMP_PREFIX):
            continue
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        try:
            if now - os.path.getmtime(path) > max_age_s:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass
