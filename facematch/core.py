"""facematch.core -- the single ArcFace scoring implementation.

Shared by the Gradio app (app/app.py), the CLI (facematch.cli), and the
pipeline tools (tools/*.py). Cosine similarity on L2-normalised ArcFace
(buffalo_l, via insightface) embeddings throughout. Rules of thumb for this
model: >0.5 same person, 0.35-0.5 plausible, <0.28 different person -- but
always read a score against a reference bank's own self-similarity (its
"ceiling"), never against 1.0.

GPU is opt-in and machine-specific by nature, so it is controlled entirely by
environment variables with safe (CPU-friendly) defaults:

    FACEMATCH_CUDA_DLLS   directory holding CUDA 12 + cuDNN 9 DLLs (a torch
                          install's ``torch\\lib`` works). Must be set before
                          onnxruntime is imported to matter, which is why
                          analyzer() calls os.add_dll_directory() itself.
    FACEMATCH_GPU         which device to expose, PCI bus order (default "0")
"""
import glob
import os
import sys
import threading
import warnings

warnings.filterwarnings("ignore")

import cv2
import numpy as np

# A reference image's biggest detected face can be the wrong person (a group
# photo, a bad crop). Members of a reference set that don't match the initial
# centroid at least this well are dropped and the centroid rebuilt from the
# survivors -- see _bank_from_embeddings.
OUTLIER_CUT = 0.35

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".3gp", ".mts", ".wmv", ".mpg", ".mpeg"}

# insightface is not documented thread-safe. Every caller in this codebase
# (the Gradio app, the dataset builder's harvest workers, the CLI) serializes
# analyzer.get() calls through this one process-wide lock.
INFER_LOCK = threading.Lock()


def _setup_cuda_dlls():
    dll_dir = os.environ.get("FACEMATCH_CUDA_DLLS", "")
    if dll_dir and os.path.isdir(dll_dir):
        os.add_dll_directory(dll_dir)
        os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")


def analyzer(gpu: bool = False):
    """Build a buffalo_l insightface FaceAnalysis.

    With gpu=True, CUDA is tried first and falls back to CPU on any failure
    (missing DLLs, no GPU, driver mismatch) so callers always get a usable
    analyzer -- they never need their own try/except around this.
    """
    import insightface
    if gpu:
        _setup_cuda_dlls()
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("FACEMATCH_GPU", "0"))
        try:
            a = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
            a.prepare(ctx_id=0, det_size=(640, 640))
            return a
        except Exception as e:
            print(f"CUDA init failed ({e!r}); falling back to CPU", file=sys.stderr)
    a = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    a.prepare(ctx_id=-1, det_size=(640, 640))
    return a


def provider_of(app):
    """Best-effort "CUDA" / "CPU" label for an analyzer built by analyzer(),
    for status messages. Never raises."""
    try:
        providers = next(iter(app.models.values())).session.get_providers()
        return "CUDA" if "CUDAExecutionProvider" in providers else "CPU"
    except Exception:
        return "unknown"


def imread(path):
    """cv2.imread chokes on non-ASCII paths on Windows; go through numpy."""
    try:
        return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def faces(app, image_path):
    """All faces detected in `image_path`, biggest first. Empty list if the
    file can't be read or no face is found."""
    img = imread(image_path)
    if img is None:
        return []
    with INFER_LOCK:
        fs = app.get(img)
    return sorted(fs, key=lambda f: -(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def gather(items):
    """Expand a list of directories / files / glob patterns into a sorted,
    deduped list of image paths. A directory yields its top-level images
    (non-recursive); anything else is passed straight to glob()."""
    files = []
    for item in items:
        if os.path.isdir(item):
            for ext in IMAGE_EXT:
                files += glob.glob(os.path.join(item, f"*{ext}"))
        else:
            files += glob.glob(item)
    return sorted(set(files))


def _bank_from_embeddings(embs, of):
    """Shared mean/outlier-rejection/ceiling math. `of` is the number of
    source files a caller attempted (for the stats' "of" field); `embs` is
    the list of per-file embeddings actually recovered."""
    if not embs:
        raise ValueError("no usable faces found")
    E = np.stack(embs)
    mean = E.mean(0)
    mean /= np.linalg.norm(mean)
    keep = (E @ mean) >= OUTLIER_CUT
    dropped = int((~keep).sum())
    if keep.sum() and not keep.all():
        E = E[keep]
        mean = E.mean(0)
        mean /= np.linalg.norm(mean)
    self_sim = E @ mean
    stats = {
        "n": int(len(E)),
        "of": of,
        "dropped": dropped,
        "self_mean": float(self_sim.mean()),
        "self_min": float(self_sim.min()),
    }
    return mean, stats


def bank_from_paths(app, paths):
    """Reference bank (mean embedding + stats) built from an explicit list of
    image paths, e.g. ad hoc uploads that don't live in one folder. Biggest
    face per image. See reference_bank() for the folder-based convenience
    wrapper and the stats field meanings."""
    embs = [fs[0].normed_embedding for fs in (faces(app, p) for p in paths) if fs]
    return _bank_from_embeddings(embs, of=len(paths))


def reference_bank(app, folder, limit=None):
    """Mean embedding over every image in `folder`, plus the spread within it
    (the ceiling -- two photos of the same person do not score 1.0 against
    each other, so nothing else should be expected to either).

    Outlier rejection: images whose biggest face doesn't match the initial
    centroid (a group photo, a bad crop) are dropped and the centroid rebuilt
    from the survivors.

    Returns (mean_embedding, stats) where stats has:
      n          reference faces kept after outlier rejection
      of         reference images that had a usable face at all
      dropped    outliers removed
      self_mean  mean cosine similarity of kept members to the final mean
      self_min   the weakest kept member's similarity to the final mean
    """
    files = gather([folder])
    if limit:
        files = files[:limit]
    return bank_from_paths(app, files)


def score_image(app, path, bank):
    """Every face detected in `path`, scored against `bank` (a mean
    embedding, or the (mean, stats) tuple reference_bank() returns).

    Returns a list of {"bbox": [x1, y1, x2, y2], "sim": float}, biggest face
    first, empty if no face was detected. Deliberately every face, not just
    the biggest or best-matching one -- that's how a character LoRA painting
    itself onto a bystander shows up.
    """
    mean = bank[0] if isinstance(bank, tuple) else bank
    return [
        {"bbox": [float(v) for v in f.bbox[:4]], "sim": float(f.normed_embedding @ mean)}
        for f in faces(app, path)
    ]


def _summarize(name, face_scores):
    sims = [f["sim"] for f in face_scores]
    ranked = sorted(sims, reverse=True)
    return {
        "file": name,
        "best": ranked[0] if ranked else None,
        "second_best": ranked[1] if len(ranked) > 1 else None,
        "n_faces": len(face_scores),
        "faces": face_scores,
    }


def score_folder(app, folder, bank):
    """Score every image in `folder` against `bank`. One row per image (see
    _summarize for the shape); NOFACE images still get a row with best=None."""
    return [_summarize(os.path.basename(p), score_image(app, p, bank)) for p in gather([folder])]


def score_video(app, video_path, bank, every_n=15, max_frames=10):
    """Sample up to `max_frames` frames (one every `every_n` decoded frames)
    from `video_path` and score each against `bank`. One row per scored
    frame, same shape as score_folder's rows plus "frame" (index) and
    "time_s"."""
    mean = bank[0] if isinstance(bank, tuple) else bank
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows = []
    idx = 0
    try:
        while len(rows) < max_frames:
            if not cap.grab():
                break
            if idx % every_n == 0:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    with INFER_LOCK:
                        fs = app.get(frame)
                    fs = sorted(fs, key=lambda f: -(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                    face_scores = [
                        {"bbox": [float(v) for v in f.bbox[:4]], "sim": float(f.normed_embedding @ mean)}
                        for f in fs
                    ]
                    row = _summarize(f"frame{idx:06d}", face_scores)
                    row["frame"] = idx
                    row["time_s"] = idx / fps
                    rows.append(row)
            idx += 1
    finally:
        cap.release()
    return rows
