"""bank_score.py -- build an ArcFace reference bank from one folder, score other images against it.

  python bank_score.py --refs BANK_DIR "glob_or_dir" [more...] [--csv out.csv] [--gpu]

Unlike score_gen.py the reference folder is an argument, so per-era banks work.
Picks the best-matching face per image (not the biggest). Prints a table and
optionally writes CSV. Never overwrites an existing CSV.
"""
import argparse, csv, glob, os, sys, warnings
import cv2, numpy as np
warnings.filterwarnings("ignore")

TORCH_CUDA_DLLS = os.environ.get("FACEMATCH_CUDA_DLLS", "")


def analyzer(gpu=False):
    if gpu:
        os.add_dll_directory(TORCH_CUDA_DLLS)
        os.environ["PATH"] = TORCH_CUDA_DLLS + os.pathsep + os.environ.get("PATH", "")
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    import insightface
    a = insightface.app.FaceAnalysis(name="buffalo_l",
                                     providers=["CUDAExecutionProvider" if gpu else "CPUExecutionProvider"])
    a.prepare(ctx_id=0 if gpu else -1, det_size=(640, 640))
    return a


def imread(p):
    try:
        return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def gather(items):
    files = []
    for item in items:
        if os.path.isdir(item):
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
                files += glob.glob(os.path.join(item, ext))
        else:
            files += glob.glob(item)
    return sorted(set(files))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--refs", required=True)
    ap.add_argument("--csv")
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    app = analyzer(args.gpu)
    embs = []
    for p in gather([args.refs]):
        img = imread(p)
        if img is None:
            continue
        fs = app.get(img)
        if fs:
            fs.sort(key=lambda f: -(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            embs.append(fs[0].normed_embedding)
    if not embs:
        sys.exit(f"FATAL: no faces in refs {args.refs}")
    E = np.stack(embs)
    ref = E.mean(0)
    ref /= np.linalg.norm(ref)
    # outlier rejection: the bank builder takes the biggest face per ref image,
    # which can be a different person -- drop members that don't match the
    # centroid, then rebuild it from the survivors
    keep = (E @ ref) >= 0.35
    if keep.sum() and not keep.all():
        E = E[keep]
        ref = E.mean(0)
        ref /= np.linalg.norm(ref)
        print(f"BANK: dropped {(~keep).sum()} outlier member(s) below 0.35")
    ss = E @ ref
    print(f"BANK {args.refs}: {len(E)} faces | ceiling {ss.mean():.3f} (min {ss.min():.3f})")

    rows = []
    for p in gather(args.inputs):
        img = imread(p)
        fs = app.get(img) if img is not None else []
        if not fs:
            rows.append((os.path.basename(p), None))
            continue
        rows.append((os.path.basename(p), max(float(f.normed_embedding @ ref) for f in fs)))
    rows.sort(key=lambda r: -(r[1] if r[1] is not None else -1))
    scored = [r[1] for r in rows if r[1] is not None]
    for name, s in rows:
        print(f"{'NOFACE' if s is None else f'{s:7.3f}'}  {name}")
    if scored:
        a = np.array(scored)
        print(f"\nn={len(a)}  mean {a.mean():.3f}  median {np.median(a):.3f}  "
              f"min {a.min():.3f}  max {a.max():.3f}  >=0.60: {(a >= 0.60).sum()}  >=0.65: {(a >= 0.65).sum()}")
    if args.csv:
        if os.path.exists(args.csv):
            sys.exit(f"refusing to overwrite {args.csv}")
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "sim"])
            w.writerows(rows)
        print(f"csv: {args.csv}")


if __name__ == "__main__":
    main()
