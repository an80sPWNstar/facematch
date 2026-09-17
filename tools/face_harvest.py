r"""face_harvest.py -- harvest training crops of ONE identity from videos and photos.

Builds an ArcFace (buffalo_l) reference embedding from a folder of real photos,
then walks input videos/images and keeps only frames where THAT person's face:
  - matches the reference bank  (--min-sim gate, identity check)
  - is large enough in pixels   (--min-face gate)
  - is sharp                    (--min-sharp gate, Laplacian variance)
Video frames are additionally best-of-windowed (sharpest frame per --window
seconds) and embedding-deduped (--dedup) so you get variety, not 400 copies of
the same pose. Crops are centered on the face with --margin headroom and saved
with sim + sharpness in the filename, plus a per-run manifest CSV.

Face selection is by BEST MATCH to the reference, not biggest face, so frames
containing other people crop the right person or nothing.

Run with the isolated scoring venv (CPU, roughly 0.2-0.5 s per analyzed frame):

  python face_harvest.py ^
      "path\to\video.mp4" "path\to\photos" --refs path\to\reference_photos --out path\to\output

Then quality-cull the output with bank_score.py.
"""
import argparse, csv, glob, os, sys, time, warnings
import cv2, numpy as np
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from facematch import core

VIDEO_EXT = core.VIDEO_EXT
IMAGE_EXT = core.IMAGE_EXT


def analyzer(gpu=False):
    return core.analyzer(gpu)


def load_image(p):
    """Read an image honoring EXIF orientation (cv2.imread ignores it) and
    tolerating non-ASCII paths."""
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


def build_bank(app, folder):
    """Reference mean embedding for `folder`, via facematch.core (outlier
    rejection included -- see core.reference_bank). Returns (mean, n_kept,
    ceiling) to match this script's original call site."""
    try:
        mean, stats = core.reference_bank(app, folder)
    except ValueError:
        sys.exit(f"FATAL: no faces found in reference folder {folder}")
    if stats["dropped"]:
        print(f"REFERENCE: dropped {stats['dropped']} outlier member(s) below {core.OUTLIER_CUT}")
    return mean, stats["n"], stats["self_mean"]


def best_match(app, img, ref):
    """Return (face, sim) for the detected face most similar to the reference,
    or (None, best_sim_seen)."""
    fs = app.get(img)
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
    a little to keep hair, clamped inside the frame."""
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


def sanitize(stem):
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in stem)[:60]


def save_crop(out_dir, stem, tag, sim, sharp, crop, fmt):
    name = f"{stem}_{tag}_s{sim:.3f}_q{int(sharp):04d}.{fmt}"
    path = os.path.join(out_dir, name)
    n = 1
    while os.path.exists(path):
        path = os.path.join(out_dir, f"{stem}_{tag}_s{sim:.3f}_q{int(sharp):04d}_{n}.{fmt}")
        n += 1
    ok, buf = cv2.imencode(f".{fmt}", crop,
                           [cv2.IMWRITE_JPEG_QUALITY, 97] if fmt == "jpg" else [])
    if not ok:
        return None
    buf.tofile(path)
    return path


def gather_inputs(inputs):
    vids, imgs, skipped = [], [], []
    paths = []
    for item in inputs:
        if os.path.isdir(item):
            for root, _, files in os.walk(item):
                paths += [os.path.join(root, f) for f in files]
        elif os.path.isfile(item):
            paths.append(item)
        else:
            paths += glob.glob(item)
    for p in sorted(set(paths)):
        ext = os.path.splitext(p)[1].lower()
        if ext in VIDEO_EXT:
            vids.append(p)
        elif ext in IMAGE_EXT:
            imgs.append(p)
        else:
            skipped.append(p)
    return vids, imgs, skipped


class Stats:
    def __init__(self):
        self.sampled = self.no_match = self.small = self.blurry = self.dup = self.kept = 0

    def line(self):
        return (f"kept {self.kept} / {self.sampled} analyzed "
                f"(rejected: no-match {self.no_match}, small {self.small}, "
                f"blurry {self.blurry}, dup {self.dup})")


ROTATIONS = [None, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180]


def evaluate(app, img, ref, args, st):
    """Run all gates on one frame. Returns (face, sim, sharp, frame) or None.
    With --auto-rotate, a frame that fails the sim gate upright is retried at
    90/270/180 (phones rotated mid-video store sideways frames, and ArcFace is
    not rotation-invariant). The returned frame is the orientation that matched
    -- crop from it, not from the original."""
    st.sampled += 1
    best = None  # (sim, face, oriented_frame)
    for rot in (ROTATIONS if args.auto_rotate else [None]):
        frame = img if rot is None else cv2.rotate(img, rot)
        face, sim = best_match(app, frame, ref)
        if face is not None and (best is None or sim > best[0]):
            best = (sim, face, frame)
        # no short-circuit: a sharp 4K face can score 0.6+ even stored sideways,
        # so every orientation gets tested and argmax wins
    if best is None or best[0] < args.min_sim:
        st.no_match += 1
        return None
    sim, face, frame = best
    x1, y1, x2, y2 = face.bbox
    if min(x2 - x1, y2 - y1) < args.min_face:
        st.small += 1
        return None
    sh = sharpness(frame, face.bbox)
    if sh < args.min_sharp:
        st.blurry += 1
        return None
    return face, sim, sh, frame


def harvest_image(app, p, ref, args, writer, st):
    img = load_image(p)
    if img is None:
        print(f"  UNREADABLE {p}")
        return
    r = evaluate(app, img, ref, args, st)
    if r is None:
        return
    face, sim, sh, img = r
    crop = make_crop(img, face.bbox, args.margin)
    out = save_crop(args.out, sanitize(os.path.splitext(os.path.basename(p))[0]),
                    "img", sim, sh, crop, args.format)
    if out:
        st.kept += 1
        writer.writerow(["image", p, "", f"{sim:.4f}", f"{sh:.1f}",
                         int(min(face.bbox[2] - face.bbox[0], face.bbox[3] - face.bbox[1])),
                         crop.shape[1], crop.shape[0], out])


def harvest_video(app, p, ref, args, writer, st):
    cap = cv2.VideoCapture(p)
    if not cap.isOpened():
        print(f"  UNREADABLE {p}")
        return
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(native / args.fps))
    stem = sanitize(os.path.splitext(os.path.basename(p))[0])
    kept_embs = []
    best = None  # (sharp, sim, emb, crop, t, face_px)
    cur_win = -1
    vid_kept_start = st.kept

    def flush():
        nonlocal best
        if best is None:
            return
        sh, sim, emb, crop, t, face_px = best
        best = None
        if args.dedup > 0 and kept_embs and max(float(emb @ k) for k in kept_embs) >= args.dedup:
            st.dup += 1
            return
        out = save_crop(args.out, stem, f"t{t:08.1f}", sim, sh, crop, args.format)
        if out:
            st.kept += 1
            kept_embs.append(emb)
            writer.writerow(["video", p, f"{t:.1f}", f"{sim:.4f}", f"{sh:.1f}",
                             face_px, crop.shape[1], crop.shape[0], out])

    idx = 0
    while True:
        if not cap.grab():
            break
        if idx % step == 0:
            if args.max_per_video and (st.kept - vid_kept_start) >= args.max_per_video:
                break
            ok, frame = cap.retrieve()
            if ok and frame is not None:
                t = idx / native
                win = int(t // args.window)
                if win != cur_win:
                    flush()
                    cur_win = win
                r = evaluate(app, frame, ref, args, st)
                if r is not None:
                    face, sim, sh, oriented = r
                    if best is None or sh > best[0]:
                        best = (sh, sim, face.normed_embedding,
                                make_crop(oriented, face.bbox, args.margin), t,
                                int(min(face.bbox[2] - face.bbox[0], face.bbox[3] - face.bbox[1])))
        idx += 1
    flush()
    cap.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="video/image files, folders (recursive), or globs")
    ap.add_argument("--out", required=True, help="output folder for crops + manifest")
    ap.add_argument("--refs", required=True, help="reference photo folder")
    ap.add_argument("--min-sim", type=float, default=0.35,
                    help="identity gate vs reference bank (default 0.35; this says 'it is her', NOT quality — cull quality afterward)")
    ap.add_argument("--min-face", type=int, default=200, help="min face bbox side in source px (default 200)")
    ap.add_argument("--min-sharp", type=float, default=45.0,
                    help="min Laplacian variance on the 256px-normalized face (default 45)")
    ap.add_argument("--fps", type=float, default=2.0, help="video sampling rate, frames analyzed per second (default 2)")
    ap.add_argument("--window", type=float, default=2.0, help="keep the sharpest passing frame per this many seconds (default 2)")
    ap.add_argument("--dedup", type=float, default=0.95,
                    help="skip a video frame whose embedding matches an already-kept frame >= this (default 0.95, 0 disables)")
    ap.add_argument("--margin", type=float, default=2.0, help="crop side = margin x face size (default 2.0 = head+shoulders)")
    ap.add_argument("--max-per-video", type=int, default=0, help="cap keeps per video (0 = unlimited)")
    ap.add_argument("--format", choices=["png", "jpg"], default="png")
    ap.add_argument("--gpu", action="store_true",
                    help="use CUDAExecutionProvider (run with .venv-gpu; pins CUDA_VISIBLE_DEVICES=0 = 5070 Ti)")
    ap.add_argument("--auto-rotate", action="store_true",
                    help="retry frames at 90/270/180 when the upright frame fails the sim gate (sideways phone video)")
    args = ap.parse_args()

    vids, imgs, skipped = gather_inputs(args.inputs)
    if not vids and not imgs:
        sys.exit("FATAL: no readable video/image inputs found")
    os.makedirs(args.out, exist_ok=True)

    print(f"Loading buffalo_l ({'GPU' if args.gpu else 'CPU'}) ...")
    app = analyzer(args.gpu)
    ref, n_ref, ceiling = build_bank(app, args.refs)
    print(f"REFERENCE {args.refs}: {n_ref} faces | bank self-sim {ceiling:.3f}")
    print(f"Inputs: {len(vids)} videos, {len(imgs)} images"
          + (f", {len(skipped)} skipped (unsupported ext, e.g. HEIC)" if skipped else ""))
    for s in skipped[:10]:
        print(f"  skipped: {s}")

    manifest = os.path.join(args.out, time.strftime("manifest_%Y%m%d_%H%M%S.csv"))
    st = Stats()
    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["kind", "source", "time_s", "sim", "sharp", "face_px", "crop_w", "crop_h", "out_file"])
        for p in imgs:
            print(f"IMAGE {p}")
            harvest_image(app, p, ref, args, writer, st)
        for p in vids:
            print(f"VIDEO {p}")
            t0 = time.time()
            k0 = st.kept
            harvest_video(app, p, ref, args, writer, st)
            print(f"  +{st.kept - k0} crops in {time.time() - t0:.0f}s")

    print(f"\nDONE: {st.line()}")
    print(f"Crops:    {args.out}")
    print(f"Manifest: {manifest}")
    print("Next: quality-cull with score_gen.py (your 0.60+ floor) before training.")


if __name__ == "__main__":
    main()
