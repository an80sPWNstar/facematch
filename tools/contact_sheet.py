"""contact_sheet.py -- numbered thumbnail grid of harvested crops for manual review.

  python contact_sheet.py "folder_or_glob" [more...] --out sheet.png [--cols 6] [--thumb 256]

Prints an index -> filename table so picks can be given as numbers.
"""
import argparse, glob, os, sys
import cv2, numpy as np


def gather(inputs):
    files = []
    for item in inputs:
        if os.path.isdir(item):
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
                files += glob.glob(os.path.join(item, ext))
        else:
            files += glob.glob(item)
    return sorted(set(files))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--thumb", type=int, default=256)
    args = ap.parse_args()

    files = gather(args.inputs)
    if not files:
        sys.exit("no images found")
    T, cols = args.thumb, args.cols
    label_h = 28
    rows = (len(files) + cols - 1) // cols
    sheet = np.full((rows * (T + label_h), cols * T, 3), 24, np.uint8)

    for i, p in enumerate(files):
        img = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        s = T / max(img.shape[:2])
        r = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))))
        row, col = divmod(i, cols)
        y0 = row * (T + label_h)
        x0 = col * T
        oy = (T - r.shape[0]) // 2
        ox = (T - r.shape[1]) // 2
        sheet[y0 + oy:y0 + oy + r.shape[0], x0 + ox:x0 + ox + r.shape[1]] = r
        name = os.path.basename(p)
        short = (name[:30] + "..") if len(name) > 32 else name
        cv2.putText(sheet, f"#{i:02d}", (x0 + 4, y0 + T + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(sheet, short, (x0 + 52, y0 + T + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        print(f"#{i:02d}  {name}")

    ok, buf = cv2.imencode(".png", sheet)
    if not ok:
        sys.exit("encode failed")
    buf.tofile(args.out)
    print(f"\nsheet: {args.out}  ({len(files)} images, {cols}x{rows})")


if __name__ == "__main__":
    main()
