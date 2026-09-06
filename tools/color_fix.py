"""color_fix.py -- gentle auto lighting/color correction for old training photos.

Gray-world white balance (gain-capped), mild CLAHE on lightness, and gamma
pulled toward a target median luminance. Deliberately conservative: this fixes
casts and murk, it does not restyle. Writes corrected copies to --out; originals
untouched.

  python color_fix.py IN_DIR --out OUT_DIR [--wb-cap 1.25] [--clahe 1.5] [--gamma-lo 0.75] [--gamma-hi 1.30]
"""
import argparse, glob, os
import cv2, numpy as np


def correct(img, wb_cap, clahe_clip, glo, ghi):
    # 1. gray-world white balance, per-channel gain capped so a genuinely
    #    colorful scene is not neutralized
    f = img.astype(np.float32)
    means = f.reshape(-1, 3).mean(0)
    gains = np.clip(means.mean() / np.maximum(means, 1e-6), 1.0 / wb_cap, wb_cap)
    f = np.clip(f * gains, 0, 255)

    # 2. gamma toward a mid-tone target, capped
    lum = cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    med = max(np.median(lum) / 255.0, 1e-3)
    gamma = np.clip(np.log(0.45) / np.log(med), glo, ghi)
    if abs(gamma - 1.0) > 0.02:
        f = np.clip(((f / 255.0) ** gamma) * 255.0, 0, 255)

    # 3. mild CLAHE on L only (contrast, not color)
    lab = cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("indir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wb-cap", type=float, default=1.25)
    ap.add_argument("--clahe", type=float, default=1.5)
    ap.add_argument("--gamma-lo", type=float, default=0.75)
    ap.add_argument("--gamma-hi", type=float, default=1.30)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    n = 0
    for p in sorted(glob.glob(os.path.join(args.indir, "*.png")) + glob.glob(os.path.join(args.indir, "*.jpg"))):
        img = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        out = correct(img, args.wb_cap, args.clahe, args.gamma_lo, args.gamma_hi)
        cv2.imencode(".png", out)[1].tofile(os.path.join(args.out, os.path.basename(p)))
        n += 1
    print(f"corrected {n} -> {args.out}")


if __name__ == "__main__":
    main()
