"""bank_score.py -- build an ArcFace reference bank from one folder, score other images against it.

  python bank_score.py --refs BANK_DIR "glob_or_dir" [more...] [--csv out.csv] [--gpu]

Unlike score_gen.py the reference folder is an argument, so per-era banks work.
Picks the best-matching face per image (not the biggest). Prints a table and
optionally writes CSV. Never overwrites an existing CSV.

Thin CLI over facematch.core -- see that module for the actual scoring/bank
implementation shared with the app and the rest of tools/*.py.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from facematch import core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--refs", required=True)
    ap.add_argument("--csv")
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    app = core.analyzer(args.gpu)
    try:
        ref, stats = core.bank_from_paths(app, core.gather([args.refs]))
    except ValueError:
        sys.exit(f"FATAL: no faces in refs {args.refs}")
    if stats["dropped"]:
        print(f"BANK: dropped {stats['dropped']} outlier member(s) below {core.OUTLIER_CUT}")
    print(f"BANK {args.refs}: {stats['n']} faces | ceiling {stats['self_mean']:.3f} (min {stats['self_min']:.3f})")

    rows = []
    for p in core.gather(args.inputs):
        face_scores = core.score_image(app, p, ref)
        best = max((f["sim"] for f in face_scores), default=None)
        rows.append((os.path.basename(p), best))
    rows.sort(key=lambda r: -(r[1] if r[1] is not None else -1))
    scored = [r[1] for r in rows if r[1] is not None]
    for name, s in rows:
        print(f"{'NOFACE' if s is None else f'{s:7.3f}'}  {name}")
    if scored:
        import numpy as np
        a = np.array(scored)
        print(f"\nn={len(a)}  mean {a.mean():.3f}  median {np.median(a):.3f}  "
              f"min {a.min():.3f}  max {a.max():.3f}  >=0.60: {(a >= 0.60).sum()}  >=0.65: {(a >= 0.65).sum()}")
    if args.csv:
        if os.path.exists(args.csv):
            sys.exit(f"refusing to overwrite {args.csv}")
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "sim"])
            w.writerows(rows)
        print(f"csv: {args.csv}")


if __name__ == "__main__":
    main()
