"""score_checkpoints.py -- rank LoRA checkpoints objectively by face-embedding
similarity, instead of by eye.

Builds a reference identity from a training set and (optionally) a held-out
set, then scores every sample image an ai-toolkit-style job produced. Because
each checkpoint renders the same handful of prompts at the same seed, the
similarity-per-step is directly comparable across checkpoints.

Also measures identity BLEED for any prompt that renders more than one
person: the second face's similarity to the reference quantifies how badly
the LoRA has started painting your subject onto everyone in frame.

Cosine similarity on L2-normalised ArcFace embeddings (buffalo_l). Rules of
thumb for this model: >0.5 same person, 0.35-0.5 plausible, <0.28 different
person -- but always read a score against the reference bank's own ceiling
(self_mean below), not against 1.0.

  python score_checkpoints.py --job JOB_DIR --train-ref TRAIN_DIR \
      [--holdout-ref HOLDOUT_DIR] [--out checkpoint_scores.json]

JOB_DIR is expected to hold a samples/ subfolder of ai-toolkit-style
"<epoch>__<9-digit step>_<prompt index>.jpg" filenames.
"""
import argparse
import glob
import json
import os
import re
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from facematch import core


def _print_ref_line(label, stats):
    print(f"{label} refs : {stats['n']}/{stats['of']} faces found | "
          f"internal spread mean {stats['self_mean']:.3f} min {stats['self_min']:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", required=True, help="job dir containing a samples/ subfolder")
    ap.add_argument("--train-ref", required=True, help="folder of training reference photos")
    ap.add_argument("--holdout-ref", default=None,
                    help="folder of held-out reference photos (optional; skips the "
                         "holdout-vs-training ceiling and per-prompt holdout column if omitted)")
    ap.add_argument("--ref-limit", type=int, default=60, help="cap reference images for speed")
    ap.add_argument("--out", default="checkpoint_scores.json", help="where to write the JSON report")
    ap.add_argument("--gpu", action="store_true", help="use CUDA if available, else CPU")
    a = ap.parse_args()

    app = core.analyzer(gpu=a.gpu)

    try:
        train_mean, train_stats = core.reference_bank(app, a.train_ref, a.ref_limit)
    except ValueError:
        sys.exit(f"FATAL: no faces found in --train-ref {a.train_ref}")
    _print_ref_line("training", train_stats)

    hold_mean = hold_stats = None
    if a.holdout_ref:
        try:
            hold_mean, hold_stats = core.reference_bank(app, a.holdout_ref)
        except ValueError:
            sys.exit(f"FATAL: no faces found in --holdout-ref {a.holdout_ref}")
        _print_ref_line("holdout ", hold_stats)
        print(f"holdout vs training mean: {float(hold_mean @ train_mean):.3f}   <- realistic ceiling\n")

    # newest file wins per (step, prompt) so re-runs supersede
    by = {}
    for p in glob.glob(os.path.join(a.job, "samples", "*.jpg")):
        m = re.match(r"(\d+)__(\d+)_(\d+)\.jpg$", os.path.basename(p))
        if not m:
            continue
        epoch, step, idx = int(m.group(1)), int(m.group(2)), int(m.group(3))
        k = (step, idx)
        if k not in by or epoch > by[k][0]:
            by[k] = (epoch, p)
    if not by:
        sys.exit(f"FATAL: no ai-toolkit-style samples found under {os.path.join(a.job, 'samples')}")

    prompt_idxs = sorted({idx for _, idx in by})
    rows = []
    for step in sorted({s for s, _ in by}):
        rec = {"step": step, "prompts": {}}
        for idx in prompt_idxs:
            if (step, idx) not in by:
                continue
            fs = core.faces(app, by[(step, idx)][1])
            if not fs:
                rec["prompts"][idx] = None
                continue
            e = fs[0].normed_embedding
            entry = {"train": float(e @ train_mean), "n_faces": len(fs),
                     "face_px": int((fs[0].bbox[2] - fs[0].bbox[0]))}
            if hold_mean is not None:
                entry["holdout"] = float(e @ hold_mean)
            if len(fs) > 1:  # bleed: is a second person in frame also scoring as her?
                e2 = fs[1].normed_embedding
                entry["second_face_train"] = float(e2 @ train_mean)
                entry["pair_sim"] = float(e @ e2)
            rec["prompts"][idx] = entry
        vals = [v["train"] for v in rec["prompts"].values() if v]
        rec["mean_train"] = float(np.mean(vals)) if vals else None
        rows.append(rec)

    report = {"train_stats": train_stats, "rows": rows}
    if hold_stats is not None:
        report["holdout_stats"] = hold_stats
        report["holdout_vs_train"] = float(hold_mean @ train_mean)
    with open(a.out, "w") as fh:
        json.dump(report, fh, indent=1)

    def fmt(v):
        return "  -  " if v is None else format(v, ".3f")

    header = ["step"] + [f"p{idx}" for idx in prompt_idxs] + ["mean"]
    if hold_mean is not None:
        header.append("vs holdout")
    header += ["2nd face", "pair"]
    print(" ".join(f"{h:>10}" for h in header))
    print("-" * (11 * len(header)))
    for r in rows:
        vals = [str(r["step"])]
        for idx in prompt_idxs:
            v = r["prompts"].get(idx)
            vals.append(fmt(v["train"]) if v else "  -  ")
        vals.append(fmt(r["mean_train"]))
        if hold_mean is not None:
            hold_vals = [v["holdout"] for v in r["prompts"].values() if v and "holdout" in v]
            vals.append(fmt(float(np.mean(hold_vals)) if hold_vals else None))
        last_bleed = next((v for v in reversed(list(r["prompts"].values())) if v and "pair_sim" in v), {})
        vals.append(fmt(last_bleed.get("second_face_train")))
        vals.append(fmt(last_bleed.get("pair_sim")))
        print(" ".join(f"{v:>10}" for v in vals))

    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
