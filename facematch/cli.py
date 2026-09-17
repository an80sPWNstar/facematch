"""facematch.cli -- `python -m facematch ...`

Human-readable output by default. `--json` switches to a single JSON object
on stdout and ONLY on stdout -- all progress/log lines go to stderr -- so an
agent driving this can pipe stdout straight into json.loads() without
scraping log noise out of it first.
"""
import argparse
import glob
import json
import os
import statistics
import sys

from . import core


def _log(msg):
    print(msg, file=sys.stderr, flush=True)


def _targets_kind(path):
    if os.path.isdir(path):
        return "folder"
    if os.path.splitext(path)[1].lower() in core.VIDEO_EXT:
        return "video"
    return "glob"


def _score_targets(app, targets, bank, every_n, max_frames):
    kind = _targets_kind(targets)
    if kind == "folder":
        return core.score_folder(app, targets, bank)
    if kind == "video":
        return core.score_video(app, targets, bank, every_n=every_n, max_frames=max_frames)
    rows = []
    for p in sorted(glob.glob(targets)):
        if os.path.splitext(p)[1].lower() in core.VIDEO_EXT:
            rows.extend(core.score_video(app, p, bank, every_n=every_n, max_frames=max_frames))
        else:
            rows.append(core._summarize(os.path.basename(p), core.score_image(app, p, bank)))
    return rows


def _print_bank_human(stats):
    print(f"bank: {stats['n']} kept / {stats['of']} usable ({stats['dropped']} outlier(s) dropped)")
    print(f"ceiling (self-similarity): mean {stats['self_mean']:.3f}, min {stats['self_min']:.3f}")


def _print_rows_human(rows):
    if not rows:
        print("No results.")
        return
    rows = sorted(rows, key=lambda r: -(r["best"] if r["best"] is not None else -1))
    for r in rows:
        best = "NOFACE" if r["best"] is None else f"{r['best']:.3f}"
        second = f"  (2nd {r['second_best']:.3f})" if r.get("second_best") is not None else ""
        name = r.get("file") or f"frame{r.get('frame')}"
        print(f"{best:>7}{second}  {name}  [{r['n_faces']} face(s)]")
    scored = [r["best"] for r in rows if r["best"] is not None]
    if scored:
        print(
            f"\nn={len(scored)}  mean {sum(scored) / len(scored):.3f}  "
            f"median {statistics.median(scored):.3f}  min {min(scored):.3f}  max {max(scored):.3f}"
        )


def cmd_bank(args):
    app = core.analyzer(gpu=args.gpu)
    _log(f"building reference bank from {args.refs}")
    _, stats = core.reference_bank(app, args.refs, limit=args.ref_limit)
    if args.json:
        print(json.dumps({"stats": stats}))
    else:
        _print_bank_human(stats)
    return 0


def cmd_score(args):
    app = core.analyzer(gpu=args.gpu)
    _log(f"building reference bank from {args.refs}")
    mean, stats = core.reference_bank(app, args.refs, limit=args.ref_limit)
    _log(f"bank ready: {stats['n']} refs, ceiling {stats['self_mean']:.3f}")
    _log(f"scoring {args.targets}")
    rows = _score_targets(app, args.targets, (mean, stats), args.every_n, args.max_frames)
    if args.json:
        print(json.dumps({"bank": stats, "rows": rows}))
    else:
        _print_bank_human(stats)
        print()
        _print_rows_human(rows)
    return 0


def build_parser():
    ap = argparse.ArgumentParser(
        prog="facematch",
        description=(
            "ArcFace face-similarity scoring: build a reference bank from photos "
            "of one person, then score other photos/videos against it."
        ),
    )
    sub = ap.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--refs", required=True, help="folder of reference photos")
    common.add_argument("--ref-limit", type=int, default=None, help="cap reference images for speed")
    common.add_argument("--gpu", action="store_true", help="use CUDA if available, else CPU")
    common.add_argument("--json", action="store_true", help="emit one JSON object to stdout instead of text")

    p_bank = sub.add_parser("bank", parents=[common], help="build a reference bank and report its stats")
    p_bank.set_defaults(func=cmd_bank)

    p_score = sub.add_parser("score", parents=[common], help="score targets against a reference bank")
    p_score.add_argument(
        "--targets", required=True, help="folder, video file, or glob of images/videos to score"
    )
    p_score.add_argument("--every-n", type=int, default=15, help="video: sample every Nth frame")
    p_score.add_argument("--max-frames", type=int, default=10, help="video: cap on frames scored")
    p_score.set_defaults(func=cmd_score)

    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
