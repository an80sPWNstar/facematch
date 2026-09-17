# FaceMatch — instructions for an LLM / agent

You are looking at a face-similarity toolkit. It answers one question: **does this
generated image or video actually look like my reference person?** It uses ArcFace
(InsightFace `buffalo_l`) and cosine similarity on L2-normalised embeddings.

## The fastest path

```bash
python -m facematch score --refs <folder-of-reference-photos> --targets <folder|video|glob> --json
```

`--json` puts **exactly one JSON document on stdout and nothing else**. All progress and
library chatter goes to stderr. Pipe stdout straight into a parser.

Real output, trimmed:

```json
{
  "bank": {"n": 20, "of": 20, "dropped": 0,
           "self_mean": 0.9105117917060852, "self_min": 0.7195218801498413},
  "rows": [
    {"file": "f01.png", "best": 0.4707188010215759, "second_best": null, "n_faces": 1,
     "faces": [{"bbox": [510.4, 170.5, 611.1, 305.5], "sim": 0.4707188010215759}]},
    {"file": "f08.png", "best": null, "second_best": null, "n_faces": 0, "faces": []}
  ]
}
```

Bank stats only:

```bash
python -m facematch bank --refs <folder> --json
# {"stats": {"n": 5, "of": 5, "dropped": 0, "self_mean": 0.923, "self_min": 0.896}}
```

Key meanings:

| key | meaning |
|---|---|
| `bank.n` / `bank.of` | reference faces kept / usable images found |
| `bank.dropped` | references rejected as outliers (a wrong person in the folder) |
| `bank.self_mean` | how tightly the reference set agrees with itself — **the ceiling** |
| `rows[].best` | best-matching face in that image. `null` means **no face detected** |
| `rows[].second_best` | next-best face, if more than one. Non-null = another person in frame |
| `rows[].n_faces` | faces detected |
| `rows[].faces[].bbox` | `[x1, y1, x2, y2]` |
| `rows[].frame` | video only: decoded frame index |
| `rows[].time_s` | video only: timestamp in seconds |

Video targets return the same row shape plus `frame` and `time_s`. `file` is still
present and is a synthetic name like `frame000000`:

```json
{"file": "frame000000", "best": 0.6002532243728638, "second_best": null,
 "n_faces": 1, "faces": [...], "frame": 0, "time_s": 0.0}
```

## How to read the numbers — read this before reporting anything

1. **A score is meaningless without the ceiling.** Always look at `bank.self_mean` first.
   It is how well the reference photos match *each other*, and it is the practical maximum.
   A typical good reference set sits around **0.85–0.92**. Against that ceiling, ~0.63 is a
   strong match and ~0.23 is a failure. Never report a raw score without it.
2. **>0.40 is the conventional "same person" line** for `buffalo_l`. Treat 0.30–0.40 as
   uncertain, not as a pass.
3. **Report mean AND max across frames.** One good frame does not make a good clip. A low
   mean with a high max usually means the subject is only recognisable in part of the shot.
4. **A low score is often bad framing, not bad likeness.** Face small in frame, turned away,
   occluded, or motion-blurred all tank the score. Check `n_faces` and `bbox` size before
   concluding the model failed to reproduce the identity.
5. **`best: null` is "no face found", not "score of zero".** Exclude those from means; report
   them separately as a detection-failure count.
6. **`second_best` is a warning.** If it is high, another face in the frame also matches the
   reference — the identity may be leaking onto a bystander.

## What not to do

- **Do not judge likeness by looking at the image and deciding.** That is the exact error this
  tool exists to prevent — eyeballing one frame reliably picks the wrong checkpoint. Score it.
- **Do not compare scores across different reference banks.** Scores are only comparable
  within one bank. Changing `--refs` or `--ref-limit` changes the scale.
- **Do not pass `--gpu` blindly.** It is CPU by default on purpose. If another job holds the
  GPU, `--gpu` will contend with it or fail on missing CUDA DLLs.
- **Do not parse the human-readable output.** Use `--json`.

## Python API

```python
from facematch.core import analyzer, reference_bank, score_folder, score_video

app = analyzer(gpu=False)
mean, stats = reference_bank(app, r"path/to/reference_photos", limit=60)
print(stats["self_mean"])                       # the ceiling

rows = score_folder(app, r"path/to/generated", (mean, stats))
rows = score_video(app, r"clip.mp4", (mean, stats), every_n=15, max_frames=10)

scored = [r["best"] for r in rows if r["best"] is not None]
print(sum(scored) / len(scored), max(scored))   # mean AND max
```

Also available: `faces(app, path)`, `score_image(app, path, bank)`,
`bank_from_paths(app, [paths])`.

## Environment variables

| var | effect if unset |
|---|---|
| `FACEMATCH_GPU` | CUDA device index used when `--gpu` is passed. Defaults to `0`. |
| `FACEMATCH_CUDA_DLLS` | directory holding CUDA/cuDNN DLLs for `onnxruntime-gpu`. Empty by default; only needed if `--gpu` fails to find them. |

Neither is required for CPU use, which is the default and needs no configuration.

## Troubleshooting

- **`best: null` on every row** — no faces detected. The images may be too low-resolution,
  the face too small in frame, or heavily stylised. Try larger images before concluding the
  identity failed.
- **Reference bank `n` much lower than `of`** — most reference photos had no detectable face.
  Check the folder actually contains face photos.
- **`dropped` > 0** — outliers were rejected. Usually a different person in the reference
  folder. Inspect before trusting the bank.
- **`onnxruntime-gpu` DLL load failure with `--gpu`** — set `FACEMATCH_CUDA_DLLS` to the
  directory containing the CUDA/cuDNN DLLs, or drop `--gpu` and run on CPU.
- **Video yields no rows** — the file may not be readable by OpenCV, or `--every-n` is larger
  than the clip. Lower `--every-n`.
- **Empty reference folder** — the command errors rather than silently scoring against nothing.
