# FaceMatch -- instructions for an LLM/agent

## 1. What this does

ArcFace face-similarity scoring: does this generated image/video look like my
reference person? You give it a folder of trusted photos of one person (the
"reference bank") and a folder/video/glob to check; it returns a cosine
similarity per detected face.

## 2. Fastest path

```
python -m facematch score --refs ./refs/alice --targets ./output/run3 --json
```

`--targets` can be a folder, a single video file, or a glob (mixed
image/video globs work). `--json` writes exactly one JSON object to stdout
and nothing else there -- all progress goes to stderr, so it is always safe
to `json.loads(stdout)` directly. Exact shape:

```json
{
  "bank": {"n": 42, "of": 45, "dropped": 3, "self_mean": 0.842, "self_min": 0.611},
  "rows": [
    {"file": "img001.jpg", "best": 0.71, "second_best": null, "n_faces": 1,
     "faces": [{"bbox": [120.0, 40.0, 340.0, 300.0], "sim": 0.71}]},
    {"file": "img002.jpg", "best": null, "second_best": null, "n_faces": 0, "faces": []}
  ]
}
```

For a video target, each row also has `"frame"` (int index) and `"time_s"`
(float). `python -m facematch bank --refs ./refs/alice --json` returns just
`{"stats": {...}}` in the same shape as `bank` above -- use it to sanity
check a reference set before spending time scoring anything against it.

## 3. How to read the numbers -- read this before reporting anything

- Scores are cosine similarity on L2-normalised ArcFace embeddings. Range is
  roughly -1..1 in theory; real faces cluster in 0.0-0.9.
- `>0.40` is the conventional "same person" floor for buffalo_l. But:
- **A raw score means nothing without the reference bank's own ceiling.**
  `bank.self_mean` is how similar the reference photos are to *each other* --
  that's the best any candidate could realistically score. A good reference
  set has `self_mean` around 0.85-0.87. Against that ceiling, 0.63 is a
  strong match and 0.23 is a clear failure -- but a 0.63 against a shaky
  0.70-ceiling bank is a much weaker result. **Always fetch/print
  `bank.self_mean` and `bank.self_min` alongside every score you report.**
  If you have a second reference set (e.g. a held-out batch of the same
  person), also compute a holdout-vs-train ceiling before judging anything
  (snippet in section 5) -- that number, not 1.0, is the realistic best case.
- For a video or a multi-frame batch, report **both the mean and the max**
  across frames/rows. A single lucky frame does not mean the clip is good;
  a single bad frame does not mean it's ruined. Both numbers, always.
- A low score can mean bad framing -- small face, turned away, occluded,
  motion blur -- rather than bad likeness. Check `faces[].bbox` size and
  `n_faces` before concluding "this doesn't look like them." A `best: null`
  row (no face detected) is not a low score; don't average it in as 0.

## 4. What not to do

- **Do not eyeball an image and decide it "looks like" the reference.**
  That subjective judgment is the exact failure mode this package exists to
  replace. Always run the scorer and report the number plus the ceiling.
- Do not compare a score from one reference bank against a score from a
  different bank -- the ceiling is different, so the numbers aren't
  comparable. Rebuild/rescore against a shared bank if you need to compare.
- Do not pass `--gpu` (or `analyzer(gpu=True)`) without checking the GPU is
  actually free -- it will silently fall back to CPU on failure, but it can
  still contend with a training job or another process for VRAM first.

## 5. Python API

```python
from facematch.core import analyzer, reference_bank, score_folder, score_video

app = analyzer(gpu=False)  # gpu=True tries CUDA, falls back to CPU on any failure
mean, stats = reference_bank(app, "./refs/alice")
print(f"ceiling: mean={stats['self_mean']:.3f} min={stats['self_min']:.3f}")

rows = score_folder(app, "./output/run3", (mean, stats))
# or: score_video(app, "./output/clip.mp4", (mean, stats), every_n=15, max_frames=10)

scored = [r["best"] for r in rows if r["best"] is not None]
print(f"mean {sum(scored)/len(scored):.3f}  max {max(scored):.3f}  n={len(scored)}")
```

Holdout-vs-train ceiling (the realistic best case, if you have two reference
sets for the same person):

```python
train_mean, train_stats = reference_bank(app, "./refs/alice_train")
hold_mean, hold_stats = reference_bank(app, "./refs/alice_holdout")
print(f"holdout vs train ceiling: {float(hold_mean @ train_mean):.3f}")
```

## 6. Environment variables

| Variable | Default if unset | Effect |
|---|---|---|
| `FACEMATCH_CUDA_DLLS` | none | CPU-only; `onnxruntime-gpu`/CUDA init is skipped entirely (`analyzer(gpu=...)` still runs, just lands on CPU). Set to a directory containing CUDA 12 + cuDNN 9 DLLs (a torch install's `...\torch\lib` works) to enable GPU. |
| `FACEMATCH_GPU` | `"0"` | Which device index (PCI bus order) to expose when GPU is requested. Ignored on CPU. |

Nothing else needs to be set. GPU is always opt-in (`gpu=True` / `--gpu`) and
always falls back to CPU silently on any init failure -- check stderr/the log
for "CUDA init failed" if a run seems slower than expected, don't assume it
crashed.

## 7. Troubleshooting

- **No faces detected in a whole folder** (`stats.n == 0`, `ValueError: no
  usable faces found`): check the folder actually has images with visible
  faces at a normal resolution -- buffalo_l's default detector size is
  640x640 and struggles on tiny thumbnails or extreme close-ups that fill
  the whole frame.
- **`onnxruntime-gpu` / CUDA DLL failures**: this is caught internally and
  falls back to CPU automatically -- it will not crash your run. If you
  actually need GPU, verify `FACEMATCH_CUDA_DLLS` points at a real directory
  containing `cudart64_*.dll`/`cudnn64_*.dll`, and that `onnxruntime-gpu` (not
  plain `onnxruntime`) is installed in the active environment.
- **Empty reference folder / bad `--refs` path**: `reference_bank`/`bank`
  raises/reports zero faces rather than silently scoring against garbage --
  treat that as a hard stop, not a 0.0 score.
- **Video with no extractable frames**: `score_video` returns `[]` if the
  file won't open (`cv2.VideoCapture.isOpened() == False`) -- check the
  codec/container is one OpenCV can decode, or extract frames yourself first
  and score them as a folder instead.
