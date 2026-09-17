# FaceMatch

A local face-similarity toolkit built around ArcFace (InsightFace buffalo_l):
a Gradio web app for interactive identity matching, a `facematch` Python
package with one scoring implementation shared by everything else, a CLI for
scripting or driving from an agent, and the pipeline tools and hard-won
knowledge for building identity LoRA training datasets.

**An LLM/agent picking this repo up cold should read
[`LLM_instructions.md`](LLM_instructions.md) first** -- it has the exact
commands, the JSON schema, and (most important) how to read a score against
a reference bank's ceiling instead of against 1.0.

## Package layout

```
facematch/            the package -- import this, or run it as a module
  core.py             the single ArcFace scoring implementation (analyzer,
                       reference banks, per-face scoring for images/folders/video)
  cli.py              argparse CLI (`python -m facematch ...`)
  __main__.py         `python -m facematch` entry point

app/
  app.py              Gradio web app: Face Match tab (build a bank, score
                       candidates, save keepers into a dataset) + Dataset
                       Builder tab (identity-gated harvesting from photos/video)
  dataset_builder.py  Dataset Builder's harvest backend
  launch_facematch.bat

tools/
  bank_score.py           CLI: build a bank, score a batch, optional CSV
  face_harvest.py         CLI: identity-gated crop harvesting (photos + video)
  score_checkpoints.py    rank LoRA checkpoints by face similarity per training step
  contact_sheet.py        numbered thumbnail grids for review
  comfy_upscale.py        batch super-resolution through a running ComfyUI instance
  color_fix.py            conservative WB/gamma correction
  caption_local_vlm.py    caption training images via a local llama.cpp vision server
  train_monitor_fm.py     watch an ai-toolkit run, grade samples via the app's
                           API, stop training at likeness peak

docs/LORA_DATASET_PLAYBOOK.md   everything learned about identity LoRAs as a
                                 function of their datasets -- measured, not assumed

run.ps1                launch the app (Windows; drive-independent)
```

All scoring -- in the app, the CLI, and every tool above -- goes through
`facematch.core`. There is exactly one implementation of "build a reference
bank" and "score a face against it"; nothing re-derives that math locally
anymore.

## Three ways to use it

**1. The GUI.**

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python app\app.py        # or: .\run.ps1
```

Open http://127.0.0.1:7861. Build a reference bank (folder path with a native
"Load folder" picker, drag-and-drop upload, or both at once), score a folder
or a batch of uploads against it, then keep the good matches into a named
training dataset. The Dataset Builder tab harvests identity-gated face crops
straight out of your own photo and video collections.

**2. The CLI**, for scripting or piping into something else:

```
python -m facematch bank  --refs path\to\references
python -m facematch score --refs path\to\references --targets path\to\candidates
python -m facematch score --refs path\to\references --targets path\to\some_video.mp4 --every-n 15 --max-frames 10
```

`--targets` accepts a folder, a single video file, or a glob -- mixed image
and video globs work too. Human-readable output by default; add `--json` for
a single machine-readable object (see **For agents/LLMs** below).

**3. `import facematch`**, from your own script or notebook:

```python
from facematch.core import analyzer, reference_bank, score_folder

app = analyzer(gpu=True)                       # CUDA if available, else CPU
mean, stats = reference_bank(app, "path/to/references")
print(f"ceiling: {stats['self_mean']:.3f}")
rows = score_folder(app, "path/to/candidates", (mean, stats))
```

## Environment variables

GPU use is opt-in and inherently machine-specific, so it's controlled
entirely through environment variables with CPU-safe defaults -- nothing in
the repo hardcodes a drive letter or a device index.

| Variable | Default | Meaning |
|---|---|---|
| `FACEMATCH_CUDA_DLLS` | unset (CPU) | Directory holding CUDA 12 + cuDNN 9 DLLs, needed by `onnxruntime-gpu`. A torch install's `...\torch\lib` works. Install `onnxruntime-gpu` instead of `onnxruntime` to use it. |
| `FACEMATCH_GPU` | `"0"` | Which device to expose (PCI bus order) when GPU is requested. |

The app always requests GPU-with-CPU-fallback (`facematch.core.analyzer(gpu=True)`);
the CLI and tools default to CPU and take `--gpu` to opt in. Either way, CUDA
init failures fall back to CPU automatically -- you never get a hard crash
for a missing DLL directory, just a slower run and a note on stderr/in the log.

## Scores

Scores are ArcFace cosine similarity: same person typically > 0.4, different
person < 0.3. A bank's mean self-similarity (its "ceiling") is your practical
ceiling — two photos of the same real person don't score 1.0 against each
other either, so read every score relative to the ceiling a run reports, not
against 1.0. Every face in an image is scored, not just the best match or the
biggest one -- that's how a LoRA painting your subject onto a bystander shows
up as a high "2nd face" score. See the playbook for the sharper edges
(rotation, frame-filling faces, expressions, cross-age matching).

## For agents/LLMs

`python -m facematch score ... --json` writes exactly one JSON object to
stdout and nothing else there (all progress goes to stderr, so redirect it
away or ignore it -- stdout is safe to parse as-is):

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

`bank` is the reference set's own stats (see Scores above -- `self_mean` is
the ceiling to judge every `best`/`sim` against). Each row's `best` is
`null` when no face was detected (`n_faces: 0`); `second_best` is the bleed
check. `python -m facematch bank --json` returns just `{"stats": {...}}` in
the same shape, for checking a reference set before committing to a scoring
run.

## Privacy

Everything runs locally. No images, banks, datasets, or score files are
tracked in this repo (see `.gitignore`); keep it that way.
