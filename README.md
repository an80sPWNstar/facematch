# FaceMatch

A local face-similarity toolkit built around ArcFace (InsightFace buffalo_l):
a Gradio web app for interactive identity matching, plus the pipeline tools and
hard-won knowledge for building identity LoRA training datasets.

## What's here

| Path | What it does |
|---|---|
| `app/app.py` | Gradio web app: build a reference "bank" from photos of one person, score candidate photos against it (table, annotated boxes, CSV), collect keepers into named datasets |
| `app/dataset_builder.py` | Identity-gated harvesting backend with framing-based crop context and auto captions |
| `tools/face_harvest.py` | Identity-gated face-crop harvesting from photos and videos (best-match face, rotation fallback, blur/size gates, video best-of-window + dedup) |
| `tools/bank_score.py` | Build an outlier-rejected reference bank from a folder; score images against it |
| `tools/contact_sheet.py` | Numbered thumbnail grids for fast human review |
| `tools/comfy_upscale.py` | Batch super-resolution through a running ComfyUI instance |
| `tools/color_fix.py` | Conservative WB/gamma correction (read the playbook before using) |
| `tools/caption_local_vlm.py` | Caption training images via a local llama.cpp vision server |
| `tools/train_monitor_fm.py` | Watches an ai-toolkit training run, grades every sample set against the dataset bank via the FaceMatch app API, and stops the run at likeness peak |
| `docs/LORA_DATASET_PLAYBOOK.md` | Everything we learned about how identity LoRAs behave as a function of their datasets — measured, not assumed |

## Quick start (app)

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python app\app.py
```

Open http://localhost:7861. For GPU inference install `onnxruntime-gpu` instead
of `onnxruntime` and set `FACEMATCH_CUDA_DLLS` to a directory containing CUDA
12 + cuDNN 9 DLLs (a torch install's `torch\lib` works). `FACEMATCH_GPU`
selects the device (PCI bus order). CPU works out of the box, just slower.

## Dataset Builder

The Dataset Builder tab harvests face crops from your own photo and video collections
for training identity LoRAs. The workflow is: build a reference bank on the Face Match
tab → point Dataset Builder at a folder of photos/videos → pick a framing (Face/Portrait/Half body/Full person)
→ harvest → review the scored gallery → save keepers into a named dataset with trigger-word captions.
Captions record the framing the crop ACTUALLY achieved, not the requested one, so training sees
the true context your dataset provides.

## Scores

Scores are ArcFace cosine similarity: same person typically > 0.4, different
person < 0.3. A bank's mean self-similarity is your practical ceiling — read
scores relative to it. See the playbook for the sharp edges (rotation,
frame-filling faces, expressions, cross-age matching).

## Privacy

Everything runs locally. No images, banks, datasets, or score files are
tracked in this repo (see `.gitignore`); keep it that way.
