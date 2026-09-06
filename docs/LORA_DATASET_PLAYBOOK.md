# Identity LoRA Dataset & Training Playbook

Field notes from building identity LoRAs (FLUX.2 Klein 9B, ai-toolkit) with an
ArcFace-based measurement harness. Everything here was measured on real runs,
not assumed. Similarities are cosine similarity on ArcFace buffalo_l embeddings
unless stated. "SUBJECT" is the person being trained; "TRIGGER" is the LoRA's
trigger token.

## 1. The dataset is the binding constraint

- A synthetic dataset that scored **better** on the identity metric (mean 0.700
  vs 0.64) produced a **worse** LoRA than the messier set (0.583 vs 0.629 final
  likeness). Optimizing dataset ArcFace score selects images close to the
  reference *embedding mean* — the most frontal, canonical, same-looking shots —
  and strips the pose/lighting/expression diversity a trainer needs.
  **Corollary: cull with a modest identity floor (~0.55 vs the era bank), never
  aggressively toward the centroid.**
- Never train on AI-generated images of the subject, even good-looking ones.
  It feeds generator drift back into the model (the same closed loop as above,
  with an extra compounding step).
- Real photos beat everything. A 20-minute phone dump of casual photos
  outperformed weeks of synthetic-data engineering.
- Video is a goldmine: 30 seconds of the subject talking yields dozens of
  distinct natural expressions/angles no photo session produces. Use best-of-N
  windowing (sharpest frame per ~2s) plus embedding dedup, or you get 400
  near-identical frames.

## 2. Harvesting gotchas (each of these cost real hours)

- **Pick the best-MATCHING face, not the biggest face.** Group photos poison
  banks and datasets otherwise (a child's face closer to the camera wins the
  "biggest" rule).
- **Phone videos contain sideways segments.** Rotation metadata changes
  mid-file and OpenCV reads the raw buffer. A sideways face of the RIGHT person
  still scores ~0.4 — above a naive 0.35 identity gate but garbage for
  training. Test all four rotations and keep the argmax; do NOT short-circuit
  on "passed the gate": a sharp 4K face can score 0.6+ while sideways.
- A whole video can look like "a different person" (sim ~0.05) purely because
  it is stored rotated. Verify rotation before concluding identity.
- **cv2.imread ignores EXIF orientation** — phone photos silently arrive
  sideways. Route image loads through PIL `ImageOps.exif_transpose`.
- **RetinaFace at det_size 640 misses frame-filling faces.** A tight close-up
  reads NOFACE. On NOFACE, downscale onto a same-size canvas (50%, then 25%)
  and re-detect.
- Extreme expressions (tongue out, squinting) tank ArcFace to ~0.06 on a photo
  that is definitely the subject. Eyeball low-sim rejects before dropping them;
  a few expression outliers are *good* training diversity.
- Faces cut off at the frame edge are correctly rejected — half a face is poor
  identity training data no matter how good the photo.

## 3. Reference banks

- Build per-era banks from REAL photos. A mixed-era bank systematically
  under-scores whichever era it under-represents, which corrupts culling.
- Always outlier-reject bank members: build centroid → drop members < 0.35 →
  rebuild. The biggest-face rule guarantees occasional wrong-person members.
- The bank's mean self-similarity is the practical **ceiling**. Read every
  score relative to it, not to 1.0. (A 112-image dataset bank had ceiling
  0.791; generated samples peaking at 0.64 ≈ 80% of ceiling — strong.)
- Small banks are noisy: 8 photos gave ceiling 0.842; the metric sharpened
  with more members.
- Cross-age matching is weak and person-dependent: the same subject 15 years
  younger scored 0.3-0.45 against their current-era centroid; another subject's
  eras cross-matched much lower. Never assume one bank spans eras.

## 4. Image cleanup

- **Pure super-resolution is identity-neutral — but measure it.** A 4x DAT2
  real-photo SR model (4xRealWebPhoto_v4_dat2) changed mean identity by 0.000
  across 111 images; a sharper alternative (Nomos RealPLKSR) drifted up to
  -0.034 and waxed skin. Skip images already ≥1024px.
- Super-resolution cannot rescue motion blur — it produces crisp blur with
  facial artifacts. Cull motion-blurred frames (normalized Laplacian variance
  < ~15) instead of upscaling them.
- **Do not batch color-correct training data.** CLAHE measurably drifted
  identity (-0.030 mean). White-balance+gamma was metric-neutral but visually
  wrecked warm-light photos (gray-world WB reads golden hour as a cast).
  Lighting variety helps generalization; fix individual photos by hand if at
  all.

## 5. Training and monitoring

- **Score every sample set against the dataset bank during training.** 5 fixed
  prompts per sample step (every 250 steps) covering: close-up, half-body,
  full-body, casual-with-activity, and a group shot with other people (the
  group shot is the drift alarm — if bystanders morph toward the subject,
  consider regularization next run).
- Step-0 baseline samples score ~0.02-0.03 (the untrained base model). That is
  the floor; every point above it is LoRA effect.
- Curve shape observed: fast rise (0.03 → 0.50 in 1000 steps), oscillating
  grind to peak (0.64 @ 2500), then genuine decline (0.58 by 3000). The peak
  is NOT the final checkpoint — keep a deep save window (20 × 250 steps) so
  the best checkpoint survives the stop.
- Stop rules that worked: OVERBAKE = last 3 sets all ≥0.02 below best (fired
  correctly at peak+750); PLATEAU = best is 12+ sets stale and nothing came
  within 0.005 since. Bias to over-training: extra steps are recoverable
  (checkpoints), a restarted run is not.
- **Trigger tokens leak English semantics.** "SUBJECT_older" rendered
  gray-haired elderly women at baseline — the text encoder reads "older"
  literally, and the LoRA wastes capacity fighting its own trigger. Use
  semantically empty tokens.
- Sample prompts are measurements — keep them single-subject, single-view.
  "Casting sheet" multi-view prompts render tiny faces that go generic and
  score near-random regardless of the LoRA.
- Vary close-up prompt lighting/angle; flat frontal studio close-ups was the
  weakest framing across a whole prior project (0.13 below other framings).
- Evaluate renders on the SAME base family the LoRA trained on, at the
  measured-best checkpoint, with the exact trigger token. Each mismatch
  silently costs likeness.

## 6. Metric honesty

- ArcFace is a proxy. It is largely invariant to hair, jawline across angles,
  and skin texture — things humans key on. Rank-order YOUR eye against the
  metric before trusting either exclusively.
- Small-batch deltas are noise: 28-image batches swing ~0.04 on seed variance
  alone; 3-seed comparisons resolve ~0.03 at best. Don't build on smaller
  differences.
- Fixed eval seeds + fixed prompts make checkpoints comparable; changing
  either invalidates the curve.

## 7. Ops notes (Windows, multi-GPU)

- One 9B qfloat8 trainer at a time: loading spikes ~50 GB commit charge and
  ~16 GB VRAM (does not fit 16 GB cards; silent process kill with no traceback
  = Windows commit limit, in-process CUDA OOM = VRAM).
- Measure commit limit at run time, never reuse a remembered number
  (auto-managed pagefiles move).
- Keep the scoring stack on a GPU the trainers never touch.
- Keep score CSVs append-only with a resume-safe reader; monitors restart, and
  the stop-rule history must survive.
