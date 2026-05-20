# `mosl/render/` — Avatar Video Rendering Subsystem

The **renderer** half of the project. The existing `mosl/` code is the
**choreographer** — it turns Arabic text into OpenPose-format motion. This
package turns that motion into photorealistic, identity-preserving avatar video
using diffusion models.

It is **strictly additive**: it imports nothing from and changes nothing in the
rest of `mosl/`. The single contract between the two halves is the **OpenPose
JSON keypoint format** already produced by `mosl/pose/`.

```
text ──► mosl/model (SignLLM) ──► pose ──► mosl/pose ──► OpenPose JSON
                                                              │
                                          ┌───────────────────┘   ← the contract
                                          ▼
   OpenPose JSON ──► mosl/render ──► photorealistic avatar video
```

---

## Why this is a new subsystem, not an edit of existing code

The original project contains **no diffusion code, no image/video generation,
no identity model, and no checkpoints for any of that**. The avatar renderer is
necessarily new. What it *does* reuse — genuinely — is:

| Reused asset | How |
|---|---|
| OpenPose JSON keypoints (`mosl/pose/`) | Direct conditioning input to ControlNet / pose-video models |
| SignLLM text→pose model | Optional motion source: text → pose → render |
| Extracted sign clips | Motion source for the avatar |
| Train/val/test splits, vocab | Unchanged, still valid |

---

## Project configuration (locked 2026-05-20)

| Decision | Value |
|---|---|
| Compute | **DGX Spark** (GB10 Grace-Blackwell, 121 GB unified, CUDA 13, aarch64) |
| Identity source | **User-provided target photos** (1–5 images) |
| Motion source | **MoSL sign poses** (extracted sign clips / SignLLM output) |
| Output | Identity-preserving photorealistic signing avatar video |

---

## Phase status

| Phase | Module | Runs on | Status |
|---|---|---|---|
| **A** | `pose_bridge.py` — OpenPose JSON → ControlNet pose frames | CPU (anywhere) | ✅ **done** |
| **B** | `dwpose_extract.py` — dense whole-body re-extraction | DGX | ✅ **code done** — run on DGX |
| **C** | `identity.py` — InstantID / IP-Adapter face embedding | DGX | ✅ **code done** — run on DGX |
| **D** | `keyframe.py` — identity embedding → photoreal reference image | DGX | ✅ **code done** — run on DGX |
| **E** | `video.py` — MimicMotion: keyframe + sign clip → avatar video | DGX | ✅ **code done** — run on DGX |
| **F** | `temporal.py` — deflicker + interpolation + upscale | DGX | ✅ **code done** — run on DGX |
| **G** | `scripts/render_avatar.py` — text → avatar video, end-to-end | DGX | ✅ **code done** — run on DGX |

Setup & validation: see [SETUP_DGX.md](SETUP_DGX.md). Dependencies:
[`requirements-render.txt`](../../requirements-render.txt).
Regression tests (CPU, stdlib only): `python -m mosl.render.test_render`
— 26 tests over every module's pure-Python logic.

---

## Phase A — Pose bridge (done)

Converts a directory of per-frame OpenPose JSON into the canonical OpenPose
**skeleton image** sequence that ControlNet-OpenPose and pose-guided video
models consume.

```bash
python -m mosl.render.pose_bridge outputs/openpose_json/<clip>_keypoints
# → outputs/pose_control/<clip>_keypoints/pose_000000.png ...  + manifest.json
```

Options: `--canvas 768` · `--fill 0.72` · `--no-face` · `--no-interp` ·
`--preview` (GIF, needs a working encoder).

What it does, and why it matters for video quality:

- **Format-agnostic** — auto-detects COCO-18 (sign clips) and BODY-25
  (synthetic clips); remaps BODY-25 → COCO-18.
- **One global fit transform per clip** — the figure is framed *identically*
  in every frame. Per-frame re-framing reads as camera jitter downstream.
- **Gap interpolation** — short keypoint dropouts are linearly filled, so the
  conditioning signal does not pop. Popping conditioning ⇒ flicker in the
  generated video. This is the first line of defense against temporal
  artifacts.
- **Confidence gating** — joints below confidence 0.10 are not drawn.

> **Known data-quality issue.** The current sign keypoints (MediaPipe Holistic:
> COCO-18 body + 21-pt hands + 478-pt face) are usable but noisy, and the body
> is COCO-18 while ControlNet/video models prefer DWPose's 134-point whole-body
> set. Phase B fixes this at the source.

---

## Phase B — DWPose re-extraction (`dwpose_extract.py`)

**Problem.** The Prompt2Sign authors themselves deprecated OpenPose for DWPose.
The current sign keypoints are noisy and inconsistent across clips.
ControlNet-OpenPose and MimicMotion are trained on **DWPose** output
(COCO-WholeBody 133: body + 68 face + 42 hands).

**Delivered.** `dwpose_extract.py` re-extracts whole-body keypoints with DWPose
and writes the *same* OpenPose-JSON layout the existing pipeline and Phase A
already consume — so nothing downstream changes, the input just gets cleaner.

```bash
# validate the DGX environment first (loads the model, no data needed)
python -m mosl.render.dwpose_extract --check

# batch-extract every video in a tree, smoothed, and render pose frames too
python -m mosl.render.dwpose_extract --video-dir data/raw/vedios-dataset \
    --npz --render
```

Key properties:

- **Backend:** DWPose via `rtmlib` (`pip install rtmlib onnxruntime-gpu
  opencv-python`). Raw COCO-WholeBody 133 is converted to OpenPose COCO-18 +
  68-face + 21/21-hands by an explicit, verifiable remap (`wholebody_to_openpose`).
- **Inputs:** single video, a video-directory tree, one image sequence, or a
  tree of image-sequence folders (`--video / --video-dir / --frames-dir /
  --frames-root`).
- **Cleaning:** reuses Phase A `interpolate_gaps`, then applies a **One-Euro**
  temporal filter — kills jitter on still joints, stays responsive on fast
  hand motion. Both are toggleable (`--no-interp`, `--no-smooth`).
- **Outputs:** per-frame `*_keypoints.json` (CMU schema), `manifest.json`,
  optional `keypoints.npz` (keys match `mosl/pose/export_openpose_json.py`),
  and optional Phase A ControlNet pose frames in the same pass (`--render`).
- **Production:** logging, per-frame and per-clip error isolation (one bad
  frame/clip never aborts the batch), resume/idempotency, `--limit`, `--check`.

> Untested in this repo's environment (needs the DGX GPU + rtmlib). The
> COCO-WholeBody→OpenPose conversion and One-Euro smoothing are unit-checked;
> run `--check` on the DGX before the first batch.

---

## Phase C — Identity (InstantID + IP-Adapter, from your photos)

**Goal.** Lock the avatar's face, skin tone, and hairstyle to the
user-provided photos across every frame.

| Component | Choice | Setting |
|---|---|---|
| Face/keyframe model | **FLUX.1-dev** (best realism) or **SDXL** | 1024×1024 |
| Identity (structure) | **InstantID** | IdentityNet 0.80, image-adapter 0.80 |
| Identity (texture) | **IP-Adapter FaceID Plus v2** | weight 0.65–0.85 |
| Pose on keyframe | ControlNet-OpenPose | weight 0.70–0.90 |

**Delivered.** `identity.py` takes a folder of target photos → detects, aligns,
and encodes each face with `insightface` ArcFace (`antelopev2`) → fuses the
per-image embeddings into one identity (outlier-filtered) → saves a reusable
embedding + face crops + metadata.

```bash
python -m mosl.render.identity --check                       # validate env
python -m mosl.render.identity --input photos/<person>/ --viz
python -m mosl.render.identity --write-default-config identity.json
```

Key properties:

- **Encoder:** `insightface` ArcFace 512-d embedding — the exact model both
  InstantID and IP-Adapter FaceID consume. Same onnxruntime backend as Phase B,
  no new framework.
- **Functions:** `load_images`, `detect_and_align_face`,
  `extract_identity_features`, `fuse_multi_image_identity`, `save_embeddings`,
  `load_embeddings` — `load_embeddings(id, root, as_torch=True)` hands Phase D
  ready tensors.
- **Multi-image fusion:** mean of L2-normalized embeddings with outlier
  rejection (a stray photo of the wrong person is dropped automatically).
- **QC metric:** mean/min pairwise cosine — warns if the photo set is
  inconsistent, *before* you waste a Phase D run on a weak identity.
- **Outputs:** `embeddings/<id>.npz`, `metadata/<id>.json`,
  `aligned/<id>/` (aligned crops + face crops + `reference_face.png`).
- **Config:** JSON config file support (`--config`), CLI overrides it.

> Phase C is independent of Phase B — identity and pose are orthogonal inputs
> that meet only at Phase D, linked by `identity_id`.
>
> **Phase D integration note.** Use `fused` (normalized) as the InstantID /
> IP-Adapter FaceID embedding. `reference_kps` is for an InstantID *keyframe*
> only — per-frame face control must come from Phase B DWPose, not the static
> reference photo. Do not push identity strength above ~0.85 or the face goes
> plastic.

---

## Phase D — Reference keyframe (`keyframe.py`)

**Problem.** Phase C produced a face *embedding*. MimicMotion (Phase E) needs a
full *reference image* of the person to animate — that image does not exist yet.

**Delivered.** `keyframe.py` turns the identity embedding into one photoreal
reference image via **InstantID** (SDXL + IdentityNet) and picks the best of N
candidates by re-scoring each against the Phase C identity.

```bash
python -m mosl.render.keyframe --check                    # validate env
python -m mosl.render.keyframe --identity-id <person> --variants 4
```

Key properties:

- **Method:** InstantID on a realism SDXL base (RealVisXL); consumes the
  Phase C ArcFace embedding directly. For a FLUX identity path, swap the
  backend for PuLID-FLUX (InstantID proper is SDXL-only).
- **Best-of-N:** generates N variants, re-encodes each face with the Phase C
  encoder, keeps the highest identity-cosine as `keyframe.png`. Warns if even
  the best is weak — before a Phase E run is spent on it.
- **Default prompts:** the still-image subset of the project's cinematic
  avatar brief; override via `--config`.
- **Outputs:** `keyframes/<id>/keyframe.png` + `variant_NN.png` +
  `face_kps.png` + `manifest.json` (per-variant identity scores).

> Untested here (needs the DGX GPU + InstantID weights). Geometry/IO is
> unit-checked; run `--check` first.

---

## Phase E — Video model (MimicMotion)

**Recommended architecture.** Do **not** hand-assemble
AnimateDiff + ControlNet + IP-Adapter — that stack flickers and drifts
identity. Use a purpose-built pose-to-video model:

| Model | Use it when | Notes |
|---|---|---|
| **MimicMotion** (Tencent) | **default** | SVD-based, native pose conditioning, confidence-aware loss, region face enhancement |
| UniAnimate | long sequences / tighter VRAM | strong long-range temporal consistency |
| StableAnimator | identity is the #1 priority | explicit ID-preservation module |
| Champ | full-body, 3D-accurate motion | SMPL-based, heavier |
| AnimateDiff + CN-OpenPose | fast prototyping only | flickers; not for final output |

> "FLUX video" and "SDXL video" are **not standalone things**. FLUX is
> image-only; SDXL animates only via AnimateDiff. Use FLUX/SDXL for the
> **reference keyframe** (Phase D), a dedicated video model for **motion**.

**Delivered.** `video.py` drives MimicMotion through its **supported entry
point** — it generates an `inference.py` YAML config and runs the official
script — rather than calling MimicMotion's internal Python API, which changes
across versions. Inputs: the Phase D `keyframe.png` + a MoSL sign clip.

```bash
python -m mosl.render.video --check                          # validate setup
python -m mosl.render.video --identity-id <person> --driving-video clip.mp4
python -m mosl.render.video --identity-id <person> --driving-dir signs/
```

- **Pose source tradeoff (important).** MimicMotion runs its *own* DWPose on
  the driving video and retargets it to the keyframe body — that preprocessing
  is coupled to its training, so this path does **not** consume the Phase B
  keypoints. The *driving video* is the MoSL motion source. Phase B's JSON
  still serves the Phase A ControlNet path and analysis. Feeding Phase B poses
  directly to `MimicMotionPipeline.image_pose` is possible but bypasses the
  trained retargeting and risks quality — not done here.
- **Long clips:** MimicMotion tiles internally (72-frame tiles, 6 overlap).
- **Known limitation:** hand fidelity on fast signing is a recognized
  MimicMotion weak point even with v1.1's regional hand loss.
- Batch over a folder of clips; per-clip error isolation; `--check`.

> Untested here (needs the DGX GPU + MimicMotion weights). Pure-Python logic is
> unit-checked; the output-file discovery is version-dependent (one-place-fixable
> in `_newest_video_since`).

### Recommended generation settings

| Parameter | Value | Reason |
|---|---|---|
| Sampler | EDM / Euler (MimicMotion native) | matches SVD training |
| CFG scale | **2.0–2.5** | high CFG flickers in video |
| Chunk length | 72 frames | MimicMotion context window |
| Chunk overlap | 6 frames | cross-fade for seam-free long clips |
| Pose (ControlNet) weight | 0.70–0.90 | lower if hands look over-constrained |
| Steps | 25 | quality/speed knee |
| Generate FPS | 12–15 | interpolate to 25–30 in Phase F |
| Resolution | 576×1024 or 768×768 | upscale in Phase F |
| Seed | fixed | reproducibility across chunks |

---

## Phase F — Temporal polish (`temporal.py`)

**Delivered.** `temporal.py` turns the raw MimicMotion output into a finished
clip. **ffmpeg is the backbone** — its `deflicker` and `minterpolate` filters
are built in and need no model weights, so the default path runs anywhere.

```bash
python -m mosl.render.temporal --check                       # validate ffmpeg
python -m mosl.render.temporal --input-video raw.mp4 --target-fps 30
python -m mosl.render.temporal --input-dir outputs/avatar_video/<id>/ --upscale
```

| Step | Default (ffmpeg) | Higher-quality opt-in |
|---|---|---|
| Deflicker | `deflicker` filter | — |
| Interpolation | `minterpolate` (motion-compensated) → 30 fps | **RIFE** (`--interp-backend rife`) |
| Upscale (opt-in) | `scale=lanczos` | **Real-ESRGAN** (`--upscale-backend realesrgan`) |
| Final encode | libx264, CRF 18, faststart | — |

Each step writes a near-lossless intermediate so it stays independently
inspectable. Batch via `--input-dir`. RIFE / Real-ESRGAN are wrapped as
subprocesses (output located by recency — version-dependent).

---

## Phase G — End-to-end pipeline (`scripts/render_avatar.py`)

**Delivered.** `scripts/render_avatar.py` runs Phases C→D→E→F as one command;
each stage is skipped if its output exists (unless `--force`).

```bash
# first run for a person: build identity from photos, drive by Arabic word
python scripts/render_avatar.py --photos photos/omar/ --text "الأذان"
# reuse the identity, drive with an explicit clip, upscale
python scripts/render_avatar.py --identity-id omar --driving-video clip.mp4 --upscale
```

```
photos ─► [C identity] ─► [D keyframe] ─┐
                                        ├─► [E MimicMotion] ─► [F polish] ─► MP4
Arabic word / sign video ───────────────┘
```

`--text` resolves an Arabic word to its MoSL clip via `data/labels.csv`
(NFC-normalized, with a stripped-form fallback).

> **Motion source.** `docs/RESULTS.md` shows the SignLLM model output loses to
> retrieval baselines, so Phase G drives the avatar with the **real MoSL sign
> clip** for the word — not the model's generated pose. The MimicMotion path is
> video-driven by design; wiring the SignLLM `text→pose` output in would need
> the direct-pose-injection path (see Phase E notes).

---

## DGX Spark environment notes (important)

The DGX Spark is **aarch64 + Blackwell (SM 12.1) + CUDA 13**. Most diffusion
tooling assumes x86_64. Lessons from this repo's `docs/DECISIONS.md` apply:

- **Use the NGC PyTorch container** (`nvcr.io/nvidia/pytorch:26.04-py3`) as the
  base — it has native SM_120 kernels. Stock `cu12x` wheels run via slow PTX-JIT.
- **`xformers` / `flash-attention` have no reliable aarch64 wheels.** Use
  PyTorch **SDPA** (`scaled_dot_product_attention`) instead — diffusers enables
  it automatically. SageAttention may build from source.
- **ComfyUI custom nodes** with compiled CUDA extensions can fail to build on
  aarch64 — prefer pure-Python nodes; test the InstantID/MimicMotion nodes
  early.
- 121 GB unified memory is generous — VRAM pressure is not the constraint;
  kernel/wheel compatibility is.

### Memory & speed optimization

`bf16` weights · `enable_model_cpu_offload()` · VAE tiling + slicing ·
SDPA attention · token merging (ToMe) for SDXL · chunked context windows ·
fixed seed across chunks · cache the FLUX/SDXL keyframe (generate once, reuse).

---

## ComfyUI workflow (node architecture)

A reference graph for the Phase D→E→F path (the actual `.json` lives in
`comfyui/avatar_workflow.json` once Phase E lands):

```
[Load Target Photos] ─► [InsightFace / FaceAnalysis]
                              │
[FLUX/SDXL Checkpoint] ───────┼─► [InstantID Apply] ─► [IP-Adapter FaceID]
                              │            │
[Pose frames: pose_bridge] ─► [ControlNet-OpenPose Apply]
                                           │
                                  [KSampler] ─► reference keyframe
                                           │
[MimicMotion Loader] ◄─ keyframe + pose-frame batch
        │
   [MimicMotion Sampler]  (chunked, overlap 6)
        │
   [RIFE VFI] ─► [Upscale (SUPIR/RealESRGAN)] ─► [Video Combine → MP4]
```

---

## Debugging strategy

| Symptom | Likely cause | Fix |
|---|---|---|
| Flicker between frames | conditioning pops; CFG too high | enable `--interp`, lower CFG to ~2, FreeInit |
| Identity drifts mid-clip | IP-Adapter weight too low; no chunk overlap | raise FaceID weight; overlap 6–8 frames |
| Melted / extra fingers | sparse or noisy hand keypoints | Phase B DWPose re-extraction (21-pt hands) |
| Plastic / uncanny face | identity strength too high | InstantID ≤ 0.85, FaceID ≤ 0.85 |
| Pose ignored | ControlNet weight too low; wrong skeleton format | raise weight; verify COCO-18 vs DWPose |
| Camera jitter | per-frame re-framing | already handled — one global fit transform |
| OOM / slow | PTX-JIT, no offload | NGC container, CPU offload, VAE tiling |

Validate each phase in isolation before chaining: a single pose frame
(Phase A) → a single keyframe (Phase D) → a short 24-frame clip (Phase E) →
full clip (Phase F).
