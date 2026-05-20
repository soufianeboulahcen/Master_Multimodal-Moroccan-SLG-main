# DGX Spark setup — avatar rendering subsystem

How to install and validate Phases B–D of `mosl/render/` on the DGX Spark
(GB10 Grace-Blackwell, aarch64, CUDA 13). Do this before running any batch.

The original `mosl/` SignLLM pipeline is untouched by all of this.

---

## 1. Container

Use the NGC PyTorch container — it ships PyTorch with native Blackwell SM_120
kernels. A plain `pip install torch` gives a PTX-JIT build that is far slower.

```bash
docker run --gpus all -it --rm \
  -v "$PWD":/workspace -w /workspace \
  nvcr.io/nvidia/pytorch:26.04-py3 bash
```

(Or extend `docker/Dockerfile` — the project already has a Docker workflow.)

## 2. Dependencies

```bash
pip install -r requirements-render.txt
```

`torch` / `torchvision` are deliberately **not** in that file — keep the
container's. If a package tries to pull torch as a dependency, add
`--no-deps` for it and install its other deps by hand.

### Phase C specific

```bash
# Core (required)
pip install insightface onnxruntime-gpu opencv-python pyyaml

# Optional — only needed if Phase D uses IP-Adapter FaceID Plus CLIP path
pip install transformers accelerate
```

Phase C saves the 112×112 ArcFace-aligned crop and the padded face crop.
Phase D computes the CLIP embedding from the crop using its own encoder —
Phase C itself has no CLIP dependency.

### aarch64 gotchas

| Symptom | Fix |
|---|---|
| `onnxruntime-gpu` has no aarch64 CUDA wheel | install `onnxruntime` (CPU) and run Phase B/C with `--device cpu`, or build ORT from source |
| `xformers` / `flash-attn` won't install | not needed — diffusers uses PyTorch SDPA automatically |
| a ComfyUI/diffusers node needs a compiled CUDA ext | prefer the pure-Python path; most diffusers pipelines are pure-Python |
| InsightFace model download blocked | pre-download the `antelopev2` pack to `~/.insightface/models/` |

## 3. Model weights

| Phase | Model | How |
|---|---|---|
| B | DWPose / RTMPose | auto-downloaded by `rtmlib` on first run |
| C | InsightFace `antelopev2` | auto-downloaded on first run (or place manually) |
| D | SDXL realism base | `huggingface-cli download SG161222/RealVisXL_V5.0` |
| D | InstantID weights | `huggingface-cli download InstantX/InstantID` (ControlNetModel + `ip-adapter.bin`) |
| D | InstantID pipeline code | `git clone https://github.com/instantX-research/InstantID` |
| E | MimicMotion + SVD-XT-1.1 | `git clone https://github.com/Tencent/MimicMotion`; `huggingface-cli download tencent/MimicMotion` |

Set a cache location with space: `export HF_HOME=/workspace/.hf_cache`.

---

## 4. Validate each phase (do this in order)

Each module has a `--check` that loads its model and runs one inference on a
blank frame — no project data needed. All three must print `OK` before batching.

```bash
# Phase B — DWPose
python -m mosl.render.dwpose_extract --check

# Phase C — identity encoder
python -m mosl.render.identity --check

# Phase D — keyframe / InstantID
python -m mosl.render.keyframe --check
```

If `--check` fails on the CUDA provider, retry with `--device cpu` to confirm
the rest works, then resolve the onnxruntime/CUDA issue separately.

## 5. Smoke tests (one clip / one identity)

```bash
# B: extract one sign clip, smoothed, and render ControlNet pose frames
python -m mosl.render.dwpose_extract --video <one_sign>.mp4 --render --npz

# C: encode an identity from a folder of photos
python -m mosl.render.identity --input photos/<person>/ --viz

# D: generate the photorealistic reference keyframe for that identity
python -m mosl.render.keyframe --identity-id <person> --variants 4
```

Inspect before scaling up:

- **B** — open `outputs/pose_control/<clip>/preview.gif` (pass `--preview`):
  the skeleton should track the signer cleanly, hands intact.
- **C** — check `outputs/identity/metadata/<id>.json`:
  `consistency.mean_pairwise_cosine` should be **> 0.55**.
- **D** — check `outputs/keyframes/<id>/manifest.json`:
  the chosen variant's `identity_cosine` should be **> 0.55** (ideally > 0.65).

## 6. Full batch

```bash
python -m mosl.render.dwpose_extract --video-dir data/raw/vedios-dataset \
    --render --npz
```

---

## Pipeline data flow

```
target photos ─► [C identity] ─► embedding ─┐
                                            ├─► [D keyframe] ─► reference image
sign videos ─► [B dwpose] ─► OpenPose JSON ─┘                          │
                    │                                                  │
                    └─► [A pose_bridge] ─► pose frames ────────────────►├─► [E MimicMotion] ─► clip
                                                                            │
                                                              [F temporal] ─┴─► final MP4
```
