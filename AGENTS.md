# AGENTS.md

Agent guidance for the **Multimodal Moroccan Sign Language Generation** project.

---

## Project overview

This repository targets automatic generation of Moroccan Sign Language (MSL/LSM) from text or speech input using multimodal deep-learning techniques. The expected stack is Python-based (PyTorch or TensorFlow), with data pipelines, model training scripts, and evaluation tooling.

> **Status:** repository initialised — no source code committed yet.  
> Update this file as the codebase grows.

---

## Repository layout

```
.
├── mosl/
│   ├── render/
│   │   ├── pose_bridge.py          # Phase A: OpenPose JSON → ControlNet skeleton PNGs
│   │   ├── dwpose_extract.py       # Phase B: DWPose whole-body keypoint extraction
│   │   ├── identity.py             # Phase C: ArcFace identity extraction & fusion
│   │   ├── keyframe.py             # Phase D: InstantID SDXL reference keyframe
│   │   ├── video.py                # Phase E: MimicMotion avatar video (SVD-based)
│   │   ├── animatediff_backend.py  # Phase E alt: AnimateDiff + ControlNet-OpenPose
│   │   ├── temporal.py             # Phase F: RIFE interpolation + Real-ESRGAN
│   │   └── test_render.py          # 32 unit tests (CPU-only, no GPU required)
│   ├── model/                      # SignLLM transformer (text → pose)
│   ├── data/                       # Dataset loaders
│   └── train/                      # Training loops
├── configs/
│   ├── keyframe_config.yaml        # Phase D production config
│   ├── animatediff_config.yaml     # Phase E AnimateDiff config
│   ├── temporal_config.yaml        # Phase F temporal polish config
│   └── comfyui_avatar_workflow.json # ComfyUI node graph
├── docs/
│   ├── PIPELINE.md                 # Data preprocessing pipeline
│   ├── AVATAR_ARCHITECTURE.md      # Full architecture analysis & roadmap
│   ├── DECISIONS.md                # Architecture decision records
│   └── PHASE_C_IDENTITY.md         # Phase C integration guide
├── scripts/
│   ├── render_avatar.py            # End-to-end orchestrator (Phases C–F)
│   └── git_push_phase.sh           # Git workflow automation
├── data/                           # CSV metadata (committed); raw video (excluded)
├── outputs/                        # All generated outputs (excluded by .gitignore)
├── identity_config.yaml            # Phase C config template
├── requirements.txt
├── requirements-render.txt
└── AGENTS.md                       # This file
```

---

## Environment setup

The dev container uses `mcr.microsoft.com/devcontainers/universal:4.0.1-noble`.  
Once source code is added, replace it with a leaner Python image and pin dependencies:

```jsonc
// .devcontainer/devcontainer.json
{
  "image": "mcr.microsoft.com/devcontainers/python:3.11",
  "postCreateCommand": "pip install -r requirements.txt"
}
```

---

## Coding conventions

- **Language:** Python 3.10+
- **Style:** PEP 8, enforced by `ruff` (line length 100)
- **Type hints:** required on all public functions and class methods
- **Docstrings:** Google style
- **Tests:** `pytest`; place under `tests/` mirroring `src/` structure
- **Notebooks:** exploratory only; strip outputs before committing

---

## Architecture decisions (non-negotiable)

These decisions are final. Do not override them without an explicit instruction
from the project owner.

### Video generation backends

Two backends exist and **both must be preserved**:

| Backend | Module | Use case |
|---|---|---|
| MimicMotion | `mosl/render/video.py` | Photorealistic full-body motion (cinematic avatar clips) |
| AnimateDiff | `mosl/render/animatediff_backend.py` | Sign-language clips — hand-critical motion accuracy |

**Do NOT:**
- Remove either backend
- Merge them into a single module
- Make one backend depend on the other's internals

**Do NOT modify MimicMotion internals.** MimicMotion runs its own DWPose
internally. Do not patch it to accept Phase B outputs — this is fragile and
version-dependent. Use AnimateDiff when Phase A/B pose outputs must be consumed
directly.

Backend selection is controlled by `--backend mimicmotion|animatediff` in
`scripts/render_avatar.py` and is fully modular.

### Frame interpolation

**RIFE is the default interpolation backend.** `ffmpeg minterpolate` is
available only as a fallback.

Rationale: `minterpolate` produces visible ghosting on fast hand motion.
Sign language requires high temporal precision. RIFE uses optical-flow
synthesis and handles rapid hand motion cleanly.

- `TemporalConfig.interp_backend` default: `"rife"`
- `render_avatar.py --interp-backend` default: `"rife"`
- ffmpeg fallback fires automatically when RIFE is not installed

**Do NOT change these defaults back to ffmpeg.**

### Motion quality priorities

When making trade-offs, apply this priority order:

1. Hand accuracy
2. Temporal consistency
3. Identity consistency
4. Smooth body motion
5. Photorealism
6. Low flickering

### Phase A/B output contract

AnimateDiff consumes Phase A skeleton PNGs (`pose_*.png`) via ControlNet-OpenPose.
MimicMotion consumes a raw driving video MP4 and runs its own DWPose internally.
These are orthogonal — do not cross-wire them.

### Engineering rules

- Prefer stable integrations over experimental patches
- Preserve upstream compatibility (MimicMotion, AnimateDiff, InstantID)
- Reuse Phase B outputs only through AnimateDiff's ControlNet path
- Keep all phase modules independently runnable (`python -m mosl.render.<module>`)
- Every new backend must expose a `--check` CLI flag for environment validation

---

## Agent rules

### General

- Read the file before editing it.
- Do not commit data files, model checkpoints, or generated outputs.
- Do not hardcode paths — use `pathlib.Path` and config files.
- Do not expose API keys or dataset credentials in code or logs.

### When adding a new model

1. Define the architecture in `src/models/`.
2. Add a config schema in `configs/`.
3. Register the model in `src/models/__init__.py`.
4. Add at least one unit test in `tests/models/`.

### When modifying data pipelines

1. Verify the pipeline against a small sample before running on the full dataset.
2. Document expected input/output formats in the module docstring.
3. Add a smoke test that runs in < 5 seconds without GPU.

### Commit messages

Follow Conventional Commits:

```
<type>(<scope>): <short description>

Types: feat | fix | refactor | test | docs | chore | perf
```

Example: `feat(models): add transformer-based pose decoder`

### Pull requests

- Target `main` unless working on a feature branch.
- PR title must follow the same Conventional Commits format.
- Include: motivation, what changed, how to test.
- Do not merge without passing CI.

---

## Testing

```bash
pytest tests/ -v --tb=short
```

For GPU-dependent tests, mark with `@pytest.mark.gpu` and skip in CI unless a GPU runner is available.

---

## Known constraints

- MSL dataset is not publicly available; agents must not attempt to download or generate synthetic data without explicit instruction.
- Model checkpoints can be large (>1 GB); store them in a separate artifact store, not in git.
