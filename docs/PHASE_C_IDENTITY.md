# Phase C — Identity Extraction

`mosl/render/identity.py` — face identity extraction and preservation for
diffusion-based avatar video generation.

---

## What it does

Encodes a person's face from a folder of photos into a 512-d ArcFace embedding
(insightface `antelopev2`) that Phase D feeds directly to InstantID or
IP-Adapter FaceID. Phase C does nothing else — it does not touch pose
extraction (Phase B) and does not run diffusion (Phase D).

---

## Output layout

```
outputs/identity/
    identity_embeddings/<identity_id>.npz   ← primary output for Phase D
    identity_metadata/<identity_id>.json    ← QC metrics and file index
    aligned/<identity_id>/
        <stem>_aligned112.png               ← 112×112 ArcFace-aligned crop
        <stem>_facecrop.png                 ← padded crop (crop_size × crop_size)
        reference_face.png                  ← best crop; Phase D CLIP input
        alignment_viz.png                   ← QC visualization (--viz only)
        aligned_grid.png                    ← contact sheet (--save-aligned-grid)
```

The NPZ contains four arrays:

| Key | Shape | Description |
|---|---|---|
| `fused` | `(512,)` | L2-normalized fused embedding — primary Phase D input |
| `fused_raw` | `(512,)` | Un-normalized fused embedding |
| `per_image` | `(N, 512)` | Per-image normalized embeddings |
| `reference_kps` | `(5, 2)` | Five-point landmarks of the best-scoring image |

---

## Quick start

```bash
# Single identity
python -m mosl.render.identity --input photos/omar/ --viz

# All sub-folders in one shot
python -m mosl.render.identity --batch photos/

# With a config file
python -m mosl.render.identity --config identity_config.yaml --input photos/omar/

# List saved identities
python -m mosl.render.identity --list

# Validate the insightface environment (no data needed)
python -m mosl.render.identity --check
```

---

## Public API

```python
from mosl.render.identity import (
    IdentityConfig, load_config,
    load_images, detect_and_align_face,
    extract_identity_features, fuse_multi_image_identity,
    save_embeddings, load_embeddings,
    process_batch, list_identities,
    visualize_alignment, build_identity,
)

# Single identity — full pipeline
cfg = IdentityConfig(input_dir="photos/omar/", identity_id="omar", device="cuda")
metadata = build_identity(cfg)

# Batch — all sub-folders
results = process_batch(Path("photos/"), cfg)

# Load for Phase D (numpy)
identity = load_embeddings("omar", Path("outputs/identity"))
face_emb = identity["fused"]          # (512,) normalized ArcFace embedding
ref_kps  = identity["reference_kps"]  # (5,2)  for InstantID IdentityNet

# Load as torch tensors (InstantID / IP-Adapter ingest directly)
identity = load_embeddings("omar", Path("outputs/identity"), as_torch=True)
face_emb = identity["fused"]          # torch.Tensor (512,)
```

---

## Phase D integration

### InstantID (SDXL)

```python
identity = load_embeddings(identity_id, out_root, as_torch=True)

# face_emb -> image-projection adapter
# reference_kps -> draw_kps() -> IdentityNet ControlNet conditioning
# Per-frame face control comes from Phase B DWPose, not from this embedding
pipeline(
    prompt=prompt,
    image_embeds=identity["fused"].unsqueeze(0),   # (1, 512)
    image=kps_image,                                # from reference_kps
    controlnet_conditioning_scale=0.8,
)
```

### IP-Adapter FaceID Plus (SDXL)

```python
identity = load_embeddings(identity_id, out_root, as_torch=True)

# faceid_embeds: the ArcFace embedding from Phase C
# face_image: reference_face.png — the adapter's CLIP encoder runs on this
pipeline.set_ip_adapter_scale(0.7)
pipeline(
    prompt=prompt,
    ip_adapter_image=Image.open(identity["reference_face_path"]),
    ip_adapter_image_embeds=[identity["fused"].unsqueeze(0)],
)
```

### PuLID-FLUX

```python
identity = load_embeddings(identity_id, out_root)
# PuLID accepts the same ArcFace embedding directly
pipeline(prompt=prompt, id_embeddings=identity["fused"])
```

---

## Integration with Phase B (DWPose)

Identity and pose are orthogonal. They meet only at Phase D, linked by
`identity_id`. The Phase B DWPose output directory for a clip is:

```
outputs/dwpose/<clip_name>/
```

Phase D resolves the identity by `identity_id` and the pose by `clip_name`.
No file-level coupling exists between Phase B and Phase C outputs.

---

## Configuration

All parameters are documented in `identity_config.yaml` at the project root.
Key parameters:

| Parameter | Default | Effect |
|---|---|---|
| `det_model` | `antelopev2` | InsightFace pack; `buffalo_l` is lighter |
| `outlier_threshold` | `0.50` | Drop images below this cosine-to-mean |
| `consistency_warn` | `0.55` | Warn if mean pairwise cosine below this |
| `face_select` | `largest` | Which face to keep when multiple are detected |
| `crop_size` | `512` | Edge of the padded crop saved for Phase D |

---

## QC signals

The metadata JSON includes a `consistency` block:

```json
"consistency": {
  "mean_pairwise_cosine": 0.91,
  "min_pairwise_cosine": 0.87,
  "n_images": 5
}
```

- **> 0.80**: good — photos are consistent, expect strong identity preservation.
- **0.55–0.80**: acceptable — some variation (lighting, angle); identity will hold.
- **< 0.55**: warning logged — photos may show different people or extreme variation.

---

## Dependencies

```
insightface          # ArcFace detection + recognition
onnxruntime-gpu      # ONNX backend for insightface (use onnxruntime for CPU)
opencv-python        # image I/O and crop geometry
pyyaml               # YAML config support
numpy                # embedding arithmetic
torch                # optional — only for as_torch=True in load_embeddings
```

Install on DGX:
```bash
pip install insightface onnxruntime-gpu opencv-python pyyaml
```
