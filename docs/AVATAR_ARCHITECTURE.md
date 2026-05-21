# Avatar Video Generation — Architecture Analysis & Roadmap

Analysis date: 2026-05-20  
Analyst: Phase D/E/F deep-read of the full `mosl/render/` subsystem.

---

## 1. Current architecture — what is already built

```
Arabic text
    │
    ▼
mosl/model/signllm.py          SignLLM Transformer (text → pose sequence)
    │  (T, 150) pose tensor
    ▼
mosl/pose/export_openpose_json.py   unpack to per-frame CMU OpenPose JSON
    │
    ├──────────────────────────────────────────────────────────────────────┐
    │  CHOREOGRAPHER (existing, complete)                                  │
    │  mosl/data/, mosl/train/, mosl/text/                                │
    └──────────────────────────────────────────────────────────────────────┘
    │
    ▼  OpenPose JSON  (the contract between the two halves)
    │
    ├── Phase A  pose_bridge.py      JSON → ControlNet skeleton PNGs   [DONE, CPU]
    ├── Phase B  dwpose_extract.py   DWPose re-extraction              [DONE, DGX]
    ├── Phase C  identity.py         ArcFace embedding extraction      [DONE, DGX]
    ├── Phase D  keyframe.py         InstantID → reference image       [DONE, DGX]
    ├── Phase E  video.py            MimicMotion → avatar video        [DONE, DGX]
    ├── Phase F  temporal.py         deflicker + RIFE + Real-ESRGAN    [DONE, DGX]
    └── Phase G  scripts/render_avatar.py   end-to-end orchestrator   [DONE]
```

### Completed modules (production-ready)

| Module | Status | Notes |
|---|---|---|
| `pose_bridge.py` | ✅ Complete | COCO-18 + BODY-25 auto-detect, gap interpolation, GIF preview |
| `dwpose_extract.py` | ✅ Complete | rtmlib backend, One-Euro smoothing, OpenPose JSON output |
| `identity.py` | ✅ Complete | ArcFace fusion, batch processing, YAML config, QC metrics |
| `keyframe.py` | ✅ Complete | InstantID SDXL, best-of-N scoring, face-kps geometry |
| `video.py` | ✅ Complete | MimicMotion via YAML config, batch driving, output discovery |
| `temporal.py` | ✅ Complete | ffmpeg deflicker + minterpolate, RIFE opt-in, Real-ESRGAN opt-in |
| `render_avatar.py` | ✅ Complete | 4-stage orchestrator, text→clip lookup, skip-if-exists |
| `test_render.py` | ✅ Complete | 32 unit tests, all passing |

---

## 2. Gap analysis — what needs improvement

### 2.1 Phase D — keyframe generation

| Gap | Severity | Fix |
|---|---|---|
| `render_avatar.py` uses old `embeddings/` path (pre-Phase C rewrite) | 🔴 Bug | Update to `identity_embeddings/` |
| No YAML config support (only JSON) | 🟡 Minor | Add YAML loader (same pattern as identity.py) |
| No IP-Adapter FaceID Plus path | 🟡 Missing | Add as alternative backend |
| `KeyframeConfig` has no `save_viz` flag | 🟡 Minor | Add face-kps visualization save |
| `identitynet_strength` default 0.80 — too high for natural poses | 🟡 Tuning | Lower to 0.65, document rationale |
| No `--list` command to show available keyframes | 🟢 Nice-to-have | Add |

### 2.2 Phase E — video generation

| Gap | Severity | Fix |
|---|---|---|
| MimicMotion runs its own DWPose — Phase B output is unused | 🟡 Design | Document clearly; add AnimateDiff path that uses Phase B/A output |
| No AnimateDiff backend | 🔴 Missing | Add `animatediff_backend.py` using Phase A pose frames as ControlNet |
| No ControlNet-OpenPose conditioning path | 🔴 Missing | AnimateDiff + ControlNet uses Phase A skeleton PNGs directly |
| `guidance_scale=2.0` — SVD default, but too low for identity lock | 🟡 Tuning | Raise to 3.0–3.5 for better identity adherence |
| No `--frames-overlap` CLI flag | 🟢 Minor | Expose in CLI |
| No progress callback / ETA | 🟢 Minor | Add tqdm wrapper |

### 2.3 Phase F — temporal polish

| Gap | Severity | Fix |
|---|---|---|
| `minterpolate` produces ghosting on fast hand motion | 🔴 Quality | Default to RIFE when available; ffmpeg as fallback |
| No face-region deflicker (whole-frame deflicker blurs edges) | 🟡 Quality | Add face-crop stabilization pass |
| No audio track handling | 🟢 Minor | Pass through audio if present |
| `_newest_video_since` is fragile for concurrent runs | 🟡 Robustness | Add explicit output path config |

### 2.4 End-to-end orchestrator

| Gap | Severity | Fix |
|---|---|---|
| `render_avatar.py` uses old embedding path | 🔴 Bug | Fix path |
| No `--animatediff` flag to select backend | 🔴 Missing | Add backend selection |
| No dry-run / status check mode | 🟡 Minor | Add `--status` flag |
| No progress summary at end | 🟢 Minor | Add timing table |

---

## 3. Recommended diffusion settings

### Phase D — InstantID SDXL keyframe

```yaml
sdxl_model: SG161222/RealVisXL_V5.0
steps: 30                    # 25 minimum, 35 for best quality
guidance_scale: 5.0          # 4.5–6.0 sweet spot for realism
identitynet_strength: 0.65   # face structure (was 0.80 — too rigid)
adapter_strength: 0.80       # face texture / identity
num_variants: 6              # more candidates = better identity match
width: 832
height: 1216                 # portrait aspect ratio
sampler: DPM++ 2M Karras     # best quality/speed for SDXL
```

**Why lower `identitynet_strength`:** 0.80 forces the face into the exact
landmark positions, which can produce an unnatural stiff expression. 0.65
gives the model room to produce a natural neutral expression while still
preserving face structure.

### Phase E — MimicMotion (SVD-based)

```yaml
num_frames: 72               # MimicMotion tile size (do not change)
frames_overlap: 6            # cross-fade between tiles
resolution: 576              # 576 is the SVD training resolution
num_inference_steps: 25      # 20 minimum, 30 for best quality
guidance_scale: 3.0          # raised from 2.0 for better identity lock
noise_aug_strength: 0.0      # 0 = cleanest; raise to 0.02 for variation
sample_stride: 2             # driving video frame stride
fps: 15                      # generate at 15, interpolate to 30 in Phase F
seed: 42
```

### Phase E — AnimateDiff + ControlNet-OpenPose (alternative)

```yaml
# Uses Phase A skeleton PNGs directly — no MimicMotion dependency
base_model: SG161222/RealVisXL_V5.0
motion_module: guoyww/animatediff-motion-adapter-v1-5-3
controlnet: lllyasviel/control_v11p_sd15_openpose
controlnet_scale: 0.85       # pose adherence
ip_adapter: h94/IP-Adapter   # FaceID for identity
ip_adapter_scale: 0.6
steps: 25
guidance_scale: 7.5
num_frames: 16               # AnimateDiff native window
fps: 8                       # generate at 8, interpolate to 25 in Phase F
```

### Phase F — temporal polish

```yaml
deflicker: true
deflicker_size: 5            # smaller window = less blur
interpolate: true
interp_backend: rife         # RIFE >> ffmpeg minterpolate for hand motion
target_fps: 30
upscale: true                # enable for delivery
upscale_factor: 2
upscale_backend: realesrgan  # Real-ESRGAN >> lanczos for skin texture
realesrgan_model: realesr-general-x4v3
crf: 18                      # H.264 quality (lower = better, 18 is near-lossless)
preset: slow
```

---

## 4. Identity preservation strategy

```
Phase C ArcFace embedding (512-d)
    │
    ├── InstantID path (Phase D)
    │       IdentityNet ControlNet: face structure lock
    │       IP-Adapter: face texture / appearance
    │       → keyframe.png (the person, neutral pose)
    │
    └── IP-Adapter FaceID Plus path (alternative Phase D)
            ArcFace embedding + CLIP crop embedding
            → keyframe.png
    │
    ▼
Phase E: keyframe.png drives MimicMotion / AnimateDiff
    │
    ├── MimicMotion: SVD-based, temporal attention, strong motion fidelity
    │   Weakness: hand detail on fast sign motion
    │
    └── AnimateDiff: motion module + ControlNet-OpenPose
        Strength: uses Phase A/B skeleton PNGs directly
        Weakness: lower temporal coherence than SVD
```

**Key insight:** Identity is locked at Phase D (the keyframe). Phase E
preserves it through SVD's reference-image conditioning. The ArcFace embedding
is NOT re-injected at Phase E — MimicMotion's SVD backbone handles identity
consistency through its reference-image attention mechanism.

---

## 5. Temporal consistency strategy

### Problem sources
1. **Frame-level flicker** — diffusion noise varies per frame
2. **Hand flicker** — fast sign motion exceeds MimicMotion's motion module capacity
3. **Face flicker** — identity drift across tiles when clip > 72 frames
4. **Lighting flicker** — inconsistent illumination between tiles

### Solutions (in order of impact)

| Solution | Phase | Impact |
|---|---|---|
| SVD temporal attention (built into MimicMotion) | E | High — handles most body flicker |
| `frames_overlap=6` tile cross-fade | E | High — eliminates tile seams |
| RIFE frame interpolation | F | High — smooths residual motion jitter |
| ffmpeg `deflicker` filter | F | Medium — removes per-frame brightness flicker |
| Real-ESRGAN upscale | F | Medium — sharpens and stabilizes texture |
| `noise_aug_strength=0.0` | E | Medium — prevents noise-induced variation |
| `guidance_scale=3.0` (not 2.0) | E | Medium — stronger identity lock per frame |
| Best-of-N keyframe selection | D | Medium — starts from a better reference |

---

## 6. Photorealism enhancement

### Prompt engineering (Phase D)

```
POSITIVE:
ultra photorealistic portrait, upper body, neutral standing pose,
facing camera, cinematic DSLR photograph, soft cinematic lighting,
volumetric light, realistic skin pores and texture, detailed eyes,
shallow depth of field, high dynamic range, sharp focus, 8k,
professional studio background, Moroccan person, natural skin tone

NEGATIVE:
flickering, distorted anatomy, deformed face, extra fingers, bad hands,
missing fingers, blurry, low resolution, CGI, 3d render, cartoon, anime,
plastic skin, waxy skin, uncanny valley, warped body, asymmetric face,
watermark, text, logo, jpeg artifacts, overexposed, underexposed
```

### Lighting (Phase D)
- Use `soft cinematic lighting` + `volumetric light` in prompt
- Avoid `studio lighting` (too flat) — prefer `natural window light` or `golden hour`
- `shallow depth of field` separates subject from background naturally

### Resolution strategy
- Generate at 832×1216 (Phase D) — portrait aspect, SDXL native
- MimicMotion at 576px (Phase E) — SVD training resolution
- Upscale 2× with Real-ESRGAN (Phase F) → 1152px delivery

---

## 7. GPU memory optimization (DGX GB10)

| Technique | Saving | Where |
|---|---|---|
| `torch.float16` throughout | ~50% VRAM | Phase D, E |
| `enable_model_cpu_offload()` | Fits in 16 GB | Phase D fallback |
| `enable_xformers_memory_efficient_attention()` | ~20% VRAM | Phase D, E |
| Sequential tile processing (not batched) | Constant VRAM | Phase E |
| `torch.compile()` (PyTorch 2.x) | ~15% speed | Phase D |
| Shared FaceAnalyzer across batch | No reload overhead | Phase C |
| Skip-if-exists in orchestrator | No redundant compute | Phase G |

---

## 8. ComfyUI workflow

See `configs/comfyui_avatar_workflow.json` for the full node graph.

Key node chain:
```
LoadImage (reference_face.png)
    → IPAdapterFaceID
    → InstantIDModelLoader
    → ApplyInstantID
    → KSampler (DPM++ 2M Karras, 30 steps, CFG 5.0)
    → VAEDecode
    → SaveImage (keyframe.png)
    │
    ▼
LoadVideo (sign clip)
    → MimicMotionLoader
    → MimicMotionRun
    → VHS_VideoCombine (15fps)
    │
    ▼
RIFE VFI (2× interpolation → 30fps)
    → VideoDeflicker
    → UpscaleModelLoader (Real-ESRGAN)
    → ImageUpscaleWithModel
    → VHS_VideoCombine (final MP4)
```

---

## 9. Training roadmap (future)

The current system is **inference-only** — it uses pretrained models (InstantID,
MimicMotion, SVD). Fine-tuning is not required for the avatar generation goal.

If fine-tuning is desired later:

| Goal | Method | Data needed |
|---|---|---|
| Better MSL hand fidelity | LoRA on MimicMotion motion module | MSL video pairs |
| Moroccan face domain | DreamBooth on SDXL | 20–50 photos per person |
| Faster inference | Consistency distillation (LCM) | None (self-distillation) |
| Higher resolution | SDXL refiner stage | None |

---

## 10. Inference roadmap (immediate)

```bash
# 1. Validate environment (run once on DGX)
python -m mosl.render.identity --check
python -m mosl.render.keyframe --check
python -m mosl.render.video --check
python -m mosl.render.temporal --check

# 2. Build identity from photos
python -m mosl.render.identity --input photos/person/ --viz

# 3. Generate reference keyframe
python -m mosl.render.keyframe --identity-id person --variants 6

# 4. Generate avatar video (MimicMotion)
python -m mosl.render.video --identity-id person --driving-video sign.mp4

# 5. Polish (RIFE + Real-ESRGAN)
python -m mosl.render.temporal --input-video outputs/avatar_video/person/sign__person.mp4 \
    --interp-backend rife --upscale --target-fps 30

# OR: one command end-to-end
python scripts/render_avatar.py --photos photos/person/ --text "سلام" --upscale
```
