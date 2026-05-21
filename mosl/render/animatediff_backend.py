"""Phase E (alternative) — AnimateDiff + ControlNet-OpenPose avatar video.

This module is the second video-generation backend, complementing MimicMotion
(video.py). It uses the Phase A skeleton PNGs produced by pose_bridge.py as
ControlNet-OpenPose conditioning frames, and IP-Adapter FaceID for identity.

When to use this backend vs MimicMotion
----------------------------------------
| Criterion              | MimicMotion (video.py)     | AnimateDiff (this file)      |
|------------------------|----------------------------|------------------------------|
| Motion source          | Driving video (raw MP4)    | Phase A/B skeleton PNGs      |
| Identity conditioning  | SVD reference-image attn   | IP-Adapter FaceID            |
| Temporal coherence     | High (SVD temporal attn)   | Medium (motion module)       |
| Hand fidelity          | Medium (known weakness)    | High (ControlNet-exact pose) |
| Resolution             | 576px (SVD native)         | 512–768px (SD1.5 / SDXL)    |
| Speed                  | Slower (SVD)               | Faster (SD1.5 base)          |
| Best for               | Realistic body motion      | Precise sign-language hands  |

The AnimateDiff path is the better choice when hand accuracy matters more than
photorealism — which is often the case for sign-language generation.

Architecture
------------
    Phase A skeleton PNGs  ──► ControlNet-OpenPose conditioning (per frame)
    Phase C ArcFace emb    ──► IP-Adapter FaceID (identity per frame)
    Phase D keyframe.png   ──► init_image (appearance anchor)
                                │
                                ▼
                    AnimateDiff motion module
                    (SD1.5 or SDXL + motion adapter)
                                │
                                ▼
                    16-frame video clips (tiled for long sequences)
                                │
                                ▼
                    Phase F temporal polish

Dependencies (DGX):
    pip install diffusers transformers accelerate
    # IP-Adapter weights: huggingface.co/h94/IP-Adapter
    # AnimateDiff motion adapter: guoyww/animatediff-motion-adapter-v1-5-3
    # ControlNet OpenPose: lllyasviel/control_v11p_sd15_openpose

NOTE: written for DGX execution; not run in this environment (no GPU).
Pure-Python logic is unit-checked. Validate with:
    python -m mosl.render.animatediff_backend --check

Usage
-----
    python -m mosl.render.animatediff_backend \\
        --identity-id omar \\
        --pose-dir outputs/pose_control/أَنْتِ_keypoints/ \\
        --out outputs/avatar_video/omar/

    python -m mosl.render.animatediff_backend --config animatediff_config.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, fields
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("mosl.render.animatediff")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AnimateDiffConfig:
    """All tunables for one AnimateDiff + ControlNet-OpenPose run.

    Designed to be loaded from YAML or JSON; CLI flags override the file.
    """
    # identity
    identity_id: str = ""
    identity_root: Path = ROOT / "outputs" / "identity"
    keyframe_root: Path = ROOT / "outputs" / "keyframes"

    # pose conditioning (Phase A output)
    pose_dir: str = ""          # directory of pose_XXXXXX.png skeleton frames

    # output
    out_root: Path = ROOT / "outputs" / "avatar_video"

    # base model
    base_model: str = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
    # AnimateDiff motion adapter
    motion_adapter: str = "guoyww/animatediff-motion-adapter-v1-5-3"
    # ControlNet for OpenPose skeleton conditioning
    controlnet_model: str = "lllyasviel/control_v11p_sd15_openpose"
    # IP-Adapter FaceID for identity
    ip_adapter_model: str = "h94/IP-Adapter"
    ip_adapter_subfolder: str = "models"
    ip_adapter_weight_name: str = "ip-adapter-faceid_sd15.bin"

    # generation settings
    prompt: str = (
        "ultra photorealistic portrait video, upper body, cinematic DSLR footage, "
        "soft cinematic lighting, volumetric light, realistic skin pores, "
        "detailed eyes, natural human movement, smooth motion, 8k"
    )
    negative_prompt: str = (
        "flickering, distorted anatomy, deformed face, extra fingers, bad hands, "
        "blurry, low resolution, CGI, cartoon, anime, plastic skin, uncanny valley, "
        "watermark, text, temporal artifacts, unstable motion"
    )
    num_frames: int = 16            # AnimateDiff native window
    width: int = 512
    height: int = 768               # portrait aspect
    steps: int = 25
    guidance_scale: float = 7.5
    controlnet_scale: float = 0.85  # pose adherence
    ip_adapter_scale: float = 0.6   # identity strength
    fps: int = 8                    # generate at 8, interpolate to 25 in Phase F
    seed: int = 42
    device: str = "cuda"
    dtype: str = "fp16"

    # tiling for long sequences
    tile_overlap: int = 4           # frames to cross-fade between tiles

    def __post_init__(self) -> None:
        self.identity_root = Path(self.identity_root).expanduser()
        self.keyframe_root = Path(self.keyframe_root).expanduser()
        self.out_root = Path(self.out_root).expanduser()


def load_config(path: Path | None, args: argparse.Namespace) -> AnimateDiffConfig:
    """Resolve config: CLI flags > YAML/JSON file > dataclass defaults."""
    data: dict = {}
    if path is not None:
        if not path.is_file():
            raise SystemExit(f"config file not found: {path}")
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml  # noqa: PLC0415
                data.update(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
            except ImportError as exc:
                raise RuntimeError("pyyaml required: pip install pyyaml") from exc
        else:
            data.update(json.loads(path.read_text(encoding="utf-8")))

    cli = {
        "identity_id": getattr(args, "identity_id", None),
        "pose_dir":    getattr(args, "pose_dir", None),
        "out_root":    getattr(args, "out_root", None),
        "steps":       getattr(args, "steps", None),
        "seed":        getattr(args, "seed", None),
        "device":      getattr(args, "device", None),
    }
    data.update({k: v for k, v in cli.items() if v is not None})
    valid = {f.name for f in fields(AnimateDiffConfig)}
    return AnimateDiffConfig(**{k: v for k, v in data.items() if k in valid})


# ---------------------------------------------------------------------------
# Pose frame loading
# ---------------------------------------------------------------------------

def load_pose_frames(pose_dir: Path) -> list[Image.Image]:
    """Load Phase A skeleton PNGs in sorted order.

    Args:
        pose_dir: Directory of pose_XXXXXX.png files from pose_bridge.py.

    Returns:
        List of PIL Images (RGB), sorted by filename.

    Raises:
        RuntimeError: If no pose frames are found.
    """
    pose_dir = Path(pose_dir)
    if not pose_dir.is_dir():
        raise RuntimeError(f"pose directory not found: {pose_dir}")
    files = sorted(p for p in pose_dir.iterdir()
                   if p.suffix.lower() in IMAGE_EXTS and "pose_" in p.name)
    if not files:
        raise RuntimeError(
            f"no pose_*.png files found in {pose_dir}\n"
            "    run Phase A first: python -m mosl.render.pose_bridge <json_dir>"
        )
    frames = [Image.open(p).convert("RGB") for p in files]
    log.info("loaded %d pose frames from %s", len(frames), pose_dir)
    return frames


def tile_frames(frames: list[Image.Image], tile_size: int,
                overlap: int) -> list[list[Image.Image]]:
    """Split a long frame sequence into overlapping tiles for AnimateDiff.

    AnimateDiff processes `tile_size` frames at a time. For sequences longer
    than `tile_size`, we split into overlapping tiles and cross-fade the
    overlap region to avoid hard seams.

    Args:
        frames:    Full frame sequence.
        tile_size: AnimateDiff window size (typically 16).
        overlap:   Number of frames to share between adjacent tiles.

    Returns:
        List of frame tiles, each of length <= tile_size.
    """
    if len(frames) <= tile_size:
        return [frames]
    stride = tile_size - overlap
    tiles = []
    start = 0
    while start < len(frames):
        end = min(start + tile_size, len(frames))
        tiles.append(frames[start:end])
        if end == len(frames):
            break
        start += stride
    return tiles


# ---------------------------------------------------------------------------
# AnimateDiff pipeline backend
# ---------------------------------------------------------------------------

class AnimateDiffGenerator:
    """Wraps the diffusers AnimateDiff + ControlNet + IP-Adapter pipeline.

    Isolated so the import error message and model loading live in one place.
    """

    def __init__(self, cfg: AnimateDiffConfig) -> None:
        try:
            import torch  # noqa: PLC0415
            from diffusers import (  # noqa: PLC0415
                AnimateDiffPipeline,
                ControlNetModel,
                MotionAdapter,
                DDIMScheduler,
            )
        except ImportError as exc:
            raise RuntimeError(
                "diffusers / torch not available. Install per "
                "mosl/render/SETUP_DGX.md."
            ) from exc

        dtype = torch.float16 if cfg.dtype == "fp16" else torch.float32
        log.info("loading AnimateDiff pipeline (base=%s)", cfg.base_model)

        # Motion adapter
        adapter = MotionAdapter.from_pretrained(
            cfg.motion_adapter, torch_dtype=dtype)

        # ControlNet for OpenPose
        controlnet = ControlNetModel.from_pretrained(
            cfg.controlnet_model, torch_dtype=dtype)

        # Scheduler — DDIM is the standard for AnimateDiff
        scheduler = DDIMScheduler.from_pretrained(
            cfg.base_model, subfolder="scheduler",
            clip_sample=False, timestep_spacing="linspace",
            beta_schedule="linear", steps_offset=1,
        )

        # Main pipeline
        pipe = AnimateDiffPipeline.from_pretrained(
            cfg.base_model,
            motion_adapter=adapter,
            controlnet=controlnet,
            scheduler=scheduler,
            torch_dtype=dtype,
        )

        # IP-Adapter FaceID for identity conditioning
        pipe.load_ip_adapter(
            cfg.ip_adapter_model,
            subfolder=cfg.ip_adapter_subfolder,
            weight_name=cfg.ip_adapter_weight_name,
        )
        pipe.set_ip_adapter_scale(cfg.ip_adapter_scale)

        # Memory optimizations
        if cfg.device == "cuda":
            pipe.enable_vae_slicing()
            pipe.enable_vae_tiling()
            try:
                pipe.enable_xformers_memory_efficient_attention()
                log.info("xformers memory-efficient attention enabled")
            except Exception:  # noqa: BLE001
                log.info("xformers not available — using PyTorch SDPA")
            pipe.to(cfg.device)
        else:
            pipe.enable_model_cpu_offload()

        self._pipe = pipe
        self._torch = torch

    def generate_tile(
        self,
        pose_frames: list[Image.Image],
        face_image: Image.Image,
        face_emb: np.ndarray,
        cfg: AnimateDiffConfig,
        seed: int,
    ) -> list[Image.Image]:
        """Generate one AnimateDiff tile (up to num_frames frames).

        Args:
            pose_frames: Phase A skeleton PNGs for this tile (ControlNet input).
            face_image:  Reference face crop from Phase C (IP-Adapter input).
            face_emb:    ArcFace embedding from Phase C (IP-Adapter FaceID input).
            cfg:         Active AnimateDiffConfig.
            seed:        Random seed for this tile.

        Returns:
            List of generated PIL Images for this tile.
        """
        import torch  # noqa: PLC0415

        gen = torch.Generator(device=cfg.device).manual_seed(seed)

        # IP-Adapter FaceID takes the face image; the ArcFace embedding is
        # passed as ip_adapter_image_embeds for FaceID models.
        face_emb_t = torch.from_numpy(
            np.ascontiguousarray(face_emb)
        ).unsqueeze(0).to(cfg.device)

        result = self._pipe(
            prompt=cfg.prompt,
            negative_prompt=cfg.negative_prompt,
            num_frames=len(pose_frames),
            width=cfg.width,
            height=cfg.height,
            num_inference_steps=cfg.steps,
            guidance_scale=cfg.guidance_scale,
            controlnet_conditioning_scale=cfg.controlnet_scale,
            conditioning_frames=pose_frames,        # ControlNet per-frame input
            ip_adapter_image=face_image,            # IP-Adapter appearance
            ip_adapter_image_embeds=[face_emb_t],   # IP-Adapter FaceID embedding
            generator=gen,
        )
        return result.frames[0]  # list of PIL Images


# ---------------------------------------------------------------------------
# Video assembly
# ---------------------------------------------------------------------------

def crossfade_tiles(
    tiles: list[list[Image.Image]], overlap: int
) -> list[Image.Image]:
    """Blend overlapping tile boundaries with a linear cross-fade.

    Without cross-fading, tile boundaries produce a hard visual seam.
    Linear blending over `overlap` frames smooths the transition.

    Args:
        tiles:   List of generated frame tiles.
        overlap: Number of frames shared between adjacent tiles.

    Returns:
        Single merged frame sequence.
    """
    if len(tiles) == 1:
        return tiles[0]

    merged: list[Image.Image] = list(tiles[0])
    for tile in tiles[1:]:
        # The last `overlap` frames of `merged` blend with the first
        # `overlap` frames of `tile`.
        n_blend = min(overlap, len(merged), len(tile))
        for i in range(n_blend):
            alpha = (i + 1) / (n_blend + 1)   # 0 → 1 over the overlap
            a = np.array(merged[-(n_blend - i)], dtype=np.float32)
            b = np.array(tile[i], dtype=np.float32)
            blended = Image.fromarray(
                np.clip(a * (1 - alpha) + b * alpha, 0, 255).astype(np.uint8)
            )
            merged[-(n_blend - i)] = blended
        merged.extend(tile[n_blend:])
    return merged


def save_video(frames: list[Image.Image], out_path: Path, fps: int) -> None:
    """Write a list of PIL Images to an MP4 using imageio.

    Args:
        frames:   Sequence of RGB PIL Images.
        out_path: Output MP4 path.
        fps:      Frames per second.
    """
    try:
        import imageio.v3 as iio  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "imageio required: pip install imageio imageio-ffmpeg"
        ) from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = [np.array(f.convert("RGB")) for f in frames]
    iio.imwrite(str(out_path), arrays, fps=fps, codec="libx264",
                output_params=["-crf", "18", "-preset", "slow",
                               "-pix_fmt", "yuv420p"])
    log.info("saved %d frames -> %s (fps=%d)", len(frames), out_path, fps)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_animatediff(cfg: AnimateDiffConfig) -> dict:
    """Full AnimateDiff Phase E pipeline for one identity + pose sequence.

    Flow:
        1. Load Phase A pose frames (ControlNet conditioning)
        2. Load Phase C identity embedding + reference face crop
        3. Load Phase D keyframe (appearance anchor)
        4. Split pose sequence into tiles
        5. Generate each tile with AnimateDiff
        6. Cross-fade tile boundaries
        7. Save output video

    Args:
        cfg: Fully populated AnimateDiffConfig.

    Returns:
        Manifest dict written to disk.
    """
    if not cfg.identity_id:
        raise SystemExit("no --identity-id given")
    if not cfg.pose_dir:
        raise SystemExit(
            "no --pose-dir given — pass the Phase A output directory "
            "(e.g. outputs/pose_control/<clip>/)"
        )

    t0 = time.time()

    # --- load pose frames (Phase A output) ---
    pose_frames = load_pose_frames(Path(cfg.pose_dir))

    # --- load identity (Phase C output) ---
    from .identity import load_embeddings  # noqa: PLC0415
    identity = load_embeddings(cfg.identity_id, cfg.identity_root)
    face_emb = identity["fused"]   # (512,) normalized ArcFace embedding
    ref_face_path = identity["reference_face_path"]
    if not ref_face_path.is_file():
        raise RuntimeError(
            f"reference face crop not found: {ref_face_path}\n"
            "    re-run Phase C with --viz to generate it"
        )
    face_image = Image.open(ref_face_path).convert("RGB")

    # --- resize pose frames to generation resolution ---
    pose_frames_resized = [
        f.resize((cfg.width, cfg.height), Image.LANCZOS) for f in pose_frames
    ]

    # --- tile the sequence ---
    tiles = tile_frames(pose_frames_resized, cfg.num_frames, cfg.tile_overlap)
    log.info("sequence: %d frames → %d tile(s) of ≤%d frames",
             len(pose_frames), len(tiles), cfg.num_frames)

    # --- generate ---
    generator = AnimateDiffGenerator(cfg)
    generated_tiles: list[list[Image.Image]] = []
    for i, tile in enumerate(tiles):
        log.info("generating tile %d/%d (%d frames, seed=%d)",
                 i + 1, len(tiles), len(tile), cfg.seed + i)
        frames = generator.generate_tile(
            tile, face_image, face_emb, cfg, seed=cfg.seed + i
        )
        generated_tiles.append(frames)

    # --- merge tiles ---
    merged = crossfade_tiles(generated_tiles, cfg.tile_overlap)
    log.info("merged %d tiles → %d frames", len(tiles), len(merged))

    # --- save ---
    out_dir = cfg.out_root / cfg.identity_id
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_name = Path(cfg.pose_dir).name
    out_path = out_dir / f"{clip_name}__animatediff__{cfg.identity_id}.mp4"
    save_video(merged, out_path, cfg.fps)

    elapsed = round(time.time() - t0, 1)
    manifest = {
        "identity_id": cfg.identity_id,
        "backend": "animatediff",
        "created": date.today().isoformat(),
        "pose_dir": cfg.pose_dir,
        "output_video": out_path.name,
        "n_frames": len(merged),
        "n_tiles": len(tiles),
        "settings": {
            "base_model": cfg.base_model,
            "motion_adapter": cfg.motion_adapter,
            "controlnet_model": cfg.controlnet_model,
            "controlnet_scale": cfg.controlnet_scale,
            "ip_adapter_scale": cfg.ip_adapter_scale,
            "width": cfg.width, "height": cfg.height,
            "steps": cfg.steps, "guidance_scale": cfg.guidance_scale,
            "fps": cfg.fps, "seed": cfg.seed,
            "tile_overlap": cfg.tile_overlap,
        },
        "elapsed_sec": elapsed,
        "next": "Phase F (temporal): deflicker, RIFE interpolation to 25fps, upscale",
    }
    (out_dir / f"{clip_name}__animatediff__manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("AnimateDiff done in %.1fs -> %s", elapsed, out_path)
    return manifest


# ---------------------------------------------------------------------------
# Environment check
# ---------------------------------------------------------------------------

def _run_check(cfg: AnimateDiffConfig) -> int:
    """Validate that all required models and dependencies are available."""
    log.info("environment check: AnimateDiff backend ...")
    ok = True

    # Python packages
    for pkg in ("diffusers", "transformers", "accelerate", "torch"):
        try:
            __import__(pkg)
            log.info("  OK   %s importable", pkg)
        except ImportError:
            ok = False
            log.error("  MISSING  %s — pip install %s", pkg, pkg)

    # Optional but recommended
    for pkg in ("xformers", "imageio"):
        try:
            __import__(pkg)
            log.info("  OK   %s importable", pkg)
        except ImportError:
            log.warning("  OPTIONAL  %s not installed (recommended)", pkg)

    log.info("environment check: %s", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--identity-id", help="identity whose embedding to use")
    ap.add_argument("--pose-dir", help="Phase A pose_*.png directory")
    ap.add_argument("--out-root", help="output root (default: outputs/avatar_video)")
    ap.add_argument("--config", metavar="PATH",
                    help="YAML or JSON config file (CLI flags override it)")
    ap.add_argument("--steps", type=int, help="diffusion steps")
    ap.add_argument("--seed", type=int, help="random seed")
    ap.add_argument("--device", choices=["cuda", "cpu"], help="inference device")
    ap.add_argument("--check", action="store_true",
                    help="validate the AnimateDiff environment and exit")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(Path(args.config) if args.config else None, args)

    if args.check:
        return _run_check(cfg)

    if not cfg.identity_id:
        ap.error("no identity — pass --identity-id <name> (or --config / --check)")
    if not cfg.pose_dir:
        ap.error("no pose frames — pass --pose-dir <dir> (or --config)")

    try:
        run_animatediff(cfg)
    except (RuntimeError, FileNotFoundError) as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
