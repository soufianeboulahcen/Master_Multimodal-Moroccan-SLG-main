"""Phase D — reference keyframe generation (keyframe.py).

Phase C produced a face *embedding*. The Phase E video model (MimicMotion)
needs a full *reference image* of the person to animate. This module bridges
that gap: it turns the identity embedding into one photorealistic reference
image of the person, which Phase E then drives with the Phase A/B pose sequence.

Method
------
InstantID (SDXL + IdentityNet ControlNet + an image-prompt adapter): it takes
the ArcFace embedding from Phase C plus a 5-point face-landmark image and
generates a new photoreal image that *is* that person. A realism SDXL
checkpoint (RealVisXL) is used as the base.

Best-of-N: several variants are generated and each is re-encoded with the
Phase C face encoder; the variant whose face is closest to the target identity
(cosine similarity) is chosen as `keyframe.png`. This guarantees the reference
actually looks like the person before a Phase E run is spent on it.

> "FLUX video" is not a real pipeline; for a FLUX identity path swap the
> backend for PuLID-FLUX. InstantID/SDXL is the stable, well-trodden route and
> consumes the exact embedding Phase C produces, so it is the default here.

Dependencies (DGX): diffusers + the InstantID repo + weights — see
mosl/render/SETUP_DGX.md. Not run in this environment (no GPU); the geometry
and IO logic is unit-checked. Validate with:

    python -m mosl.render.keyframe --check

Usage
-----
    python -m mosl.render.keyframe --identity-id <person>
    python -m mosl.render.keyframe --identity-id <person> --variants 6 --seed 0
    python -m mosl.render.keyframe --config keyframe.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sys
import time
from dataclasses import dataclass, fields
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image

from .identity import FaceAnalyzer, IdentityConfig, load_embeddings

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("mosl.render.keyframe")

# Default prompts — the still-image subset of the project's cinematic avatar
# brief (motion-only terms like "eye blinking" belong to Phase E, not here).
DEFAULT_PROMPT = (
    "ultra photorealistic portrait of a person, upper body, neutral standing "
    "pose, facing the camera, cinematic DSLR photograph, soft cinematic "
    "lighting, volumetric light, realistic skin pores and texture, detailed "
    "eyes, shallow depth of field, high dynamic range, sharp focus, 8k, "
    "professional studio background"
)
DEFAULT_NEGATIVE = (
    "flickering, distorted anatomy, deformed face, extra fingers, bad hands, "
    "missing fingers, blurry, low resolution, lowres, CGI, 3d render, cartoon, "
    "anime, plastic skin, waxy skin, uncanny valley, warped body, asymmetric "
    "face, watermark, text, logo, jpeg artifacts"
)

# Canonical front-facing 5-point face, InstantID landmark order
# (left eye, right eye, nose, left mouth, right mouth), normalized.
CANONICAL_FACE_KPS = np.array([
    [0.32, 0.36], [0.68, 0.36], [0.50, 0.55], [0.36, 0.74], [0.64, 0.74],
], dtype=np.float32)

# InstantID draw_kps topology and colours
_KPS_LIMBS = [(0, 2), (1, 2), (3, 2), (4, 2)]
_KPS_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]


# --- configuration ----------------------------------------------------------

@dataclass
class KeyframeConfig:
    """All tunables for one keyframe-generation run."""
    identity_id: str = ""
    identity_root: Path = ROOT / "outputs" / "identity"
    out_root: Path = ROOT / "outputs" / "keyframes"
    sdxl_model: str = "SG161222/RealVisXL_V5.0"
    instantid_repo: str = "checkpoints/InstantID"      # cloned InstantID repo
    instantid_weights: str = "checkpoints/InstantID"   # ControlNetModel + ip-adapter.bin
    prompt: str = DEFAULT_PROMPT
    negative_prompt: str = DEFAULT_NEGATIVE
    width: int = 832
    height: int = 1216
    steps: int = 30
    guidance_scale: float = 5.0
    identitynet_strength: float = 0.65     # ControlNet (face structure); 0.80 is too rigid
    adapter_strength: float = 0.80         # IP-adapter (face texture/identity)
    num_variants: int = 4
    seed: int = 42
    kps_source: str = "canonical"          # canonical | reference
    face_height_frac: float = 0.14         # eye-to-mouth span / canvas height
    face_cx: float = 0.50
    face_cy: float = 0.32
    min_identity_cosine: float = 0.50      # warn if the best variant is below
    device: str = "cuda"
    dtype: str = "fp16"
    save_viz: bool = True                  # save face_kps.png alongside keyframe

    _FILE_FIELDS = (
        "identity_id", "identity_root", "out_root", "sdxl_model",
        "instantid_repo", "instantid_weights", "prompt", "negative_prompt",
        "width", "height", "steps", "guidance_scale", "identitynet_strength",
        "adapter_strength", "num_variants", "seed", "kps_source",
        "face_height_frac", "face_cx", "face_cy", "min_identity_cosine",
        "device", "dtype", "save_viz",
    )

    def __post_init__(self) -> None:
        self.identity_root = Path(self.identity_root).expanduser()
        self.out_root = Path(self.out_root).expanduser()


def load_config(path: Path | None, args: argparse.Namespace) -> KeyframeConfig:
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
                raise RuntimeError("pyyaml required for YAML configs: pip install pyyaml") from exc
        else:
            data.update(json.loads(path.read_text(encoding="utf-8")))
    cli = {
        "identity_id": args.identity_id, "out_root": args.out_root,
        "sdxl_model": args.sdxl_model, "num_variants": args.variants,
        "seed": args.seed, "steps": args.steps, "device": args.device,
    }
    data.update({k: v for k, v in cli.items() if v is not None})
    valid = {f.name for f in fields(KeyframeConfig)}
    return KeyframeConfig(**{k: v for k, v in data.items() if k in valid})


# --- face-landmark control image -------------------------------------------

def place_kps(ref_kps: np.ndarray, cfg: KeyframeConfig) -> np.ndarray:
    """Position a 5-point face into the keyframe canvas.

    `canonical` uses a front-facing template (a clean neutral keyframe that
    Phase E can re-pose freely); `reference` keeps the head orientation of the
    source photos. Either way the face is scaled and centred consistently.
    """
    src = None
    if cfg.kps_source == "reference":
        ref = np.asarray(ref_kps, dtype=np.float32)
        if ref.shape == (5, 2):
            span_y = ref[:, 1].max() - ref[:, 1].min()
            span_x = ref[:, 0].max() - ref[:, 0].min()
            if span_y > 1e-3 and span_x > 1e-3:
                src = ref.copy()
    if src is None:  # canonical, or reference kps were degenerate
        src = CANONICAL_FACE_KPS * np.array([cfg.width, cfg.height], np.float32)

    span_y = max(src[:, 1].max() - src[:, 1].min(), 1e-6)
    scale = (cfg.face_height_frac * cfg.height) / span_y
    src = src * scale
    cx = (src[:, 0].max() + src[:, 0].min()) / 2.0
    cy = (src[:, 1].max() + src[:, 1].min()) / 2.0
    src[:, 0] += cfg.face_cx * cfg.width - cx
    src[:, 1] += cfg.face_cy * cfg.height - cy
    return src


def draw_face_kps(kps: np.ndarray, width: int, height: int) -> Image.Image:
    """Render the InstantID 5-point face-landmark control image."""
    import cv2  # noqa: PLC0415

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    kps = np.asarray(kps, dtype=np.float32)
    stickwidth = 4
    for i, (a, b) in enumerate(_KPS_LIMBS):
        xa, ya = kps[a]
        xb, yb = kps[b]
        mx, my = (xa + xb) / 2.0, (ya + yb) / 2.0
        length = math.hypot(xa - xb, ya - yb)
        angle = math.degrees(math.atan2(ya - yb, xa - xb))
        poly = cv2.ellipse2Poly((int(mx), int(my)),
                                (int(length / 2), stickwidth),
                                int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, poly, [int(c * 0.6) for c in _KPS_COLORS[i]])
    for i, (px, py) in enumerate(kps):
        cv2.circle(canvas, (int(px), int(py)), 10, _KPS_COLORS[i], -1)
    return Image.fromarray(canvas)


# --- InstantID backend ------------------------------------------------------

class KeyframeGenerator:
    """Wraps the InstantID SDXL pipeline. Isolated for a one-place fix."""

    def __init__(self, cfg: KeyframeConfig) -> None:
        try:
            import torch  # noqa: PLC0415
            from diffusers.models import ControlNetModel  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "diffusers / torch not available. Install per "
                "mosl/render/SETUP_DGX.md (use the NGC container's torch)."
            ) from exc

        repo = Path(cfg.instantid_repo).expanduser()
        if not repo.is_dir():
            raise RuntimeError(
                f"InstantID repo not found at {repo}\n"
                "    git clone https://github.com/instantX-research/InstantID")
        sys.path.insert(0, str(repo))
        try:
            from pipeline_stable_diffusion_xl_instantid import (  # noqa: PLC0415
                StableDiffusionXLInstantIDPipeline)
        except ImportError as exc:
            raise RuntimeError(
                f"could not import the InstantID pipeline from {repo}") from exc

        weights = Path(cfg.instantid_weights).expanduser()
        adapter = weights / "ip-adapter.bin"
        controlnet_dir = weights / "ControlNetModel"
        if not adapter.is_file() or not controlnet_dir.is_dir():
            raise RuntimeError(
                f"InstantID weights missing under {weights} — expected "
                "ControlNetModel/ and ip-adapter.bin "
                "(huggingface.co/InstantX/InstantID)")

        dtype = torch.float16 if cfg.dtype == "fp16" else torch.float32
        log.info("loading InstantID (base=%s dtype=%s)", cfg.sdxl_model, cfg.dtype)
        controlnet = ControlNetModel.from_pretrained(
            str(controlnet_dir), torch_dtype=dtype)
        pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
            cfg.sdxl_model, controlnet=controlnet, torch_dtype=dtype)
        pipe.load_ip_adapter_instantid(str(adapter))
        if cfg.device == "cuda":
            pipe.cuda()
        else:
            pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe
        self._torch = torch

    def generate(self, face_emb: np.ndarray, kps_image: Image.Image,
                 cfg: KeyframeConfig, seed: int) -> Image.Image:
        gen = self._torch.Generator(
            device=cfg.device if cfg.device == "cuda" else "cpu"
        ).manual_seed(seed)
        result = self._pipe(
            prompt=cfg.prompt, negative_prompt=cfg.negative_prompt,
            image_embeds=face_emb, image=kps_image,
            controlnet_conditioning_scale=float(cfg.identitynet_strength),
            ip_adapter_scale=float(cfg.adapter_strength),
            num_inference_steps=cfg.steps,
            guidance_scale=cfg.guidance_scale,
            width=cfg.width, height=cfg.height, generator=gen,
        )
        return result.images[0]


# --- identity scoring (reuses the Phase C encoder) --------------------------

def score_identity(image: Image.Image, target_norm: np.ndarray,
                   analyzer: FaceAnalyzer) -> float:
    """Cosine similarity between the generated face and the target identity."""
    import cv2  # noqa: PLC0415

    bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    faces = analyzer.get(bgr)
    if not faces:
        return -1.0
    face = max(faces, key=lambda f:
               (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    emb = np.asarray(face.normed_embedding, dtype=np.float32)
    target = target_norm / (np.linalg.norm(target_norm) + 1e-9)
    return float(emb @ target)


# --- discovery --------------------------------------------------------------

def list_keyframes(out_root: Path) -> list[dict]:
    """Return a summary of all saved keyframes under out_root.

    Reads each manifest.json and returns lightweight summaries so Phase G
    can discover available keyframes without loading images.
    """
    out_root = Path(out_root).expanduser()
    summaries: list[dict] = []
    for manifest_path in sorted(out_root.rglob("manifest.json")):
        try:
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
            summaries.append({
                "identity_id": m.get("identity_id", manifest_path.parent.name),
                "keyframe": str(manifest_path.parent / m.get("keyframe", "keyframe.png")),
                "best_identity_cosine": m.get("best_variant", {}).get("identity_cosine"),
                "created": m.get("created", ""),
            })
        except Exception as exc:  # noqa: BLE001
            log.warning("could not read manifest %s: %s", manifest_path, exc)
    return summaries


# --- orchestration ----------------------------------------------------------

def generate_keyframe(cfg: KeyframeConfig) -> dict:
    """Full Phase D pipeline: identity -> best-of-N -> keyframe.png."""
    if not cfg.identity_id:
        raise SystemExit("no --identity-id given")
    t0 = time.time()

    identity = load_embeddings(cfg.identity_id, cfg.identity_root)
    face_emb = identity["fused_raw"]      # InstantID ingests the raw ArcFace emb
    target_norm = identity["fused"]       # normalized — used for scoring
    kps = place_kps(identity["reference_kps"], cfg)
    kps_image = draw_face_kps(kps, cfg.width, cfg.height)

    out_dir = cfg.out_root / cfg.identity_id
    out_dir.mkdir(parents=True, exist_ok=True)
    kps_image.save(out_dir / "face_kps.png")

    generator = KeyframeGenerator(cfg)
    analyzer = FaceAnalyzer(IdentityConfig(input_dir="", device=cfg.device))

    variants: list[dict] = []
    for v in range(cfg.num_variants):
        seed = cfg.seed + v
        image = generator.generate(face_emb, kps_image, cfg, seed)
        score = score_identity(image, target_norm, analyzer)
        fname = f"variant_{v:02d}.png"
        image.save(out_dir / fname)
        variants.append({"variant": v, "seed": seed,
                         "identity_cosine": round(score, 4), "file": fname})
        log.info("variant %d (seed %d): identity_cosine=%.3f", v, seed, score)

    best = max(variants, key=lambda d: d["identity_cosine"])
    shutil.copyfile(out_dir / best["file"], out_dir / "keyframe.png")
    log.info("best variant: %s (identity_cosine=%.3f)",
             best["file"], best["identity_cosine"])
    if best["identity_cosine"] < cfg.min_identity_cosine:
        log.warning("best keyframe identity_cosine %.3f < %.2f — the reference "
                    "may not match the person well; try more --variants, a "
                    "different --seed, or stronger identity photos",
                    best["identity_cosine"], cfg.min_identity_cosine)

    manifest = {
        "identity_id": cfg.identity_id,
        "created": date.today().isoformat(),
        "keyframe": "keyframe.png",
        "best_variant": best,
        "variants": variants,
        "settings": {
            "sdxl_model": cfg.sdxl_model,
            "resolution": [cfg.width, cfg.height],
            "steps": cfg.steps, "guidance_scale": cfg.guidance_scale,
            "identitynet_strength": cfg.identitynet_strength,
            "adapter_strength": cfg.adapter_strength,
            "kps_source": cfg.kps_source,
        },
        "elapsed_sec": round(time.time() - t0, 1),
        "next": "Phase E (MimicMotion) animates keyframe.png with the pose "
                "sequence from Phase A/B",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("keyframe for '%s' done in %.1fs -> %s",
             cfg.identity_id, manifest["elapsed_sec"], out_dir / "keyframe.png")
    return manifest


def _run_check(cfg: KeyframeConfig) -> int:
    """Load the InstantID pipeline and run a 2-step generation — no data needed."""
    log.info("environment check: loading InstantID pipeline ...")
    try:
        generator = KeyframeGenerator(cfg)
    except RuntimeError as exc:
        log.error("%s", exc)
        return 1
    rng = np.random.default_rng(0)
    dummy_emb = rng.normal(size=512).astype(np.float32)
    probe = KeyframeConfig(**{**cfg.__dict__, "width": 768, "height": 768,
                              "steps": 2})
    kps_image = draw_face_kps(place_kps(np.zeros((5, 2), np.float32), probe),
                              768, 768)
    generator.generate(dummy_emb, kps_image, probe, seed=0)
    log.info("backend OK — InstantID pipeline loaded and ran")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--identity-id", help="identity to generate a keyframe for")
    ap.add_argument("--out-root", help="output root (default: outputs/keyframes)")
    ap.add_argument("--config", help="JSON config file (CLI flags override it)")
    ap.add_argument("--sdxl-model", help="SDXL base checkpoint")
    ap.add_argument("--variants", type=int, help="number of candidates (best wins)")
    ap.add_argument("--steps", type=int, help="diffusion steps")
    ap.add_argument("--seed", type=int, help="base random seed")
    ap.add_argument("--device", choices=["cuda", "cpu"], help="inference device")
    ap.add_argument("--list", action="store_true",
                    help="list saved keyframes and exit")
    ap.add_argument("--check", action="store_true",
                    help="validate the InstantID environment and exit")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    cfg = load_config(Path(args.config) if args.config else None, args)

    if args.list:
        kfs = list_keyframes(cfg.out_root)
        print(json.dumps(kfs, indent=2, ensure_ascii=False))
        return 0

    if args.check:
        return _run_check(cfg)
    if not cfg.identity_id:
        ap.error("no identity — pass --identity-id <name> (or --config / --check)")

    try:
        generate_keyframe(cfg)
    except (RuntimeError, FileNotFoundError) as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
