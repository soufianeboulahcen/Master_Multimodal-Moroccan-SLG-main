"""Phase E — MimicMotion avatar video generation (video.py).

Animates the Phase D reference keyframe with sign-language motion, producing the
identity-preserving avatar video. This is the heaviest stage of the renderer.

Method
------
**MimicMotion** (Tencent, SVD-based): given a reference image of a person and a
driving video, it generates a video of that person performing the driving
motion, with confidence-aware pose guidance and a regional hand refinement.

This module drives MimicMotion through its **supported entry point** — it
generates an `inference.py` YAML config and runs the official script — rather
than calling MimicMotion's internal Python API. The config interface is far
more stable across versions than the internals, which makes this integration
robust to MimicMotion updates.

Inputs:
  * reference keyframe  — Phase D `keyframe.png` (the person)
  * driving video       — a MoSL sign clip (the motion)

Relationship to Phase B
-----------------------
MimicMotion runs its **own** DWPose on the driving video and retargets the pose
to the keyframe's body — that preprocessing is coupled to how the model was
trained, so this path does not feed it the Phase B keypoints. Phase B's DWPose
JSON still serves the Phase A ControlNet path and any analysis; for the
MimicMotion path the *driving video itself* is the MoSL motion source.

Known limitation: hand fidelity on fast sign-language motion is a recognized
MimicMotion weak point (v1.1's regional hand loss improves it but does not
fully solve it).

NOTE: written for DGX execution; not run in this environment (no GPU /
MimicMotion). Pure-Python logic is unit-checked. The output-file discovery is
version-dependent — see `_newest_video_since`. Validate with:

    python -m mosl.render.video --check

Usage
-----
    python -m mosl.render.video --identity-id <person> --driving-video clip.mp4
    python -m mosl.render.video --identity-id <person> --driving-dir signs/
    python -m mosl.render.video --config video.json
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, fields
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("mosl.render.video")

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


# --- configuration ----------------------------------------------------------

@dataclass
class VideoConfig:
    """All tunables for one MimicMotion run."""
    identity_id: str = ""
    keyframe_path: str = ""                      # default: derived from identity_id
    keyframe_root: Path = ROOT / "outputs" / "keyframes"
    out_root: Path = ROOT / "outputs" / "avatar_video"
    mimicmotion_repo: str = "checkpoints/MimicMotion"
    base_model: str = "stabilityai/stable-video-diffusion-img2vid-xt-1-1"
    ckpt_path: str = "checkpoints/MimicMotion/MimicMotion_1-1.pth"
    # generation settings (see mosl/render/README.md — Phase E table)
    num_frames: int = 72                         # MimicMotion tile size
    frames_overlap: int = 6                      # tile cross-fade for long clips
    resolution: int = 576
    num_inference_steps: int = 25
    guidance_scale: float = 2.0                  # low — high CFG flickers in video
    noise_aug_strength: float = 0.0
    sample_stride: int = 2                       # driving-video frame stride
    fps: int = 15                                # generate low, interpolate in Phase F
    seed: int = 42
    device: str = "cuda"

    _FILE_FIELDS = (
        "identity_id", "keyframe_path", "keyframe_root", "out_root",
        "mimicmotion_repo", "base_model", "ckpt_path", "num_frames",
        "frames_overlap", "resolution", "num_inference_steps",
        "guidance_scale", "noise_aug_strength", "sample_stride", "fps",
        "seed", "device",
    )

    def __post_init__(self) -> None:
        self.keyframe_root = Path(self.keyframe_root).expanduser()
        self.out_root = Path(self.out_root).expanduser()


def load_config(path: Path | None, args: argparse.Namespace) -> VideoConfig:
    """Resolve config: CLI flags > JSON file > dataclass defaults."""
    data: dict = {}
    if path is not None:
        if not path.is_file():
            raise SystemExit(f"config file not found: {path}")
        data.update(json.loads(path.read_text(encoding="utf-8")))
    cli = {
        "identity_id": args.identity_id, "keyframe_path": args.keyframe,
        "out_root": args.out_root, "seed": args.seed,
        "num_inference_steps": args.steps, "device": args.device,
    }
    data.update({k: v for k, v in cli.items() if v is not None})
    valid = {f.name for f in fields(VideoConfig)}
    return VideoConfig(**{k: v for k, v in data.items() if k in valid})


# --- helpers ----------------------------------------------------------------

def resolve_keyframe(cfg: VideoConfig) -> Path:
    """Locate the Phase D reference keyframe for this identity."""
    if cfg.keyframe_path:
        kf = Path(cfg.keyframe_path).expanduser()
    else:
        kf = cfg.keyframe_root / cfg.identity_id / "keyframe.png"
    if not kf.is_file():
        raise RuntimeError(
            f"keyframe not found: {kf}\n"
            f"    run Phase D first:  python -m mosl.render.keyframe "
            f"--identity-id {cfg.identity_id or '<person>'}")
    return kf


def write_inference_yaml(cfg: VideoConfig, keyframe: Path,
                         driving: Path, yaml_path: Path) -> None:
    """Write the MimicMotion `inference.py` config.

    OmegaConf handles quoting — MoSL clip names contain spaces and Arabic
    characters, which a hand-written YAML writer would mangle.
    """
    from omegaconf import OmegaConf  # noqa: PLC0415

    conf = OmegaConf.create({
        "base_model_path": cfg.base_model,
        "ckpt_path": str(Path(cfg.ckpt_path).expanduser().resolve()),
        "test_case": [{
            "ref_video_path": str(driving.resolve()),
            "ref_image_path": str(keyframe.resolve()),
            "num_frames": cfg.num_frames,
            "resolution": cfg.resolution,
            "frames_overlap": cfg.frames_overlap,
            "num_inference_steps": cfg.num_inference_steps,
            "noise_aug_strength": cfg.noise_aug_strength,
            "guidance_scale": cfg.guidance_scale,
            "sample_stride": cfg.sample_stride,
            "fps": cfg.fps,
            "seed": cfg.seed,
        }],
    })
    OmegaConf.save(conf, yaml_path)


def _newest_video_since(root: Path, since: float) -> Path | None:
    """Find the most recent video file written under `root` after `since`.

    MimicMotion's output path/naming varies by version, so rather than assume
    it, we pick the newest video the run produced. One-place-fixable if a
    future version writes elsewhere.
    """
    newest: tuple[float, Path] | None = None
    for p in root.rglob("*"):
        if p.suffix.lower() in VIDEO_EXTS and p.is_file():
            mtime = p.stat().st_mtime
            if mtime >= since and (newest is None or mtime > newest[0]):
                newest = (mtime, p)
    return newest[1] if newest else None


def _stream_subprocess(cmd: list[str], cwd: Path) -> int:
    """Run a command, streaming its output through the logger. Returns exit code."""
    proc = subprocess.Popen(
        cmd, cwd=str(cwd), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        log.info("[mimicmotion] %s", line.rstrip())
    return proc.wait()


# --- core -------------------------------------------------------------------

def render_video(cfg: VideoConfig, driving: Path) -> dict:
    """Generate one avatar video: keyframe + one driving clip -> MP4."""
    repo = Path(cfg.mimicmotion_repo).expanduser()
    inference = repo / "inference.py"
    if not inference.is_file():
        raise RuntimeError(
            f"MimicMotion repo not found at {repo}\n"
            "    git clone https://github.com/Tencent/MimicMotion")
    if not Path(cfg.ckpt_path).expanduser().is_file():
        raise RuntimeError(f"MimicMotion checkpoint not found: {cfg.ckpt_path}")
    if not driving.is_file():
        raise RuntimeError(f"driving video not found: {driving}")

    keyframe = resolve_keyframe(cfg)
    out_dir = cfg.out_root / cfg.identity_id
    out_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = out_dir / f"{driving.stem}_mimicmotion.yaml"
    write_inference_yaml(cfg, keyframe, driving, yaml_path)

    log.info("MimicMotion: identity=%s keyframe=%s motion=%s",
             cfg.identity_id, keyframe.name, driving.name)
    t0 = time.time()
    rc = _stream_subprocess(
        [sys.executable, "inference.py", "--inference_config",
         str(yaml_path.resolve())], cwd=repo)
    if rc != 0:
        raise RuntimeError(f"MimicMotion inference.py exited with code {rc}")

    produced = _newest_video_since(repo, t0)
    if produced is None:
        raise RuntimeError(
            "MimicMotion finished but no output video was found under "
            f"{repo} — check the script's output directory")
    final = out_dir / f"{driving.stem}__{cfg.identity_id}.mp4"
    shutil.copyfile(produced, final)
    elapsed = round(time.time() - t0, 1)
    log.info("done %s -> %s (%.1fs)", driving.name, final, elapsed)

    manifest = {
        "identity_id": cfg.identity_id,
        "created": date.today().isoformat(),
        "keyframe": str(keyframe),
        "driving_video": str(driving),
        "output_video": final.name,
        "mimicmotion_raw_output": str(produced),
        "settings": {
            "base_model": cfg.base_model,
            "num_frames": cfg.num_frames,
            "frames_overlap": cfg.frames_overlap,
            "resolution": cfg.resolution,
            "num_inference_steps": cfg.num_inference_steps,
            "guidance_scale": cfg.guidance_scale,
            "sample_stride": cfg.sample_stride,
            "fps": cfg.fps, "seed": cfg.seed,
        },
        "elapsed_sec": elapsed,
        "next": "Phase F (temporal): deflicker, interpolate to 25-30 fps, upscale",
    }
    (out_dir / f"{driving.stem}__manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def discover_driving(args: argparse.Namespace) -> list[Path]:
    """Resolve --driving-video / --driving-dir into a list of clips."""
    clips: list[Path] = []
    if args.driving_video:
        clips.append(Path(args.driving_video).expanduser())
    if args.driving_dir:
        root = Path(args.driving_dir).expanduser()
        clips.extend(sorted(p for p in root.rglob("*")
                            if p.suffix.lower() in VIDEO_EXTS))
    return clips


# --- CLI --------------------------------------------------------------------

def _run_check(cfg: VideoConfig) -> int:
    """Static validation of the MimicMotion setup — no inference run."""
    log.info("environment check: MimicMotion setup ...")
    ok = True
    repo = Path(cfg.mimicmotion_repo).expanduser()
    for label, path, hint in [
        ("MimicMotion repo", repo,
         "git clone https://github.com/Tencent/MimicMotion"),
        ("inference.py", repo / "inference.py", "(comes with the repo)"),
        ("MimicMotion checkpoint", Path(cfg.ckpt_path).expanduser(),
         "huggingface-cli download tencent/MimicMotion"),
    ]:
        if path.exists():
            log.info("  OK   %s: %s", label, path)
        else:
            ok = False
            log.error("  MISSING  %s: %s  — %s", label, path, hint)
    try:
        import omegaconf  # noqa: F401, PLC0415
        log.info("  OK   omegaconf importable")
    except ImportError:
        ok = False
        log.error("  MISSING  omegaconf — pip install -r requirements-render.txt")
    log.info("environment check: %s", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--identity-id", help="identity whose keyframe to animate")
    ap.add_argument("--keyframe", help="explicit keyframe path (overrides identity-id lookup)")
    ap.add_argument("--driving-video", help="a single MoSL sign clip")
    ap.add_argument("--driving-dir", help="a directory of sign clips (batch)")
    ap.add_argument("--out-root", help="output root (default: outputs/avatar_video)")
    ap.add_argument("--config", help="JSON config file (CLI flags override it)")
    ap.add_argument("--steps", type=int, help="diffusion steps")
    ap.add_argument("--seed", type=int, help="random seed")
    ap.add_argument("--device", choices=["cuda", "cpu"], help="inference device")
    ap.add_argument("--check", action="store_true",
                    help="validate the MimicMotion setup and exit")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    cfg = load_config(Path(args.config) if args.config else None, args)

    if args.check:
        return _run_check(cfg)
    if not cfg.identity_id:
        ap.error("no identity — pass --identity-id <name> (or --config / --check)")

    clips = discover_driving(args)
    if not clips:
        ap.error("no driving motion — pass --driving-video or --driving-dir")

    log.info("rendering %d clip(s) for identity '%s'", len(clips), cfg.identity_id)
    done = failed = 0
    t_start = time.time()
    for clip in clips:
        try:
            render_video(cfg, clip)
            done += 1
        except (RuntimeError, FileNotFoundError) as exc:
            failed += 1
            log.error("FAILED %s: %s", clip.name, exc)
    log.info("finished — %d done, %d failed in %.1fs",
             done, failed, time.time() - t_start)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
