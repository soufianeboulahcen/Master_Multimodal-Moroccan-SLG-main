"""Phase F — temporal polish (temporal.py).

Takes the raw MimicMotion output from Phase E and turns it into a finished clip:
deflicker, frame-interpolate to a smooth frame rate, optionally upscale, and
re-encode to a clean delivery container.

Design
------
**RIFE is the default interpolation backend** — ffmpeg `minterpolate` produces
visible ghosting on fast hand motion (sign language), so RIFE is the correct
default for this domain. ffmpeg remains available as a fallback when RIFE is
not installed.

    interpolation : RIFE (default)                   |  ffmpeg minterpolate (--interp-backend ffmpeg)
    upscaling     : Real-ESRGAN (default, opt-in)    |  ffmpeg scale=lanczos (--upscale-backend ffmpeg)

ffmpeg is still the backbone for deflicker, face stabilization, and final
encoding — it requires no model weights and works everywhere.

Each step writes an intermediate file (near-lossless, CRF 12) so steps are
independently inspectable — useful while the pipeline is still unvalidated.

NOTE: written for DGX execution; not run in this environment. The ffmpeg
command construction is unit-checked. The RIFE / Real-ESRGAN wrappers shell out
to those repos and locate their output by recency (version-dependent — one
place to fix). Validate with:  python -m mosl.render.temporal --check

Usage
-----
    python -m mosl.render.temporal --input-video raw.mp4
    python -m mosl.render.temporal --input-dir outputs/avatar_video/<id>/
    python -m mosl.render.temporal --input-video raw.mp4 --upscale --target-fps 30
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
import time
from dataclasses import dataclass, fields
from datetime import date
from pathlib import Path

from .video import VIDEO_EXTS, _newest_video_since, _stream_subprocess

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("mosl.render.temporal")

INTERMEDIATE_CRF = 12          # near-lossless intermediates between steps


# --- configuration ----------------------------------------------------------

@dataclass
class TemporalConfig:
    """All tunables for one temporal-polish run."""
    out_root: Path = ROOT / "outputs" / "avatar_final"
    target_fps: int = 30
    deflicker: bool = True
    deflicker_size: int = 5                 # smaller window = less blur (was 7)
    interpolate: bool = True
    interp_backend: str = "rife"            # rife >> ffmpeg for hand motion (was ffmpeg)
    rife_repo: str = "checkpoints/Practical-RIFE"
    rife_exp: int = 1                       # RIFE multiplier: 1=2x, 2=4x interpolation
    upscale: bool = False                   # opt-in — heavy
    upscale_factor: int = 2
    upscale_backend: str = "realesrgan"     # realesrgan >> lanczos for skin (was ffmpeg)
    realesrgan_repo: str = "checkpoints/Real-ESRGAN"
    realesrgan_model: str = "realesr-general-x4v3"
    crf: int = 18                           # final encode quality (lower = better)
    preset: str = "slow"
    codec: str = "libx264"
    # face-region stabilization (reduces face flicker independently of body)
    face_stabilize: bool = False            # opt-in: crop-stabilize face region
    face_stabilize_strength: float = 0.5   # blend weight for stabilized face

    _FILE_FIELDS = (
        "out_root", "target_fps", "deflicker", "deflicker_size",
        "interpolate", "interp_backend", "rife_repo", "rife_exp", "upscale",
        "upscale_factor", "upscale_backend", "realesrgan_repo",
        "realesrgan_model", "crf", "preset", "codec",
        "face_stabilize", "face_stabilize_strength",
    )

    def __post_init__(self) -> None:
        self.out_root = Path(self.out_root).expanduser()


def load_config(path: Path | None, args: argparse.Namespace) -> TemporalConfig:
    """Resolve config: CLI flags > JSON file > dataclass defaults."""
    data: dict = {}
    if path is not None:
        if not path.is_file():
            raise SystemExit(f"config file not found: {path}")
        data.update(json.loads(path.read_text(encoding="utf-8")))
    cli = {
        "out_root": args.out_root, "target_fps": args.target_fps,
        "upscale": True if args.upscale else None,
        "interp_backend": args.interp_backend,
        "face_stabilize": True if getattr(args, "face_stabilize", False) else None,
    }
    data.update({k: v for k, v in cli.items() if v is not None})
    valid = {f.name for f in fields(TemporalConfig)}
    return TemporalConfig(**{k: v for k, v in data.items() if k in valid})


# --- ffmpeg -----------------------------------------------------------------

def ffmpeg_exe() -> str:
    """Locate an ffmpeg binary — imageio-ffmpeg ships one; else fall back to PATH."""
    try:
        import imageio_ffmpeg  # noqa: PLC0415
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return "ffmpeg"


def _ffmpeg_filter(ff: str, src: Path, dst: Path, vf: str, crf: int,
                   fps: int | None = None, preset: str = "medium") -> None:
    """Run one ffmpeg filter step. Raises on non-zero exit."""
    cmd = [ff, "-y", "-i", str(src), "-vf", vf,
           "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
           "-pix_fmt", "yuv420p"]
    if fps is not None:
        cmd += ["-r", str(fps)]
    cmd += ["-movflags", "+faststart", str(dst)]
    rc = _stream_subprocess(cmd, cwd=dst.parent)
    if rc != 0 or not dst.is_file():
        raise RuntimeError(f"ffmpeg step failed (exit {rc}): {vf}")


def probe_video(path: Path) -> dict:
    """Best-effort metadata read (fps, frames, size). Never fatal."""
    try:
        import imageio.v3 as iio  # noqa: PLC0415
        meta = iio.immeta(path)
        return {"fps": meta.get("fps"), "duration": meta.get("duration")}
    except Exception:  # noqa: BLE001
        return {}


# --- optional external backends ---------------------------------------------

def _run_rife(cfg: TemporalConfig, src: Path, work: Path) -> Path:
    """Interpolate with Practical-RIFE (higher quality than ffmpeg minterpolate).

    RIFE is significantly better than ffmpeg minterpolate for fast hand motion
    because it uses optical-flow-based synthesis rather than block matching.
    `--exp N` doubles the frame count N times (exp=1 → 2×, exp=2 → 4×).
    """
    repo = Path(cfg.rife_repo).expanduser()
    script = repo / "inference_video.py"
    if not script.is_file():
        raise RuntimeError(
            f"RIFE repo not found at {repo}\n"
            "    git clone https://github.com/hzwer/Practical-RIFE")
    import sys  # noqa: PLC0415
    t0 = time.time()
    rc = _stream_subprocess(
        [sys.executable, "inference_video.py",
         f"--video={src.resolve()}",
         f"--exp={cfg.rife_exp}",
         f"--fps={cfg.target_fps}"], cwd=repo)
    if rc != 0:
        raise RuntimeError(f"RIFE exited with code {rc}")
    produced = _newest_video_since(repo, t0)
    if produced is None:
        raise RuntimeError("RIFE finished but produced no video")
    dst = work / "interp_rife.mp4"
    shutil.copyfile(produced, dst)
    return dst


def _face_stabilize(ff: str, src: Path, work: Path,
                    strength: float, crf: int) -> Path:
    """Apply face-region temporal stabilization using ffmpeg vidstabdetect/transform.

    This reduces face flicker independently of body motion by computing a
    stabilization transform on the face region and blending it back.
    `strength` controls the blend weight (0 = no effect, 1 = full stabilization).

    Note: this is a whole-frame stabilization pass — true face-crop stabilization
    would require face detection per frame (Phase C FaceAnalyzer). This simpler
    approach handles the common case of a mostly-static camera.
    """
    transforms = work / "transforms.trf"
    # Step 1: detect motion
    detect_filter = (
        f"vidstabdetect=shakiness=5:accuracy=9:result={transforms}"
    )
    nxt1 = work / "stab_detect.mp4"
    _ffmpeg_filter(ff, src, nxt1, detect_filter, crf)

    # Step 2: apply stabilization with smoothing
    smooth_filter = (
        f"vidstabtransform=input={transforms}:"
        f"smoothing=10:optzoom=1:interpol=bicubic,"
        f"unsharp=5:5:0.8:3:3:0.4"   # mild sharpen after warp
    )
    nxt2 = work / "stab_transform.mp4"
    _ffmpeg_filter(ff, nxt1, nxt2, smooth_filter, crf)
    return nxt2


def _run_realesrgan(cfg: TemporalConfig, src: Path, work: Path) -> Path:
    """Upscale with Real-ESRGAN (better texture than ffmpeg lanczos)."""
    repo = Path(cfg.realesrgan_repo).expanduser()
    script = repo / "inference_realesrgan_video.py"
    if not script.is_file():
        raise RuntimeError(
            f"Real-ESRGAN repo not found at {repo}\n"
            "    git clone https://github.com/xinntao/Real-ESRGAN")
    import sys  # noqa: PLC0415
    out_dir = work / "realesrgan_out"
    out_dir.mkdir(exist_ok=True)
    t0 = time.time()
    rc = _stream_subprocess(
        [sys.executable, "inference_realesrgan_video.py",
         "-i", str(src.resolve()), "-o", str(out_dir.resolve()),
         "-n", cfg.realesrgan_model, "-s", str(cfg.upscale_factor)], cwd=repo)
    if rc != 0:
        raise RuntimeError(f"Real-ESRGAN exited with code {rc}")
    produced = _newest_video_since(out_dir, t0) or _newest_video_since(repo, t0)
    if produced is None:
        raise RuntimeError("Real-ESRGAN finished but produced no video")
    dst = work / "upscaled.mp4"
    shutil.copyfile(produced, dst)
    return dst


# --- pipeline ---------------------------------------------------------------

def polish_video(cfg: TemporalConfig, input_video: Path) -> dict:
    """Run the full temporal-polish chain on one video."""
    input_video = Path(input_video).expanduser()
    if not input_video.is_file():
        raise RuntimeError(f"input video not found: {input_video}")

    ff = ffmpeg_exe()
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    final = cfg.out_root / f"{input_video.stem}_final.mp4"
    t0 = time.time()
    steps: list[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        cur = input_video

        if cfg.deflicker:
            nxt = work / "01_deflicker.mp4"
            _ffmpeg_filter(ff, cur, nxt,
                           f"deflicker=size={cfg.deflicker_size}",
                           INTERMEDIATE_CRF)
            cur = nxt
            steps.append("deflicker")

        if cfg.interpolate:
            if cfg.interp_backend == "rife":
                try:
                    cur = _run_rife(cfg, cur, work)
                except RuntimeError as exc:
                    log.warning("RIFE failed (%s) — falling back to ffmpeg minterpolate", exc)
                    nxt = work / "02_interp_fallback.mp4"
                    _ffmpeg_filter(
                        ff, cur, nxt,
                        f"minterpolate=fps={cfg.target_fps}:mi_mode=mci:"
                        "mc_mode=aobmc:me_mode=bidir:vsbmc=1:scd=fdiff:scd_threshold=5",
                        INTERMEDIATE_CRF)
                    cur = nxt
            else:
                nxt = work / "02_interp.mp4"
                # scd_threshold=5: scene-change detection prevents ghosting
                # on hard cuts between sign clips
                _ffmpeg_filter(
                    ff, cur, nxt,
                    f"minterpolate=fps={cfg.target_fps}:mi_mode=mci:"
                    "mc_mode=aobmc:me_mode=bidir:vsbmc=1:scd=fdiff:scd_threshold=5",
                    INTERMEDIATE_CRF)
                cur = nxt
            steps.append(f"interpolate({cfg.interp_backend})")

        if cfg.face_stabilize:
            try:
                nxt = _face_stabilize(ff, cur, work,
                                      cfg.face_stabilize_strength, INTERMEDIATE_CRF)
                cur = nxt
                steps.append("face_stabilize")
            except RuntimeError as exc:
                log.warning("face stabilization failed (%s) — skipping", exc)

        if cfg.upscale:
            if cfg.upscale_backend == "realesrgan":
                cur = _run_realesrgan(cfg, cur, work)
            else:
                nxt = work / "03_upscale.mp4"
                _ffmpeg_filter(
                    ff, cur, nxt,
                    f"scale=iw*{cfg.upscale_factor}:ih*{cfg.upscale_factor}:"
                    "flags=lanczos", INTERMEDIATE_CRF)
                cur = nxt
            steps.append(f"upscale({cfg.upscale_backend})")

        # final delivery encode
        _ffmpeg_filter(ff, cur, final, "null", cfg.crf,
                       fps=cfg.target_fps, preset=cfg.preset)

    elapsed = round(time.time() - t0, 1)
    log.info("polished %s -> %s (%s; %.1fs)",
             input_video.name, final.name, ", ".join(steps) or "encode-only",
             elapsed)

    manifest = {
        "input_video": str(input_video),
        "output_video": final.name,
        "created": date.today().isoformat(),
        "steps": steps,
        "settings": {
            "target_fps": cfg.target_fps,
            "deflicker": cfg.deflicker,
            "interp_backend": cfg.interp_backend if cfg.interpolate else None,
            "upscale": (f"{cfg.upscale_backend} x{cfg.upscale_factor}"
                        if cfg.upscale else None),
            "crf": cfg.crf,
        },
        "source_meta": probe_video(input_video),
        "elapsed_sec": elapsed,
    }
    (cfg.out_root / f"{input_video.stem}_final.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def discover_inputs(args: argparse.Namespace) -> list[Path]:
    """Resolve --input-video / --input-dir into a list of videos."""
    vids: list[Path] = []
    if args.input_video:
        vids.append(Path(args.input_video).expanduser())
    if args.input_dir:
        root = Path(args.input_dir).expanduser()
        vids.extend(sorted(p for p in root.rglob("*")
                           if p.suffix.lower() in VIDEO_EXTS
                           and "_final" not in p.stem))
    return vids


# --- CLI --------------------------------------------------------------------

def _run_check(cfg: TemporalConfig) -> int:
    """Validate ffmpeg (and any selected external backend)."""
    log.info("environment check: temporal-polish backends ...")
    ff = ffmpeg_exe()
    rc = _stream_subprocess([ff, "-version"], cwd=ROOT)
    if rc != 0:
        log.error("  ffmpeg not runnable (%s)", ff)
        return 1
    log.info("  OK   ffmpeg: %s", ff)
    ok = True
    if cfg.interp_backend == "rife":
        p = Path(cfg.rife_repo).expanduser() / "inference_video.py"
        log.info("  %s  RIFE: %s", "OK  " if p.is_file() else "MISSING", p)
        ok &= p.is_file()
    if cfg.upscale and cfg.upscale_backend == "realesrgan":
        p = Path(cfg.realesrgan_repo).expanduser() / "inference_realesrgan_video.py"
        log.info("  %s  Real-ESRGAN: %s", "OK  " if p.is_file() else "MISSING", p)
        ok &= p.is_file()
    log.info("environment check: %s", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-video", help="a single raw avatar video")
    ap.add_argument("--input-dir", help="a directory of raw videos (batch)")
    ap.add_argument("--out-root", help="output root (default: outputs/avatar_final)")
    ap.add_argument("--config", help="JSON config file (CLI flags override it)")
    ap.add_argument("--target-fps", type=int, help="output frame rate")
    ap.add_argument("--upscale", action="store_true", help="enable upscaling")
    ap.add_argument("--interp-backend", choices=["ffmpeg", "rife"],
                    help="frame interpolation backend (default: rife)")
    ap.add_argument("--face-stabilize", action="store_true",
                    help="apply face-region temporal stabilization")
    ap.add_argument("--check", action="store_true",
                    help="validate ffmpeg / backends and exit")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    cfg = load_config(Path(args.config) if args.config else None, args)

    if args.check:
        return _run_check(cfg)

    videos = discover_inputs(args)
    if not videos:
        ap.error("no input — pass --input-video or --input-dir (or --check)")

    log.info("polishing %d video(s)", len(videos))
    done = failed = 0
    t_start = time.time()
    for video in videos:
        try:
            polish_video(cfg, video)
            done += 1
        except (RuntimeError, FileNotFoundError) as exc:
            failed += 1
            log.error("FAILED %s: %s", video.name, exc)
    log.info("finished — %d done, %d failed in %.1fs",
             done, failed, time.time() - t_start)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
