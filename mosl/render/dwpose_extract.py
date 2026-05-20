"""Phase B — DWPose whole-body keypoint extraction.

Re-extracts dense, consistent whole-body keypoints (body + face + hands) from
sign-language videos or image sequences, and writes them in the **OpenPose JSON
layout** the rest of this project already consumes:

    <out_root>/<clip>/<clip>_<frame:012d>_keypoints.json   (CMU OpenPose schema)
    <out_root>/<clip>/manifest.json                        (clip metadata)
    <out_root>/<clip>/keypoints.npz        (--npz, stacked arrays, optional)

Why this phase exists
---------------------
The original `mosl/pose/` extraction (Hzzone OpenPose / MediaPipe) produces
noisy keypoints with an inconsistent body model across clips. ControlNet-OpenPose
and pose-guided video models (MimicMotion / AnimateDiff / SVD) are trained on
**DWPose** output. This module re-extracts with DWPose so the conditioning
signal is clean and standard — without changing any existing code.

Compatibility
-------------
* Output JSON schema is identical to `mosl/pose/extract_one.py` — body in
  OpenPose COCO-18 order (54 floats), hands 21 pts (63), face 68 pts (204).
* The NPZ uses the exact keys `mosl/pose/export_openpose_json.py` expects.
* It reuses Phase A's `interpolate_gaps` for gap filling and can chain Phase A's
  `bridge_clip` to emit ControlNet-ready pose frames in one pass (--render).

Backend
-------
DWPose is run via **rtmlib** (the maintained runner for the RTMPose/DWPose
whole-body models). It is the only external dependency this module adds:

    pip install rtmlib onnxruntime-gpu opencv-python

`Wholebody` returns raw COCO-WholeBody 133-keypoint output; the COCO-WholeBody
-> OpenPose conversion is done explicitly here (see `wholebody_to_openpose`) so
it is verifiable and not dependent on a library-internal remap.

NOTE: this module is written for DGX execution and has not been run in this
environment (no GPU / no rtmlib here). Validate the environment first with:

    python -m mosl.render.dwpose_extract --check

Usage
-----
    # single video
    python -m mosl.render.dwpose_extract --video clip.mp4
    # every video in a directory tree (batch)
    python -m mosl.render.dwpose_extract --video-dir data/raw/vedios-dataset
    # one image sequence  /  a tree of image-sequence folders
    python -m mosl.render.dwpose_extract --frames-dir frames/clip01
    python -m mosl.render.dwpose_extract --frames-root outputs/frames
    # also emit ControlNet pose frames (Phase A) in the same pass
    python -m mosl.render.dwpose_extract --video clip.mp4 --render
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .pose_bridge import CONF_THR, bridge_clip, interpolate_gaps

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("mosl.render.dwpose")

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# --- COCO-WholeBody 133 layout ----------------------------------------------
# 0-16 body | 17-22 foot | 23-90 face (68) | 91-111 left hand | 112-132 right
WB_FACE = slice(23, 91)
WB_LHAND = slice(91, 112)
WB_RHAND = slice(112, 133)

# OpenPose-18 body index -> COCO-17 body index (None marks the synthesized
# neck joint, computed as the shoulder midpoint).
OP18_FROM_COCO17 = [0, None, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]


# --- configuration ----------------------------------------------------------

@dataclass
class ExtractConfig:
    """All tunables for one extraction run."""
    out_root: Path
    mode: str = "balanced"            # rtmlib: performance | lightweight | balanced
    backend: str = "onnxruntime"      # rtmlib: onnxruntime | opencv | openvino
    device: str = "cuda"              # cuda | cpu | mps
    interpolate: bool = True          # gap-fill missing keypoints (Phase A)
    smooth: bool = True               # One-Euro temporal smoothing
    min_cutoff: float = 1.0           # One-Euro: lower => smoother (more lag)
    beta: float = 0.15                # One-Euro: higher => less lag on fast motion
    npz: bool = False                 # also write stacked keypoints.npz
    render: bool = False              # also emit Phase A ControlNet pose frames
    render_root: Path = ROOT / "outputs" / "pose_control"
    canvas: int = 768                 # render canvas size (when --render)
    fill: float = 0.72                # render figure fill fraction
    overwrite: bool = False           # re-process clips already complete
    limit: int = 0                    # cap number of clips (0 = all)


# --- One-Euro temporal smoothing -------------------------------------------

def _alpha(cutoff: float, freq: float) -> float:
    tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
    return 1.0 / (1.0 + tau * freq)


def one_euro_smooth(seq: np.ndarray, freq: float, min_cutoff: float,
                    beta: float, d_cutoff: float = 1.0) -> np.ndarray:
    """One-Euro filter over time, vectorised across channels.

    `seq` is (T, D); each of the D channels is filtered independently along T.
    One-Euro adapts its cutoff to motion speed: it smooths still keypoints hard
    (kills jitter) while staying responsive on fast hand motion (low lag) — the
    right trade-off for sign-language conditioning fed to a video diffusion
    model, where both jitter and lag turn into temporal artifacts.
    """
    T, _ = seq.shape
    if T < 3:
        return seq.copy()
    out = np.empty_like(seq)
    out[0] = seq[0]
    x_prev = seq[0].astype(np.float64)
    dx_prev = np.zeros_like(x_prev)
    a_d = _alpha(d_cutoff, freq)
    for t in range(1, T):
        x = seq[t].astype(np.float64)
        dx = (x - x_prev) * freq
        dx_hat = a_d * dx + (1.0 - a_d) * dx_prev
        cutoff = min_cutoff + beta * np.abs(dx_hat)
        a = np.array([_alpha(c, freq) for c in cutoff])
        x_hat = a * x + (1.0 - a) * x_prev
        out[t] = x_hat
        x_prev, dx_prev = x_hat, dx_hat
    return out


def smooth_part(part: np.ndarray, freq: float, cfg: ExtractConfig) -> np.ndarray:
    """Apply One-Euro to the (x, y) channels of a (T, K, 3) keypoint array."""
    if part.shape[1] == 0 or part.shape[0] < 3:
        return part
    out = part.copy()
    T, K, _ = out.shape
    xy = out[:, :, :2].reshape(T, K * 2)
    out[:, :, :2] = one_euro_smooth(
        xy, freq, cfg.min_cutoff, cfg.beta).reshape(T, K, 2)
    return out


# --- DWPose backend ---------------------------------------------------------

class DWPoseEstimator:
    """Thin wrapper over rtmlib's whole-body model.

    Isolated on purpose: if the rtmlib API shifts, this class is the only place
    that needs changing. `__call__` returns the single best-scoring person.
    """

    def __init__(self, cfg: ExtractConfig) -> None:
        try:
            from rtmlib import Wholebody  # noqa: PLC0415 - optional heavy dep
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "rtmlib is not installed. On the DGX run:\n"
                "    pip install rtmlib onnxruntime-gpu opencv-python"
            ) from exc

        log.info("loading DWPose (mode=%s backend=%s device=%s)",
                 cfg.mode, cfg.backend, cfg.device)
        # to_openpose=False -> raw COCO-WholeBody 133; we convert explicitly.
        self._model = Wholebody(
            mode=cfg.mode, to_openpose=False,
            backend=cfg.backend, device=cfg.device,
        )

    def __call__(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (keypoints (133,2), scores (133,)) for the best person, or None."""
        keypoints, scores = self._model(image_bgr)
        if keypoints is None or len(keypoints) == 0:
            return None
        # pick the person with the highest mean body-keypoint confidence
        best = int(np.argmax(scores[:, :17].mean(axis=1)))
        return (np.asarray(keypoints[best], dtype=np.float32),
                np.asarray(scores[best], dtype=np.float32))


# --- COCO-WholeBody -> OpenPose conversion ----------------------------------

def wholebody_to_openpose(kpts: np.ndarray, scores: np.ndarray) -> dict[str, np.ndarray]:
    """Convert one frame of COCO-WholeBody 133 to OpenPose-format parts.

    Returns body (18,3), face (68,3), hand_l (21,3), hand_r (21,3); the third
    channel is confidence. Pixel coordinates are preserved unchanged.
    """
    body = np.zeros((18, 3), dtype=np.float32)
    for op, coco in enumerate(OP18_FROM_COCO17):
        if coco is None:
            continue
        body[op, :2] = kpts[coco]
        body[op, 2] = scores[coco]
    # synthesized neck = shoulder midpoint; confidence = min of both shoulders
    l_sh, r_sh = kpts[5], kpts[6]
    body[1, :2] = (l_sh + r_sh) * 0.5
    body[1, 2] = float(min(scores[5], scores[6]))

    def part(sl: slice) -> np.ndarray:
        return np.concatenate([kpts[sl], scores[sl, None]], axis=1).astype(np.float32)

    return {
        "body": body,
        "face": part(WB_FACE),
        "hand_l": part(WB_LHAND),
        "hand_r": part(WB_RHAND),
    }


# --- frame readers ----------------------------------------------------------

def read_video(path: Path) -> tuple[list[np.ndarray], float, tuple[int, int]]:
    """Decode a video to a list of BGR frames. Returns (frames, fps, (w, h))."""
    import cv2  # noqa: PLC0415 - heavy, only needed at run time

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    h, w = frames[0].shape[:2]
    return frames, float(fps), (w, h)


def read_image_sequence(directory: Path) -> tuple[list[np.ndarray], float, tuple[int, int]]:
    """Load a sorted directory of images as BGR frames (fps defaults to 25)."""
    import cv2  # noqa: PLC0415

    files = sorted(p for p in directory.iterdir()
                   if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise RuntimeError(f"no images in {directory}")
    frames = []
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            log.warning("unreadable image skipped: %s", p)
            continue
        frames.append(img)
    if not frames:
        raise RuntimeError(f"no readable images in {directory}")
    h, w = frames[0].shape[:2]
    return frames, 25.0, (w, h)


# --- writers ----------------------------------------------------------------

def _frame_record(body: np.ndarray, face: np.ndarray,
                  hand_l: np.ndarray, hand_r: np.ndarray) -> dict:
    """Build one CMU-OpenPose-schema JSON record for a single frame."""
    return {
        "version": 1.3,
        "people": [{
            "person_id": [-1],
            "pose_keypoints_2d": body.reshape(-1).astype(float).tolist(),
            "face_keypoints_2d": face.reshape(-1).astype(float).tolist(),
            "hand_left_keypoints_2d": hand_l.reshape(-1).astype(float).tolist(),
            "hand_right_keypoints_2d": hand_r.reshape(-1).astype(float).tolist(),
            "pose_keypoints_3d": [],
            "face_keypoints_3d": [],
            "hand_left_keypoints_3d": [],
            "hand_right_keypoints_3d": [],
        }],
    }


def write_clip_outputs(out_dir: Path, clip: str, body: np.ndarray, face: np.ndarray,
                       hand_l: np.ndarray, hand_r: np.ndarray, cfg: ExtractConfig) -> None:
    """Write per-frame OpenPose JSON and, if requested, a stacked NPZ."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(body.shape[0]):
        record = _frame_record(body[i], face[i], hand_l[i], hand_r[i])
        # <stem>_<frame>_keypoints.json — the naming locked in docs/DECISIONS.md
        name = f"{clip}_{i:012d}_keypoints.json"
        with open(out_dir / name, "w", encoding="utf-8") as fh:
            json.dump(record, fh)

    if cfg.npz:
        # keys match mosl/pose/export_openpose_json.py's expected NPZ schema
        np.savez_compressed(
            out_dir / "keypoints.npz",
            pose_keypoints_2d=body.reshape(body.shape[0], -1),
            face_keypoints_2d=face.reshape(face.shape[0], -1),
            hand_left_keypoints_2d=hand_l.reshape(hand_l.shape[0], -1),
            hand_right_keypoints_2d=hand_r.reshape(hand_r.shape[0], -1),
        )


# --- clip processing --------------------------------------------------------

def _is_complete(out_dir: Path, n_frames: int) -> bool:
    return out_dir.is_dir() and \
        len(list(out_dir.glob("*_keypoints.json"))) == n_frames


def process_clip(frames: list[np.ndarray], clip: str, fps: float,
                 resolution: tuple[int, int], source: str,
                 estimator: DWPoseEstimator, cfg: ExtractConfig) -> dict:
    """Extract, clean, and write one clip. Returns its manifest dict."""
    out_dir = cfg.out_root / clip
    n = len(frames)
    if not cfg.overwrite and _is_complete(out_dir, n):
        log.info("skip (already complete): %s", clip)
        return {"clip_name": clip, "n_frames": n, "status": "skipped"}

    t0 = time.time()
    body, face, hand_l, hand_r = [], [], [], []
    n_missing = 0
    for i, frame in enumerate(frames):
        try:
            result = estimator(frame)
        except Exception as exc:  # noqa: BLE001 - one bad frame must not abort
            log.warning("%s frame %d: estimator error: %s", clip, i, exc)
            result = None
        if result is None:
            n_missing += 1
            body.append(np.zeros((18, 3), dtype=np.float32))
            face.append(np.zeros((68, 3), dtype=np.float32))
            hand_l.append(np.zeros((21, 3), dtype=np.float32))
            hand_r.append(np.zeros((21, 3), dtype=np.float32))
        else:
            parts = wholebody_to_openpose(*result)
            body.append(parts["body"])
            face.append(parts["face"])
            hand_l.append(parts["hand_l"])
            hand_r.append(parts["hand_r"])

    arrays = {k: np.stack(v) for k, v in
              {"body": body, "face": face, "hand_l": hand_l, "hand_r": hand_r}.items()}

    if cfg.interpolate:  # reuse Phase A gap-fill
        arrays = {k: interpolate_gaps(v) for k, v in arrays.items()}
    if cfg.smooth:       # One-Euro temporal smoothing
        arrays = {k: smooth_part(v, fps, cfg) for k, v in arrays.items()}

    write_clip_outputs(out_dir, clip, arrays["body"], arrays["face"],
                       arrays["hand_l"], arrays["hand_r"], cfg)

    manifest = {
        "clip_name": clip,
        "source": source,
        "n_frames": n,
        "fps": round(fps, 3),
        "width": resolution[0],
        "height": resolution[1],
        "estimator": "dwpose/rtmlib",
        "mode": cfg.mode,
        "backend": cfg.backend,
        "device": cfg.device,
        "keypoint_format": {"body": "openpose-coco18", "face": 68, "hands": 21},
        "frames_no_person": n_missing,
        "interpolated": cfg.interpolate,
        "smoothed": cfg.smooth,
        "one_euro": {"min_cutoff": cfg.min_cutoff, "beta": cfg.beta},
        "npz": cfg.npz,
        "elapsed_sec": round(time.time() - t0, 2),
        "status": "ok",
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    if n_missing:
        log.warning("%s: no person detected in %d/%d frames", clip, n_missing, n)
    log.info("done %s — %d frames in %.1fs", clip, n, manifest["elapsed_sec"])

    if cfg.render:  # chain Phase A: emit ControlNet-ready pose frames
        try:
            bridge_clip(out_dir, cfg.render_root, canvas=cfg.canvas,
                        fill=cfg.fill, fps=int(round(fps)))
            manifest["rendered"] = True
        except Exception as exc:  # noqa: BLE001
            log.error("%s: pose-frame render failed: %s", clip, exc)
            manifest["rendered"] = False
    return manifest


# --- input discovery --------------------------------------------------------

def discover_clips(args: argparse.Namespace) -> list[tuple[str, Path, str]]:
    """Resolve CLI input flags to a list of (clip_name, path, kind) jobs."""
    jobs: list[tuple[str, Path, str]] = []
    if args.video:
        p = Path(args.video)
        jobs.append((p.stem, p, "video"))
    if args.video_dir:
        root = Path(args.video_dir)
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() in VIDEO_EXTS:
                jobs.append((p.stem, p, "video"))
    if args.frames_dir:
        p = Path(args.frames_dir)
        jobs.append((p.name, p, "frames"))
    if args.frames_root:
        root = Path(args.frames_root)
        for p in sorted(root.iterdir()):
            if p.is_dir() and any(f.suffix.lower() in IMAGE_EXTS
                                  for f in p.iterdir()):
                jobs.append((p.name, p, "frames"))
    return jobs


# --- CLI --------------------------------------------------------------------

def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def _run_check(cfg: ExtractConfig) -> int:
    """Validate that the DWPose backend loads and runs — no data needed."""
    log.info("environment check: loading DWPose backend ...")
    try:
        estimator = DWPoseEstimator(cfg)
    except RuntimeError as exc:
        log.error("%s", exc)
        return 1
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    out = estimator(dummy)
    log.info("backend OK — inference ran (persons on blank frame: %s)",
             0 if out is None else 1)
    log.info("ready: rtmlib + %s on %s", cfg.backend, cfg.device)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("inputs (choose one or more)")
    src.add_argument("--video", help="a single video file")
    src.add_argument("--video-dir", help="directory tree of video files")
    src.add_argument("--frames-dir", help="one image-sequence directory")
    src.add_argument("--frames-root", help="directory of image-sequence folders")

    out = ap.add_argument_group("output")
    out.add_argument("--out-root", default=str(ROOT / "outputs" / "dwpose_json"),
                     help="root for per-clip JSON output")
    out.add_argument("--npz", action="store_true",
                     help="also write a stacked keypoints.npz per clip")
    out.add_argument("--render", action="store_true",
                     help="also emit Phase A ControlNet pose frames")
    out.add_argument("--render-root",
                     default=str(ROOT / "outputs" / "pose_control"),
                     help="root for --render pose frames")
    out.add_argument("--canvas", type=int, default=768)
    out.add_argument("--fill", type=float, default=0.72)

    mdl = ap.add_argument_group("DWPose backend")
    mdl.add_argument("--mode", default="balanced",
                     choices=["performance", "lightweight", "balanced"])
    mdl.add_argument("--backend", default="onnxruntime",
                     choices=["onnxruntime", "opencv", "openvino"])
    mdl.add_argument("--device", default="cuda",
                     choices=["cuda", "cpu", "mps"])

    proc = ap.add_argument_group("processing")
    proc.add_argument("--no-interp", action="store_true",
                      help="disable keypoint gap interpolation")
    proc.add_argument("--no-smooth", action="store_true",
                      help="disable One-Euro temporal smoothing")
    proc.add_argument("--min-cutoff", type=float, default=1.0)
    proc.add_argument("--beta", type=float, default=0.15)
    proc.add_argument("--overwrite", action="store_true",
                      help="re-process clips already complete")
    proc.add_argument("--limit", type=int, default=0,
                      help="process at most N clips (0 = all)")

    ap.add_argument("--check", action="store_true",
                    help="validate the DWPose environment and exit")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    _setup_logging(args.log_level)
    cfg = ExtractConfig(
        out_root=Path(args.out_root),
        mode=args.mode, backend=args.backend, device=args.device,
        interpolate=not args.no_interp, smooth=not args.no_smooth,
        min_cutoff=args.min_cutoff, beta=args.beta,
        npz=args.npz, render=args.render, render_root=Path(args.render_root),
        canvas=args.canvas, fill=args.fill,
        overwrite=args.overwrite, limit=args.limit,
    )

    if args.check:
        return _run_check(cfg)

    jobs = discover_clips(args)
    if not jobs:
        ap.error("no inputs — pass --video / --video-dir / --frames-dir / "
                 "--frames-root (or --check)")
    if cfg.limit > 0:
        jobs = jobs[:cfg.limit]
    log.info("discovered %d clip(s); output -> %s", len(jobs), cfg.out_root)

    try:
        estimator = DWPoseEstimator(cfg)
    except RuntimeError as exc:
        log.error("%s", exc)
        return 1
    done = skipped = failed = total_frames = 0
    t_start = time.time()
    for clip, path, kind in jobs:
        try:
            if kind == "video":
                frames, fps, res = read_video(path)
            else:
                frames, fps, res = read_image_sequence(path)
            m = process_clip(frames, clip, fps, res, str(path), estimator, cfg)
            if m["status"] == "skipped":
                skipped += 1
            else:
                done += 1
                total_frames += m["n_frames"]
        except Exception as exc:  # noqa: BLE001 - one bad clip must not abort
            failed += 1
            log.error("FAILED %s (%s): %s", clip, path, exc)

    log.info("finished — %d done, %d skipped, %d failed; %d frames in %.1fs",
             done, skipped, failed, total_frames, time.time() - t_start)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
