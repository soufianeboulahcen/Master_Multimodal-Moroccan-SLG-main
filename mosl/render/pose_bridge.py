"""Phase A — OpenPose JSON  ->  ControlNet-ready pose-conditioning frames.

This is the keystone that connects the existing pose pipeline to a diffusion
video renderer. It reads the per-frame OpenPose JSON our `mosl/pose/` stage
already produces and draws the canonical OpenPose *skeleton image* (coloured
limbs on black) that ControlNet-OpenPose and pose-guided video models
(MimicMotion / UniAnimate / AnimateDiff) consume as conditioning.

It does not require a GPU, opencv, torch, or any diffusion dependency — only
numpy + Pillow — so it runs identically on a laptop and on the DGX Spark.

What it handles
---------------
* Body format auto-detection: COCO-18 (54 floats, sign clips) and
  BODY-25 (75 floats, synthetic clips) — BODY-25 is remapped to COCO-18.
* 21-point hands (full topology) and sparse hands (points only).
* Optional face landmarks (any count — MediaPipe 478 or OpenPose 70).
* Confidence-gated drawing: low-confidence joints/limbs are skipped.
* Gap interpolation: short keypoint dropouts are linearly filled so the
  conditioning signal does not flicker between frames (--no-interp to disable).
* One global fit transform for the whole clip, so the figure is framed
  identically in every frame — a prerequisite for temporal stability.

Outputs (under <out_dir>/<clip_name>/)
    pose_000000.png ...     one RGB skeleton frame per input frame
    preview.gif             animated preview of the sequence
    manifest.json           {n_frames, canvas, fps, body_format, ...}

Usage
    python -m mosl.render.pose_bridge <json_dir> [<out_dir>] [options]
    python -m mosl.render.pose_bridge outputs/openpose_json/أَنْتِ_keypoints
"""
from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]

# --- keypoint topology (canonical OpenPose / ControlNet convention) ---------

# COCO-18 limb pairs (0-indexed) and the standard 18-colour OpenPose palette.
COCO18_LIMBS = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9),
    (9, 10), (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16),
    (0, 15), (15, 17),
]
COCO18_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170), (255, 0, 85),
]

# BODY-25 -> COCO-18 row remap. Indices 0..7 are identical; the rest differ
# because BODY-25 inserts a mid-hip joint at index 8.
BODY25_TO_COCO18 = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

# 21-point hand bone chains (0-indexed), thumb + four fingers.
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
    (15, 16), (0, 17), (17, 18), (18, 19), (19, 20),
]

CONF_THR = 0.10  # joints below this confidence are treated as missing


# --- loading ----------------------------------------------------------------

def _frame_files(json_dir: Path) -> list[Path]:
    files = sorted(json_dir.glob("*_keypoints.json"))
    if not files:
        raise FileNotFoundError(f"no *_keypoints.json files in {json_dir}")
    return files


def _person(record: dict) -> dict | None:
    people = record.get("people") or []
    return people[0] if people else None


def _kpts(person: dict | None, key: str, n_expected: int | None = None) -> np.ndarray:
    """Return an (K, 3) float array for one keypoint field; zeros if absent."""
    flat = (person or {}).get(key) or []
    if not flat:
        return np.zeros((n_expected or 0, 3), dtype=np.float32)
    a = np.asarray(flat, dtype=np.float32).reshape(-1, 3)
    return a


def load_clip(json_dir: Path) -> dict:
    """Load a clip's keypoints into stacked arrays.

    Returns dict with body (T,18,3), hand_l/hand_r (T,21or11,3),
    face (T,F,3) and metadata. Body is normalised to COCO-18 ordering.
    """
    files = _frame_files(json_dir)
    body, hl, hr, face = [], [], [], []
    body_format = "coco18"

    for fp in files:
        with open(fp, encoding="utf-8") as fh:
            person = _person(json.load(fh))
        raw_body = _kpts(person, "pose_keypoints_2d")
        if raw_body.shape[0] == 25:               # BODY-25 -> COCO-18
            body_format = "body25"
            raw_body = raw_body[BODY25_TO_COCO18]
        elif raw_body.shape[0] == 0:
            raw_body = np.zeros((18, 3), dtype=np.float32)
        body.append(raw_body)
        hl.append(_kpts(person, "hand_left_keypoints_2d"))
        hr.append(_kpts(person, "hand_right_keypoints_2d"))
        face.append(_kpts(person, "face_keypoints_2d"))

    def _stack(seq: list[np.ndarray]) -> np.ndarray:
        width = max((a.shape[0] for a in seq), default=0)
        if width == 0:
            return np.zeros((len(seq), 0, 3), dtype=np.float32)
        out = np.zeros((len(seq), width, 3), dtype=np.float32)
        for i, a in enumerate(seq):
            if a.shape[0]:
                out[i, : a.shape[0]] = a
        return out

    return {
        "body": _stack(body),
        "hand_l": _stack(hl),
        "hand_r": _stack(hr),
        "face": _stack(face),
        "n_frames": len(files),
        "body_format": body_format,
        "clip_name": json_dir.name,
    }


# --- preprocessing ----------------------------------------------------------

def interpolate_gaps(kpts: np.ndarray) -> np.ndarray:
    """Linearly fill short keypoint dropouts along time.

    `kpts` is (T, K, 3). A keypoint is 'missing' in a frame when its
    confidence <= CONF_THR. Missing runs are linearly interpolated between the
    nearest valid frames; leading/trailing gaps hold the nearest valid value.
    This removes the per-frame popping that otherwise propagates into the
    diffusion output as flicker.
    """
    if kpts.shape[1] == 0:
        return kpts
    out = kpts.copy()
    T, K, _ = out.shape
    for k in range(K):
        valid = out[:, k, 2] > CONF_THR
        if valid.sum() < 2:
            continue
        idx = np.arange(T)
        good = idx[valid]
        for c in (0, 1):  # x, y
            out[:, k, c] = np.interp(idx, good, out[good, k, c])
        out[:, k, 2] = np.interp(idx, good, out[good, k, 2])
    return out


def compute_transform(clip: dict, canvas: int, fill: float) -> tuple[float, float, float]:
    """One scale+offset for the whole clip, from the global keypoint bbox.

    Using a single transform for every frame keeps the avatar framed
    identically across the clip — drift in framing reads as camera jitter.
    Face points are excluded so dense face meshes do not shrink the body.
    """
    pts = []
    for key in ("body", "hand_l", "hand_r"):
        a = clip[key]
        if a.shape[1] == 0:
            continue
        m = a[:, :, 2] > CONF_THR
        if m.any():
            pts.append(a[:, :, :2][m])
    if not pts:
        return 1.0, 0.0, 0.0
    p = np.concatenate(pts, axis=0)
    x0, y0 = p[:, 0].min(), p[:, 1].min()
    x1, y1 = p[:, 0].max(), p[:, 1].max()
    span = max(x1 - x0, y1 - y0, 1e-6)
    scale = fill * canvas / span
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    off_x = canvas / 2 - cx * scale
    off_y = canvas / 2 - cy * scale
    return scale, off_x, off_y


def _apply(kpts: np.ndarray, s: float, ox: float, oy: float) -> np.ndarray:
    out = kpts.copy()
    out[..., 0] = out[..., 0] * s + ox
    out[..., 1] = out[..., 1] * s + oy
    return out


# --- drawing ----------------------------------------------------------------

def _hand_edge_color(i: int, n: int) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(i / max(n, 1), 1.0, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def draw_pose_frame(
    body: np.ndarray, hand_l: np.ndarray, hand_r: np.ndarray,
    face: np.ndarray, canvas: int, draw_face: bool,
) -> Image.Image:
    """Draw one OpenPose skeleton frame (coloured limbs on black).

    All coordinates are cast to native Python floats before they reach Pillow:
    passing numpy scalars into ImageDraw segfaults on some Pillow builds.
    """
    img = Image.new("RGB", (canvas, canvas), (0, 0, 0))
    d = ImageDraw.Draw(img)
    sw = max(2, canvas // 170)          # limb stroke width
    jr = max(2, canvas // 200)          # joint radius

    def line(p: np.ndarray, q: np.ndarray, color, width: int) -> None:
        d.line([(float(p[0]), float(p[1])), (float(q[0]), float(q[1]))],
               fill=color, width=int(width))

    def dot(p: np.ndarray, r: int, color) -> None:
        x, y = float(p[0]), float(p[1])
        d.ellipse([x - r, y - r, x + r, y + r], fill=color)

    # body limbs + joints
    for li, (a, b) in enumerate(COCO18_LIMBS):
        if body[a, 2] > CONF_THR and body[b, 2] > CONF_THR:
            line(body[a], body[b], COCO18_COLORS[li], sw)
    for j in range(body.shape[0]):
        if body[j, 2] > CONF_THR:
            dot(body[j], jr, COCO18_COLORS[j])

    # hands — full bone chains when 21 points are present, else points only
    for hand in (hand_l, hand_r):
        if hand.shape[0] >= 21:
            for ei, (a, b) in enumerate(HAND_EDGES):
                if hand[a, 2] > CONF_THR and hand[b, 2] > CONF_THR:
                    line(hand[a], hand[b],
                         _hand_edge_color(ei, len(HAND_EDGES)), max(2, sw - 2))
        for j in range(hand.shape[0]):
            if hand[j, 2] > CONF_THR:
                dot(hand[j], max(1, jr - 2), (0, 0, 255))

    # face landmarks — light dots; canonical 70-point face comes in Phase B.
    # Drawn as 1px ellipses, not ImageDraw.point() (segfaults on some builds).
    if draw_face and face.shape[0] > 0:
        for j in range(face.shape[0]):
            if face[j, 2] > CONF_THR:
                dot(face[j], 1, (255, 255, 255))

    return img


# --- driver -----------------------------------------------------------------

def bridge_clip(
    json_dir: Path, out_dir: Path, canvas: int = 768, fill: float = 0.72,
    fps: int = 25, draw_face: bool = True, interpolate: bool = True,
    preview: bool = False,
) -> dict:
    """Convert one OpenPose JSON directory to a ControlNet pose-frame sequence."""
    clip = load_clip(json_dir)
    if interpolate:
        for key in ("body", "hand_l", "hand_r", "face"):
            clip[key] = interpolate_gaps(clip[key])

    s, ox, oy = compute_transform(clip, canvas, fill)
    out_dir = out_dir / clip["clip_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: list[Image.Image] = []
    for t in range(clip["n_frames"]):
        img = draw_pose_frame(
            _apply(clip["body"][t], s, ox, oy),
            _apply(clip["hand_l"][t], s, ox, oy),
            _apply(clip["hand_r"][t], s, ox, oy),
            _apply(clip["face"][t], s, ox, oy) if clip["face"].shape[1] else clip["face"][t],
            canvas, draw_face,
        )
        # compress_level=1: still lossless, and sidesteps a segfault in the
        # default-level zlib path on some Pillow/Python builds.
        img.save(out_dir / f"pose_{t:06d}.png", compress_level=1)
        frames.append(img)

    # Manifest is written *before* the optional preview: the PNG frames are
    # the real deliverable and the manifest must survive even if the GIF
    # encoder crashes the interpreter (a SIGSEGV can't be caught here).
    manifest = {
        "clip_name": clip["clip_name"],
        "source": str(json_dir),
        "n_frames": clip["n_frames"],
        "canvas": canvas,
        "fps": fps,
        "body_format": clip["body_format"],
        "has_face": bool(clip["face"].shape[1]) and draw_face,
        "hand_points": int(clip["hand_l"].shape[1]),
        "interpolated": interpolate,
        "fit": {"scale": float(s), "offset_x": float(ox), "offset_y": float(oy)},
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    # preview.gif is opt-in: on some Pillow builds the animated-GIF encoder
    # segfaults, which would kill the process. Only attempt it on request.
    if preview and frames:
        frames[0].save(
            out_dir / "preview.gif", save_all=True, append_images=frames[1:],
            duration=int(1000 / max(fps, 1)), loop=0,
        )
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json_dir", help="directory of *_keypoints.json files for one clip")
    ap.add_argument("out_dir", nargs="?",
                    default=str(ROOT / "outputs" / "pose_control"),
                    help="output root (default: outputs/pose_control)")
    ap.add_argument("--canvas", type=int, default=768, help="square output size")
    ap.add_argument("--fill", type=float, default=0.72,
                    help="fraction of the canvas the figure should occupy")
    ap.add_argument("--fps", type=int, default=25, help="fps for preview.gif")
    ap.add_argument("--no-face", action="store_true", help="skip face landmarks")
    ap.add_argument("--no-interp", action="store_true",
                    help="disable gap interpolation")
    ap.add_argument("--preview", action="store_true",
                    help="also write preview.gif (needs a working GIF encoder)")
    args = ap.parse_args()

    m = bridge_clip(
        Path(args.json_dir), Path(args.out_dir),
        canvas=args.canvas, fill=args.fill, fps=args.fps,
        draw_face=not args.no_face, interpolate=not args.no_interp,
        preview=args.preview,
    )
    print(json.dumps(m, ensure_ascii=False, indent=2))
    print(f"\nwrote {m['n_frames']} pose frames -> "
          f"{Path(args.out_dir) / m['clip_name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
