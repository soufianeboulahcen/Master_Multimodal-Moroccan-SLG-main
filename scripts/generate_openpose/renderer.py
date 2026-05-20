"""
OpenPose-style renderer with visual effects.

Provides:
  render_skeleton_frame()  — skeleton on black background
  render_overlay_frame()   — skeleton drawn over a synthetic background
  apply_glow()             — bloom/glow post-process
  apply_cinematic_bars()   — letterbox bars
  camera_offset()          — subtle camera drift
"""

import cv2
import numpy as np
from rig import LIMBS, KP_COLOUR, KP_RADIUS


# ── Glow / bloom ──────────────────────────────────────────────────────────

def apply_glow(frame: np.ndarray, strength: float = 0.55, blur_k: int = 21) -> np.ndarray:
    """Additive bloom: blur a brightened copy and blend back."""
    bright = np.clip(frame.astype(np.float32) * 1.4, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(bright, (blur_k, blur_k), 0)
    glow = cv2.addWeighted(frame, 1.0, blurred, strength, 0)
    return np.clip(glow, 0, 255).astype(np.uint8)


# ── Cinematic letterbox ───────────────────────────────────────────────────

def apply_cinematic_bars(frame: np.ndarray, ratio: float = 0.08) -> np.ndarray:
    """Black bars at top and bottom (2.35:1 feel)."""
    h = frame.shape[0]
    bar = int(h * ratio)
    out = frame.copy()
    out[:bar] = 0
    out[-bar:] = 0
    return out


# ── Camera drift ──────────────────────────────────────────────────────────

def camera_offset(t: float, amp_x: float = 6.0, amp_y: float = 4.0) -> tuple:
    """Slow sinusoidal camera drift. Returns (dx, dy) in pixels."""
    dx = int(amp_x * np.sin(2 * np.pi * t * 0.3))
    dy = int(amp_y * np.cos(2 * np.pi * t * 0.2))
    return dx, dy


def _shift_frame(frame: np.ndarray, dx: int, dy: int) -> np.ndarray:
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))


# ── Synthetic background ───────────────────────────────────────────────────

def _make_background(h: int, w: int, frame_idx: int, n_frames: int) -> np.ndarray:
    """
    Dark gradient background with a subtle vignette and animated colour shift.
    Gives the overlay version a cinematic studio feel.
    """
    t = frame_idx / max(n_frames - 1, 1)

    # Base dark gradient (top lighter, bottom darker)
    bg = np.zeros((h, w, 3), dtype=np.float32)
    for row in range(h):
        v = 18 + 12 * (1 - row / h)
        bg[row] = [v * 0.8, v * 0.9, v]   # slight blue tint

    # Slow colour pulse
    pulse = 0.5 + 0.5 * np.sin(2 * np.pi * t)
    bg[:, :, 0] += pulse * 4   # red channel breathes
    bg[:, :, 2] += (1 - pulse) * 4

    # Vignette
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    vignette = np.clip(1 - dist * 0.6, 0, 1)[..., np.newaxis]
    bg = bg * vignette

    return np.clip(bg, 0, 255).astype(np.uint8)


# ── Core drawing ──────────────────────────────────────────────────────────

def _to_px(kp_norm: np.ndarray, w: int, h: int) -> np.ndarray:
    """Convert normalised [0,1] coords to pixel coords."""
    px = kp_norm.copy()
    px[:, 0] *= w
    px[:, 1] *= h
    return px.astype(np.int32)


def _draw_skeleton(canvas: np.ndarray, kp_px: np.ndarray,
                   line_thickness: int = 3, glow: bool = True) -> np.ndarray:
    """Draw limbs and keypoints onto canvas (in-place). Returns canvas."""
    # Draw limbs
    for (a, b, colour) in LIMBS:
        pt_a = tuple(kp_px[a])
        pt_b = tuple(kp_px[b])
        # Skip if either point is off-screen
        h, w = canvas.shape[:2]
        if not (0 <= pt_a[0] < w and 0 <= pt_a[1] < h and
                0 <= pt_b[0] < w and 0 <= pt_b[1] < h):
            continue
        if glow:
            # Thick dim under-layer for glow feel
            cv2.line(canvas, pt_a, pt_b,
                     tuple(int(c * 0.35) for c in colour), line_thickness + 4)
        cv2.line(canvas, pt_a, pt_b, colour, line_thickness,
                 lineType=cv2.LINE_AA)

    # Draw keypoints
    for i, (x, y) in enumerate(kp_px):
        h, w = canvas.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            continue
        if glow:
            cv2.circle(canvas, (x, y), KP_RADIUS + 3,
                       tuple(int(c * 0.3) for c in KP_COLOUR), -1)
        cv2.circle(canvas, (x, y), KP_RADIUS, KP_COLOUR, -1,
                   lineType=cv2.LINE_AA)

    return canvas


# ── Public API ────────────────────────────────────────────────────────────

def render_skeleton_frame(kp_norm: np.ndarray,
                          width: int = 1280, height: int = 720,
                          frame_idx: int = 0, n_frames: int = 1,
                          cinematic: bool = True) -> np.ndarray:
    """Skeleton on black background."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    kp_px = _to_px(kp_norm, width, height)
    _draw_skeleton(canvas, kp_px, line_thickness=3, glow=True)
    canvas = apply_glow(canvas, strength=0.6)
    if cinematic:
        t = frame_idx / max(n_frames - 1, 1)
        dx, dy = camera_offset(t)
        canvas = _shift_frame(canvas, dx, dy)
        canvas = apply_cinematic_bars(canvas)
    return canvas


def render_overlay_frame(kp_norm: np.ndarray,
                         width: int = 1280, height: int = 720,
                         frame_idx: int = 0, n_frames: int = 1,
                         cinematic: bool = True) -> np.ndarray:
    """Skeleton overlaid on a synthetic dark studio background."""
    canvas = _make_background(height, width, frame_idx, n_frames)
    kp_px = _to_px(kp_norm, width, height)
    _draw_skeleton(canvas, kp_px, line_thickness=3, glow=True)
    canvas = apply_glow(canvas, strength=0.45)
    if cinematic:
        t = frame_idx / max(n_frames - 1, 1)
        dx, dy = camera_offset(t)
        canvas = _shift_frame(canvas, dx, dy)
        canvas = apply_cinematic_bars(canvas)
    return canvas
