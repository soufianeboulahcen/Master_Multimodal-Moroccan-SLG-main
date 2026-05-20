"""
Main generation script.

Produces for each motion:
  outputs/videos/skeleton/<motion>_skeleton.mp4
  outputs/videos/overlay/<motion>_overlay.mp4
  outputs/videos/slowmo/<motion>_slowmo.mp4
  outputs/openpose_json/<motion>_keypoints/XXXXXX_keypoints.json
  outputs/frames/<motion>/XXXXXX.jpg

Usage:
  python generate.py [--motions walking running ...] [--no-frames] [--no-json]
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

# Allow imports from this directory
sys.path.insert(0, os.path.dirname(__file__))

from motions import MOTIONS
from renderer import render_skeleton_frame, render_overlay_frame

# ── Output paths ──────────────────────────────────────────────────────────

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "outputs")
)

DIRS = {
    "skeleton":  os.path.join(ROOT, "videos", "skeleton"),
    "overlay":   os.path.join(ROOT, "videos", "overlay"),
    "slowmo":    os.path.join(ROOT, "videos", "slowmo"),
    "json":      os.path.join(ROOT, "openpose_json"),
    "frames":    os.path.join(ROOT, "frames"),
}

# ── Video settings ────────────────────────────────────────────────────────

WIDTH, HEIGHT = 1280, 720
FPS = 30
SLOWMO_FACTOR = 3          # slow-motion playback speed divisor
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")


# ── Keypoint → OpenPose JSON ──────────────────────────────────────────────

def kp_to_openpose_json(kp_norm: np.ndarray, frame_idx: int,
                        width: int, height: int) -> dict:
    """
    Convert a (52, 2) normalised array to an OpenPose-compatible JSON dict.
    Confidence is set to 1.0 for all generated keypoints.
    """
    def flat(indices):
        out = []
        for i in indices:
            x = float(kp_norm[i, 0] * width)
            y = float(kp_norm[i, 1] * height)
            out.extend([x, y, 1.0])
        return out

    body_indices  = list(range(25))
    face_indices  = list(range(25, 30))
    lhand_indices = list(range(30, 41))
    rhand_indices = list(range(41, 52))

    return {
        "version": 1.3,
        "people": [{
            "person_id": [-1],
            "pose_keypoints_2d":       flat(body_indices),
            "face_keypoints_2d":       flat(face_indices),
            "hand_left_keypoints_2d":  flat(lhand_indices),
            "hand_right_keypoints_2d": flat(rhand_indices),
        }]
    }


# ── Video writer helper ───────────────────────────────────────────────────

def _open_writer(path: str) -> cv2.VideoWriter:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = cv2.VideoWriter(path, FOURCC, FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {path}")
    return writer


# ── Per-motion generation ─────────────────────────────────────────────────

def generate_motion(name: str, export_frames: bool, export_json: bool):
    print(f"\n[{name}] Generating frames …", flush=True)
    t0 = time.time()

    gen_fn = MOTIONS[name]
    frames_kp = gen_fn(n_frames=150, fps=FPS)   # list of (52,2) arrays
    n = len(frames_kp)

    # ── Open video writers ────────────────────────────────────────────────
    skel_path = os.path.join(DIRS["skeleton"], f"{name}_skeleton.mp4")
    over_path = os.path.join(DIRS["overlay"],  f"{name}_overlay.mp4")
    slow_path = os.path.join(DIRS["slowmo"],   f"{name}_slowmo.mp4")

    w_skel = _open_writer(skel_path)
    w_over = _open_writer(over_path)
    # Slow-mo: same frames, lower FPS → longer playback
    os.makedirs(os.path.dirname(slow_path), exist_ok=True)
    w_slow = cv2.VideoWriter(slow_path, FOURCC,
                             max(1, FPS // SLOWMO_FACTOR), (WIDTH, HEIGHT))

    # ── JSON / frame dirs ─────────────────────────────────────────────────
    json_dir   = os.path.join(DIRS["json"],   f"{name}_keypoints")
    frames_dir = os.path.join(DIRS["frames"], name)
    if export_json:
        os.makedirs(json_dir, exist_ok=True)
    if export_frames:
        os.makedirs(frames_dir, exist_ok=True)

    # ── Frame loop ────────────────────────────────────────────────────────
    for i, kp in enumerate(frames_kp):
        skel_frame = render_skeleton_frame(kp, WIDTH, HEIGHT, i, n)
        over_frame = render_overlay_frame(kp, WIDTH, HEIGHT, i, n)

        w_skel.write(skel_frame)
        w_over.write(over_frame)
        w_slow.write(skel_frame)   # slow-mo uses skeleton version

        if export_json:
            data = kp_to_openpose_json(kp, i, WIDTH, HEIGHT)
            jpath = os.path.join(json_dir, f"{i:06d}_keypoints.json")
            with open(jpath, "w") as f:
                json.dump(data, f, indent=2)

        if export_frames:
            fpath = os.path.join(frames_dir, f"{i:06d}.jpg")
            cv2.imwrite(fpath, over_frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

        if (i + 1) % 30 == 0 or i == n - 1:
            print(f"  frame {i+1}/{n}", flush=True)

    w_skel.release()
    w_over.release()
    w_slow.release()

    elapsed = time.time() - t0
    print(f"[{name}] Done in {elapsed:.1f}s  →  {skel_path}", flush=True)
    print(f"                              →  {over_path}", flush=True)
    print(f"                              →  {slow_path}", flush=True)


# ── Entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate OpenPose-style demo videos")
    parser.add_argument("--motions", nargs="+", default=list(MOTIONS.keys()),
                        choices=list(MOTIONS.keys()),
                        help="Which motions to generate (default: all)")
    parser.add_argument("--no-frames", action="store_true",
                        help="Skip per-frame JPEG export")
    parser.add_argument("--no-json", action="store_true",
                        help="Skip per-frame JSON keypoint export")
    args = parser.parse_args()

    # Create top-level output dirs
    for d in DIRS.values():
        os.makedirs(d, exist_ok=True)

    print(f"Generating {len(args.motions)} motion(s): {', '.join(args.motions)}")
    print(f"Output root: {ROOT}")
    print(f"Resolution : {WIDTH}×{HEIGHT} @ {FPS} fps")
    print(f"Frame export : {'no' if args.no_frames else 'yes'}")
    print(f"JSON export  : {'no' if args.no_json else 'yes'}")

    total_t0 = time.time()
    for name in args.motions:
        generate_motion(name,
                        export_frames=not args.no_frames,
                        export_json=not args.no_json)

    print(f"\n✅ All done in {time.time() - total_t0:.1f}s")
    print(f"   Outputs in: {ROOT}")


if __name__ == "__main__":
    main()
