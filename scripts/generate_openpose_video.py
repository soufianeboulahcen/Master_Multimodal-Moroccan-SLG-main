"""OpenPose-style pose tracking and video generation pipeline.

Produces three outputs for every input video:

  <out_dir>/<stem>_overlay.mp4    — original frames + skeleton drawn on top
  <out_dir>/<stem>_skeleton.mp4   — pure black-background skeleton animation
  <out_dir>/<stem>_keypoints/     — per-frame OpenPose-compatible JSON files
      000000_keypoints.json
      000001_keypoints.json
      ...

Keypoint sets detected (via MediaPipe Holistic Tasks API):
  * 33 body pose landmarks  (mapped to 18-joint COCO subset for display)
  * 21 left-hand landmarks
  * 21 right-hand landmarks
  * face mesh landmarks (drawn as subtle dots)

Colour scheme matches the canonical CMU OpenPose rainbow palette.

Usage:
    python scripts/generate_openpose_video.py <input_video_or_dir> [options]

    # single video (use absolute path for Arabic filenames):
    python scripts/generate_openpose_video.py /abs/path/to/video.mp4 \\
        --out-dir openpose_output

    # batch over a directory (up to --limit clips):
    python scripts/generate_openpose_video.py \\
        data/vedios-dataset/mosl_videos_dataset_Pronouns/ \\
        --out-dir openpose_output --limit 5
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core import base_options as mp_base

ROOT          = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "holistic_landmarker.task"

# ── OpenPose-style rainbow palette (BGR) ─────────────────────────────────────
LIMB_COLORS_BGR = [
    (  0,215,255),(  0,255,170),(  0,255, 85),(  0,255,  0),
    (  0,170,255),(  0, 85,255),(  0,  0,255),( 85,  0,255),
    (170,  0,255),(255,  0,255),(255,  0,170),(255,  0, 85),
    (255,  0,  0),(255, 85,  0),(255,170,  0),(255,255,  0),
    (170,255,  0),( 85,255,  0),
]

# COCO-18 body edges: (joint_a, joint_b, colour_index)
# Joints: 0=Nose 1=Neck 2=RShoulder 3=RElbow 4=RWrist
#         5=LShoulder 6=LElbow 7=LWrist 8=RHip 9=RKnee 10=RAnkle
#         11=LHip 12=LKnee 13=LAnkle 14=REye 15=LEye 16=REar 17=LEar
BODY_EDGES = [
    (0,1,0),(1,2,1),(2,3,2),(3,4,3),(1,5,4),(5,6,5),(6,7,6),
    (1,8,7),(8,9,8),(9,10,9),(1,11,10),(11,12,11),(12,13,12),
    (0,14,13),(14,16,14),(0,15,15),(15,17,16),
]

_HAND_FINGERS = [[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]]
HAND_EDGES    = [(a,b) for f in _HAND_FINGERS for a,b in zip(f,f[1:])]

LHAND_BONE=(255,200,0); LHAND_JOINT=(255,255,0)
RHAND_BONE=(0,0,255);   RHAND_JOINT=(50,50,255)
FACE_COLOR=(160,160,160)
JOINT_R=4; BONE_T=2

# ── MediaPipe PoseLandmark → COCO-18 index map ───────────────────────────────
_PL = mp_vision.PoseLandmark
MP_TO_COCO18 = {
    0:_PL.NOSE,
    2:_PL.RIGHT_SHOULDER, 3:_PL.RIGHT_ELBOW, 4:_PL.RIGHT_WRIST,
    5:_PL.LEFT_SHOULDER,  6:_PL.LEFT_ELBOW,  7:_PL.LEFT_WRIST,
    8:_PL.RIGHT_HIP,  9:_PL.RIGHT_KNEE,  10:_PL.RIGHT_ANKLE,
    11:_PL.LEFT_HIP, 12:_PL.LEFT_KNEE,  13:_PL.LEFT_ANKLE,
    14:_PL.RIGHT_EYE, 15:_PL.LEFT_EYE,
    16:_PL.RIGHT_EAR, 17:_PL.LEFT_EAR,
}

# ── Landmark extraction ───────────────────────────────────────────────────────
def extract_body(pose_lms, w, h):
    """MediaPipe pose landmark list → (18, 3) float32 (x_px, y_px, visibility)."""
    kpts = np.zeros((18,3), dtype=np.float32)
    if pose_lms is None:
        return kpts
    for ci, mi in MP_TO_COCO18.items():
        lm = pose_lms[mi]
        kpts[ci] = [lm.x*w, lm.y*h, float(getattr(lm, 'visibility', 1.0))]
    # Neck = midpoint of shoulders
    rs, ls = kpts[2], kpts[5]
    if rs[2] > 0.1 and ls[2] > 0.1:
        kpts[1] = [(rs[0]+ls[0])/2, (rs[1]+ls[1])/2, (rs[2]+ls[2])/2]
    return kpts

def extract_hand(hand_lms, w, h):
    """MediaPipe hand landmark list → (21, 3) float32 (x_px, y_px, 1.0)."""
    kpts = np.zeros((21,3), dtype=np.float32)
    if hand_lms is None:
        return kpts
    for i, lm in enumerate(hand_lms):
        kpts[i] = [lm.x*w, lm.y*h, 1.0]
    return kpts

def extract_face(face_lms, w, h):
    """MediaPipe face landmark list → (N, 3) float32 (x_px, y_px, 1.0)."""
    if face_lms is None:
        return np.zeros((0,3), dtype=np.float32)
    return np.array([[lm.x*w, lm.y*h, 1.0] for lm in face_lms], dtype=np.float32)

# ── Drawing ───────────────────────────────────────────────────────────────────
def draw_body(canvas, kpts, thr=0.25):
    for a, b, ci in BODY_EDGES:
        if kpts[a,2] > thr and kpts[b,2] > thr:
            cv2.line(canvas,
                     (int(kpts[a,0]), int(kpts[a,1])),
                     (int(kpts[b,0]), int(kpts[b,1])),
                     LIMB_COLORS_BGR[ci % len(LIMB_COLORS_BGR)],
                     BONE_T, cv2.LINE_AA)
    for i in range(18):
        if kpts[i,2] > thr:
            pt  = (int(kpts[i,0]), int(kpts[i,1]))
            col = LIMB_COLORS_BGR[i % len(LIMB_COLORS_BGR)]
            cv2.circle(canvas, pt, JOINT_R,     col,       -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, JOINT_R + 1, (0,0,0),    1, cv2.LINE_AA)

def draw_hand(canvas, kpts, bc, jc, thr=0.1):
    for a, b in HAND_EDGES:
        if kpts[a,2] > thr and kpts[b,2] > thr:
            cv2.line(canvas,
                     (int(kpts[a,0]), int(kpts[a,1])),
                     (int(kpts[b,0]), int(kpts[b,1])),
                     bc, max(1, BONE_T-1), cv2.LINE_AA)
    for i in range(21):
        if kpts[i,2] > thr:
            cv2.circle(canvas,
                       (int(kpts[i,0]), int(kpts[i,1])),
                       max(1, JOINT_R-1), jc, -1, cv2.LINE_AA)

def draw_face(canvas, kpts):
    for i in range(len(kpts)):
        if kpts[i,2] > 0.1:
            cv2.circle(canvas,
                       (int(kpts[i,0]), int(kpts[i,1])),
                       1, FACE_COLOR, -1, cv2.LINE_AA)

def compose(base, body, lhand, rhand, face, overlay):
    """Render one output frame: overlay on original or on black background."""
    canvas = base.copy() if overlay else np.zeros_like(base)
    draw_face(canvas, face)
    draw_body(canvas, body)
    draw_hand(canvas, lhand, LHAND_BONE, LHAND_JOINT)
    draw_hand(canvas, rhand, RHAND_BONE, RHAND_JOINT)
    return canvas

# ── JSON serialisation (OpenPose-compatible schema) ───────────────────────────
def flat(kpts):
    return [round(float(v), 4) for v in kpts.flatten()]

def frame_json(idx, body, lhand, rhand, face, w, h):
    return {
        "version": 1.3,
        "frame_index": idx,
        "image_size": {"width": w, "height": h},
        "people": [{
            "person_id":               [-1],
            "pose_keypoints_2d":       flat(body),   # 18*3 = 54 values
            "face_keypoints_2d":       flat(face),   # N*3 values
            "hand_left_keypoints_2d":  flat(lhand),  # 21*3 = 63 values
            "hand_right_keypoints_2d": flat(rhand),  # 21*3 = 63 values
            "pose_keypoints_3d":       [],
            "face_keypoints_3d":       [],
            "hand_left_keypoints_3d":  [],
            "hand_right_keypoints_3d": [],
        }],
    }

# ── Per-video pipeline ────────────────────────────────────────────────────────
def process_video(video_path: Path, out_dir: Path, model_path: Path,
                  write_json=True, write_overlay=True, write_skeleton=True) -> dict:
    """Run the full pipeline on one video. Returns a summary dict."""
    stem    = video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    ov_path  = out_dir / f"{stem}_overlay.mp4"
    sk_path  = out_dir / f"{stem}_skeleton.mp4"
    json_dir = out_dir / f"{stem}_keypoints"
    if write_json:
        json_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    ov_wr  = cv2.VideoWriter(str(ov_path), fourcc, fps, (W,H)) if write_overlay  else None
    sk_wr  = cv2.VideoWriter(str(sk_path), fourcc, fps, (W,H)) if write_skeleton else None

    # Build HolisticLandmarker with VIDEO running mode for temporal smoothing.
    # Parameter names verified against mediapipe 0.10.35 signature.
    opts = mp_vision.HolisticLandmarkerOptions(
        base_options=mp_base.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_suppression_threshold=0.5,
        min_pose_landmarks_confidence=0.5,
        min_face_detection_confidence=0.5,
        min_face_suppression_threshold=0.5,
        min_face_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
        output_face_blendshapes=False,
        output_segmentation_mask=False,
    )
    landmarker = mp_vision.HolisticLandmarker.create_from_options(opts)

    idx = 0; n_det = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break

        # MediaPipe Tasks API requires RGB mp.Image
        rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms  = int(idx * 1000 / fps)
        res    = landmarker.detect_for_video(mp_img, ts_ms)

        body  = extract_body(res.pose_landmarks,       W, H)
        lhand = extract_hand(res.left_hand_landmarks,  W, H)
        rhand = extract_hand(res.right_hand_landmarks, W, H)
        face  = extract_face(res.face_landmarks,       W, H)

        if body.max() > 0:
            n_det += 1

        if write_json:
            rec = frame_json(idx, body, lhand, rhand, face, W, H)
            with open(json_dir / f"{idx:06d}_keypoints.json", "w") as f:
                json.dump(rec, f, separators=(",", ":"))

        if ov_wr:
            ov_wr.write(compose(bgr, body, lhand, rhand, face, overlay=True))
        if sk_wr:
            sk_wr.write(compose(bgr, body, lhand, rhand, face, overlay=False))

        idx += 1
        if idx % 25 == 0:
            pct = 100 * idx / max(n_tot, 1)
            print(f"  [{pct:5.1f}%] frame {idx}/{n_tot}", flush=True)

    cap.release()
    landmarker.close()
    if ov_wr:  ov_wr.release()
    if sk_wr:  sk_wr.release()

    return {
        "video":         str(video_path),
        "frames":        idx,
        "fps":           fps,
        "resolution":    f"{W}x{H}",
        "body_det_rate": f"{100*n_det/max(idx,1):.1f}%",
        "overlay_mp4":   str(ov_path)  if write_overlay  else None,
        "skeleton_mp4":  str(sk_path)  if write_skeleton else None,
        "json_dir":      str(json_dir) if write_json     else None,
        "json_files":    idx           if write_json     else 0,
    }

# ── CLI ───────────────────────────────────────────────────────────────────────
def collect_videos(p: Path, limit: int) -> list:
    if p.is_file():
        return [p]
    # Use os.listdir to preserve exact filesystem encoding of Arabic filenames
    vids = sorted(
        p / f for f in os.listdir(str(p))
        if f.lower().endswith(".mp4") and (p / f).is_file()
    )
    return vids[:limit] if limit > 0 else vids

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input",
                    help="Input .mp4 file or directory of .mp4 files")
    ap.add_argument("--out-dir",     default="openpose_output")
    ap.add_argument("--model",       default=str(DEFAULT_MODEL),
                    help="Path to holistic_landmarker.task")
    ap.add_argument("--limit",       type=int, default=0,
                    help="Max videos when input is a directory (0=all)")
    ap.add_argument("--no-overlay",  action="store_true")
    ap.add_argument("--no-skeleton", action="store_true")
    ap.add_argument("--no-json",     action="store_true")
    args = ap.parse_args()

    inp   = Path(os.path.abspath(args.input))
    model = Path(args.model)

    if not inp.exists():
        print(f"[error] not found: {inp}", file=sys.stderr)
        return 1
    if not model.exists():
        print(f"[error] model not found: {model}", file=sys.stderr)
        print("  Download with:\n  wget https://storage.googleapis.com/mediapipe-models/"
              "holistic_landmarker/holistic_landmarker/float16/latest/"
              "holistic_landmarker.task -O models/holistic_landmarker.task",
              file=sys.stderr)
        return 1

    videos = collect_videos(inp, args.limit)
    if not videos:
        print(f"[error] no .mp4 files found under {inp}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(videos)} video(s) → {out_dir}/")
    print(f"  overlay={not args.no_overlay}  "
          f"skeleton={not args.no_skeleton}  "
          f"json={not args.no_json}\n")

    summaries = []; t0 = time.perf_counter()
    for i, vp in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {vp.name}")
        try:
            s = process_video(
                vp, out_dir, model,
                write_json=not args.no_json,
                write_overlay=not args.no_overlay,
                write_skeleton=not args.no_skeleton,
            )
            summaries.append(s)
            print(f"  ✓ {s['frames']} frames  "
                  f"body_det={s['body_det_rate']}  fps={s['fps']:.1f}")
            for k, lbl in [("overlay_mp4","overlay"),
                            ("skeleton_mp4","skeleton"),
                            ("json_dir","JSON")]:
                if s.get(k):
                    extra = f" ({s['json_files']} files)" if k == "json_dir" else ""
                    print(f"  → [{lbl}] {s[k]}{extra}")
        except Exception as e:
            import traceback
            print(f"  ✗ FAILED: {e}", file=sys.stderr)
            traceback.print_exc()
        print()

    elapsed = time.perf_counter() - t0
    print(f"Done — {len(summaries)}/{len(videos)} succeeded in {elapsed:.1f}s")

    sp = out_dir / "run_summary.json"
    with open(sp, "w") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"Summary → {sp}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
