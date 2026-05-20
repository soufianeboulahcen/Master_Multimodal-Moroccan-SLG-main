# OpenPose-style Video Generator

Generates synthetic human skeleton tracking videos using procedural animation.
No real video input required — all motion is mathematically defined.

## Files

| File | Purpose |
|------|---------|
| `rig.py` | Skeleton definition: 52 keypoints (BODY_25 + hands + face), limb pairs, colours |
| `motions.py` | Motion generators: walking, running, dancing, jumping, hand_face, waving |
| `renderer.py` | OpenPose-style drawing, glow, cinematic bars, camera drift, background |
| `generate.py` | Orchestrator: runs all motions, writes MP4s, JSON, and frame JPEGs |

## Usage

```bash
# Generate all motions (default)
python scripts/generate_openpose/generate.py

# Generate specific motions only
python scripts/generate_openpose/generate.py --motions walking dancing

# Skip frame/JSON export (faster, videos only)
python scripts/generate_openpose/generate.py --no-frames --no-json
```

## Output per motion

- `outputs/videos/skeleton/<motion>_skeleton.mp4` — skeleton on black, 30 fps
- `outputs/videos/overlay/<motion>_overlay.mp4`   — skeleton on studio background, 30 fps
- `outputs/videos/slowmo/<motion>_slowmo.mp4`     — skeleton at 10 fps (3× slow)
- `outputs/openpose_json/<motion>_keypoints/`     — one JSON per frame (OpenPose 1.3)
- `outputs/frames/<motion>/`                      — one JPEG per frame (overlay version)

## Adding a new motion

1. Add a `generate_<name>(n_frames, fps) -> list[np.ndarray]` function to `motions.py`
2. Register it in the `MOTIONS` dict at the bottom of `motions.py`
3. Run `generate.py --motions <name>`

## Dependencies

- opencv-python
- numpy
- scipy
