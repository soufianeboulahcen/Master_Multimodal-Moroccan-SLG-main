# Multimodal Moroccan Sign Language Generation

A paper-faithful PyTorch reimplementation of **SignLLM** trained on the **MoSL** isolated-word dataset, extended with a full OpenPose-style skeleton tracking and video generation pipeline.

---

## Overview

| Phase | Description |
|-------|-------------|
| Dataset processing | Label extraction, train/val/test split, vocabulary |
| Pose extraction | OpenPose keypoint extraction to NPZ per clip |
| 2D to 3D lifting | Prompt2Sign `.skels` format conversion |
| Model | SignLLM encoder-decoder transformer (35M params) |
| Training | MSE / RL / RL+PLC ablation runs |
| Evaluation | Teacher-forced MSE + autoregressive DTW vs baselines |
| Visualization | OpenPose-style skeleton videos, GIFs, JSON keypoints |

---

## Results

**Test AR DTW (lower is better):**

| Method | DTW |
|--------|-----|
| Nearest-Neighbor (baseline) | 0.7817 |
| Mean-Pose (baseline) | 0.8682 |
| Random-Clip (baseline) | 0.9620 |
| SignLLM MSE | 1.0447 |
| SignLLM RL+PLC | 1.2117 |
| SignLLM RL | 1.3182 |

![Training curves](images/training_curves.png)
![Baseline comparison](images/baseline_comparison.png)

---

## Generated Outputs

### Skeleton Tracking Videos

Six motion types with full body, hand, and face tracking (1280x720, 30 fps):

| Motion | Skeleton | Overlay | Slow-motion |
|--------|----------|---------|-------------|
| Walking | `outputs/videos/skeleton/walking_skeleton.mp4` | `outputs/videos/overlay/walking_overlay.mp4` | `outputs/videos/slowmo/walking_slowmo.mp4` |
| Running | `outputs/videos/skeleton/running_skeleton.mp4` | `outputs/videos/overlay/running_overlay.mp4` | `outputs/videos/slowmo/running_slowmo.mp4` |
| Dancing | `outputs/videos/skeleton/dancing_skeleton.mp4` | `outputs/videos/overlay/dancing_overlay.mp4` | `outputs/videos/slowmo/dancing_slowmo.mp4` |
| Jumping | `outputs/videos/skeleton/jumping_skeleton.mp4` | `outputs/videos/overlay/jumping_overlay.mp4` | `outputs/videos/slowmo/jumping_slowmo.mp4` |
| Hand & Face | `outputs/videos/skeleton/hand_face_skeleton.mp4` | `outputs/videos/overlay/hand_face_overlay.mp4` | `outputs/videos/slowmo/hand_face_slowmo.mp4` |
| Waving | `outputs/videos/skeleton/waving_skeleton.mp4` | `outputs/videos/overlay/waving_overlay.mp4` | `outputs/videos/slowmo/waving_slowmo.mp4` |

Visual effects: glow/bloom, cinematic letterbox bars, camera drift, animated studio background.

### Visualizations

![Predicted pose frames](images/predicted_frames.png)
![Sample skeleton](images/sample_skeleton.png)
![Predicted pose animation](images/predicted_pose.gif)

---

## Repository Structure

```
.
├── assets/                         # Model weights (Git LFS)
│   └── holistic_landmarker.task
├── data/                           # Dataset metadata
│   ├── labels.csv
│   ├── splits.csv
│   └── video_meta.csv
├── datasets/                       # Dataset documentation and download instructions
├── docs/                           # Technical documentation
├── images/                         # Figures and visualizations (Git LFS)
├── models/                         # Model files (Git LFS)
├── mosl/                           # Python package
│   ├── data/                       # Dataset loading and splitting
│   ├── model/                      # SignLLM transformer
│   ├── pose/                       # Keypoint extraction
│   ├── text/                       # Arabic word tokenizer
│   └── train/                      # Training loop, losses, evaluation
├── notebooks/                      # Jupyter notebooks
│   └── final_project.ipynb         # Full pipeline (67 cells)
├── outputs/                        # All generated results
│   ├── frames/                     # Per-frame JPEG exports (Git LFS)
│   ├── openpose_json/              # Per-frame OpenPose 1.3 JSON keypoints
│   └── videos/                     # All generated MP4 videos (Git LFS)
│       ├── skeleton/               # Skeleton on black background
│       ├── overlay/                # Skeleton on studio background
│       ├── slowmo/                 # 3x slow-motion
│       ├── heatmap/
│       ├── mosaic/
│       ├── neon/
│       ├── studio/
│       └── demo/                   # Per-sign demo outputs
├── scripts/                        # CLI entry points
│   └── generate_openpose/          # OpenPose-style video generator
├── docker/                         # Docker environment for GPU pipeline
├── final_project.ipynb             # Main notebook
└── requirements.txt
```

---

## Quick Start

### Generate OpenPose-style skeleton videos (no GPU required)

```bash
pip install -r requirements.txt
python scripts/generate_openpose/generate.py
```

Outputs are written to `outputs/videos/`, `outputs/openpose_json/`, and `outputs/frames/`.

Generate specific motions:

```bash
python scripts/generate_openpose/generate.py --motions walking dancing
python scripts/generate_openpose/generate.py --no-frames --no-json   # videos only
```

### Run the notebook

```bash
jupyter notebook notebooks/final_project.ipynb
```

### Full training pipeline (requires Docker + GPU + MoSL dataset)

```bash
# Place MoSL dataset at data/vedios-dataset/
docker/run.sh python scripts/extract_dataset.py
docker/run.sh python scripts/train_signllm.py --run baseline_mse
docker/run.sh python scripts/evaluate_runs.py
```

---

## OpenPose JSON Format

```json
{
  "version": 1.3,
  "people": [{
    "person_id": [-1],
    "pose_keypoints_2d":       [...],
    "face_keypoints_2d":       [...],
    "hand_left_keypoints_2d":  [...],
    "hand_right_keypoints_2d": [...]
  }]
}
```

- **Body:** 25 keypoints (BODY_25), pixel coordinates at 1280x720
- **Face:** 5 landmarks (eyes, mouth)
- **Hands:** 11 keypoints per hand
- **Confidence:** 1.0 for all generated keypoints

---

## Model Architecture

```
Arabic text -> WordTokenizer -> Encoder (2 layers, d=768, h=12)
                                      |
                              Decoder (2 layers, d=768, h=12)
                                      |
                          Linear -> (T, 150) pose sequence
                                   [50 joints x (x, y, z)]
```

Parameters: ~35M | Optimizer: Adam + Noam LR (warmup=4000)

![Positional encoding](images/positional_encoding.png)

---

## Dataset

**MoSL — Moroccan Sign Language Video Dataset**
Ben Zaid et al. (2026). Mendeley Data. DOI: [10.17632/23phgyt3mt.1](https://doi.org/10.17632/23phgyt3mt.1)

| Stat | Value |
|------|-------|
| Total clips | 2,216 |
| Unique signs | 1,631 |
| Train / Val / Test | 1,674 / 430 / 112 |
| Singletons | 74% of signs have 1 clip |

Raw video files are not included. See `datasets/README.md` for download instructions.

---

## Git LFS

Large binary files are tracked via Git LFS. After cloning:

```bash
git lfs pull
```

Tracked: `.mp4`, `.gif`, `.png`, `.jpg`, `.task`, `.npz`, `.pt`

---

## Application Scenarios

The pipeline combines OpenPose-based skeleton tracking, a Transformer encoder-decoder
for sign language production, and a procedural avatar rendering engine. The techniques
developed here address a broad set of real-world problems across accessibility,
healthcare, HCI, and AI-driven media production.

| # | Scenario | Core Capability Used |
|---|----------|----------------------|
| 1 | Assistive communication for the Deaf | Text → MoSL skeleton animation |
| 2 | Sign language learning and Deaf education | Generation + DTW-based learner scoring |
| 3 | Clinical rehabilitation and physical therapy | Marker-free joint tracking + DTW deviation |
| 4 | Gesture-based human–computer interaction | 21-joint hand keypoint classification |
| 5 | Avatar telepresence and virtual communication | Compact keypoint stream (150 floats/frame) |
| 6 | Intelligent video surveillance | Skeleton-based anomaly detection |
| 7 | Sports biomechanics and performance analysis | DTW scoring against reference trajectories |
| 8 | AI animation and virtual production | Text → pose → avatar rendering |

### 1. Assistive Communication for the Deaf and Hard-of-Hearing

The primary application is **automatic sign language production**: given a written
Arabic word or phrase, the system generates a temporally coherent skeleton animation
that a Deaf user can read as a sign. Deployed as a browser or mobile widget, this
enables public-sector services — government portals, hospital kiosks, educational
platforms — to provide MoSL output at scale without requiring human interpreters.
The MoSL dataset covers 1,631 isolated signs across five semantic categories,
providing a practical lexical foundation for a signing assistant.

### 2. Sign Language Learning and Deaf Education

An avatar that renders any sign on demand is a natural component of an **interactive
sign language tutor**. A learner types a word; the system generates the reference MoSL
sign; the learner's webcam recording is scored by computing the DTW distance between
their pose sequence and the model's output. This closes the loop between generation
(SignLLM) and recognition (OpenPose extraction) in a single self-contained pedagogical
tool, requiring no additional infrastructure beyond a webcam.

### 3. Clinical Rehabilitation and Physical Therapy

The 52-keypoint skeleton tracker (full body, both hands, face at 30 fps) enables
**marker-free clinical motion analysis**. Physiotherapists can quantify range of
motion, detect compensatory movement patterns, and track recovery progress over time
without attaching physical markers to the patient. The DTW-based comparison metric
used for sign evaluation transfers directly: a patient's recorded keypoint sequence is
scored against a normative reference trajectory, yielding an objective deviation index
per joint per session.

### 4. Gesture-Based Human–Computer Interaction

The 21-joint hand keypoint chains provide sufficient resolution to distinguish a
vocabulary of static and dynamic hand gestures for **touchless interface control** in
environments where physical contact is undesirable — operating theatres, cleanrooms,
automotive dashboards, and public kiosks. A lightweight classifier trained on
normalised keypoint sequences maps gestures to application commands, with the OpenPose
extraction layer serving as a reusable front-end for any gesture-based HCI system.

### 5. Avatar-Based Telepresence and Virtual Communication

The rendering pipeline produces skeleton animations that can be mapped onto a rigged
3D avatar for **real-time telepresence**. Transmitting a compact keypoint stream
(150 floats per frame) instead of raw video reduces bandwidth by two to three orders
of magnitude while preserving the gestural and postural information that carries
communicative meaning — a property particularly valuable for Deaf users signing over
a network connection.

### 6. Intelligent Video Surveillance and Anomaly Detection

Skeleton-based action recognition is invariant to clothing, lighting, and camera
viewpoint in a way that pixel-level methods are not, making it well-suited to
**intelligent surveillance**. The temporal pose sequences can be fed to a sequence
classifier trained to detect anomalous activities (falls, intrusions, crowd crush).
The procedural motion generators in Section 11 provide a controlled source of
synthetic training data for rare events that cannot be collected from real footage.

### 7. Sports Biomechanics and Athletic Performance Analysis

High-speed pose estimation enables **objective, marker-free biomechanical analysis**
of athletic movement. Coaches can measure joint angles, stride length, arm swing
symmetry, and centre-of-mass displacement across sessions, identifying technique
deviations that correlate with injury risk or performance loss. The DTW metric
provides a single interpretable performance index by comparing an athlete's attempt
against a stored reference motion (e.g., an elite sprinter's stride cycle).

### 8. AI Animation and Virtual Production

The pose-to-avatar pipeline is a lightweight alternative to optical motion capture
for **AI-driven animation**. A director describes a motion in natural language; a
language model translates the description into pose keyframes; the rendering engine
produces a preview in seconds rather than the hours required by traditional keyframe
animation. The SignLLM architecture — which maps text tokens to pose sequences —
transfers directly to this broader motion synthesis task by replacing the sign
vocabulary with a motion vocabulary and retraining on a motion-captioned dataset
such as HumanML3D or BABEL.

---

## References

- Fang et al. (2024). *SignLLM: Sign Languages Production Large Language Models*. arXiv:2405.10718
- Ben Zaid et al. (2026). *Moroccan Sign Language Video Dataset*. Mendeley Data
- Vaswani et al. (2017). *Attention is All You Need*. NeurIPS 2017
- Saunders et al. (2020). *Progressive Transformers for End-to-End Sign Language Production*. ECCV 2020
- Hzzone (2019). *pytorch-openpose*. GitHub

---

## License

See [LICENSE](LICENSE).
