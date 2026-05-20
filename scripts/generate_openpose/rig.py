"""
Skeleton rig definition.

Keypoint indices follow the BODY_25 / OpenPose convention:
  0  Nose
  1  Neck
  2  RShoulder  3  RElbow  4  RWrist
  5  LShoulder  6  LElbow  7  LWrist
  8  MidHip
  9  RHip  10 RKnee  11 RAnkle
  12 LHip  13 LKnee  14 LAnkle
  15 REye  16 LEye  17 REar  18 LEar
  19 LBigToe  20 LSmallToe  21 LHeel
  22 RBigToe  23 RSmallToe  24 RHeel

Face keypoints (25-92) are simplified to 5 landmarks:
  25 LeftEyeOuter  26 RightEyeOuter
  27 MouthLeft     28 MouthRight  29 MouthCenter

Hand keypoints (30-51) — 11 per hand (wrist + 4 finger tips + 4 knuckles + thumb):
  Left hand  : 30-40
  Right hand : 41-51
"""

import numpy as np

# ── Limb pairs (index_a, index_b, colour_BGR) ──────────────────────────────
LIMBS = [
    # Torso
    (1,  2,  (255, 100,  50)),
    (1,  5,  (255, 100,  50)),
    (2,  8,  (255, 100,  50)),
    (5,  8,  (255, 100,  50)),
    (1,  8,  (200,  80,  30)),
    # Right arm
    (2,  3,  ( 50, 200, 255)),
    (3,  4,  ( 50, 150, 255)),
    # Left arm
    (5,  6,  (255, 200,  50)),
    (6,  7,  (255, 150,  50)),
    # Right leg
    (8,  9,  ( 50, 255, 150)),
    (9, 10,  ( 50, 255, 100)),
    (10,11,  ( 50, 200,  80)),
    (11,22,  ( 30, 180,  60)),
    (22,23,  ( 20, 160,  50)),
    (11,24,  ( 20, 160,  50)),
    # Left leg
    (8, 12,  (200,  50, 255)),
    (12,13,  (180,  50, 255)),
    (13,14,  (150,  50, 200)),
    (14,19,  (120,  30, 180)),
    (19,20,  (100,  20, 160)),
    (14,21,  (100,  20, 160)),
    # Head
    (0,  1,  (255, 255, 100)),
    (0, 15,  (200, 255, 100)),
    (0, 16,  (200, 255, 100)),
    (15,17,  (180, 220,  80)),
    (16,18,  (180, 220,  80)),
    # Face
    (25,26,  (255, 255, 200)),
    (27,28,  (255, 200, 200)),
    (27,29,  (255, 200, 200)),
    (28,29,  (255, 200, 200)),
    # Left hand fingers (wrist→knuckle→tip chains)
    (7, 30,  (255, 220, 100)),
    (30,31,  (255, 220, 100)), (31,32,  (255, 220, 100)),
    (30,33,  (255, 200,  80)), (33,34,  (255, 200,  80)),
    (30,35,  (255, 180,  60)), (35,36,  (255, 180,  60)),
    (30,37,  (255, 160,  40)), (37,38,  (255, 160,  40)),
    (30,39,  (255, 140,  20)), (39,40,  (255, 140,  20)),
    # Right hand fingers
    (4, 41,  (100, 220, 255)),
    (41,42,  (100, 220, 255)), (42,43,  (100, 220, 255)),
    (41,44,  ( 80, 200, 255)), (44,45,  ( 80, 200, 255)),
    (41,46,  ( 60, 180, 255)), (46,47,  ( 60, 180, 255)),
    (41,48,  ( 40, 160, 255)), (48,49,  ( 40, 160, 255)),
    (41,50,  ( 20, 140, 255)), (50,51,  ( 20, 140, 255)),
]

# Keypoint colours (BGR)
KP_COLOUR = (255, 255, 255)
KP_RADIUS = 4

N_KEYPOINTS = 52  # 0-51


def neutral_pose(cx: float = 0.5, cy: float = 0.5, scale: float = 0.28) -> np.ndarray:
    """
    Return a (52, 2) array of normalised [0,1] coordinates for a T-pose.
    cx, cy  : centre of the figure (normalised)
    scale   : height of the figure as a fraction of frame height
    """
    s = scale
    kp = np.zeros((N_KEYPOINTS, 2), dtype=np.float32)

    # ── Body ──────────────────────────────────────────────────────────────
    kp[0]  = [cx,        cy - s*0.48]   # Nose
    kp[1]  = [cx,        cy - s*0.35]   # Neck
    kp[2]  = [cx + s*0.18, cy - s*0.33] # RShoulder
    kp[3]  = [cx + s*0.30, cy - s*0.15] # RElbow
    kp[4]  = [cx + s*0.30, cy + s*0.05] # RWrist
    kp[5]  = [cx - s*0.18, cy - s*0.33] # LShoulder
    kp[6]  = [cx - s*0.30, cy - s*0.15] # LElbow
    kp[7]  = [cx - s*0.30, cy + s*0.05] # LWrist
    kp[8]  = [cx,        cy + s*0.00]   # MidHip
    kp[9]  = [cx + s*0.10, cy + s*0.00] # RHip
    kp[10] = [cx + s*0.12, cy + s*0.22] # RKnee
    kp[11] = [cx + s*0.12, cy + s*0.45] # RAnkle
    kp[12] = [cx - s*0.10, cy + s*0.00] # LHip
    kp[13] = [cx - s*0.12, cy + s*0.22] # LKnee
    kp[14] = [cx - s*0.12, cy + s*0.45] # LAnkle
    kp[15] = [cx + s*0.05, cy - s*0.50] # REye
    kp[16] = [cx - s*0.05, cy - s*0.50] # LEye
    kp[17] = [cx + s*0.09, cy - s*0.47] # REar
    kp[18] = [cx - s*0.09, cy - s*0.47] # LEar
    # Feet
    kp[19] = [cx - s*0.14, cy + s*0.48] # LBigToe
    kp[20] = [cx - s*0.16, cy + s*0.48] # LSmallToe
    kp[21] = [cx - s*0.10, cy + s*0.47] # LHeel
    kp[22] = [cx + s*0.14, cy + s*0.48] # RBigToe
    kp[23] = [cx + s*0.16, cy + s*0.48] # RSmallToe
    kp[24] = [cx + s*0.10, cy + s*0.47] # RHeel

    # ── Face (simplified) ─────────────────────────────────────────────────
    kp[25] = [cx - s*0.06, cy - s*0.50] # LeftEyeOuter
    kp[26] = [cx + s*0.06, cy - s*0.50] # RightEyeOuter
    kp[27] = [cx - s*0.03, cy - s*0.42] # MouthLeft
    kp[28] = [cx + s*0.03, cy - s*0.42] # MouthRight
    kp[29] = [cx,          cy - s*0.42] # MouthCenter

    # ── Left hand (wrist + 5 finger chains: knuckle, tip) ─────────────────
    lw = kp[7].copy()
    kp[30] = lw                                          # wrist mirror
    kp[31] = lw + [-s*0.04,  s*0.06]                    # index knuckle
    kp[32] = lw + [-s*0.04,  s*0.11]                    # index tip
    kp[33] = lw + [-s*0.01,  s*0.07]                    # middle knuckle
    kp[34] = lw + [-s*0.01,  s*0.12]                    # middle tip
    kp[35] = lw + [ s*0.02,  s*0.06]                    # ring knuckle
    kp[36] = lw + [ s*0.02,  s*0.11]                    # ring tip
    kp[37] = lw + [ s*0.05,  s*0.05]                    # pinky knuckle
    kp[38] = lw + [ s*0.05,  s*0.09]                    # pinky tip
    kp[39] = lw + [-s*0.07,  s*0.03]                    # thumb knuckle
    kp[40] = lw + [-s*0.10,  s*0.06]                    # thumb tip

    # ── Right hand ────────────────────────────────────────────────────────
    rw = kp[4].copy()
    kp[41] = rw
    kp[42] = rw + [ s*0.04,  s*0.06]
    kp[43] = rw + [ s*0.04,  s*0.11]
    kp[44] = rw + [ s*0.01,  s*0.07]
    kp[45] = rw + [ s*0.01,  s*0.12]
    kp[46] = rw + [-s*0.02,  s*0.06]
    kp[47] = rw + [-s*0.02,  s*0.11]
    kp[48] = rw + [-s*0.05,  s*0.05]
    kp[49] = rw + [-s*0.05,  s*0.09]
    kp[50] = rw + [ s*0.07,  s*0.03]
    kp[51] = rw + [ s*0.10,  s*0.06]

    return kp
