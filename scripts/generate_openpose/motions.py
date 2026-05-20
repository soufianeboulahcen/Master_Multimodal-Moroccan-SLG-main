"""
Motion generators.

Each generator is a function:
    generate_<name>(n_frames, fps) -> list[np.ndarray]

Returns a list of (52, 2) normalised keypoint arrays, one per frame.
All coordinates are in [0, 1] (x=right, y=down).
"""

import numpy as np
from scipy.interpolate import CubicSpline
from rig import neutral_pose, N_KEYPOINTS


# ── helpers ───────────────────────────────────────────────────────────────

def _sin(t, freq=1.0, phase=0.0, amp=1.0):
    return amp * np.sin(2 * np.pi * freq * t + phase)


def _cos(t, freq=1.0, phase=0.0, amp=1.0):
    return amp * np.cos(2 * np.pi * freq * t + phase)


def _smooth_loop(n, keyframes_t, keyframes_v):
    """Cubic-spline interpolation over a looping sequence."""
    cs = CubicSpline(keyframes_t, keyframes_v, bc_type='periodic')
    return cs(np.linspace(0, keyframes_t[-1], n, endpoint=False))


def _build_frames(n_frames, pose_fn):
    """Call pose_fn(t) for t in [0,1) and return list of kp arrays."""
    ts = np.linspace(0, 1, n_frames, endpoint=False)
    return [pose_fn(t) for t in ts]


def _attach_hands(kp, s=0.28):
    """Recompute hand keypoints relative to wrists after body animation."""
    lw = kp[7]
    kp[30] = lw
    kp[31] = lw + np.array([-s*0.04,  s*0.06])
    kp[32] = lw + np.array([-s*0.04,  s*0.11])
    kp[33] = lw + np.array([-s*0.01,  s*0.07])
    kp[34] = lw + np.array([-s*0.01,  s*0.12])
    kp[35] = lw + np.array([ s*0.02,  s*0.06])
    kp[36] = lw + np.array([ s*0.02,  s*0.11])
    kp[37] = lw + np.array([ s*0.05,  s*0.05])
    kp[38] = lw + np.array([ s*0.05,  s*0.09])
    kp[39] = lw + np.array([-s*0.07,  s*0.03])
    kp[40] = lw + np.array([-s*0.10,  s*0.06])
    rw = kp[4]
    kp[41] = rw
    kp[42] = rw + np.array([ s*0.04,  s*0.06])
    kp[43] = rw + np.array([ s*0.04,  s*0.11])
    kp[44] = rw + np.array([ s*0.01,  s*0.07])
    kp[45] = rw + np.array([ s*0.01,  s*0.12])
    kp[46] = rw + np.array([-s*0.02,  s*0.06])
    kp[47] = rw + np.array([-s*0.02,  s*0.11])
    kp[48] = rw + np.array([-s*0.05,  s*0.05])
    kp[49] = rw + np.array([-s*0.05,  s*0.09])
    kp[50] = rw + np.array([ s*0.07,  s*0.03])
    kp[51] = rw + np.array([ s*0.10,  s*0.06])
    return kp


# ── 1. Walking ─────────────────────────────────────────────────────────────

def generate_walking(n_frames=150, fps=30):
    cx, cy, s = 0.5, 0.52, 0.28

    def pose(t):
        kp = neutral_pose(cx, cy, s).copy()
        ph = 2 * np.pi * t          # one full gait cycle per loop

        # Vertical body bob
        bob = _sin(t, freq=2, amp=s*0.015)
        kp[[0,1,2,3,4,5,6,7,8,15,16,17,18,25,26,27,28,29], 1] += bob

        # Hip sway
        sway = _sin(t, freq=2, amp=s*0.025)
        kp[[0,1,2,3,4,5,6,7,8,15,16,17,18,25,26,27,28,29], 0] += sway

        # Leg swing (opposite phase)
        rleg = np.sin(ph)
        lleg = -np.sin(ph)

        # Right leg
        kp[10, 0] += rleg * s*0.08   # RKnee x
        kp[10, 1] -= abs(rleg) * s*0.04
        kp[11, 0] += rleg * s*0.12
        kp[11, 1] -= max(0, rleg) * s*0.06
        kp[22, 0] = kp[11, 0] + s*0.02
        kp[23, 0] = kp[11, 0] + s*0.04
        kp[24, 0] = kp[11, 0]

        # Left leg
        kp[13, 0] += lleg * s*0.08
        kp[13, 1] -= abs(lleg) * s*0.04
        kp[14, 0] += lleg * s*0.12
        kp[14, 1] -= max(0, lleg) * s*0.06
        kp[19, 0] = kp[14, 0] - s*0.02
        kp[20, 0] = kp[14, 0] - s*0.04
        kp[21, 0] = kp[14, 0]

        # Arm swing (counter to legs)
        kp[3, 0] += -rleg * s*0.06   # RElbow
        kp[4, 0] += -rleg * s*0.08   # RWrist
        kp[6, 0] += -lleg * s*0.06
        kp[7, 0] += -lleg * s*0.08

        return _attach_hands(kp, s)

    return _build_frames(n_frames, pose)


# ── 2. Running ─────────────────────────────────────────────────────────────

def generate_running(n_frames=120, fps=30):
    cx, cy, s = 0.5, 0.52, 0.28

    def pose(t):
        kp = neutral_pose(cx, cy, s).copy()
        ph = 2 * np.pi * t

        bob = _sin(t, freq=2, amp=s*0.04)
        kp[[0,1,2,3,4,5,6,7,8,15,16,17,18,25,26,27,28,29], 1] += bob

        # Forward lean
        lean = s * 0.04
        kp[[0,1,2,3,4,5,6,7,15,16,17,18,25,26,27,28,29], 0] += lean

        rleg = np.sin(ph)
        lleg = -np.sin(ph)

        # Exaggerated leg drive
        kp[10, 0] += rleg * s*0.14
        kp[10, 1] -= abs(rleg) * s*0.10
        kp[11, 0] += rleg * s*0.18
        kp[11, 1] -= max(0, rleg) * s*0.14
        kp[22, 0] = kp[11, 0] + s*0.02
        kp[23, 0] = kp[11, 0] + s*0.04
        kp[24, 0] = kp[11, 0]

        kp[13, 0] += lleg * s*0.14
        kp[13, 1] -= abs(lleg) * s*0.10
        kp[14, 0] += lleg * s*0.18
        kp[14, 1] -= max(0, lleg) * s*0.14
        kp[19, 0] = kp[14, 0] - s*0.02
        kp[20, 0] = kp[14, 0] - s*0.04
        kp[21, 0] = kp[14, 0]

        # Pumping arms (bent at elbow)
        kp[3, 0] += -rleg * s*0.10
        kp[3, 1] += -abs(rleg) * s*0.06
        kp[4, 0] += -rleg * s*0.14
        kp[4, 1] += -abs(rleg) * s*0.10
        kp[6, 0] += -lleg * s*0.10
        kp[6, 1] += -abs(lleg) * s*0.06
        kp[7, 0] += -lleg * s*0.14
        kp[7, 1] += -abs(lleg) * s*0.10

        return _attach_hands(kp, s)

    return _build_frames(n_frames, pose)


# ── 3. Dancing ─────────────────────────────────────────────────────────────

def generate_dancing(n_frames=180, fps=30):
    cx, cy, s = 0.5, 0.52, 0.28

    def pose(t):
        kp = neutral_pose(cx, cy, s).copy()
        ph = 2 * np.pi * t

        # Hip sway + bounce
        sway = _sin(t, freq=2, amp=s*0.06)
        bounce = abs(_sin(t, freq=4, amp=s*0.03))
        kp[:, 0] += sway
        kp[[0,1,2,3,4,5,6,7,8,15,16,17,18,25,26,27,28,29], 1] -= bounce

        # Arm wave — raised and flowing
        wave_r = np.sin(ph + 0.5)
        wave_l = np.sin(ph - 0.5)

        kp[2, 1] -= s*0.10           # raise shoulders
        kp[5, 1] -= s*0.10
        kp[3, 0] += s*0.12 + wave_r * s*0.06
        kp[3, 1] -= s*0.18 + wave_r * s*0.04
        kp[4, 0] += s*0.18 + wave_r * s*0.10
        kp[4, 1] -= s*0.28 + wave_r * s*0.06
        kp[6, 0] -= s*0.12 + wave_l * s*0.06
        kp[6, 1] -= s*0.18 + wave_l * s*0.04
        kp[7, 0] -= s*0.18 + wave_l * s*0.10
        kp[7, 1] -= s*0.28 + wave_l * s*0.06

        # Leg step side-to-side
        step = _sin(t, freq=2, amp=s*0.06)
        kp[10, 0] += step * 0.5
        kp[11, 0] += step
        kp[13, 0] -= step * 0.5
        kp[14, 0] -= step

        # Head tilt
        tilt = _sin(t, freq=1, amp=s*0.02)
        kp[0, 0] += tilt
        kp[15, 0] += tilt
        kp[16, 0] += tilt

        return _attach_hands(kp, s)

    return _build_frames(n_frames, pose)


# ── 4. Jumping ─────────────────────────────────────────────────────────────

def generate_jumping(n_frames=120, fps=30):
    cx, cy, s = 0.5, 0.52, 0.28

    def pose(t):
        kp = neutral_pose(cx, cy, s).copy()
        ph = 2 * np.pi * t

        # Vertical jump arc (parabolic feel via abs-sin)
        jump_h = max(0, _sin(t, freq=1, amp=s*0.30))
        kp[:, 1] -= jump_h

        # Tuck legs in air
        tuck = max(0, _sin(t, freq=1, amp=1.0))
        kp[10, 1] -= tuck * s*0.12
        kp[11, 1] -= tuck * s*0.20
        kp[13, 1] -= tuck * s*0.12
        kp[14, 1] -= tuck * s*0.20

        # Arms raise on takeoff
        arm_raise = max(0, _sin(t, freq=1, amp=1.0))
        kp[3, 1] -= arm_raise * s*0.12
        kp[4, 1] -= arm_raise * s*0.22
        kp[6, 1] -= arm_raise * s*0.12
        kp[7, 1] -= arm_raise * s*0.22

        # Spread arms wide
        kp[3, 0] += arm_raise * s*0.06
        kp[4, 0] += arm_raise * s*0.10
        kp[6, 0] -= arm_raise * s*0.06
        kp[7, 0] -= arm_raise * s*0.10

        return _attach_hands(kp, s)

    return _build_frames(n_frames, pose)


# ── 5. Hand & face gestures ────────────────────────────────────────────────

def generate_hand_face(n_frames=150, fps=30):
    cx, cy, s = 0.5, 0.52, 0.28

    def pose(t):
        kp = neutral_pose(cx, cy, s).copy()
        ph = 2 * np.pi * t

        # Gentle body sway
        kp[:, 0] += _sin(t, freq=0.5, amp=s*0.02)

        # Raise both hands to face level
        kp[3, 1] -= s*0.25
        kp[4, 1] -= s*0.40
        kp[6, 1] -= s*0.25
        kp[7, 1] -= s*0.40
        kp[3, 0] += s*0.05
        kp[4, 0] += s*0.08
        kp[6, 0] -= s*0.05
        kp[7, 0] -= s*0.08

        # Animated finger spread (right hand)
        spread_r = _sin(t, freq=2, amp=s*0.03)
        rw = kp[4]
        kp[41] = rw
        kp[42] = rw + np.array([ s*0.04 + spread_r,  s*0.06])
        kp[43] = rw + np.array([ s*0.04 + spread_r,  s*0.11])
        kp[44] = rw + np.array([ s*0.01,              s*0.07 + spread_r])
        kp[45] = rw + np.array([ s*0.01,              s*0.12 + spread_r])
        kp[46] = rw + np.array([-s*0.02 - spread_r,  s*0.06])
        kp[47] = rw + np.array([-s*0.02 - spread_r,  s*0.11])
        kp[48] = rw + np.array([-s*0.05,              s*0.05 - spread_r])
        kp[49] = rw + np.array([-s*0.05,              s*0.09 - spread_r])
        kp[50] = rw + np.array([ s*0.07 + spread_r,  s*0.03])
        kp[51] = rw + np.array([ s*0.10 + spread_r,  s*0.06])

        # Animated finger curl (left hand)
        curl = _sin(t, freq=3, amp=s*0.025)
        lw = kp[7]
        kp[30] = lw
        kp[31] = lw + np.array([-s*0.04,  s*0.06 + curl])
        kp[32] = lw + np.array([-s*0.04,  s*0.11 + curl*2])
        kp[33] = lw + np.array([-s*0.01,  s*0.07 + curl])
        kp[34] = lw + np.array([-s*0.01,  s*0.12 + curl*2])
        kp[35] = lw + np.array([ s*0.02,  s*0.06 + curl])
        kp[36] = lw + np.array([ s*0.02,  s*0.11 + curl*2])
        kp[37] = lw + np.array([ s*0.05,  s*0.05 + curl])
        kp[38] = lw + np.array([ s*0.05,  s*0.09 + curl*2])
        kp[39] = lw + np.array([-s*0.07,  s*0.03])
        kp[40] = lw + np.array([-s*0.10,  s*0.06 + curl])

        # Face: blinking eyes (move eye kps slightly)
        blink = max(0, _sin(t, freq=1.5, amp=s*0.008))
        kp[15, 1] += blink
        kp[16, 1] += blink
        kp[25, 1] += blink
        kp[26, 1] += blink

        # Mouth open/close
        mouth = abs(_sin(t, freq=2, amp=s*0.012))
        kp[27, 1] += mouth
        kp[28, 1] += mouth
        kp[29, 1] += mouth * 0.5

        return kp

    return _build_frames(n_frames, pose)


# ── 6. Waving ──────────────────────────────────────────────────────────────

def generate_waving(n_frames=120, fps=30):
    cx, cy, s = 0.5, 0.52, 0.28

    def pose(t):
        kp = neutral_pose(cx, cy, s).copy()
        ph = 2 * np.pi * t

        # Gentle weight shift
        shift = _sin(t, freq=0.5, amp=s*0.02)
        kp[:, 0] += shift

        # Raise right arm fully
        kp[2, 1] -= s*0.08
        kp[3, 0] += s*0.22
        kp[3, 1] -= s*0.22
        kp[4, 0] += s*0.28
        kp[4, 1] -= s*0.38

        # Wrist wave rotation
        wave = _sin(t, freq=3, amp=s*0.06)
        kp[4, 0] += wave

        # Animated right hand fingers during wave
        rw = kp[4]
        finger_wave = _sin(t, freq=4, amp=s*0.03)
        kp[41] = rw
        kp[42] = rw + np.array([ s*0.04,  s*0.06 + finger_wave])
        kp[43] = rw + np.array([ s*0.04,  s*0.11 + finger_wave*1.5])
        kp[44] = rw + np.array([ s*0.01,  s*0.07 + finger_wave*0.8])
        kp[45] = rw + np.array([ s*0.01,  s*0.12 + finger_wave*1.2])
        kp[46] = rw + np.array([-s*0.02,  s*0.06 + finger_wave*0.6])
        kp[47] = rw + np.array([-s*0.02,  s*0.11 + finger_wave])
        kp[48] = rw + np.array([-s*0.05,  s*0.05 + finger_wave*0.4])
        kp[49] = rw + np.array([-s*0.05,  s*0.09 + finger_wave*0.8])
        kp[50] = rw + np.array([ s*0.07,  s*0.03 + finger_wave*1.2])
        kp[51] = rw + np.array([ s*0.10,  s*0.06 + finger_wave*1.5])

        # Left arm relaxed at side
        lw = kp[7]
        kp[30] = lw
        kp[31] = lw + np.array([-s*0.04,  s*0.06])
        kp[32] = lw + np.array([-s*0.04,  s*0.11])
        kp[33] = lw + np.array([-s*0.01,  s*0.07])
        kp[34] = lw + np.array([-s*0.01,  s*0.12])
        kp[35] = lw + np.array([ s*0.02,  s*0.06])
        kp[36] = lw + np.array([ s*0.02,  s*0.11])
        kp[37] = lw + np.array([ s*0.05,  s*0.05])
        kp[38] = lw + np.array([ s*0.05,  s*0.09])
        kp[39] = lw + np.array([-s*0.07,  s*0.03])
        kp[40] = lw + np.array([-s*0.10,  s*0.06])

        return kp

    return _build_frames(n_frames, pose)


# ── Registry ───────────────────────────────────────────────────────────────

MOTIONS = {
    "walking":    generate_walking,
    "running":    generate_running,
    "dancing":    generate_dancing,
    "jumping":    generate_jumping,
    "hand_face":  generate_hand_face,
    "waving":     generate_waving,
}
