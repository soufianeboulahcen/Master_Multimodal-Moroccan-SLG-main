"""Avatar video rendering subsystem.

This package is the *renderer* half of the project. The existing `mosl/` code
is the *choreographer*: it turns Arabic text into OpenPose-format motion. This
package turns that motion into photorealistic avatar video via diffusion.

It is strictly additive — it imports nothing from and modifies nothing in the
rest of `mosl/`. The only contract between the two halves is the OpenPose JSON
keypoint format produced by `mosl/pose/`.

Phases (see render/README.md):
    A  pose_bridge   OpenPose JSON  -> ControlNet-ready pose frames   [done, CPU]
    B  dwpose        re-extract dense whole-body keypoints            [DGX]
    C  identity      InstantID / IP-Adapter face embedding            [DGX]
    D  keyframe      identity embedding -> photoreal reference image  [DGX]
    E  video         MimicMotion keyframe+sign clip -> avatar video   [DGX]
    F  temporal      deflicker + frame interpolation + upscale        [DGX]
    G  render_avatar end-to-end orchestrator (scripts/)               [DGX]
"""
