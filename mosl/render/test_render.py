"""Regression tests for the avatar rendering subsystem (mosl/render/).

Covers the pure-Python / CPU logic of every phase — keypoint conversion,
temporal smoothing, identity fusion, keyframe geometry, file discovery. The
GPU / diffusion paths (DWPose, insightface, InstantID, MimicMotion, ffmpeg
filters) are NOT covered here; they are validated on the DGX with each module's
`--check`.

Run:
    python -m mosl.render.test_render
    python -m unittest mosl.render.test_render -v
"""
from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from mosl.render import dwpose_extract as dw
from mosl.render import identity as idt
from mosl.render import keyframe as kf
from mosl.render import pose_bridge as pb
from mosl.render import temporal as tmp
from mosl.render import video as vid

ROOT = Path(__file__).resolve().parents[2]


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)


# --- Phase A: pose_bridge ---------------------------------------------------

class TestPoseBridge(unittest.TestCase):

    def test_interpolate_gaps_fills_dropouts(self):
        # one keypoint, present at t=0 and t=4, missing (conf 0) between
        arr = np.zeros((5, 1, 3), dtype=np.float32)
        arr[0, 0] = [0.0, 0.0, 1.0]
        arr[4, 0] = [40.0, 80.0, 1.0]
        out = pb.interpolate_gaps(arr)
        self.assertAlmostEqual(out[2, 0, 0], 20.0, places=3)
        self.assertAlmostEqual(out[2, 0, 1], 40.0, places=3)

    def test_interpolate_gaps_keeps_single_point_clips(self):
        arr = np.zeros((1, 5, 3), dtype=np.float32)
        out = pb.interpolate_gaps(arr)
        self.assertEqual(out.shape, arr.shape)

    def test_compute_transform_centers_figure(self):
        clip = {
            "body": np.array([[[100, 100, 1.0], [300, 500, 1.0]]], dtype=np.float32),
            "hand_l": np.zeros((1, 0, 3), dtype=np.float32),
            "hand_r": np.zeros((1, 0, 3), dtype=np.float32),
        }
        s, ox, oy = pb.compute_transform(clip, canvas=768, fill=0.72)
        self.assertGreater(s, 0.0)
        # the bbox centre must map to the canvas centre
        cx = ((100 + 300) / 2) * s + ox
        self.assertAlmostEqual(cx, 384.0, places=3)

    def test_draw_pose_frame_returns_sized_image(self):
        body = np.zeros((18, 3), dtype=np.float32)
        body[1] = [128, 100, 1.0]   # neck
        body[2] = [160, 140, 1.0]   # r-shoulder
        img = pb.draw_pose_frame(body, np.zeros((21, 3), np.float32),
                                 np.zeros((21, 3), np.float32),
                                 np.zeros((0, 3), np.float32), 256, False)
        self.assertEqual(img.size, (256, 256))
        self.assertGreater(np.array(img).sum(), 0)   # something was drawn

    def test_load_clip_detects_body_format(self):
        # uses the real example keypoints committed to the repo
        sign = ROOT / "outputs" / "openpose_json" / "أَنْتِ_keypoints"
        if sign.is_dir():
            clip = pb.load_clip(sign)
            self.assertEqual(clip["body_format"], "coco18")
            self.assertEqual(clip["body"].shape[1], 18)


# --- Phase B: dwpose_extract ------------------------------------------------

class TestDWPoseConvert(unittest.TestCase):

    def test_wholebody_to_openpose_shapes(self):
        kpts = np.random.default_rng(0).uniform(0, 500, (133, 2)).astype(np.float32)
        scores = np.random.default_rng(1).uniform(0.5, 1, 133).astype(np.float32)
        parts = dw.wholebody_to_openpose(kpts, scores)
        self.assertEqual(parts["body"].shape, (18, 3))
        self.assertEqual(parts["face"].shape, (68, 3))
        self.assertEqual(parts["hand_l"].shape, (21, 3))
        self.assertEqual(parts["hand_r"].shape, (21, 3))

    def test_wholebody_neck_is_shoulder_midpoint(self):
        kpts = np.zeros((133, 2), dtype=np.float32)
        kpts[5], kpts[6] = [10, 20], [30, 60]          # L / R shoulder
        scores = np.ones(133, dtype=np.float32)
        parts = dw.wholebody_to_openpose(kpts, scores)
        np.testing.assert_allclose(parts["body"][1, :2], [20, 40])

    def test_wholebody_body_remap_is_correct(self):
        # COCO-17 indices placed at recognizable coordinates
        kpts = np.zeros((133, 2), dtype=np.float32)
        for i in range(17):
            kpts[i] = [i, i * 10]
        scores = np.ones(133, dtype=np.float32)
        body = dw.wholebody_to_openpose(kpts, scores)["body"]
        # nose op0<-coco0, r-wrist op4<-coco10, l-wrist op7<-coco9, r-ear op16<-coco4
        for op, coco in [(0, 0), (4, 10), (7, 9), (16, 4), (14, 2)]:
            np.testing.assert_allclose(body[op, :2], [coco, coco * 10],
                                       err_msg=f"op{op} should map coco{coco}")

    def test_one_euro_reduces_jitter(self):
        rng = np.random.default_rng(2)
        noisy = rng.normal(0, 5.0, (60, 4))
        smoothed = dw.one_euro_smooth(noisy, 25.0, 1.0, 0.15)
        self.assertLess(smoothed.std(), noisy.std())

    def test_one_euro_short_sequence_passthrough(self):
        seq = np.array([[1.0], [2.0]])
        np.testing.assert_array_equal(dw.one_euro_smooth(seq, 25, 1, 0.1), seq)

    def test_smooth_part_preserves_shape_and_confidence(self):
        rng = np.random.default_rng(3)
        part = rng.uniform(0, 500, (40, 21, 3)).astype(np.float32)
        conf = part[:, :, 2].copy()
        out = dw.smooth_part(part, 25.0,
                             dw.ExtractConfig(out_root=Path("."), min_cutoff=1.0,
                                              beta=0.15))
        self.assertEqual(out.shape, part.shape)
        np.testing.assert_array_equal(out[:, :, 2], conf)


# --- Phase C: identity ------------------------------------------------------

class TestIdentity(unittest.TestCase):

    def _rec(self, name, emb):
        return idt.FaceRecord(name, np.array([0, 0, 100, 100], "f4"),
                              np.zeros((5, 2), "f4"), 0.9, emb, _unit(emb),
                              np.zeros((112, 112, 3), "u1"),
                              np.zeros((8, 8, 3), "u1"))

    def test_fusion_drops_outlier_and_normalizes(self):
        rng = np.random.default_rng(0)
        base = _unit(rng.normal(size=512))
        recs = [self._rec(f"g{i}.jpg", _unit(base + 0.05 * rng.normal(size=512)))
                for i in range(3)]
        recs.append(self._rec("bad.jpg", _unit(rng.normal(size=512))))
        cfg = idt.IdentityConfig(input_dir="x", outlier_threshold=0.5)
        fused = idt.fuse_multi_image_identity(recs, cfg)
        self.assertEqual(fused.used[:3], [True, True, True])
        self.assertFalse(fused.used[3])
        self.assertAlmostEqual(float(np.linalg.norm(fused.fused)), 1.0, places=4)

    def test_fusion_keeps_best_when_all_flagged(self):
        rng = np.random.default_rng(1)
        recs = [self._rec(f"{i}.jpg", _unit(rng.normal(size=512))) for i in range(3)]
        cfg = idt.IdentityConfig(input_dir="x", outlier_threshold=0.999)
        fused = idt.fuse_multi_image_identity(recs, cfg)
        self.assertEqual(sum(fused.used), 1)   # never returns an empty identity

    def test_consistency_metric_ordering(self):
        rng = np.random.default_rng(2)
        base = _unit(rng.normal(size=512))
        tight = np.stack([_unit(base + 0.02 * rng.normal(size=512)) for _ in range(4)])
        loose = np.stack([_unit(rng.normal(size=512)) for _ in range(4)])
        self.assertGreater(idt._consistency(tight)["mean_pairwise_cosine"],
                           idt._consistency(loose)["mean_pairwise_cosine"])

    def test_identity_id_defaults_to_folder_name(self):
        self.assertEqual(idt.IdentityConfig(input_dir="photos/omar").identity_id,
                         "omar")

    def test_load_embeddings_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "identity_embeddings").mkdir()
            (root / "identity_metadata").mkdir()
            np.savez(root / "identity_embeddings" / "omar.npz",
                     fused=np.ones(512, "f4"), fused_raw=np.ones(512, "f4"),
                     per_image=np.ones((2, 512), "f4"),
                     reference_kps=np.zeros((5, 2), "f4"))
            (root / "identity_metadata" / "omar.json").write_text('{"identity_id":"omar"}')
            loaded = idt.load_embeddings("omar", root)
            self.assertEqual(loaded["fused"].shape, (512,))
            self.assertEqual(loaded["metadata"]["identity_id"], "omar")

    def test_load_embeddings_missing_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                idt.load_embeddings("ghost", Path(d))

    def test_load_embeddings_includes_reference_face_path(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "identity_embeddings").mkdir()
            (root / "identity_metadata").mkdir()
            np.savez(root / "identity_embeddings" / "omar.npz",
                     fused=np.ones(512, "f4"), fused_raw=np.ones(512, "f4"),
                     per_image=np.ones((2, 512), "f4"),
                     reference_kps=np.zeros((5, 2), "f4"))
            (root / "identity_metadata" / "omar.json").write_text('{"identity_id":"omar"}')
            loaded = idt.load_embeddings("omar", root)
            self.assertIn("reference_face_path", loaded)
            self.assertIn("omar", str(loaded["reference_face_path"]))

    def test_load_embeddings_new_layout(self):
        """Embeddings must live under identity_embeddings/, not embeddings/."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "identity_embeddings").mkdir()
            np.savez(root / "identity_embeddings" / "test.npz",
                     fused=np.ones(512, "f4"), fused_raw=np.ones(512, "f4"),
                     per_image=np.ones((1, 512), "f4"),
                     reference_kps=np.zeros((5, 2), "f4"))
            loaded = idt.load_embeddings("test", root)
            self.assertEqual(loaded["fused"].shape, (512,))

    def test_list_identities_empty(self):
        with tempfile.TemporaryDirectory() as d:
            result = idt.list_identities(Path(d))
            self.assertEqual(result, [])

    def test_list_identities_returns_summaries(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            meta_dir = root / "identity_metadata"
            meta_dir.mkdir()
            meta = {
                "identity_id": "fatima",
                "n_images": 3,
                "n_used_in_fusion": 3,
                "consistency": {"mean_pairwise_cosine": 0.92},
                "created": "2026-05-20",
                "files": {},
            }
            (meta_dir / "fatima.json").write_text(json.dumps(meta))
            summaries = idt.list_identities(root)
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0]["identity_id"], "fatima")
            self.assertEqual(summaries[0]["n_images"], 3)

    def test_identity_config_directory_properties(self):
        cfg = idt.IdentityConfig(input_dir="photos/omar", identity_id="omar")
        self.assertTrue(str(cfg.embeddings_dir).endswith("identity_embeddings"))
        self.assertTrue(str(cfg.metadata_dir).endswith("identity_metadata"))
        self.assertIn("omar", str(cfg.aligned_dir))

    def test_yaml_config_loaded(self):
        """load_config must accept a YAML file when pyyaml is available."""
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("pyyaml not installed")
        import io
        with tempfile.TemporaryDirectory() as d:
            cfg_path = Path(d) / "cfg.yaml"
            cfg_path.write_text("det_model: buffalo_l\ndevice: cpu\n")
            args = type("A", (), {
                "input": None, "identity_id": None, "out_root": None,
                "det_model": None, "device": None, "crop_size": None, "viz": False,
            })()
            cfg = idt.load_config(cfg_path, args)
            self.assertEqual(cfg.det_model, "buffalo_l")
            self.assertEqual(cfg.device, "cpu")


# --- Phase D: keyframe ------------------------------------------------------

class TestKeyframe(unittest.TestCase):

    def test_place_kps_canonical_centered_and_scaled(self):
        cfg = kf.KeyframeConfig(identity_id="x")
        kps = kf.place_kps(np.zeros((5, 2), "f4"), cfg)   # -> canonical
        cx = (kps[:, 0].min() + kps[:, 0].max()) / 2
        span_y = kps[:, 1].max() - kps[:, 1].min()
        self.assertAlmostEqual(cx, cfg.face_cx * cfg.width, delta=1.0)
        self.assertAlmostEqual(span_y, cfg.face_height_frac * cfg.height, delta=1.0)
        self.assertTrue((kps >= 0).all())

    def test_place_kps_reference_preserves_asymmetry(self):
        cfg = kf.KeyframeConfig(identity_id="x", kps_source="reference")
        ref = np.array([[400, 300], [500, 320], [450, 360],
                        [410, 420], [490, 430]], "f4")
        kps = kf.place_kps(ref, cfg)
        self.assertNotAlmostEqual(kps[0, 1], kps[1, 1], places=2)  # eyes not level

    def test_place_kps_degenerate_reference_falls_back(self):
        cfg = kf.KeyframeConfig(identity_id="x", kps_source="reference")
        kps = kf.place_kps(np.zeros((5, 2), "f4"), cfg)  # degenerate -> canonical
        self.assertGreater(kps[:, 1].max() - kps[:, 1].min(), 1.0)


# --- Phase E: video ---------------------------------------------------------

class TestVideo(unittest.TestCase):

    def test_newest_video_since_picks_recent(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            old = root / "old.mp4"
            old.write_bytes(b"x")
            import os
            os.utime(old, (1000, 1000))
            t0 = time.time()
            time.sleep(0.02)
            (root / "sub").mkdir()
            new = root / "sub" / "new.mp4"
            new.write_bytes(b"y")
            self.assertEqual(vid._newest_video_since(root, t0), new)

    def test_newest_video_since_none_when_empty(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(vid._newest_video_since(Path(d), time.time()))

    def test_resolve_keyframe_missing_raises(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = vid.VideoConfig(identity_id="ghost", keyframe_root=Path(d))
            with self.assertRaises(RuntimeError):
                vid.resolve_keyframe(cfg)


# --- Phase F: temporal ------------------------------------------------------

class TestTemporal(unittest.TestCase):

    def test_config_defaults(self):
        c = tmp.TemporalConfig()
        self.assertEqual(c.target_fps, 30)
        self.assertTrue(c.deflicker)
        self.assertEqual(c.interp_backend, "ffmpeg")

    def test_discover_inputs_skips_final_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "raw.mp4").write_bytes(b"x")
            (root / "clip_final.mp4").write_bytes(b"y")  # must be skipped
            args = type("A", (), {"input_video": None, "input_dir": str(root)})()
            found = tmp.discover_inputs(args)
            names = {p.name for p in found}
            self.assertIn("raw.mp4", names)
            self.assertNotIn("clip_final.mp4", names)

    def test_ffmpeg_exe_returns_path(self):
        self.assertTrue(tmp.ffmpeg_exe())


if __name__ == "__main__":
    unittest.main(verbosity=2)
