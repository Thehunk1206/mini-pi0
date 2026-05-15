"""Tests for selecting the render camera used by eval grid videos."""

import unittest

from mini_pi0.config.io import load_config
from mini_pi0.eval.core import _resolve_grid_cameras


class EvalGridCameraTests(unittest.TestCase):
    def test_resolve_grid_camera_defaults_to_first_simulator_camera(self) -> None:
        cfg = load_config()
        cfg.simulator.camera_names = ["base_camera", "hand_camera"]

        cameras = _resolve_grid_cameras(cfg)

        self.assertEqual(cameras, ["base_camera"])

    def test_resolve_grid_camera_accepts_configured_camera_name(self) -> None:
        cfg = load_config(overrides=["eval.grid_cameras='hand_camera'"])
        cfg.simulator.camera_names = ["base_camera", "hand_camera"]

        cameras = _resolve_grid_cameras(cfg)

        self.assertEqual(cameras, ["hand_camera"])

    def test_resolve_grid_camera_accepts_configured_image_key(self) -> None:
        cfg = load_config(overrides=["eval.grid_cameras='robot0_eye_in_hand_image'"])
        cfg.robot.image_keys = ["agentview_image", "robot0_eye_in_hand_image"]

        cameras = _resolve_grid_cameras(cfg)

        self.assertEqual(cameras, ["robot0_eye_in_hand_image"])

    def test_resolve_grid_camera_accepts_multiple_configured_cameras(self) -> None:
        cfg = load_config(overrides=["eval.grid_cameras=['base_camera', 'hand_camera']"])
        cfg.simulator.camera_names = ["base_camera", "hand_camera"]

        cameras = _resolve_grid_cameras(cfg)

        self.assertEqual(cameras, ["base_camera", "hand_camera"])

    def test_resolve_grid_camera_rejects_unknown_camera(self) -> None:
        cfg = load_config(overrides=["eval.grid_cameras='side_camera'"])
        cfg.simulator.camera_names = ["base_camera", "hand_camera"]

        with self.assertRaisesRegex(ValueError, "eval grid camera"):
            _resolve_grid_cameras(cfg)


if __name__ == "__main__":
    unittest.main()
