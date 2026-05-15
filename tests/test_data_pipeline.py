import tempfile
import unittest
from pathlib import Path

import numpy as np
import h5py

from mini_pi0.config.io import load_config
from mini_pi0.dataset.episodes import (
    list_supported_dataset_formats,
    load_episodes_from_config,
    load_episodes_robomimic,
)
from mini_pi0.dataset.stats import ActionStats
from mini_pi0.dataset.torch_dataset import ActionChunkDataset


class DataPipelineTests(unittest.TestCase):
    def test_supported_formats(self):
        self.assertEqual(set(list_supported_dataset_formats()), {"robomimic_hdf5", "lerobot_hf"})

    def test_robomimic_hdf5_loading(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            h5_path = root / "demo.hdf5"
            t = 10
            with h5py.File(h5_path, "w") as f:
                data = f.create_group("data")
                demo = data.create_group("demo_0")
                demo.create_dataset("actions", data=np.random.randn(t, 7).astype(np.float32))
                obs = demo.create_group("obs")
                obs.create_dataset("robot0_eef_pos", data=np.random.randn(t, 3).astype(np.float32))
                obs.create_dataset("robot0_eef_quat", data=np.random.randn(t, 4).astype(np.float32))
                obs.create_dataset("robot0_gripper_qpos", data=np.random.randn(t, 2).astype(np.float32))
                obs.create_dataset("agentview_image", data=np.random.randint(0, 255, size=(t, 84, 84, 3), dtype=np.uint8))
                obs.create_dataset(
                    "robot0_eye_in_hand_image",
                    data=np.random.randint(0, 255, size=(t, 84, 84, 3), dtype=np.uint8),
                )

            episodes = load_episodes_robomimic(
                hdf5_path=str(h5_path),
                image_keys=["agentview_image"],
                proprio_keys=["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
                limit=None,
                data_group="data",
                fallback_image_hw=(84, 84),
            )
            self.assertEqual(len(episodes), 1)
            self.assertEqual(episodes[0].actions.shape, (t, 7))
            self.assertEqual(episodes[0].obs[0]["agentview_image"].shape, (84, 84, 3))

            stats = ActionStats.from_actions(episodes[0].actions)
            ds = ActionChunkDataset(
                episodes=episodes,
                chunk_size=4,
                image_key="agentview_image",
                image_keys=None,
                proprio_keys=["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
                action_stats=stats,
            )
            self.assertEqual(len(ds), t - 4 + 1)
            img, prop, chunk = ds[0]
            self.assertEqual(tuple(img.shape), (3, 84, 84))
            self.assertEqual(tuple(prop.shape), (9,))
            self.assertEqual(tuple(chunk.shape), (4, 7))

            episodes_multi = load_episodes_robomimic(
                hdf5_path=str(h5_path),
                image_keys=["agentview_image", "robot0_eye_in_hand_image"],
                proprio_keys=["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
                limit=None,
                data_group="data",
                fallback_image_hw=(84, 84),
            )
            ds_multi = ActionChunkDataset(
                episodes=episodes_multi,
                chunk_size=4,
                image_key="agentview_image",
                image_keys=["agentview_image", "robot0_eye_in_hand_image"],
                proprio_keys=["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
                action_stats=stats,
            )
            img_multi, _, _ = ds_multi[0]
            self.assertEqual(tuple(img_multi.shape), (3, 84, 168))

            ds_hist = ActionChunkDataset(
                episodes=episodes_multi,
                chunk_size=4,
                image_key="agentview_image",
                image_keys=["agentview_image", "robot0_eye_in_hand_image"],
                proprio_keys=["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
                action_stats=stats,
                obs_horizon=2,
                preserve_camera_dim=True,
            )
            img_hist, prop_hist, chunk_hist = ds_hist[0]
            self.assertEqual(tuple(img_hist.shape), (2, 2, 3, 84, 84))
            self.assertEqual(tuple(prop_hist.shape), (2, 9))
            self.assertEqual(tuple(chunk_hist.shape), (4, 7))
            self.assertTrue(np.allclose(prop_hist[0].numpy(), prop_hist[1].numpy()))

if __name__ == "__main__":
    unittest.main()
