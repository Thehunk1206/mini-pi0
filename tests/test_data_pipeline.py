import tempfile
import unittest
from pathlib import Path

import numpy as np
import h5py

from mini_pi0.config.io import load_config
from mini_pi0.dataset.episodes import (
    list_supported_dataset_formats,
    load_episodes,
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

            episodes = load_episodes_robomimic(
                hdf5_path=str(h5_path),
                image_key="agentview_image",
                proprio_keys=["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
                limit=None,
                data_group="data",
                fallback_image_hw=(84, 84),
            )
            self.assertEqual(len(episodes), 1)
            self.assertEqual(episodes[0].actions.shape, (t, 7))
            self.assertEqual(episodes[0].obs[0]["agentview_image"].shape, (84, 84, 3))

            # convenience wrapper should behave identically
            episodes2 = load_episodes(
                hdf5_path=str(h5_path),
                image_key="agentview_image",
                proprio_keys=["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
                limit=None,
                data_group="data",
                fallback_image_hw=(84, 84),
            )
            self.assertEqual(len(episodes2), 1)

            stats = ActionStats.from_actions(episodes2[0].actions)
            ds = ActionChunkDataset(
                episodes=episodes2,
                chunk_size=4,
                image_key="agentview_image",
                proprio_keys=["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
                action_stats=stats,
            )
            self.assertEqual(len(ds), t - 4 + 1)
            img, prop, chunk = ds[0]
            self.assertEqual(tuple(img.shape), (3, 84, 84))
            self.assertEqual(tuple(prop.shape), (9,))
            self.assertEqual(tuple(chunk.shape), (4, 7))

    def test_precomputed_feature_attach(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            h5_path = root / "demo.hdf5"
            feat_path = root / "vision_feats.npz"
            t = 6
            with h5py.File(h5_path, "w") as f:
                data = f.create_group("data")
                demo = data.create_group("demo_0")
                demo.create_dataset("actions", data=np.random.randn(t, 7).astype(np.float32))
                obs = demo.create_group("obs")
                obs.create_dataset("robot0_eef_pos", data=np.random.randn(t, 3).astype(np.float32))
                obs.create_dataset("robot0_eef_quat", data=np.random.randn(t, 4).astype(np.float32))
                obs.create_dataset("robot0_gripper_qpos", data=np.random.randn(t, 2).astype(np.float32))
                obs.create_dataset("agentview_image", data=np.random.randint(0, 255, size=(t, 84, 84, 3), dtype=np.uint8))

            np.savez_compressed(feat_path, ep_000000=np.random.randn(t, 32).astype(np.float32))
            cfg = load_config(
                overrides=[
                    "data.format=robomimic_hdf5",
                    f"data.robomimic_hdf5='{str(h5_path)}'",
                    "data.robomimic_data_group='data'",
                    "data.observation_mode='precomputed'",
                    f"data.precomputed_features_path='{str(feat_path)}'",
                    "data.precomputed_feature_key='vision_feat'",
                    "data.n_demos=1",
                ]
            )
            episodes = load_episodes_from_config(cfg)
            self.assertEqual(len(episodes), 1)
            self.assertEqual(episodes[0].obs[0]["vision_feat"].shape, (32,))


if __name__ == "__main__":
    unittest.main()
