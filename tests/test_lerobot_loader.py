import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from mini_pi0.dataset.episodes import load_episodes_lerobot


class _StubLeRobotDataset:
    def __init__(self, repo_id=None, split=None, local_files_only=False, **kwargs):
        self.repo_id = repo_id
        self.split = split
        self.local_files_only = local_files_only
        self._samples = [
            {
                "episode_index": np.array(0),
                "action": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                "observation": {
                    "images": {"base_0_rgb": np.random.rand(16, 16, 3).astype(np.float32)},
                    "state": {
                        "eef_pos": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                        "eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                        "tool": np.array([0.5, -0.5], dtype=np.float32),
                    },
                },
            },
            {
                "episode_index": np.array(0),
                "action": np.array([0.4, 0.5, 0.6], dtype=np.float32),
                "observation": {
                    "images": {"base_0_rgb": np.random.rand(16, 16, 3).astype(np.float32)},
                    "state": {
                        "eef_pos": np.array([1.1, 2.1, 3.1], dtype=np.float32),
                        "eef_quat": np.array([0.0, 0.1, 0.0, 0.99], dtype=np.float32),
                        "tool": np.array([0.4, -0.4], dtype=np.float32),
                    },
                },
            },
            {
                "episode_index": np.array(1),
                "action": np.array([0.7, 0.8, 0.9], dtype=np.float32),
                "observation": {
                    "images": {"base_0_rgb": np.random.rand(16, 16, 3).astype(np.float32)},
                    "state": {
                        "eef_pos": np.array([2.0, 3.0, 4.0], dtype=np.float32),
                        "eef_quat": np.array([0.1, 0.0, 0.0, 0.99], dtype=np.float32),
                        "tool": np.array([0.3, -0.3], dtype=np.float32),
                    },
                },
            },
        ]

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        return self._samples[idx]


class LeRobotLoaderTests(unittest.TestCase):
    def test_load_episodes_lerobot_with_stub_module(self):
        mod_root = types.ModuleType("lerobot")
        mod_datasets = types.ModuleType("lerobot.datasets")
        mod_ld = types.ModuleType("lerobot.datasets.lerobot_dataset")
        mod_ld.LeRobotDataset = _StubLeRobotDataset

        with patch.dict(
            sys.modules,
            {
                "lerobot": mod_root,
                "lerobot.datasets": mod_datasets,
                "lerobot.datasets.lerobot_dataset": mod_ld,
            },
            clear=False,
        ):
            episodes = load_episodes_lerobot(
                repo_id="robotgeneralist/robosuite_lift_ph",
                image_keys=["observation.images.base_0_rgb"],
                proprio_keys=[
                    "observation.state.eef_pos",
                    "observation.state.eef_quat",
                    "observation.state.tool",
                ],
                action_key="action",
                episode_index_key="episode_index",
                limit=None,
                fallback_image_hw=(84, 84),
                local_files_only=True,
            )

        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[0].actions.shape, (2, 3))
        self.assertEqual(episodes[1].actions.shape, (1, 3))
        self.assertEqual(
            episodes[0].obs[0]["observation.images.base_0_rgb"].dtype,
            np.uint8,
        )
        self.assertEqual(
            episodes[0].obs[0]["observation.state.eef_pos"].shape,
            (3,),
        )


if __name__ == "__main__":
    unittest.main()
