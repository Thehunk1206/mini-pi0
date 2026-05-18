import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch

from mini_pi0.dataset.stats import ActionStats
from mini_pi0.dataset.lerobot_v3 import (
    LeRobotActionStatsComputer,
    LeRobotFeatureSpec,
    LeRobotPolicyDataset,
    LeRobotTemporalConfig,
)
from mini_pi0.dataset.robomimic_to_lerobot import RobomimicToLeRobotConfig, convert_robomimic_to_lerobot


class _FakeLeRobotDataset:
    def __init__(self):
        self.samples = []
        for idx in range(5):
            action_pad = idx > 2
            self.samples.append(
                {
                    "episode_index": np.array(0),
                    "action": np.stack(
                        [
                            np.array([min(idx + offset, 4), min(idx + offset, 4) + 0.5], dtype=np.float32)
                            for offset in range(3)
                        ],
                        axis=0,
                    ),
                    "action_is_pad": torch.tensor([False, action_pad, action_pad]),
                    "observation.state": np.stack(
                        [
                            np.array([max(idx - 1, 0), max(idx - 1, 0) + 1, max(idx - 1, 0) + 2], dtype=np.float32),
                            np.array([idx, idx + 1, idx + 2], dtype=np.float32),
                        ],
                        axis=0,
                    ),
                    "observation.images.agentview_image": np.stack(
                        [
                            np.full((3, 8, 8), max(idx - 1, 0), dtype=np.uint8),
                            np.full((3, 8, 8), idx, dtype=np.uint8),
                        ],
                        axis=0,
                    ),
                }
            )
        self.features = {
            "action": {},
            "observation.state": {},
            "observation.images.agentview_image": {},
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class _FakeCreatedLeRobotDataset:
    last = None

    def __init__(self, root, features):
        self.root = Path(root)
        self.features = features
        self.frames = []
        self.episodes = 0
        self.finalized = False

    @classmethod
    def create(cls, **kwargs):
        root = Path(kwargs["root"])
        root.mkdir(parents=True, exist_ok=False)
        obj = cls(root=root, features=kwargs["features"])
        cls.last = obj
        return obj

    def add_frame(self, frame):
        self.frames.append(dict(frame))

    def save_episode(self):
        self.episodes += 1

    def finalize(self):
        self.finalized = True


class LeRobotV3Tests(unittest.TestCase):
    def test_temporal_config_builds_delta_timestamps(self):
        spec = LeRobotFeatureSpec.from_keys(
            action_key="action",
            state_key="observation.state",
            image_keys=["agentview_image"],
        )
        temporal = LeRobotTemporalConfig(fps=20, obs_horizon=3, chunk_size=4)

        deltas = temporal.delta_timestamps(spec)

        self.assertEqual(deltas["action"], [0.0, 0.05, 0.1, 0.15])
        self.assertEqual(deltas["observation.state"], [-0.1, -0.05, 0.0])
        self.assertEqual(deltas["observation.images.agentview_image"], [-0.1, -0.05, 0.0])

    def test_lazy_dataset_returns_expected_shapes(self):
        base = _FakeLeRobotDataset()
        spec = LeRobotFeatureSpec.from_keys(
            action_key="action",
            state_key="observation.state",
            image_keys=["agentview_image"],
        )

        ds = LeRobotPolicyDataset(
            dataset=base,
            spec=spec,
            chunk_size=3,
            obs_horizon=2,
            preserve_camera_dim=True,
        )

        img, prop, chunk = ds[0]

        self.assertEqual(len(ds), 3)
        self.assertEqual(tuple(img.shape), (2, 1, 3, 8, 8))
        self.assertEqual(tuple(prop.shape), (2, 3))
        self.assertEqual(tuple(chunk.shape), (3, 2))
        self.assertEqual(chunk.dtype, torch.float32)

    def test_streaming_action_stats_matches_numpy(self):
        base = _FakeLeRobotDataset()
        raw_actions = [sample["action"] for sample in base.samples]
        expected = np.concatenate(raw_actions, axis=0)

        stats = ActionStats.from_iterable(LeRobotActionStatsComputer(base, "action").iter_actions())

        self.assertTrue(np.allclose(stats.mean, expected.mean(axis=0)))
        self.assertTrue(np.allclose(stats.std, expected.std(axis=0) + 1e-6))

    def test_convert_robomimic_to_lerobot_writes_expected_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            h5_path = root / "source.hdf5"
            out_dir = root / "lerobot"
            with h5py.File(h5_path, "w") as f:
                data = f.create_group("data")
                demo = data.create_group("demo_0")
                demo.create_dataset("actions", data=np.ones((4, 2), dtype=np.float32))
                obs = demo.create_group("obs")
                obs.create_dataset("agentview_image", data=np.ones((4, 8, 8, 3), dtype=np.uint8))
                obs.create_dataset("robot0_eef_pos", data=np.ones((4, 3), dtype=np.float32))
                obs.create_dataset("robot0_eef_quat", data=np.ones((4, 4), dtype=np.float32))

            mod_root = types.ModuleType("lerobot")
            mod_datasets = types.ModuleType("lerobot.datasets")
            mod_ld = types.ModuleType("lerobot.datasets.lerobot_dataset")
            mod_ld.LeRobotDataset = _FakeCreatedLeRobotDataset
            with patch.dict(
                sys.modules,
                {
                    "lerobot": mod_root,
                    "lerobot.datasets": mod_datasets,
                    "lerobot.datasets.lerobot_dataset": mod_ld,
                },
                clear=False,
            ):
                summary = convert_robomimic_to_lerobot(
                    RobomimicToLeRobotConfig(
                        input_hdf5=str(h5_path),
                        output_dir=str(out_dir),
                        repo_id="local/test",
                        image_keys=("agentview_image",),
                        state_keys=("robot0_eef_pos", "robot0_eef_quat"),
                        use_videos=False,
                    )
                )

        created = _FakeCreatedLeRobotDataset.last
        self.assertIsNotNone(created)
        assert created is not None
        self.assertTrue(created.finalized)
        self.assertEqual(created.episodes, 1)
        self.assertEqual(len(created.frames), 4)
        self.assertEqual(summary["episodes"], 1)
        self.assertEqual(summary["state_key"], "observation.state")
        self.assertIn("observation.images.agentview_image", summary["image_keys"])


if __name__ == "__main__":
    unittest.main()
