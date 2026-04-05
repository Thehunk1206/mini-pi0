import unittest

import numpy as np

from mini_pi0.config.io import load_config
from mini_pi0.dataset.episodes import EpisodeData
from mini_pi0.deploy.sim_runner import _resolve_deploy_rollout_controls
from mini_pi0.eval.core import _resolve_eval_rollout_controls
from mini_pi0.train.runner import _curate_episodes


def _make_episode(length: int, action_scale: float, object_delta: float) -> EpisodeData:
    actions = (np.random.randn(length, 7) * action_scale).astype(np.float32)
    obs = []
    for t in range(length):
        frac = float(t) / max(1.0, float(length - 1))
        obj = np.array([object_delta * frac, 0.0, 0.0], dtype=np.float32)
        obs.append(
            {
                "observation.state.object": obj,
                "observation.state.eef_pos": np.array([0.0, 0.0, 0.0], dtype=np.float32),
                "observation.images.right_wrist_0_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
            }
        )
    return EpisodeData(obs=obs, actions=actions)


class TrainingStabilityControlTests(unittest.TestCase):
    def test_episode_curation_filters_low_quality_episodes(self):
        good = _make_episode(length=40, action_scale=0.5, object_delta=0.15)
        short = _make_episode(length=8, action_scale=0.5, object_delta=0.15)
        no_progress = _make_episode(length=40, action_scale=1e-4, object_delta=1e-4)

        cfg = load_config(
            overrides=[
                "data.filter_min_episode_length=16",
                "data.filter_min_action_std=0.01",
                "data.filter_min_state_delta=0.02",
                "data.filter_state_delta_key='observation.state.object'",
                "data.filter_drop_nan=true",
            ]
        )
        kept, summary = _curate_episodes([good, short, no_progress], cfg)
        self.assertEqual(len(kept), 1)
        self.assertEqual(summary["before_episodes"], 3)
        self.assertEqual(summary["after_episodes"], 1)
        self.assertEqual(summary["removed_episodes"], 2)
        self.assertTrue(summary["reasons"])

    def test_eval_warmup_rollout_controls(self):
        cfg = load_config(
            overrides=[
                "eval.execute_steps=4",
                "eval.n_flow_steps=10",
                "eval.action_smoothing_alpha=0.0",
                "eval.stability_warmup_steps=20",
                "eval.stability_warmup_execute_steps=2",
                "eval.stability_warmup_n_flow_steps=16",
                "eval.stability_warmup_action_smoothing_alpha=0.2",
            ]
        )
        warm = _resolve_eval_rollout_controls(cfg, env_steps_done=5)
        steady = _resolve_eval_rollout_controls(cfg, env_steps_done=25)
        self.assertEqual(warm, (2, 16, 0.2))
        self.assertEqual(steady, (4, 10, 0.0))

    def test_deploy_warmup_rollout_controls(self):
        cfg = load_config(
            overrides=[
                "deploy.execute_steps=4",
                "deploy.n_flow_steps=10",
                "deploy.action_smoothing_alpha=0.0",
                "deploy.stability_warmup_steps=30",
                "deploy.stability_warmup_execute_steps=1",
                "deploy.stability_warmup_n_flow_steps=20",
                "deploy.stability_warmup_action_smoothing_alpha=0.3",
            ]
        )
        warm = _resolve_deploy_rollout_controls(cfg, env_steps_done=0)
        steady = _resolve_deploy_rollout_controls(cfg, env_steps_done=35)
        self.assertEqual(warm, (1, 20, 0.3))
        self.assertEqual(steady, (4, 10, 0.0))


if __name__ == "__main__":
    unittest.main()

