"""Tests for backend-independent ReinFlow rollout behavior."""

from __future__ import annotations

# ruff: noqa: E402 - keep torch optional at test collection time.

import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from mini_pi0.config.io import load_config
from mini_pi0.rl import flow_policy, flow_ppo
from mini_pi0.rl.flow_ppo import ReinFlowPPOUpdateStats
from mini_pi0.rl.rewards import NativeReward
from mini_pi0.rl.runner import (
    _execute_macro_action,
    _evaluate_actor,
    _make_obs_processor,
    _policy_actions_to_env,
    _run_reinflow_loop,
    run_rl_smoke,
)
from mini_pi0.sim.base import StepOutput
from mini_pi0.sim.batched import SerialBatchAdapter


class _TinyActionTransformer(nn.Module):
    def __init__(self, action_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.cond_proj = nn.Linear(cond_dim, action_dim)

    def forward(self, actions: torch.Tensor, tau: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        del tau
        return 0.1 * actions + self.cond_proj(cond).unsqueeze(1)


class _TinyFlowPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.action_transformer = _TinyActionTransformer(action_dim=2, cond_dim=4)

    def _encode_conditioning(self, image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        del image
        return self.encoder(proprio)


class _FakeAdapter:
    backend_name = "fake_backend"

    def __init__(self, cfg, *, terminate_after: int | None = None) -> None:
        self.cfg = cfg
        self.seen_backend = cfg.simulator.backend
        self.terminate_after = terminate_after
        self.episode_step = 0
        self.step_calls = 0
        self.reset_seeds: list[int] = []

    def reset(self, seed: int | None = None):
        self.episode_step = 0
        self.reset_seeds.append(int(seed) if seed is not None else -1)
        return self._observation()

    def step(self, action: np.ndarray) -> StepOutput:
        self.episode_step += 1
        self.step_calls += 1
        terminated = self.terminate_after is not None and self.episode_step >= self.terminate_after
        return StepOutput(
            obs=self._observation(),
            reward=1.0,
            terminated=terminated,
            truncated=False,
            info={"success": False, "action_sum": float(np.asarray(action).sum())},
        )

    def action_spec(self):
        return -np.ones(2, dtype=np.float32), np.ones(2, dtype=np.float32)

    def render(self, camera: str = "agentview", width: int = 512, height: int = 512):
        del camera
        return np.zeros((height, width, 3), dtype=np.uint8)

    def check_success(self, info=None, obs=None):
        del obs
        return bool((info or {}).get("success", False))

    def close(self) -> None:
        return None

    def _observation(self) -> dict[str, np.ndarray]:
        return {
            "agentview_image": np.zeros((8, 8, 3), dtype=np.uint8),
            "robot0_eef_pos": np.array([self.episode_step, 0.0, 0.0], dtype=np.float32),
        }


class _FakeNoise:
    def __init__(self) -> None:
        self.current_std_max = torch.tensor(0.1)

    def set_training_progress(self, progress: float) -> None:
        del progress


class _FakeRolloutActor:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.deterministic_batch_sizes: list[int] = []
        self.actor = SimpleNamespace(
            kernel=SimpleNamespace(noise=_FakeNoise()),
            likelihood_mode=lambda: None,
        )

    def sample_path(self, image: torch.Tensor, proprio: torch.Tensor, *, bounds=None):
        del proprio, bounds
        batch = image.shape[0]
        path = torch.zeros(
            batch,
            self.cfg.rl.flow_steps + 1,
            self.cfg.model.chunk_size,
            self.cfg.model.action_dim,
        )
        return SimpleNamespace(
            path=path,
            action_chunk=path[:, -1],
            log_prob=torch.zeros(batch),
            value=torch.zeros(batch),
        )

    def value(self, image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        del proprio
        return torch.zeros(image.shape[0])

    def deterministic_sample(
        self,
        image: torch.Tensor,
        proprio: torch.Tensor,
        *,
        bounds=None,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del proprio, bounds, initial_noise
        self.deterministic_batch_sizes.append(int(image.shape[0]))
        return torch.zeros(image.shape[0], self.cfg.model.chunk_size, self.cfg.model.action_dim)


class _FakeUpdater:
    def update(self, *args, **kwargs) -> ReinFlowPPOUpdateStats:
        del args, kwargs
        return ReinFlowPPOUpdateStats(
            policy_loss=0.0,
            value_loss=0.0,
            entropy=0.0,
            approx_kl=0.0,
            clip_fraction=0.0,
            ratio_min=1.0,
            ratio_max=1.0,
            reference_w2=0.0,
            transition_kl=0.0,
            velocity_anchor=0.0,
            actor_grad_norm=0.0,
            critic_grad_norm=0.0,
            explained_variance=0.0,
            actor_updated=True,
        )


def _config(*, execution_horizon: int = 1):
    return load_config(
        overrides=[
            "experiment.name='runner-test'",
            "simulator.backend='fake_backend'",
            "simulator.task='fake_task'",
            "robot.action_dim=2",
            "robot.image_key='agentview_image'",
            "robot.image_keys=['agentview_image']",
            "robot.state_keys=['robot0_eef_pos']",
            "robot.proprio_keys=['robot0_eef_pos']",
            "model.action_dim=2",
            "model.chunk_size=3",
            "model.prop_dim=3",
            "model.cond_dim=4",
            "model.d_model=8",
            "rl.algorithm='reinflow_ppo'",
            "rl.init_mode='scratch'",
            "rl.action_normalization='env_bounds'",
            "rl.flow_steps=2",
            f"rl.execution_horizon={execution_horizon}",
            "rl.device='cpu'",
        ]
    )


def test_rl_smoke_uses_configured_backend() -> None:
    cfg = _config()
    adapters: list[_FakeAdapter] = []

    def factory(factory_cfg):
        adapter = _FakeAdapter(factory_cfg)
        adapters.append(adapter)
        return adapter

    with patch("mini_pi0.rl.flow_policy.make_model", return_value=_TinyFlowPolicy()):
        with patch("mini_pi0.rl.runner.make_sim_adapter", side_effect=factory):
            summary = run_rl_smoke(cfg)

    assert summary["backend"] == "fake_backend"
    assert adapters[0].seen_backend == "fake_backend"
    assert summary["path_shape"] == [1, 3, 3, 2]


def test_macro_action_stops_each_environment_at_its_episode_boundary() -> None:
    cfg = _config(execution_horizon=3)
    first = _FakeAdapter(cfg, terminate_after=1)
    second = _FakeAdapter(cfg)
    adapter = SerialBatchAdapter([first, second])
    observations = adapter.reset([0, 1])
    processor = _make_obs_processor(cfg, device="cpu")
    action_chunk = torch.zeros(2, 3, 2)

    macro = _execute_macro_action(
        cfg=cfg,
        adapter=adapter,
        action_chunk=action_chunk,
        observations=observations,
        processor=processor,
        reward_strategy=NativeReward(),
        low=-np.ones(2, dtype=np.float32),
        high=np.ones(2, dtype=np.float32),
        device=torch.device("cpu"),
    )

    assert macro.durations.tolist() == [1, 3]
    assert macro.terminated.tolist() == [True, False]
    assert macro.episode_rewards.tolist() == pytest.approx([1.0, 3.0])
    assert macro.training_rewards.tolist() == pytest.approx([1.0, 1.0 + 0.99 + 0.99**2])
    assert first.step_calls == 1
    assert second.step_calls == 3


def test_policy_actions_apply_binary_gripper_after_denormalization() -> None:
    cfg = _config()
    cfg.rl.binary_gripper = True
    cfg.rl.binary_gripper_index = 1
    actions = torch.tensor([[[-2.0, -0.2], [2.0, 0.2]]])

    converted, clipped = _policy_actions_to_env(
        actions,
        cfg=cfg,
        processor=None,
        low=-np.ones(2, dtype=np.float32),
        high=np.ones(2, dtype=np.float32),
        device=torch.device("cpu"),
    )

    assert converted[0, :, 1].tolist() == [-1.0, 1.0]
    assert not clipped.any()


def test_episode_state_spans_updates_and_reset_seeds_are_unique(tmp_path: Path) -> None:
    cfg = _config()
    cfg.rl.total_updates = 2
    cfg.rl.rollout_decisions_per_update = 1
    scalar_adapter = _FakeAdapter(cfg, terminate_after=2)
    adapter = SerialBatchAdapter([scalar_adapter])

    with patch("mini_pi0.rl.checkpointing.save_reinflow_checkpoint"):
        summary = _run_reinflow_loop(
            cfg=cfg,
            run_dir=tmp_path,
            actor=_FakeRolloutActor(cfg),
            updater=_FakeUpdater(),
            adapter=adapter,
            device=torch.device("cpu"),
        )

    assert summary["latest"]["completed_episodes"] == 1
    assert summary["latest"]["reward_mean"] == pytest.approx(2.0)
    assert summary["latest"]["rollout_macro_reward_mean"] == pytest.approx(1.0)
    assert summary["latest"]["rollout_native_reward_per_primitive_step"] == pytest.approx(1.0)
    assert summary["latest"]["macro_duration_mean"] == pytest.approx(1.0)
    assert summary["latest"]["phase"] == "ppo"
    assert scalar_adapter.reset_seeds == [0, 1]


def test_update_logging_distinguishes_incomplete_episode_from_zero_reward(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = _config()
    cfg.rl.total_updates = 1
    cfg.rl.rollout_decisions_per_update = 1
    adapter = SerialBatchAdapter([_FakeAdapter(cfg, terminate_after=2)])

    with patch("mini_pi0.rl.checkpointing.save_reinflow_checkpoint"):
        summary = _run_reinflow_loop(
            cfg=cfg,
            run_dir=tmp_path,
            actor=_FakeRolloutActor(cfg),
            updater=_FakeUpdater(),
            adapter=adapter,
            device=torch.device("cpu"),
        )

    latest = summary["latest"]
    output = capsys.readouterr().out
    assert latest["completed_episodes"] == 0
    assert latest["completed_episode_return_mean"] is None
    assert latest["rollout_macro_reward_mean"] == pytest.approx(1.0)
    assert "ReinFlow update 0001/0001" in output
    assert "Phase" in output and "ppo" in output
    assert "Actor" in output and "ON" in output
    assert "Macro reward" in output and "1.000" in output
    assert "Episode return" in output and "n/a" in output
    assert "Completed episodes" in output and "0" in output
    table_lines = [line for line in output.splitlines() if line.startswith(("+", "|"))]
    assert table_lines
    assert max(map(len, table_lines)) == 79


def test_serial_batch_rejects_inconsistent_action_bounds() -> None:
    cfg = _config()
    first = _FakeAdapter(cfg)
    second = _FakeAdapter(cfg)
    second.action_spec = lambda: (-np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32))

    with pytest.raises(ValueError, match="bounds differ"):
        SerialBatchAdapter([first, second])


def test_deterministic_evaluation_uses_fixed_episode_seeds() -> None:
    cfg = _config()
    cfg.rl.eval_episodes = 2
    cfg.rl.eval_seed_start = 100
    cfg.simulator.horizon = 3
    fake = _FakeAdapter(cfg, terminate_after=2)

    with patch("mini_pi0.rl.runner._make_batched_adapter", return_value=SerialBatchAdapter([fake])):
        summary = _evaluate_actor(cfg, _FakeRolloutActor(cfg), torch.device("cpu"))

    assert summary["return_mean"] == pytest.approx(2.0)
    assert summary["episode_length_mean"] == pytest.approx(2.0)
    assert fake.reset_seeds == [100, 101]


def test_deterministic_evaluation_runs_vector_envs_and_exact_episode_count() -> None:
    cfg = _config()
    cfg.rl.eval_episodes = 5
    cfg.rl.eval_seed_start = 100
    cfg.rl.num_envs = 2
    cfg.simulator.horizon = 3
    first = _FakeAdapter(cfg, terminate_after=1)
    second = _FakeAdapter(cfg, terminate_after=2)
    actor = _FakeRolloutActor(cfg)

    with patch(
        "mini_pi0.rl.runner._make_batched_adapter",
        return_value=SerialBatchAdapter([first, second]),
    ):
        summary = _evaluate_actor(cfg, actor, torch.device("cpu"))

    assert summary["episodes"] == 5
    assert summary["num_envs"] == 2
    assert summary["returns"] == pytest.approx([1.0, 2.0, 1.0, 1.0, 2.0])
    assert summary["lengths"] == [1, 2, 1, 1, 2]
    assert first.reset_seeds == [100, 102, 103]
    assert second.reset_seeds == [101, 104]
    assert set(actor.deterministic_batch_sizes) == {2}


def test_deterministic_evaluation_uses_nominal_simulator_overrides() -> None:
    cfg = _config()
    cfg.rl.eval_episodes = 1
    cfg.rl.num_envs = 16
    cfg.rl.eval_num_envs = 1
    cfg.rl.eval_sim_backend = "physx_cpu"
    cfg.rl.eval_disable_domain_randomization = True
    cfg.simulator.env_kwargs = {
        "sim_backend": "physx_cuda",
        "domain_randomization": {"enabled": True, "profile": "strong"},
    }
    fake = _FakeAdapter(cfg, terminate_after=1)
    captured = []

    def make_adapter(eval_cfg):
        captured.append(eval_cfg)
        return SerialBatchAdapter([fake])

    with patch("mini_pi0.rl.runner._make_batched_adapter", side_effect=make_adapter):
        _evaluate_actor(cfg, _FakeRolloutActor(cfg), torch.device("cpu"))

    eval_cfg = captured[0]
    assert eval_cfg.rl.num_envs == 1
    assert eval_cfg.simulator.env_kwargs["sim_backend"] == "physx_cpu"
    assert not eval_cfg.simulator.env_kwargs["domain_randomization"]["enabled"]


def test_algorithm_modules_do_not_import_simulator_backends() -> None:
    source = inspect.getsource(flow_policy) + inspect.getsource(flow_ppo)

    assert "isaaclab" not in source.lower()
    assert "mani_skill" not in source.lower()
