"""Unit tests for vector-row canonicalization and contact extraction."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mini_pi0.config.io import load_config
from mini_pi0.sim.contact_features import collect_contact_features
from mini_pi0.sim.isaaclab_adapter import IsaacLabAdapter
from mini_pi0.sim.maniskill3_adapter import ManiSkill3Adapter


class _ForceEntity:
    def __init__(self, name: str, forces: np.ndarray) -> None:
        self.name = name
        self._forces = forces

    def get_net_contact_forces(self) -> np.ndarray:
        return self._forces


class _Robot:
    def __init__(self, link: _ForceEntity) -> None:
        self.links_map = {"panda_leftfinger": link}

    def get_qf(self) -> np.ndarray:
        return np.arange(18, dtype=np.float32).reshape(2, 9)

    def compute_passive_force(self) -> np.ndarray:
        return np.ones((2, 9), dtype=np.float32)


class _Scene:
    timestep = 0.01

    def get_pairwise_contact_forces(self, left: object, right: object) -> np.ndarray:
        del left, right
        return np.array([[1.0, 0.0, 0.0], [3.0, 4.0, 0.0]], dtype=np.float32)


class _MatrixPose:
    def __init__(self, matrices: torch.Tensor) -> None:
        self.matrices = matrices

    def to_transformation_matrix(self) -> torch.Tensor:
        return self.matrices


class _InversePose:
    def __init__(self, relative: torch.Tensor) -> None:
        self.relative = relative

    def __mul__(self, other: object):
        del other
        return SimpleNamespace(p=self.relative)


class _HolePose(_MatrixPose):
    def __init__(self, matrices: torch.Tensor, relative: torch.Tensor) -> None:
        super().__init__(matrices)
        self.relative = relative

    def inv(self) -> _InversePose:
        return _InversePose(self.relative)


class _PegAgent:
    def is_grasping(self, peg: object) -> torch.Tensor:
        del peg
        return torch.tensor([True, False])


class _PegScene:
    def get_pairwise_contact_forces(self, peg: object, box: object) -> torch.Tensor:
        del peg, box
        return torch.tensor([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])


def test_contact_features_select_requested_vector_environment() -> None:
    link = _ForceEntity("leftfinger", np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32))
    peg = _ForceEntity("peg", np.zeros((2, 3), dtype=np.float32))
    box = _ForceEntity("box", np.zeros((2, 3), dtype=np.float32))
    env = SimpleNamespace(
        agent=SimpleNamespace(robot=_Robot(link)),
        scene=_Scene(),
        peg=peg,
        box=box,
    )

    features = collect_contact_features(
        env,
        env_index=1,
        link_names=("panda_leftfinger",),
        object_names=("peg", "box"),
    )

    assert features["robot_qf"].tolist() == list(np.arange(9, 18, dtype=np.float32))
    assert features["leftfinger_force_norm"].item() == 2.0
    assert features["pair_leftfinger_peg_force_norm"].item() == 5.0


def test_peg_diagnostics_compute_all_vector_rows_in_one_batch() -> None:
    matrices = torch.eye(4).repeat(2, 1, 1)
    hole_matrices = matrices.clone()
    hole_matrices[1, :3, 0] = torch.tensor([0.0, 1.0, 0.0])
    relative = torch.tensor([[-0.01, 0.0, 0.0], [0.02, 0.03, 0.04]])
    env = SimpleNamespace(
        num_envs=2,
        peg=SimpleNamespace(pose=_MatrixPose(matrices)),
        box=object(),
        peg_head_pose=object(),
        box_hole_pose=_HolePose(hole_matrices, relative),
        peg_half_sizes=torch.tensor([[0.05, 0.01, 0.01], [0.06, 0.01, 0.01]]),
        agent=_PegAgent(),
        scene=_PegScene(),
    )
    adapter = ManiSkill3Adapter.__new__(ManiSkill3Adapter)
    adapter.env = SimpleNamespace(unwrapped=env)
    adapter.cfg = load_config(overrides=["simulator.control_freq=20"])
    adapter._previous_insertion_depth = np.array([0.005, 0.035], dtype=np.float32)
    adapter._jam_steps = np.array([0, 2], dtype=np.int64)

    diagnostics = adapter._peg_diagnostics_batch(update_jam=True)

    assert len(diagnostics) == 2
    assert diagnostics[0]["observation.state.insertion_depth"].item() == pytest.approx(0.005)
    assert diagnostics[1]["observation.state.peg_hole_alignment_error"].item() == pytest.approx(0.05)
    assert diagnostics[1]["observation.state.peg_axis_error"].item() == pytest.approx(np.pi / 2)
    assert diagnostics[1]["observation.state.peg_box_contact_force"].item() == pytest.approx(5.0)
    assert diagnostics[1]["observation.state.jam_duration"].item() == pytest.approx(0.15)


def test_isaac_canonical_observation_selects_requested_vector_row() -> None:
    cfg = load_config(
        overrides=[
            "robot.image_keys=['agentview_image']",
            "robot.state_keys=['robot0_joint_pos', 'observation.state.policy']",
            "robot.proprio_keys=['robot0_joint_pos', 'observation.state.policy']",
        ]
    )
    adapter = IsaacLabAdapter.__new__(IsaacLabAdapter)
    adapter.cfg = cfg
    adapter._image_keys = ["agentview_image"]
    adapter._state_keys = ["robot0_joint_pos", "observation.state.policy"]
    raw = {
        "policy": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "joint_pos": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        "rgb": np.stack(
            [np.zeros((4, 4, 3), dtype=np.uint8), np.full((4, 4, 3), 255, dtype=np.uint8)]
        ),
    }

    observation = adapter._canonical_obs(raw, env_index=1)

    assert observation["robot0_joint_pos"].tolist() == [7.0, 8.0]
    assert observation["observation.state.policy"].tolist() == [3.0, 4.0]
    assert int(observation["agentview_image"].mean()) == 255
