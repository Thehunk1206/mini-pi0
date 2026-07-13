"""Unit tests for vector-row canonicalization and contact extraction."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mini_pi0.config.io import load_config
from mini_pi0.sim.contact_features import collect_contact_features
from mini_pi0.sim.isaaclab_adapter import IsaacLabAdapter


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
