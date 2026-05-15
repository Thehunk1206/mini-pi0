"""Tests for converting ManiSkill replay trajectories to robomimic-style HDF5."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from mini_pi0.dataset.episodes import load_episodes_robomimic
from mini_pi0.dataset.maniskill_convert import (
    ManiSkillConversionConfig,
    ManiSkillConversionError,
    convert_maniskill_trajectory_to_robomimic,
)


def _write_metadata(path: Path, *, success: bool = True) -> None:
    """Write minimal ManiSkill JSON metadata."""

    payload = {
        "env_info": {
            "env_id": "StackCube-v1",
            "env_kwargs": {
                "obs_mode": "rgbd",
                "control_mode": "pd_ee_delta_pose",
            },
        },
        "episodes": [
            {
                "episode_id": 0,
                "episode_seed": 11,
                "elapsed_steps": 4,
                "control_mode": "pd_ee_delta_pose",
                "success": success,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_replayed_maniskill_hdf5(path: Path, *, include_obs: bool = True) -> None:
    """Write a tiny ManiSkill-like trajectory file."""

    steps = 4
    obs_steps = steps + 1
    with h5py.File(path, "w") as h5:
        traj = h5.create_group("traj_0")
        traj.create_dataset("actions", data=np.arange(steps * 7, dtype=np.float32).reshape(steps, 7))
        traj.create_dataset("success", data=np.asarray([False, False, False, True], dtype=np.bool_))
        traj.create_dataset("terminated", data=np.asarray([False, False, False, True], dtype=np.bool_))

        env_states = traj.create_group("env_states")
        actors = env_states.create_group("actors")
        actors.create_dataset("cubeA", data=np.ones((obs_steps, 13), dtype=np.float32))
        actors.create_dataset("cubeB", data=np.full((obs_steps, 13), 2.0, dtype=np.float32))

        if not include_obs:
            return

        obs = traj.create_group("obs")
        sensor_data = obs.create_group("sensor_data")
        for camera_uid in ("base_camera", "hand_camera"):
            camera = sensor_data.create_group(camera_uid)
            camera.create_dataset(
                "rgb",
                data=np.random.randint(0, 255, size=(obs_steps, 8, 8, 3), dtype=np.uint8),
            )
        extra = obs.create_group("extra")
        tcp_pose = np.zeros((obs_steps, 7), dtype=np.float32)
        tcp_pose[:, 3] = 1.0
        extra.create_dataset("tcp_pose", data=tcp_pose)
        agent = obs.create_group("agent")
        agent.create_dataset("qpos", data=np.ones((obs_steps, 9), dtype=np.float32))
        agent.create_dataset("qvel", data=np.full((obs_steps, 9), 0.5, dtype=np.float32))


def test_convert_maniskill_trajectory_to_robomimic_writes_trainable_dataset(tmp_path: Path) -> None:
    # Arrange
    input_hdf5 = tmp_path / "trajectory.rgbd.pd_ee_delta_pose.physx_cpu.h5"
    output_hdf5 = tmp_path / "stackcube.hdf5"
    _write_replayed_maniskill_hdf5(input_hdf5)
    _write_metadata(input_hdf5.with_suffix(".json"))

    # Act
    summary = convert_maniskill_trajectory_to_robomimic(
        ManiSkillConversionConfig(
            input_hdf5=str(input_hdf5),
            output_hdf5=str(output_hdf5),
            overwrite=True,
        )
    )

    # Assert
    assert summary["episodes"] == 1
    assert summary["total_samples"] == 4
    episodes = load_episodes_robomimic(
        hdf5_path=str(output_hdf5),
        image_keys=["agentview_image", "robot0_eye_in_hand_image"],
        proprio_keys=["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
        data_group="data",
        fallback_image_hw=(8, 8),
    )
    assert len(episodes) == 1
    assert episodes[0].actions.shape == (4, 7)
    assert episodes[0].obs[0]["agentview_image"].shape == (8, 8, 3)
    assert episodes[0].obs[0]["robot0_eef_pos"].shape == (3,)
    assert episodes[0].obs[0]["robot0_eef_quat"].shape == (4,)
    assert episodes[0].obs[0]["robot0_gripper_qpos"].shape == (2,)


def test_convert_maniskill_trajectory_writes_joint_velocity_state(tmp_path: Path) -> None:
    # Arrange
    input_hdf5 = tmp_path / "trajectory.rgbd.pd_ee_delta_pose.physx_cpu.h5"
    output_hdf5 = tmp_path / "stackcube.hdf5"
    _write_replayed_maniskill_hdf5(input_hdf5)
    _write_metadata(input_hdf5.with_suffix(".json"))

    # Act
    summary = convert_maniskill_trajectory_to_robomimic(
        ManiSkillConversionConfig(
            input_hdf5=str(input_hdf5),
            output_hdf5=str(output_hdf5),
            state_keys=("robot0_eef_pos", "robot0_joint_vel"),
            overwrite=True,
        )
    )

    # Assert
    assert summary["state_keys"] == ["robot0_eef_pos", "robot0_joint_vel"]
    episodes = load_episodes_robomimic(
        hdf5_path=str(output_hdf5),
        image_keys=["agentview_image", "robot0_eye_in_hand_image"],
        proprio_keys=["robot0_eef_pos", "robot0_joint_vel"],
        data_group="data",
        fallback_image_hw=(8, 8),
    )
    assert episodes[0].obs[0]["robot0_joint_vel"].shape == (9,)
    np.testing.assert_allclose(episodes[0].obs[0]["robot0_joint_vel"], np.full((9,), 0.5, dtype=np.float32))


def test_convert_maniskill_trajectory_raises_when_observations_missing(tmp_path: Path) -> None:
    # Arrange
    input_hdf5 = tmp_path / "trajectory.none.pd_ee_delta_pose.physx_cuda.h5"
    output_hdf5 = tmp_path / "stackcube.hdf5"
    _write_replayed_maniskill_hdf5(input_hdf5, include_obs=False)
    _write_metadata(input_hdf5.with_suffix(".json"))

    # Act / Assert
    with pytest.raises(ManiSkillConversionError, match="missing obs"):
        convert_maniskill_trajectory_to_robomimic(
            ManiSkillConversionConfig(
                input_hdf5=str(input_hdf5),
                output_hdf5=str(output_hdf5),
                overwrite=True,
            )
        )
