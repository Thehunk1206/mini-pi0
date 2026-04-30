"""Shared data structures and utilities for ManiSkill dataset collection.

This module centralizes:
- episode buffering
- tensor/array normalization helpers
- canonical observation extraction
- HDF5 episode writing and summary statistics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class EpisodeBuffer:
    """In-memory container for one collected trajectory."""

    obs: list[dict[str, np.ndarray]]
    actions: list[np.ndarray]
    rewards: list[float]
    dones: list[int]
    info_rows: list[dict[str, Any]]


def to_numpy(value: Any) -> np.ndarray:
    """Convert numpy/torch/scalar-like values into a numpy array.

    Args:
        value: Input value (numpy array, torch tensor, scalar, or sequence).

    Returns:
        Numpy representation of `value`.
    """
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def summarize_collection_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-episode rows into a compact collection report.

    Args:
        rows: List of per-episode rows emitted by collector orchestrator.

    Returns:
        Summary dictionary with success/partial/fail rates, trajectory lengths,
        mean success fraction, and object-count histogram.
    """
    if not rows:
        return {
            "episodes": 0,
            "success_rate": 0.0,
            "partial_rate": 0.0,
            "fail_rate": 0.0,
            "mean_success_fraction": 0.0,
            "episode_len_mean": 0.0,
            "episode_len_min": 0,
            "episode_len_max": 0,
            "object_count_histogram": {},
        }
    lens = np.asarray([int(r["num_samples"]) for r in rows], dtype=np.int32)
    succ = np.asarray([int(r["success_bool"]) for r in rows], dtype=np.int32)
    frac = np.asarray([float(r["final_success_fraction"]) for r in rows], dtype=np.float32)
    total_obj = np.asarray([int(r["total_objects"]) for r in rows], dtype=np.int32)
    partial = (frac > 1e-6) & (frac < 1.0 - 1e-6)
    fail = frac <= 1e-6
    hist: dict[str, int] = {}
    for v in total_obj.tolist():
        key = str(int(v))
        hist[key] = hist.get(key, 0) + 1
    return {
        "episodes": int(len(rows)),
        "success_rate": float(np.mean(succ)),
        "partial_rate": float(np.mean(partial.astype(np.float32))),
        "fail_rate": float(np.mean(fail.astype(np.float32))),
        "mean_success_fraction": float(np.mean(frac)),
        "episode_len_mean": float(np.mean(lens)),
        "episode_len_min": int(np.min(lens)),
        "episode_len_max": int(np.max(lens)),
        "object_count_histogram": hist,
    }


def write_episode(group, demo_idx: int, ep: EpisodeBuffer, final_info: dict[str, Any]) -> None:
    """Write one `EpisodeBuffer` into robomimic-style HDF5 demo layout.

    Args:
        group: Open HDF5 group (typically `/data`).
        demo_idx: Sequential demo index used for key naming (`demo_{idx}`).
        ep: In-memory trajectory payload.
        final_info: Final per-episode metrics and metadata.
    """
    key = f"demo_{demo_idx}"
    if key in group:
        del group[key]
    demo = group.create_group(key)

    t = len(ep.actions)
    demo.attrs["num_samples"] = int(t)
    demo.attrs["success_bool"] = int(bool(final_info.get("success", False)))
    demo.attrs["placed_count"] = int(final_info.get("placed_count", 0))
    demo.attrs["total_objects"] = int(final_info.get("total_objects", 0))
    demo.attrs["final_success_fraction"] = float(final_info.get("success_fraction", 0.0))
    demo.attrs["collector_type"] = str(final_info.get("collector_type", "scripted"))

    demo.create_dataset("actions", data=np.asarray(ep.actions, dtype=np.float32))
    demo.create_dataset("rewards", data=np.asarray(ep.rewards, dtype=np.float32))
    demo.create_dataset("dones", data=np.asarray(ep.dones, dtype=np.int32))

    info_grp = demo.create_group("info")
    for name in (
        "success_fraction",
        "reward_total",
        "reward_progress",
        "reward_place",
        "reward_terminal",
        "reward_shaping",
        "reward_penalties",
        "reward_step",
    ):
        info_grp.create_dataset(name, data=np.asarray([float(row.get(name, 0.0)) for row in ep.info_rows], dtype=np.float32))

    obs_grp = demo.create_group("obs")
    keys = sorted(ep.obs[0].keys()) if ep.obs else []
    for k in keys:
        arr = np.stack([np.asarray(o[k]) for o in ep.obs], axis=0)
        obs_grp.create_dataset(k, data=arr)


def canonical_obs_batch_from_raw_env(env, image_keys: list[str], state_keys: list[str], raw_obs: Any) -> list[dict[str, np.ndarray]]:
    """Build canonical per-env observations from ManiSkill raw env tensors.

    Args:
        env: ManiSkill environment instance (possibly vectorized).
        image_keys: Image keys expected by downstream pipeline.
        state_keys: State keys expected by downstream pipeline.
        raw_obs: Raw observation object returned by env reset/step.

    Returns:
        List of per-env observation dicts in canonical mini-pi0 format.
    """
    uw = env.unwrapped
    num_envs = int(uw.num_envs)
    tcp_p = to_numpy(uw.agent.tcp.pose.p).astype(np.float32)
    tcp_q = to_numpy(uw.agent.tcp.pose.q).astype(np.float32)
    qpos = to_numpy(uw.agent.robot.get_qpos()).astype(np.float32)
    obj_pos = to_numpy(uw._get_object_pos_tensor()).reshape(num_envs, -1).astype(np.float32)
    active = to_numpy(uw._active_object_mask).astype(np.float32)
    placed = to_numpy(uw._placed_mask).astype(np.float32)
    frac = to_numpy(uw._last_success_fraction).astype(np.float32)

    frame = None
    if isinstance(raw_obs, dict):
        try:
            rgb = raw_obs["sensor_data"]["base_camera"]["rgb"]
            arr = to_numpy(rgb)
            if arr.ndim == 3:
                arr = arr[None, ...]
            frame = np.clip(arr, 0, 255).astype(np.uint8)
        except Exception:
            frame = None
    if frame is None:
        frame = np.zeros((num_envs, 128, 128, 3), dtype=np.uint8)

    out_batch: list[dict[str, np.ndarray]] = []
    for i in range(num_envs):
        default_state = {
            "robot0_eef_pos": tcp_p[i],
            "observation.state.eef_pos": tcp_p[i],
            "robot0_eef_quat": tcp_q[i],
            "observation.state.eef_quat": tcp_q[i],
            "robot0_gripper_qpos": qpos[i][-2:] if qpos.shape[1] >= 2 else qpos[i],
            "observation.state.tool": qpos[i][-2:] if qpos.shape[1] >= 2 else qpos[i],
            "observation.state.object": obj_pos[i],
            "observation.state.object_mask": active[i],
            "observation.state.placed_mask": placed[i],
            "observation.state.task_progress": np.array([frac[i]], dtype=np.float32),
        }
        obs_i: dict[str, np.ndarray] = {}
        for key in image_keys:
            obs_i[key] = frame[i][..., :3]
        for key in state_keys:
            obs_i[key] = np.asarray(default_state.get(key, np.zeros((1,), dtype=np.float32)), dtype=np.float32)
        for key in ("observation.state.object", "observation.state.object_mask", "observation.state.placed_mask", "observation.state.task_progress"):
            obs_i[key] = np.asarray(default_state[key], dtype=np.float32)
        out_batch.append(obs_i)
    return out_batch


def normalize_info_batched(info: dict[str, Any], num_envs: int) -> list[dict[str, Any]]:
    """Convert batched env info dict into `num_envs` JSON-safe row dicts.

    Args:
        info: Batched info dict from vectorized env step.
        num_envs: Number of environment rows expected.

    Returns:
        List of `num_envs` dict objects with scalar/list values.
    """
    out = [dict() for _ in range(num_envs)]
    for k, v in info.items():
        arr = to_numpy(v)
        if arr.shape == ():
            for i in range(num_envs):
                out[i][k] = float(arr)
            continue
        if arr.shape[0] != num_envs:
            for i in range(num_envs):
                out[i][k] = arr.tolist()
            continue
        for i in range(num_envs):
            item = arr[i]
            out[i][k] = float(item) if np.isscalar(item) else item.tolist()
    return out


def canonical_obs_from_raw_env(env, image_keys: list[str], state_keys: list[str], last_raw_obs: Any) -> dict[str, np.ndarray]:
    """Single-env convenience wrapper for canonical observation extraction.

    Args:
        env: ManiSkill environment instance.
        image_keys: Image keys expected by downstream pipeline.
        state_keys: State keys expected by downstream pipeline.
        last_raw_obs: Raw observation object returned by env reset/step.

    Returns:
        Canonical single-env observation dict.
    """
    return canonical_obs_batch_from_raw_env(env, image_keys=image_keys, state_keys=state_keys, raw_obs=last_raw_obs)[0]


def normalize_info(info: dict[str, Any]) -> dict[str, Any]:
    """Convert env info values into JSON/HDF5-friendly scalar/list types.

    Args:
        info: Raw info dict from single env step/evaluate.

    Returns:
        Dict with scalar/list values compatible with JSON/HDF5 serialization.
    """
    out: dict[str, Any] = {}
    for k, v in info.items():
        arr = to_numpy(v)
        if arr.shape == ():
            out[k] = float(arr)
        elif arr.ndim > 0 and arr.shape[0] == 1:
            item = arr[0]
            out[k] = float(item) if np.isscalar(item) else item.tolist()
        else:
            out[k] = arr.tolist() if isinstance(arr, np.ndarray) else v
    return out
