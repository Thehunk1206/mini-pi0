"""Convert ManiSkill replay trajectories into mini-pi0 training datasets.

Key exports:
- :class:`ManiSkillConversionConfig` describes one conversion job.
- :func:`convert_maniskill_trajectory_to_robomimic` writes robomimic-style HDF5.
- :func:`convert_maniskill_trajectories_to_robomimic` merges replay shards into
  one robomimic-style HDF5.

Example:
    >>> cfg = ManiSkillConversionConfig(
    ...     input_hdf5="demos/StackCube-v1/motionplanning/trajectory.rgbd.pd_joint_pos.physx_cpu.h5",
    ...     output_hdf5="data/robomimic/maniskill/stackcube/ph/rgbd.hdf5",
    ... )
    >>> summary = convert_maniskill_trajectory_to_robomimic(cfg)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import h5py
import numpy as np


DEFAULT_IMAGE_CAMERA_MAP: Final[dict[str, str]] = {
    "agentview_image": "base_camera",
    "robot0_eye_in_hand_image": "hand_camera",
}
DEFAULT_STATE_KEYS: Final[tuple[str, ...]] = (
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)


class ManiSkillConversionError(RuntimeError):
    """Raised when a ManiSkill trajectory cannot be converted safely."""


@dataclass(frozen=True)
class ManiSkillConversionConfig:
    """Configuration for converting one ManiSkill HDF5 trajectory.

    Attributes:
        input_hdf5: Source ManiSkill trajectory file. It must include an ``obs``
            group, which usually means the raw demo was replayed with
            ``mani_skill.trajectory.replay_trajectory --save-traj -o rgbd``.
        output_hdf5: Destination robomimic-style HDF5 file.
        input_json: Optional source JSON metadata path. Defaults to the input
            HDF5 path with ``.json`` suffix.
        data_group: Top-level output group containing ``demo_N`` episodes.
        image_camera_map: Output image-key to ManiSkill camera UID mapping.
        state_keys: State keys to write under each output ``obs`` group.
        limit: Optional maximum number of trajectories to convert.
        only_success: Skip episodes that are not marked successful.
        overwrite: Replace an existing output file.
    """

    input_hdf5: str
    output_hdf5: str
    input_json: str | None = None
    data_group: str = "data"
    image_camera_map: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_IMAGE_CAMERA_MAP))
    state_keys: tuple[str, ...] = DEFAULT_STATE_KEYS
    limit: int | None = None
    only_success: bool = True
    overwrite: bool = False


@dataclass(frozen=True)
class ManiSkillMultiConversionConfig:
    """Configuration for converting one or more ManiSkill replay files.

    Attributes:
        input_hdf5s: Source ManiSkill trajectory files. Vectorized replay may
            write multiple shard files for a single replay command.
        output_hdf5: Destination robomimic-style HDF5 file.
        input_jsons: Optional JSON metadata paths matching ``input_hdf5s``.
            Missing entries default to the HDF5 path with ``.json`` suffix.
        data_group: Top-level output group containing ``demo_N`` episodes.
        image_camera_map: Output image-key to ManiSkill camera UID mapping.
        state_keys: State keys to write under each output ``obs`` group.
        limit: Optional maximum number of converted trajectories across all
            input files.
        only_success: Skip episodes that are not marked successful.
        overwrite: Replace an existing output file.
    """

    input_hdf5s: tuple[str, ...]
    output_hdf5: str
    input_jsons: tuple[str | None, ...] | None = None
    data_group: str = "data"
    image_camera_map: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_IMAGE_CAMERA_MAP))
    state_keys: tuple[str, ...] = DEFAULT_STATE_KEYS
    limit: int | None = None
    only_success: bool = True
    overwrite: bool = False


@dataclass(frozen=True)
class _EpisodeMeta:
    """JSON metadata for one ManiSkill trajectory episode."""

    episode_id: int
    success: bool | None
    elapsed_steps: int | None
    raw: dict[str, object]


@dataclass(frozen=True)
class _TrajectorySource:
    """Resolved ManiSkill replay source and metadata."""

    input_hdf5: Path
    input_json: Path
    metadata: dict[str, object]
    episode_meta: dict[int, _EpisodeMeta]


def convert_maniskill_trajectory_to_robomimic(cfg: ManiSkillConversionConfig) -> dict[str, object]:
    """Convert a ManiSkill trajectory into robomimic-style HDF5.

    Args:
        cfg: Conversion configuration.

    Returns:
        Summary dictionary containing source, destination, episode count, and
        sample count.

    Raises:
        FileNotFoundError: If required input files are missing.
        FileExistsError: If the output exists and ``overwrite`` is false.
        ManiSkillConversionError: If the source file lacks required observation
            data or no episodes are converted.
    """

    summary = convert_maniskill_trajectories_to_robomimic(
        ManiSkillMultiConversionConfig(
            input_hdf5s=(cfg.input_hdf5,),
            output_hdf5=cfg.output_hdf5,
            input_jsons=(cfg.input_json,),
            data_group=cfg.data_group,
            image_camera_map=cfg.image_camera_map,
            state_keys=cfg.state_keys,
            limit=cfg.limit,
            only_success=cfg.only_success,
            overwrite=cfg.overwrite,
        )
    )
    summary["input_hdf5"] = summary["input_hdf5s"][0]
    summary["input_json"] = summary["input_jsons"][0]
    return summary


def convert_maniskill_trajectories_to_robomimic(cfg: ManiSkillMultiConversionConfig) -> dict[str, object]:
    """Convert one or more ManiSkill replay files into one robomimic HDF5.

    Args:
        cfg: Conversion configuration. Multiple input files are appended into a
            single output group with contiguous ``demo_N`` numbering.

    Returns:
        Summary dictionary containing source files, destination, episode count,
        and sample count.

    Raises:
        FileNotFoundError: If any required source file is missing.
        FileExistsError: If the output exists and ``overwrite`` is false.
        ManiSkillConversionError: If source metadata is invalid, observations
            are missing, or no episodes are converted.
    """

    output_hdf5 = Path(cfg.output_hdf5)
    output_hdf5.parent.mkdir(parents=True, exist_ok=True)
    sources = _resolve_sources(input_hdf5s=cfg.input_hdf5s, input_jsons=cfg.input_jsons)
    _validate_output_path(output_hdf5=output_hdf5, overwrite=cfg.overwrite)

    rows: list[dict[str, object]] = []
    mode = "w" if cfg.overwrite else "x"
    remaining = cfg.limit
    with h5py.File(output_hdf5, mode) as dst:
        data = dst.create_group(cfg.data_group)
        data.attrs["source_format"] = "maniskill_trajectory"
        data.attrs["source_hdf5"] = str(sources[0].input_hdf5)
        data.attrs["source_json"] = str(sources[0].input_json)
        data.attrs["source_hdf5s"] = json.dumps([str(source.input_hdf5) for source in sources])
        data.attrs["source_jsons"] = json.dumps([str(source.input_json) for source in sources])
        data.attrs["env_args"] = json.dumps(sources[0].metadata.get("env_info", {}), sort_keys=True)

        for source in sources:
            if remaining is not None and remaining <= 0:
                break
            with h5py.File(source.input_hdf5, "r") as src:
                for traj_key in _selected_trajectory_keys(src, remaining):
                    traj = src[traj_key]
                    meta = source.episode_meta.get(_trajectory_episode_id(traj_key))
                    if cfg.only_success and meta is not None and meta.success is False:
                        continue
                    row = _write_demo(
                        data_group=data,
                        demo_idx=len(rows),
                        traj_key=traj_key,
                        traj=traj,
                        meta=meta,
                        image_camera_map=cfg.image_camera_map,
                        state_keys=cfg.state_keys,
                    )
                    row["source_hdf5"] = str(source.input_hdf5)
                    rows.append(row)
                    if remaining is not None:
                        remaining -= 1
                        if remaining <= 0:
                            break

    if not rows:
        output_hdf5.unlink(missing_ok=True)
        raise ManiSkillConversionError(
            f"No episodes converted from {', '.join(str(source.input_hdf5) for source in sources)}. "
            "Check --only_success, --limit, and whether trajectory groups exist."
        )

    total_samples = int(sum(int(row["num_samples"]) for row in rows))
    return {
        "input_hdf5s": [str(source.input_hdf5) for source in sources],
        "input_jsons": [str(source.input_json) for source in sources],
        "output_hdf5": str(output_hdf5),
        "data_group": cfg.data_group,
        "episodes": len(rows),
        "total_samples": total_samples,
        "image_keys": sorted(cfg.image_camera_map.keys()),
        "state_keys": list(cfg.state_keys),
    }


def _resolve_sources(
    input_hdf5s: tuple[str, ...],
    input_jsons: tuple[str | None, ...] | None,
) -> tuple[_TrajectorySource, ...]:
    """Resolve and validate replay source files."""

    if not input_hdf5s:
        raise ManiSkillConversionError("At least one ManiSkill HDF5 input is required.")
    json_paths = input_jsons or tuple(None for _ in input_hdf5s)
    if len(json_paths) != len(input_hdf5s):
        raise ManiSkillConversionError(f"Expected {len(input_hdf5s)} JSON metadata paths, got {len(json_paths)}.")

    sources: list[_TrajectorySource] = []
    for input_hdf5_raw, input_json_raw in zip(input_hdf5s, json_paths, strict=True):
        input_hdf5 = Path(input_hdf5_raw)
        input_json = Path(input_json_raw) if input_json_raw else input_hdf5.with_suffix(".json")
        if not input_hdf5.exists():
            raise FileNotFoundError(f"ManiSkill HDF5 not found: {input_hdf5}")
        if not input_json.exists():
            raise FileNotFoundError(f"ManiSkill JSON metadata not found: {input_json}")
        metadata = _load_metadata(input_json)
        sources.append(
            _TrajectorySource(
                input_hdf5=input_hdf5,
                input_json=input_json,
                metadata=metadata,
                episode_meta=_episode_meta_by_id(metadata),
            )
        )
    return tuple(sources)


def _validate_output_path(output_hdf5: Path, overwrite: bool) -> None:
    """Validate the destination path before opening HDF5 files."""

    if output_hdf5.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_hdf5}. Pass overwrite=True to replace it.")


def _load_metadata(path: Path) -> dict[str, object]:
    """Load ManiSkill JSON metadata."""

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ManiSkillConversionError(f"Expected object metadata in {path}")
    return payload


def _episode_meta_by_id(metadata: dict[str, object]) -> dict[int, _EpisodeMeta]:
    """Index JSON episode metadata by episode id."""

    episodes = metadata.get("episodes", [])
    if not isinstance(episodes, list):
        return {}

    out: dict[int, _EpisodeMeta] = {}
    for item in episodes:
        if not isinstance(item, dict) or "episode_id" not in item:
            continue
        raw_item = dict(item)
        episode_id = int(raw_item["episode_id"])
        success = raw_item.get("success")
        out[episode_id] = _EpisodeMeta(
            episode_id=episode_id,
            success=(bool(success) if success is not None else None),
            elapsed_steps=(int(raw_item["elapsed_steps"]) if raw_item.get("elapsed_steps") is not None else None),
            raw=raw_item,
        )
    return out


def _selected_trajectory_keys(src: h5py.File, limit: int | None) -> list[str]:
    """Return numerically sorted ManiSkill trajectory group keys."""

    keys = sorted((key for key in src.keys() if key.startswith("traj_")), key=_trajectory_episode_id)
    if limit is not None:
        return keys[: max(0, int(limit))]
    return keys


def _trajectory_episode_id(traj_key: str) -> int:
    """Parse the numeric episode id from a ``traj_N`` key."""

    try:
        return int(str(traj_key).split("_", maxsplit=1)[1])
    except (IndexError, ValueError) as exc:
        raise ManiSkillConversionError(f"Invalid ManiSkill trajectory key: {traj_key}") from exc


def _write_demo(
    data_group: h5py.Group,
    demo_idx: int,
    traj_key: str,
    traj: h5py.Group,
    meta: _EpisodeMeta | None,
    image_camera_map: dict[str, str],
    state_keys: tuple[str, ...],
) -> dict[str, object]:
    """Write one converted demo group."""

    if "actions" not in traj:
        raise ManiSkillConversionError(f"{traj_key} is missing actions")
    if "obs" not in traj:
        raise ManiSkillConversionError(
            f"{traj_key} is missing obs. Replay the raw ManiSkill file with --save-traj and an obs mode such as rgbd."
        )

    actions = np.asarray(traj["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] == 0:
        raise ManiSkillConversionError(f"{traj_key}/actions must have shape [T, A], got {actions.shape}")
    length = _common_length(traj=traj, actions=actions, image_camera_map=image_camera_map)
    if length <= 0:
        raise ManiSkillConversionError(f"{traj_key} has no aligned action/observation frames")

    demo = data_group.create_group(f"demo_{demo_idx}")
    demo.attrs["num_samples"] = int(length)
    demo.attrs["source_traj_key"] = traj_key
    demo.attrs["episode_id"] = int(meta.episode_id if meta is not None else _trajectory_episode_id(traj_key))
    demo.attrs["success_bool"] = int(_final_bool(traj, "success", default=meta.success if meta is not None else None))
    if meta is not None and meta.elapsed_steps is not None:
        demo.attrs["elapsed_steps"] = int(meta.elapsed_steps)

    demo.create_dataset("actions", data=actions[:length].astype(np.float32))
    demo.create_dataset("dones", data=_dones_for(traj, length))
    demo.create_dataset("rewards", data=_rewards_for(traj, length))

    obs = demo.create_group("obs")
    _write_images(obs=obs, traj=traj, image_camera_map=image_camera_map, length=length)
    _write_states(obs=obs, traj=traj, state_keys=state_keys, length=length)
    _write_object_state(obs=obs, traj=traj, length=length)
    _write_extra_contact(obs=obs, traj=traj, length=length)

    return {
        "demo_key": demo.name,
        "num_samples": int(length),
        "success": bool(demo.attrs["success_bool"]),
    }


def _common_length(traj: h5py.Group, actions: np.ndarray, image_camera_map: dict[str, str]) -> int:
    """Calculate the shared valid length across required arrays."""

    lengths = [int(actions.shape[0])]
    obs = traj["obs"]
    sensor_data = _required_group(obs, "sensor_data")
    for camera_uid in image_camera_map.values():
        camera = _required_group(sensor_data, camera_uid)
        if "rgb" not in camera:
            raise ManiSkillConversionError(f"obs/sensor_data/{camera_uid} is missing rgb")
        lengths.append(int(camera["rgb"].shape[0]))
    extra = _required_group(obs, "extra")
    agent = _required_group(obs, "agent")
    if "tcp_pose" not in extra:
        raise ManiSkillConversionError("obs/extra is missing tcp_pose")
    if "qpos" not in agent:
        raise ManiSkillConversionError("obs/agent is missing qpos")
    lengths.extend([int(extra["tcp_pose"].shape[0]), int(agent["qpos"].shape[0])])
    if "qvel" in agent:
        lengths.append(int(agent["qvel"].shape[0]))
    return min(lengths)


def _required_group(parent: h5py.Group, key: str) -> h5py.Group:
    """Return a required child group."""

    if key not in parent or not isinstance(parent[key], h5py.Group):
        raise ManiSkillConversionError(f"Missing required group: {parent.name}/{key}")
    return parent[key]


def _write_images(obs: h5py.Group, traj: h5py.Group, image_camera_map: dict[str, str], length: int) -> None:
    """Write mapped RGB camera frames."""

    sensor_data = traj["obs"]["sensor_data"]
    for output_key, camera_uid in image_camera_map.items():
        rgb = np.asarray(sensor_data[camera_uid]["rgb"][:length])
        if rgb.ndim != 4 or rgb.shape[-1] < 3:
            raise ManiSkillConversionError(f"{camera_uid}/rgb must have shape [T, H, W, C], got {rgb.shape}")
        obs.create_dataset(output_key, data=np.clip(rgb[..., :3], 0, 255).astype(np.uint8))


def _write_states(obs: h5py.Group, traj: h5py.Group, state_keys: tuple[str, ...], length: int) -> None:
    """Write canonical proprioceptive state datasets."""

    tcp_pose = np.asarray(traj["obs"]["extra"]["tcp_pose"][:length], dtype=np.float32)
    qpos = np.asarray(traj["obs"]["agent"]["qpos"][:length], dtype=np.float32)
    agent = traj["obs"]["agent"]
    if "qvel" in agent:
        qvel = np.asarray(agent["qvel"][:length], dtype=np.float32)
    else:
        qvel = np.zeros_like(qpos, dtype=np.float32)
    values = {
        "robot0_eef_pos": tcp_pose[:, :3],
        "robot0_eef_quat": tcp_pose[:, 3:7],
        "robot0_gripper_qpos": qpos[:, -2:] if qpos.shape[1] >= 2 else qpos,
        "robot0_joint_vel": qvel,
        "observation.state.eef_pos": tcp_pose[:, :3],
        "observation.state.eef_quat": tcp_pose[:, 3:7],
        "observation.state.tool": qpos[:, -2:] if qpos.shape[1] >= 2 else qpos,
        "observation.state.joint_vel": qvel,
    }
    for key in state_keys:
        if key not in values:
            raise ManiSkillConversionError(f"Unsupported state key for ManiSkill conversion: {key}")
        obs.create_dataset(key, data=np.asarray(values[key], dtype=np.float32))


def _write_object_state(obs: h5py.Group, traj: h5py.Group, length: int) -> None:
    """Write a compact object-state vector when actor states are available."""

    if "env_states" not in traj or "actors" not in traj["env_states"]:
        return
    actors = traj["env_states"]["actors"]
    actor_keys = sorted(key for key in actors.keys() if key.startswith("cube"))
    if not actor_keys:
        return
    parts = [np.asarray(actors[key][:length, :7], dtype=np.float32) for key in actor_keys]
    obs.create_dataset("object-state", data=np.concatenate(parts, axis=1))
    obs.create_dataset("observation.state.object", data=np.concatenate(parts, axis=1))


def _write_extra_contact(obs: h5py.Group, traj: h5py.Group, length: int) -> None:
    """Copy compact contact observations into the robomimic obs group."""

    source = traj.get("obs", {}).get("extra_contact") if isinstance(traj.get("obs"), h5py.Group) else None
    if not isinstance(source, h5py.Group):
        return
    for key, dataset in source.items():
        if not isinstance(dataset, h5py.Dataset):
            continue
        values = np.asarray(dataset[:length], dtype=np.float32)
        obs.create_dataset(str(key), data=values)


def _dones_for(traj: h5py.Group, length: int) -> np.ndarray:
    """Build robomimic-style done flags."""

    if "terminated" in traj:
        dones = np.asarray(traj["terminated"][:length], dtype=np.int32)
    else:
        dones = np.zeros((length,), dtype=np.int32)
    dones[-1] = 1
    return dones


def _rewards_for(traj: h5py.Group, length: int) -> np.ndarray:
    """Read rewards or return zero rewards when absent."""

    if "rewards" in traj:
        return np.asarray(traj["rewards"][:length], dtype=np.float32)
    return np.zeros((length,), dtype=np.float32)


def _final_bool(traj: h5py.Group, key: str, default: bool | None) -> bool:
    """Read the final boolean value from a trajectory dataset."""

    if key in traj and len(traj[key]) > 0:
        return bool(np.asarray(traj[key])[-1])
    return bool(default) if default is not None else False
