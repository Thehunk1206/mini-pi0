"""Replay ManiSkill trajectories and record compact contact feedback.

This utility augments a replayed ManiSkill HDF5 with low-dimensional physics
signals that are useful as policy proprioception. It intentionally records
stable summary signals instead of full contact manifolds.

Example:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/replay_maniskill_contacts.py \
      --traj-path demos/maniskill/PegInsertionSide-v1/motionplanning/trajectory.h5 \
      --base-hdf5 demos/maniskill/PegInsertionSide-v1/motionplanning/trajectory.rgbd.pd_ee_delta_pose.physx_cpu.h5 \
      --out-hdf5 demos/maniskill/PegInsertionSide-v1/motionplanning/trajectory.rgbd.pd_ee_delta_pose.contacts.h5 \
      --obs-mode rgbd \
      --control-mode pd_ee_delta_pose \
      --sim-backend physx_cpu \
      --use-first-env-state
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, Protocol

import gymnasium as gym
import h5py
import numpy as np
import torch

import mani_skill.envs  # noqa: F401 - registers ManiSkill environments.
import mini_pi0.sim.maniskill3_custom_env  # noqa: F401 - registers repo-local environments.
import mini_pi0.sim.maniskill3_peginsertion_env  # noqa: F401 - registers repo-local environments.
from mani_skill.trajectory import utils as trajectory_utils


DEFAULT_LINKS: Final[tuple[str, ...]] = (
    "panda_leftfinger",
    "panda_rightfinger",
    "panda_hand",
    "panda_hand_tcp",
    "panda_link8",
)
DEFAULT_OBJECTS: Final[tuple[str, ...]] = ("peg", "box")
CONTACT_GROUP: Final[str] = "extra_contact"


@dataclass(frozen=True)
class ReplayContactConfig:
    """Arguments for recording contact feedback from a ManiSkill replay."""

    traj_path: Path
    out_hdf5: Path
    base_hdf5: Path | None
    obs_mode: str
    control_mode: str
    sim_backend: str
    reward_mode: str | None
    render_mode: str
    links: tuple[str, ...]
    objects: tuple[str, ...]
    contact_threshold: float
    use_first_env_state: bool
    limit: int | None
    overwrite: bool

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> ReplayContactConfig:
        """Create a validated config from parsed CLI arguments.

        Args:
            args: Parsed command-line namespace.

        Returns:
            Replay configuration ready for execution.
        """

        reward_mode = None if str(args.reward_mode).lower() == "none" else str(args.reward_mode)
        return cls(
            traj_path=args.traj_path,
            out_hdf5=args.out_hdf5,
            base_hdf5=args.base_hdf5,
            obs_mode=str(args.obs_mode),
            control_mode=str(args.control_mode),
            sim_backend=str(args.sim_backend),
            reward_mode=reward_mode,
            render_mode=str(args.render_mode),
            links=tuple(item.strip() for item in str(args.links).split(",") if item.strip()),
            objects=tuple(item.strip() for item in str(args.objects).split(",") if item.strip()),
            contact_threshold=float(args.contact_threshold),
            use_first_env_state=bool(args.use_first_env_state),
            limit=args.limit,
            overwrite=bool(args.overwrite),
        )


class ContactLink(Protocol):
    """Minimal ManiSkill link interface used by the contact recorder."""

    def get_net_contact_forces(self) -> object:
        """Return net contact force for this link."""


class ContactRobot(Protocol):
    """Minimal ManiSkill robot interface used by the contact recorder."""

    links_map: Mapping[str, ContactLink]

    def get_qf(self) -> object:
        """Return articulation generalized force."""


@dataclass(frozen=True)
class ContactTarget:
    """One body or actor whose raw PhysX contact impulse is tracked."""

    key: str
    aliases: tuple[str, ...]

    def matches(self, body_name: str) -> bool:
        """Return whether a raw PhysX contact body belongs to this target.

        Args:
            body_name: Name of a contact body entity from SAPIEN/PhysX.

        Returns:
            True when the body name matches one of this target's aliases.
        """

        normalized = body_name.lower()
        return any(alias.lower() in normalized for alias in self.aliases)


@dataclass(frozen=True)
class TrajectoryEpisode:
    """Metadata needed to replay one source trajectory episode."""

    episode_id: int
    episode_seed: int | None
    reset_kwargs: Mapping[str, object]

    @property
    def traj_key(self) -> str:
        """Return the HDF5 trajectory group key."""

        return f"traj_{self.episode_id}"

    def reset_options(self) -> dict[str, object]:
        """Build reset kwargs for ManiSkill.

        Returns:
            A mutable reset-kwargs dictionary with the episode seed filled in
            when the metadata did not already provide one.
        """

        out = dict(self.reset_kwargs)
        if "seed" not in out and self.episode_seed is not None:
            out["seed"] = self.episode_seed
        return out


@dataclass(frozen=True)
class TrajectoryMetadata:
    """Parsed ManiSkill trajectory metadata."""

    env_id: str
    episodes: tuple[TrajectoryEpisode, ...]

    @classmethod
    def load(cls, path: Path, limit: int | None) -> TrajectoryMetadata:
        """Load and validate metadata from a ManiSkill JSON file.

        Args:
            path: Metadata JSON path next to ``trajectory.h5``.
            limit: Optional maximum number of episodes to return.

        Returns:
            Parsed trajectory metadata.

        Raises:
            ValueError: If the metadata shape is invalid.
        """

        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected object metadata in {path}")

        env_info = payload.get("env_info", {})
        if not isinstance(env_info, dict) or not env_info.get("env_id"):
            raise ValueError("Metadata is missing env_info.env_id")

        episodes_raw = payload.get("episodes", [])
        if not isinstance(episodes_raw, list):
            raise ValueError("Metadata field episodes must be a list")
        episodes = tuple(cls._parse_episode(item) for item in episodes_raw if isinstance(item, dict))
        if limit is not None:
            episodes = episodes[:limit]
        return cls(env_id=str(env_info["env_id"]), episodes=episodes)

    @staticmethod
    def _parse_episode(item: Mapping[str, object]) -> TrajectoryEpisode:
        """Parse one episode metadata object."""

        if "episode_id" not in item:
            raise ValueError("Episode metadata is missing episode_id")
        reset_kwargs = item.get("reset_kwargs", {})
        return TrajectoryEpisode(
            episode_id=int(item["episode_id"]),
            episode_seed=(int(item["episode_seed"]) if item.get("episode_seed") is not None else None),
            reset_kwargs=reset_kwargs if isinstance(reset_kwargs, Mapping) else {},
        )


@dataclass(frozen=True)
class ContactFrame:
    """One timestep of compact contact observations."""

    values: Mapping[str, np.ndarray]


class ContactSignalCollector:
    """Collect low-dimensional physics feedback from configured robot links."""

    def __init__(
        self,
        base_env: object,
        robot: ContactRobot,
        link_names: Sequence[str],
        object_names: Sequence[str],
        contact_threshold: float,
    ) -> None:
        """Initialize the collector.

        Args:
            base_env: Unwrapped ManiSkill environment.
            robot: ManiSkill robot articulation wrapper.
            link_names: Robot link names to query.
            object_names: Environment actor attributes to query, e.g. ``peg``.
            contact_threshold: Force norm threshold for binary contact flags.

        Raises:
            ValueError: If any requested link is missing.
        """

        self._base_env = base_env
        self._robot = robot
        self._contact_threshold = float(contact_threshold)
        self._links = self._resolve_links(robot, tuple(link_names))
        self._targets = self._build_targets(base_env, self._links, tuple(object_names))
        self._target_pairs = self._build_target_pairs(self._targets)

    def collect(self) -> ContactFrame:
        """Collect one timestep of contact signals.

        Returns:
            Contact frame containing generalized robot force and per-link force,
            force norm, and binary contact flag.
        """

        robot_qf = self._first_env_array(self._robot.get_qf())
        robot_passive_qf = self._first_env_array(self._robot.compute_passive_force())
        values: dict[str, np.ndarray] = {
            "robot_qf": robot_qf,
            "robot_arm_qf": robot_qf[:7],
            "robot_gripper_qf": robot_qf[7:],
            "robot_passive_qf": robot_passive_qf,
            "robot_arm_passive_qf": robot_passive_qf[:7],
            "robot_gripper_passive_qf": robot_passive_qf[7:],
        }
        raw = self._raw_contact_summary()
        timestep = float(getattr(self._base_env.scene, "timestep", 1.0))

        for target in self._targets:
            impulse = raw["target_impulses"][target.key]
            force = impulse / max(timestep, 1e-8)
            norm = np.asarray([np.linalg.norm(force)], dtype=np.float32)
            values[f"{target.key}_force"] = force.astype(np.float32)
            values[f"{target.key}_force_norm"] = norm
            values[f"{target.key}_contact"] = (norm > self._contact_threshold).astype(np.float32)
            values[f"{target.key}_contact_count"] = np.asarray(
                [raw["target_counts"][target.key]],
                dtype=np.float32,
            )

        for pair_key, impulse in raw["pair_impulses"].items():
            force = impulse / max(timestep, 1e-8)
            norm = np.asarray([np.linalg.norm(force)], dtype=np.float32)
            values[f"pair_{pair_key}_force"] = force.astype(np.float32)
            values[f"pair_{pair_key}_force_norm"] = norm
            values[f"pair_{pair_key}_contact"] = (norm > self._contact_threshold).astype(np.float32)
            values[f"pair_{pair_key}_contact_count"] = np.asarray(
                [raw["pair_counts"][pair_key]],
                dtype=np.float32,
            )
        return ContactFrame(values=values)

    @staticmethod
    def stack(frames: Sequence[ContactFrame]) -> dict[str, np.ndarray]:
        """Stack frames into HDF5-ready arrays.

        Args:
            frames: Ordered contact frames from one replayed episode.

        Returns:
            Mapping from signal name to ``[T, ...]`` float32 arrays.
        """

        if not frames:
            raise ValueError("Cannot stack an empty contact frame sequence.")
        keys = sorted(frames[0].values.keys())
        return {
            key: np.stack([frame.values[key] for frame in frames], axis=0).astype(np.float32)
            for key in keys
        }

    @staticmethod
    def _resolve_links(robot: ContactRobot, link_names: tuple[str, ...]) -> dict[str, ContactLink]:
        """Resolve link names against the robot link map."""

        missing = [name for name in link_names if name not in robot.links_map]
        if missing:
            available = ", ".join(sorted(robot.links_map.keys()))
            raise ValueError(f"Missing robot links {missing}. Available links: {available}")
        return {name: robot.links_map[name] for name in link_names}

    @classmethod
    def _build_targets(
        cls,
        base_env: object,
        links: Mapping[str, ContactLink],
        object_names: tuple[str, ...],
    ) -> tuple[ContactTarget, ...]:
        """Create raw-contact matchers for robot links and task actors."""

        targets: list[ContactTarget] = []
        for link_name, link in links.items():
            key = cls._signal_prefix(link_name)
            aliases = {link_name, key, cls._entity_name(link)}
            targets.append(ContactTarget(key=key, aliases=tuple(alias for alias in aliases if alias)))

        for object_name in object_names:
            actor = getattr(base_env, object_name, None)
            if actor is None:
                continue
            key = cls._signal_prefix(object_name)
            actor_name = str(getattr(actor, "name", ""))
            aliases = {object_name, key, actor_name, cls._entity_name(actor)}
            targets.append(ContactTarget(key=key, aliases=tuple(alias for alias in aliases if alias)))
        return tuple(targets)

    @staticmethod
    def _build_target_pairs(targets: Sequence[ContactTarget]) -> tuple[tuple[ContactTarget, ContactTarget], ...]:
        """Return all unique target pairs for pairwise contact summaries."""

        pairs: list[tuple[ContactTarget, ContactTarget]] = []
        for left_idx, left in enumerate(targets):
            for right in targets[left_idx + 1 :]:
                pairs.append((left, right))
        return tuple(pairs)

    def _raw_contact_summary(self) -> dict[str, dict[str, np.ndarray] | dict[str, float]]:
        """Aggregate raw PhysX contact impulses by target and target pair."""

        target_impulses = {target.key: np.zeros(3, dtype=np.float32) for target in self._targets}
        target_counts = {target.key: 0.0 for target in self._targets}
        pair_impulses = {
            f"{left.key}_{right.key}": np.zeros(3, dtype=np.float32)
            for left, right in self._target_pairs
        }
        pair_counts = {key: 0.0 for key in pair_impulses}

        px = getattr(self._base_env.scene, "px", None)
        if px is None or not hasattr(px, "get_contacts"):
            return {
                "target_impulses": target_impulses,
                "target_counts": target_counts,
                "pair_impulses": pair_impulses,
                "pair_counts": pair_counts,
            }

        for contact in px.get_contacts():
            body0 = self._contact_body_name(contact, 0)
            body1 = self._contact_body_name(contact, 1)
            impulse = self._contact_impulse(contact)
            for target in self._targets:
                if target.matches(body0):
                    target_impulses[target.key] += impulse
                    target_counts[target.key] += 1.0
                if target.matches(body1):
                    target_impulses[target.key] -= impulse
                    target_counts[target.key] += 1.0
            for left, right in self._target_pairs:
                pair_key = f"{left.key}_{right.key}"
                if left.matches(body0) and right.matches(body1):
                    pair_impulses[pair_key] += impulse
                    pair_counts[pair_key] += 1.0
                elif left.matches(body1) and right.matches(body0):
                    pair_impulses[pair_key] -= impulse
                    pair_counts[pair_key] += 1.0
        return {
            "target_impulses": target_impulses,
            "target_counts": target_counts,
            "pair_impulses": pair_impulses,
            "pair_counts": pair_counts,
        }

    @staticmethod
    def _contact_body_name(contact: object, index: int) -> str:
        """Return the entity name for one body in a raw PhysX contact."""

        body = contact.bodies[index]
        entity = getattr(body, "entity", None)
        return str(getattr(entity, "name", ""))

    @staticmethod
    def _contact_impulse(contact: object) -> np.ndarray:
        """Sum point impulses for one raw PhysX contact."""

        if not getattr(contact, "points", None):
            return np.zeros(3, dtype=np.float32)
        return np.sum([point.impulse for point in contact.points], axis=0).astype(np.float32)

    @staticmethod
    def _entity_name(obj: object) -> str:
        """Best-effort entity name for a ManiSkill wrapper object."""

        bodies = getattr(obj, "_bodies", None)
        if bodies:
            entity = getattr(bodies[0], "entity", None)
            return str(getattr(entity, "name", ""))
        return str(getattr(obj, "name", ""))

    @staticmethod
    def _first_env_array(value: object) -> np.ndarray:
        """Convert a ManiSkill tensor-like value to the first-env numpy row."""

        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)
        if array.ndim >= 2 and array.shape[0] == 1:
            array = array[0]
        return np.asarray(array, dtype=np.float32)

    @staticmethod
    def _signal_prefix(link_name: str) -> str:
        """Create a compact HDF5 key prefix from a robot link name."""

        return str(link_name).replace("panda_", "").replace("-", "_")


class ContactHdf5Writer:
    """Create and update the output HDF5 file with contact observations."""

    def __init__(self, cfg: ReplayContactConfig) -> None:
        """Initialize the writer.

        Args:
            cfg: Replay output configuration.
        """

        self._cfg = cfg

    def prepare(self) -> None:
        """Create the output HDF5 or copy the base replay HDF5."""

        self._cfg.out_hdf5.parent.mkdir(parents=True, exist_ok=True)
        if self._cfg.out_hdf5.exists():
            self._cfg.out_hdf5.unlink()
        if self._cfg.base_hdf5 is not None:
            shutil.copy2(self._cfg.base_hdf5, self._cfg.out_hdf5)
            return
        with h5py.File(self._cfg.out_hdf5, "w"):
            pass

    def write_episode(self, dst: h5py.File, traj_key: str, contact: Mapping[str, np.ndarray]) -> None:
        """Write one episode's contact arrays.

        Args:
            dst: Open output HDF5 file.
            traj_key: Source trajectory key, such as ``traj_0``.
            contact: Contact arrays returned by ``ContactSignalCollector.stack``.
        """

        traj = dst.require_group(traj_key)
        obs = traj.require_group("obs")
        if CONTACT_GROUP in obs:
            del obs[CONTACT_GROUP]
        group = obs.create_group(CONTACT_GROUP)
        for key, value in contact.items():
            group.create_dataset(key, data=value, compression="gzip")


class ManiSkillContactReplayApp:
    """Replay ManiSkill trajectories and record contact observations."""

    def __init__(self, cfg: ReplayContactConfig) -> None:
        """Initialize the replay application.

        Args:
            cfg: Replay and output configuration.
        """

        self._cfg = cfg
        self._writer = ContactHdf5Writer(cfg)

    def run(self) -> None:
        """Execute replay and contact recording."""

        self._validate_inputs()
        metadata = TrajectoryMetadata.load(self._cfg.traj_path.with_suffix(".json"), self._cfg.limit)
        self._writer.prepare()

        env = self._make_env(metadata.env_id)
        try:
            collector = ContactSignalCollector(
                base_env=env.unwrapped,
                robot=env.unwrapped.agent.robot,
                link_names=self._cfg.links,
                object_names=self._cfg.objects,
                contact_threshold=self._cfg.contact_threshold,
            )
            self._record_all(env=env, collector=collector, episodes=metadata.episodes)
        finally:
            env.close()

    def _validate_inputs(self) -> None:
        """Validate source and output paths."""

        if not self._cfg.traj_path.exists():
            raise FileNotFoundError(f"Source trajectory not found: {self._cfg.traj_path}")
        metadata_path = self._cfg.traj_path.with_suffix(".json")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Source metadata not found: {metadata_path}")
        if self._cfg.base_hdf5 is not None and not self._cfg.base_hdf5.exists():
            raise FileNotFoundError(f"Base replay HDF5 not found: {self._cfg.base_hdf5}")
        if self._cfg.out_hdf5.exists() and not self._cfg.overwrite:
            raise FileExistsError(f"Output already exists: {self._cfg.out_hdf5}. Pass --overwrite to replace it.")

    def _make_env(self, env_id: str) -> gym.Env:
        """Create the ManiSkill environment used for replay."""

        return gym.make(
            env_id,
            obs_mode=self._cfg.obs_mode,
            control_mode=self._cfg.control_mode,
            render_mode=self._cfg.render_mode,
            reward_mode=self._cfg.reward_mode,
            sim_backend=self._cfg.sim_backend,
            num_envs=1,
        )

    def _record_all(
        self,
        *,
        env: gym.Env,
        collector: ContactSignalCollector,
        episodes: Sequence[TrajectoryEpisode],
    ) -> None:
        """Record contact observations for all selected episodes."""

        with h5py.File(self._cfg.traj_path, "r") as src, h5py.File(self._cfg.out_hdf5, "a") as dst:
            for episode in episodes:
                if episode.traj_key not in src:
                    continue
                contact = self._replay_episode(env=env, collector=collector, traj=src[episode.traj_key], episode=episode)
                self._writer.write_episode(dst=dst, traj_key=episode.traj_key, contact=contact)

    def _replay_episode(
        self,
        *,
        env: gym.Env,
        collector: ContactSignalCollector,
        traj: h5py.Group,
        episode: TrajectoryEpisode,
    ) -> dict[str, np.ndarray]:
        """Replay one source episode and return stacked contact arrays."""

        env.reset(**episode.reset_options())
        if self._cfg.use_first_env_state:
            states = trajectory_utils.dict_to_list_of_dicts(traj["env_states"])
            env.unwrapped.set_state_dict(states[0])

        frames = [collector.collect()]
        actions = np.asarray(traj["actions"], dtype=np.float32)
        for action in actions:
            env.step(action)
            frames.append(collector.collect())
        return ContactSignalCollector.stack(frames)


class ReplayContactCli:
    """Command-line interface for the contact replay tool."""

    @staticmethod
    def parse() -> ReplayContactConfig:
        """Parse command-line arguments into a replay config."""

        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--traj-path", required=True, type=Path, help="Source ManiSkill trajectory.h5 path.")
        parser.add_argument("--out-hdf5", required=True, type=Path, help="Output HDF5 path with contact observations.")
        parser.add_argument(
            "--base-hdf5",
            type=Path,
            default=None,
            help="Optional replayed RGB/RGBD HDF5 to copy and augment.",
        )
        parser.add_argument("--obs-mode", default="rgbd", help="ManiSkill observation mode for replay.")
        parser.add_argument("--control-mode", default="pd_joint_pos", help="ManiSkill control mode for replay.")
        parser.add_argument("--sim-backend", default="physx_cpu", help="ManiSkill simulation backend.")
        parser.add_argument("--reward-mode", default="dense", help="Reward mode passed to gym.make; use 'none' to omit.")
        parser.add_argument("--render-mode", default="rgb_array", help="Render mode passed to gym.make.")
        parser.add_argument("--links", default=",".join(DEFAULT_LINKS), help="Comma-separated robot link names to record.")
        parser.add_argument("--objects", default=",".join(DEFAULT_OBJECTS), help="Comma-separated task actor attributes to record.")
        parser.add_argument("--contact-threshold", type=float, default=1e-3, help="Force norm threshold for contact flags.")
        parser.add_argument("--use-first-env-state", action="store_true", help="Set each replay to its recorded initial state.")
        parser.add_argument("--limit", type=int, default=None, help="Maximum number of episodes to replay.")
        parser.add_argument("--overwrite", action="store_true", help="Replace out-hdf5 if it already exists.")
        return ReplayContactConfig.from_args(parser.parse_args())

def main() -> int:
    """Run the CLI entrypoint."""

    cfg = ReplayContactCli.parse()
    ManiSkillContactReplayApp(cfg).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
