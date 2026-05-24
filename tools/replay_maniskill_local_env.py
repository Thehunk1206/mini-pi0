"""Replay ManiSkill trajectories after registering repo-local environments.

ManiSkill's replay CLI reads the environment id from the trajectory JSON. This
wrapper imports mini-pi0 environments first and can stage a JSON copy with a
different env id, which is useful when replaying built-in demos through a
repo-local sensor variant.

Example:
    .venv/bin/python tools/replay_maniskill_local_env.py \
      --env-id MiniPi0PegInsertionSide-v1 \
      --work-dir tmp/peginsertion_holecam_replay \
      --traj-path demos/maniskill/PegInsertionSide-v1/motionplanning/trajectory.h5 \
      --obs-mode rgbd \
      --target-control-mode pd_ee_delta_pose \
      --save-traj \
      --sim-backend physx_cpu
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import mini_pi0.sim.maniskill3_custom_env  # noqa: F401
import mini_pi0.sim.maniskill3_peginsertion_env  # noqa: F401
from mani_skill.trajectory import replay_trajectory


class ReplaySetupError(RuntimeError):
    """Raised when local replay staging cannot be prepared."""


def main(argv: list[str] | None = None) -> int:
    """Run ManiSkill replay with repo-local environments registered."""

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    local_args, replay_argv = _parse_local_args(raw_argv)
    if local_args.env_id or local_args.robot_uids:
        replay_argv = _stage_replay_inputs(
            replay_argv=replay_argv,
            env_id=local_args.env_id,
            robot_uids=local_args.robot_uids,
            work_dir=local_args.work_dir,
            copy_h5=local_args.copy_h5,
        )
    replay_trajectory.main(replay_trajectory.parse_args(replay_argv))
    return 0


def _parse_local_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--robot-uids", default=None)
    parser.add_argument("--work-dir", type=Path, default=Path("tmp/maniskill_local_replay"))
    parser.add_argument("--copy-h5", action="store_true")
    return parser.parse_known_args(argv)


def _stage_replay_inputs(
    replay_argv: list[str],
    env_id: str | None,
    robot_uids: str | None,
    work_dir: Path,
    copy_h5: bool,
) -> list[str]:
    traj_path = _extract_traj_path(replay_argv)
    source_h5 = Path(traj_path)
    source_json = source_h5.with_suffix(".json")
    if not source_h5.exists():
        raise FileNotFoundError(f"Trajectory HDF5 not found: {source_h5}")
    if not source_json.exists():
        raise FileNotFoundError(f"Trajectory JSON not found: {source_json}")

    work_dir.mkdir(parents=True, exist_ok=True)
    staged_h5 = work_dir / source_h5.name
    staged_json = work_dir / source_json.name
    _stage_h5(source=source_h5, destination=staged_h5, copy_h5=copy_h5)
    _write_replay_json(source=source_json, destination=staged_json, env_id=env_id, robot_uids=robot_uids)
    return _replace_traj_path(replay_argv, staged_h5)


def _extract_traj_path(argv: list[str]) -> str:
    for idx, value in enumerate(argv):
        if value == "--traj-path" and idx + 1 < len(argv):
            return argv[idx + 1]
        if value.startswith("--traj-path="):
            return value.split("=", maxsplit=1)[1]
    raise ReplaySetupError("Missing required ManiSkill replay argument: --traj-path")


def _replace_traj_path(argv: list[str], traj_path: Path) -> list[str]:
    out = list(argv)
    for idx, value in enumerate(out):
        if value == "--traj-path" and idx + 1 < len(out):
            out[idx + 1] = str(traj_path)
            return out
        if value.startswith("--traj-path="):
            out[idx] = f"--traj-path={traj_path}"
            return out
    raise ReplaySetupError("Missing required ManiSkill replay argument: --traj-path")


def _stage_h5(source: Path, destination: Path, copy_h5: bool) -> None:
    if destination.exists():
        return
    if copy_h5:
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source.resolve())


def _write_replay_json(source: Path, destination: Path, env_id: str | None, robot_uids: str | None) -> None:
    with source.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or not isinstance(payload.get("env_info"), dict):
        raise ReplaySetupError(f"Invalid ManiSkill trajectory metadata: {source}")
    if env_id:
        payload["env_info"]["env_id"] = env_id
    if robot_uids:
        env_kwargs = payload["env_info"].setdefault("env_kwargs", {})
        if not isinstance(env_kwargs, dict):
            raise ReplaySetupError(f"Invalid env_kwargs in ManiSkill trajectory metadata: {source}")
        env_kwargs["robot_uids"] = robot_uids
    with destination.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())
