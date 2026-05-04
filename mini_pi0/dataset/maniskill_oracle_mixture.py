"""Success-only oracle mixture collection for ManiSkill scripted datasets."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from mini_pi0.config.schema import RootConfig, effective_image_keys, effective_state_keys
from mini_pi0.dataset.maniskill_collect import _episode_stats_row, _next_demo_index
from mini_pi0.dataset.maniskill_collectors.backends import (
    VectorizedOracleSettings,
    _project_scripted_action,
    _resolve_scripted_control_mode,
    collect_vectorized_scripted_episodes,
)
from mini_pi0.dataset.maniskill_collectors.common import EpisodeBuffer, summarize_collection_stats, write_episode
from mini_pi0.dataset.maniskill_collectors.policy import OracleOptions, OracleProfile, ScriptedMultiObjectOracle
from mini_pi0.sim.registry import make_sim_adapter
from mani_skill.utils.structs import Pose


@dataclass(frozen=True)
class ProfileSpec:
    """Internal collection settings for one oracle mixture profile."""

    profile: OracleProfile
    target_episodes: int
    options: OracleOptions
    perturbation_types: tuple[str, ...]


@dataclass(frozen=True)
class DifficultySpec:
    """Internal perturbation/noise magnitudes for a named difficulty."""

    action_noise_std: float
    action_noise_clip: float
    speed_scale: float
    grasp_pose_noise_xy: float
    grasp_pose_noise_z: float
    grasp_angle_jitter_deg: float
    displacement_scale: float
    max_retries: int


DIFFICULTIES: dict[str, DifficultySpec] = {
    "safe": DifficultySpec(0.018, 0.07, 0.85, 0.006, 0.003, 12.0, 0.75, 4),
    "balanced": DifficultySpec(0.030, 0.10, 0.75, 0.010, 0.005, 20.0, 1.0, 4),
    "aggressive": DifficultySpec(0.045, 0.14, 0.65, 0.014, 0.007, 30.0, 1.25, 5),
}


def allocate_profile_counts(total: int, mix: dict[str, float]) -> dict[str, int]:
    """Convert profile ratios into integer episode counts."""
    keys = ["core", "recovery", "suboptimal"]
    weights = np.asarray([max(0.0, float(mix.get(k, 0.0))) for k in keys], dtype=np.float64)
    if float(weights.sum()) <= 0.0:
        weights = np.asarray([0.65, 0.25, 0.10], dtype=np.float64)
    weights = weights / weights.sum()
    raw = weights * int(total)
    counts = np.floor(raw).astype(np.int64)
    remainder = int(total) - int(counts.sum())
    for idx in np.argsort(raw - counts)[::-1][:remainder]:
        counts[idx] += 1
    return {k: int(v) for k, v in zip(keys, counts)}


def _profile_specs(total: int, mix: dict[str, float], difficulty: str) -> list[ProfileSpec]:
    diff = DIFFICULTIES.get(str(difficulty), DIFFICULTIES["balanced"])
    counts = allocate_profile_counts(total, mix)
    return [
        ProfileSpec(OracleProfile.CORE, counts["core"], OracleOptions(profile=OracleProfile.CORE), ("none",)),
        ProfileSpec(
            OracleProfile.RECOVERY,
            counts["recovery"],
            OracleOptions(
                profile=OracleProfile.RECOVERY,
                grasp_pose_noise_xy=diff.grasp_pose_noise_xy,
                grasp_pose_noise_z=diff.grasp_pose_noise_z,
                allow_regrasp=True,
            ),
            ("object_displace_2cm", "object_displace_5cm", "midtask_nudge", "bowl_escape", "grasp_noise"),
        ),
        ProfileSpec(
            OracleProfile.SUBOPTIMAL,
            counts["suboptimal"],
            OracleOptions(
                profile=OracleProfile.SUBOPTIMAL,
                action_noise_std=diff.action_noise_std,
                action_noise_clip=diff.action_noise_clip,
                speed_scale=diff.speed_scale,
                grasp_angle_jitter_deg=diff.grasp_angle_jitter_deg,
                allow_regrasp=True,
            ),
            ("action_noise",),
        ),
    ]


def _episode_quality_ok(
    final_info: dict[str, Any],
    num_samples: int,
    accepted_lengths: list[int],
    *,
    reject_long: bool,
    max_retries: int,
) -> tuple[bool, str]:
    if not bool(final_info.get("success", False)) or float(final_info.get("success_fraction", 0.0)) < 1.0 - 1e-6:
        return False, "not_success"
    if int(final_info.get("oracle_retry_count", 0)) > int(max_retries):
        return False, "too_many_retries"
    if int(final_info.get("oracle_phase_timeout_count", 0)) > int(max_retries):
        return False, "phase_timeouts"
    if reject_long:
        limit = 550.0 if not accepted_lengths else min(550.0, 2.0 * float(np.median(np.asarray(accepted_lengths))))
        if int(num_samples) > int(limit):
            return False, "too_long"
    return True, "accepted"


def _first_unplaced_target(obs: dict[str, np.ndarray]) -> int | None:
    mask = np.asarray(obs.get("observation.state.object_mask", []), dtype=np.float32).reshape(-1)
    placed = np.asarray(obs.get("observation.state.placed_mask", np.zeros_like(mask)), dtype=np.float32).reshape(-1)
    for idx in range(min(len(mask), len(placed))):
        if mask[idx] > 0.5 and placed[idx] < 0.5:
            return idx
    return None


def _nudge_object(adapter: Any, obj_idx: int, magnitude: float, rng: np.random.Generator) -> None:
    uw = adapter.unwrapped
    if obj_idx < 0 or obj_idx >= len(uw.objects):
        return
    actor = uw.objects[obj_idx]
    pose_p = actor.pose.p.clone()
    pose_q = actor.pose.q.clone()
    theta = float(rng.uniform(0.0, 2.0 * np.pi))
    delta = torch.tensor([magnitude * np.cos(theta), magnitude * np.sin(theta), 0.0], dtype=torch.float32, device=pose_p.device)
    pose_p[0, :3] = pose_p[0, :3] + delta
    pose_p[0, 0] = torch.clamp(pose_p[0, 0], -0.10, 0.18)
    pose_p[0, 1] = torch.clamp(pose_p[0, 1], -0.36, 0.36)
    actor.set_pose(Pose.create_from_pq(p=pose_p, q=pose_q))
    try:
        actor.set_linear_velocity(torch.zeros_like(pose_p))
        actor.set_angular_velocity(torch.zeros_like(pose_p))
    except Exception:
        pass


def _move_object_out_of_bowl(adapter: Any, obj_idx: int, rng: np.random.Generator) -> float:
    """Place an active object visibly outside the source bowl for recovery demos."""
    uw = adapter.unwrapped
    if obj_idx < 0 or obj_idx >= len(uw.objects):
        return 0.0
    actor = uw.objects[obj_idx]
    pose_p = actor.pose.p.clone()
    pose_q = actor.pose.q.clone()
    old_xy = pose_p[0, :2].clone()
    bowl_center = np.asarray(getattr(uw, "bowl_center_np", [0.05, -0.28, 0.0]), dtype=np.float32)
    x_offset = float(rng.uniform(-0.055, 0.055))
    target_xy = torch.tensor(
        [float(bowl_center[0] + x_offset), float(bowl_center[1] + 0.18)],
        dtype=torch.float32,
        device=pose_p.device,
    )
    pose_p[0, 0] = target_xy[0]
    pose_p[0, 1] = target_xy[1]
    pose_p[0, 2] = torch.clamp(pose_p[0, 2], 0.035, 0.055)
    actor.set_pose(Pose.create_from_pq(p=pose_p, q=pose_q))
    try:
        actor.set_linear_velocity(torch.zeros_like(pose_p))
        actor.set_angular_velocity(torch.zeros_like(pose_p))
    except Exception:
        pass
    return float(torch.linalg.norm(target_xy - old_xy).item())


def _collect_profile_trial(
    cfg: RootConfig,
    spec: ProfileSpec,
    *,
    seed: int,
    max_steps: int,
    image_keys: list[str],
    state_keys: list[str],
    difficulty: str,
    force_perturbation_type: str | None = None,
) -> tuple[EpisodeBuffer, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    run_cfg = copy.deepcopy(cfg)
    run_cfg.experiment.seed = int(seed)
    run_cfg.simulator.controller = _resolve_scripted_control_mode(run_cfg)
    run_cfg.simulator.env_kwargs = dict(run_cfg.simulator.env_kwargs or {})
    run_cfg.simulator.env_kwargs.pop("scripted_control_mode", None)
    run_cfg.simulator.env_kwargs["control_mode"] = str(run_cfg.simulator.controller)
    if spec.profile == OracleProfile.CORE:
        run_cfg.simulator.env_kwargs["robot_init_qpos_noise"] = max(float(run_cfg.simulator.env_kwargs.get("robot_init_qpos_noise", 0.0)), 0.015)

    adapter = make_sim_adapter(run_cfg)
    obs = adapter.reset(seed=seed)
    tray_center = np.asarray(run_cfg.simulator.env_kwargs.get("tray_center", [0.62, 0.0, 0.0]), dtype=np.float32)
    policy = ScriptedMultiObjectOracle(tray_center=tray_center, options=spec.options, rng=rng)
    policy.reset()
    low, high = adapter.action_spec()

    perturbation_type = str(force_perturbation_type or rng.choice(spec.perturbation_types))
    trigger_step = int(rng.choice([55, 110, 165])) if spec.profile == OracleProfile.RECOVERY else -1
    perturbation_applied = perturbation_type in {"none", "grasp_noise", "action_noise"}
    perturbation_magnitude = 0.0
    if perturbation_type == "object_displace_2cm":
        perturbation_magnitude = 0.02 * DIFFICULTIES.get(difficulty, DIFFICULTIES["balanced"]).displacement_scale
    elif perturbation_type in {"object_displace_5cm", "midtask_nudge"}:
        perturbation_magnitude = 0.05 * DIFFICULTIES.get(difficulty, DIFFICULTIES["balanced"]).displacement_scale
    elif perturbation_type == "bowl_escape":
        perturbation_magnitude = 0.18

    buf = EpisodeBuffer(obs=[], actions=[], rewards=[], dones=[], info_rows=[])
    final_info: dict[str, Any] = {"success": False, "success_fraction": 0.0, "placed_count": 0, "total_objects": 0}

    for step_idx in range(int(max_steps)):
        buf.obs.append({
            k: np.asarray(obs[k])
            for k in set(image_keys + state_keys + [
                "observation.state.object",
                "observation.state.object_mask",
                "observation.state.placed_mask",
                "observation.state.place_targets",
                "observation.state.task_progress",
            ])
            if k in obs
        })
        if not perturbation_applied and step_idx >= trigger_step:
            target_idx = policy.target_idx if policy.target_idx is not None else _first_unplaced_target(obs)
            if target_idx is not None:
                if perturbation_type == "bowl_escape":
                    perturbation_magnitude = _move_object_out_of_bowl(adapter, int(target_idx), rng)
                else:
                    _nudge_object(adapter, int(target_idx), float(perturbation_magnitude), rng)
            perturbation_applied = True
        action7 = policy.act(obs)
        action = _project_scripted_action(action7, low, high)
        step = adapter.step(action)
        buf.actions.append(action.astype(np.float32))
        buf.rewards.append(float(step.reward))
        done = bool(step.done or adapter.check_success(step.info, step.obs))
        buf.dones.append(1 if done else 0)
        buf.info_rows.append(dict(step.info))
        obs = step.obs
        final_info = dict(step.info)
        if done:
            break

    adapter.close()
    final_info.update(policy.telemetry())
    final_info.update(
        {
            "profile_type": spec.profile.value,
            "difficulty": str(difficulty),
            "seed": int(seed),
            "perturbation_type": perturbation_type,
            "perturbation_magnitude": float(perturbation_magnitude),
        }
    )
    return buf, final_info


def collect_maniskill_oracle_mixture(cfg: RootConfig, *, overwrite: bool = False) -> dict[str, Any]:
    """Collect a success-only oracle mixture dataset from simplified config."""
    dc = cfg.dataset_collection
    out_path = Path(dc.output_hdf5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output HDF5 already exists: {out_path}. Pass --overwrite to replace it.")

    difficulty = str(dc.difficulty).lower().strip() or "balanced"
    force_perturbation_type = str(dc.force_perturbation_type).strip() if dc.force_perturbation_type else None
    diff = DIFFICULTIES.get(difficulty, DIFFICULTIES["balanced"])
    specs = _profile_specs(int(dc.total_episodes), dict(dc.mix), difficulty)
    image_keys = effective_image_keys(cfg.robot)
    state_keys = effective_state_keys(cfg.robot)
    num_envs = int(max(1, getattr(dc, "num_envs", 1)))
    summary: dict[str, Any] = {
        "path": str(out_path),
        "summary_path": str(out_path.with_name(out_path.stem + "_summary.json")),
        "difficulty": difficulty,
        "num_envs": int(num_envs),
        "profiles": {},
    }
    rows: list[dict[str, Any]] = []
    saved_total = 0
    seed_cursor = int(cfg.experiment.seed)
    t0 = time.perf_counter()

    with h5py.File(out_path, "w") as h5:
        data_group = h5.require_group("data")
        data_group.attrs["total"] = 0
        data_group.attrs["env_args"] = json.dumps(
            {
                "env_name": cfg.simulator.task,
                "env_type": "maniskill3_custom",
                "env_kwargs": dict(cfg.simulator.env_kwargs),
                "collector_name": "mini_pi0_oracle_mixture",
                "dataset_collection": dict(dc.__dict__),
                "num_envs": int(num_envs),
            }
        )
        demo_idx = _next_demo_index(data_group)
        for spec in specs:
            if spec.target_episodes <= 0:
                continue
            accepted_lengths: list[int] = []
            rejections: dict[str, int] = {}
            trials = 0
            saved = 0
            max_trials = max(10, spec.target_episodes * 20)
            print(f"[mixture] profile_start name={spec.profile.value} target={spec.target_episodes}", flush=True)
            while saved < spec.target_episodes and trials < max_trials:
                trials += 1
                seed = seed_cursor
                seed_cursor += num_envs
                if num_envs > 1:
                    run_cfg = copy.deepcopy(cfg)
                    run_cfg.experiment.seed = int(seed)
                    run_cfg.simulator.controller = _resolve_scripted_control_mode(run_cfg)
                    run_cfg.simulator.env_kwargs = dict(run_cfg.simulator.env_kwargs or {})
                    run_cfg.simulator.env_kwargs.pop("scripted_control_mode", None)
                    run_cfg.simulator.env_kwargs["control_mode"] = str(run_cfg.simulator.controller)
                    batch = collect_vectorized_scripted_episodes(
                        run_cfg,
                        image_keys=image_keys,
                        state_keys=state_keys,
                        num_envs=num_envs,
                        episodes_target=int(spec.target_episodes) - int(saved),
                        max_steps=int(dc.max_steps),
                        only_success=bool(dc.only_success),
                        oracle_settings=VectorizedOracleSettings(
                            options=spec.options,
                            profile_type=spec.profile.value,
                            difficulty=difficulty,
                            perturbation_types=spec.perturbation_types,
                            force_perturbation_type=force_perturbation_type if spec.profile == OracleProfile.RECOVERY else None,
                            displacement_scale=float(diff.displacement_scale),
                            seed=int(seed),
                        ),
                    )
                else:
                    batch = [
                        _collect_profile_trial(
                            cfg,
                            spec,
                            seed=seed,
                            max_steps=int(dc.max_steps),
                            image_keys=image_keys,
                            state_keys=state_keys,
                            difficulty=difficulty,
                            force_perturbation_type=force_perturbation_type if spec.profile == OracleProfile.RECOVERY else None,
                        )
                    ]
                accepted_this_trial = 0
                for ep, final_info in batch:
                    ok, reason = _episode_quality_ok(
                        final_info,
                        len(ep.actions),
                        accepted_lengths,
                        reject_long=bool(dc.reject_long_episodes),
                        max_retries=int(diff.max_retries),
                    )
                    if bool(dc.only_success) and not ok:
                        rejections[reason] = rejections.get(reason, 0) + 1
                        continue
                    final_info["collector_type"] = "scripted_oracle_mixture"
                    write_episode(data_group, demo_idx, ep, final_info)
                    data_group.attrs["total"] = int(data_group.attrs.get("total", 0)) + len(ep.actions)
                    rows.append(_episode_stats_row(len(ep.actions), final_info))
                    accepted_lengths.append(len(ep.actions))
                    demo_idx += 1
                    saved += 1
                    saved_total += 1
                    accepted_this_trial += 1
                    print(
                        "[mixture] accepted "
                        f"profile={spec.profile.value} saved={saved}/{spec.target_episodes} "
                        f"len={len(ep.actions)} seed={final_info.get('seed', seed)} "
                        f"perturbation={final_info.get('perturbation_type')}",
                        flush=True,
                    )
                    if saved >= spec.target_episodes:
                        break
                if num_envs > 1 and accepted_this_trial <= 0:
                    rejections["empty_vectorized_batch"] = rejections.get("empty_vectorized_batch", 0) + 1
            summary["profiles"][spec.profile.value] = {
                "target": int(spec.target_episodes),
                "saved": int(saved),
                "trials": int(trials),
                "rejections": rejections,
                "median_length": float(np.median(np.asarray(accepted_lengths))) if accepted_lengths else 0.0,
            }

    summary.update(
        {
            "episodes_saved": int(saved_total),
            "total_requested": int(dc.total_episodes),
            "elapsed_s": float(time.perf_counter() - t0),
            "stats": summarize_collection_stats(rows),
        }
    )
    summary_path = Path(summary["summary_path"])
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary
