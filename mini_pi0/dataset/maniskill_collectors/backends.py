"""Low-level collector backends for ManiSkill data generation.

Backends here execute rollout mechanics (scripted or mplib planning) while
task plugins decide which backend to call.
"""

from __future__ import annotations

import copy
import subprocess
import sys
import time
from typing import Any

import gymnasium as gym
import numpy as np
import sapien

from mini_pi0.config.schema import RootConfig
from mini_pi0.sim.registry import make_sim_adapter
from .common import (
    EpisodeBuffer,
    canonical_obs_batch_from_raw_env,
    canonical_obs_from_raw_env,
    normalize_info,
    normalize_info_batched,
    to_numpy,
)
from .policy import ScriptedMultiObjectOracle


def mplib_runtime_check() -> bool:
    """Return True when mplib planner can be constructed in current runtime.

    This runs a tiny subprocess probe so a native mplib initialization failure
    cannot crash the current Python process.
    """
    code = (
        "import mplib\n"
        "urdf='"
        ".venv/lib/python3.11/site-packages/mani_skill/assets/robots/panda/panda_v2.urdf"
        "'\n"
        "mplib.Planner(urdf=urdf, move_group='panda_hand_tcp')\n"
        "print('ok')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    return proc.returncode == 0 and "ok" in proc.stdout


def collect_single_scripted_episode(
    ep_cfg: RootConfig,
    *,
    image_keys: list[str],
    state_keys: list[str],
    max_steps: int,
) -> tuple[EpisodeBuffer, dict[str, Any]]:
    """Collect one episode with the scripted oracle through sim adapter API.

    Args:
        ep_cfg: Runtime config for this episode seed.
        image_keys: Canonical image keys to capture.
        state_keys: Canonical state keys to capture.
        max_steps: Maximum environment steps before forced stop.

    Returns:
        `(EpisodeBuffer, final_info)` for one trajectory.
    """
    tray_center = np.asarray(ep_cfg.simulator.env_kwargs.get("tray_center", [0.62, 0.0, 0.0]), dtype=np.float32)
    policy = ScriptedMultiObjectOracle(tray_center=tray_center)
    adapter = make_sim_adapter(ep_cfg)
    obs = adapter.reset(seed=ep_cfg.experiment.seed)
    policy.reset()

    buf = EpisodeBuffer(obs=[], actions=[], rewards=[], dones=[], info_rows=[])
    final_info: dict[str, Any] = {"success": False, "success_fraction": 0.0, "placed_count": 0, "total_objects": 0}

    for _ in range(int(max_steps)):
        buf.obs.append({
            k: np.asarray(obs[k])
            for k in set(image_keys + state_keys + [
                "observation.state.object",
                "observation.state.object_mask",
                "observation.state.placed_mask",
                "observation.state.task_progress",
            ])
            if k in obs
        })
        action = policy.act(obs)
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
    return buf, final_info


def collect_vectorized_scripted_episodes(
    ep_cfg: RootConfig,
    *,
    image_keys: list[str],
    state_keys: list[str],
    num_envs: int,
    episodes_target: int,
    max_steps: int,
    only_success: bool,
) -> list[tuple[EpisodeBuffer, dict[str, Any]]]:
    """Collect multiple episodes using ManiSkill vectorized env stepping.

    Args:
        ep_cfg: Runtime config for current collection trial.
        image_keys: Canonical image keys to capture.
        state_keys: Canonical state keys to capture.
        num_envs: Number of vectorized environments to step in parallel.
        episodes_target: Target number of finalized episodes to return.
        max_steps: Maximum rollout steps before stopping this trial.
        only_success: Whether to retain only successful completed episodes.

    Returns:
        List of finalized `(EpisodeBuffer, final_info)` episodes.
    """
    import mini_pi0.sim.maniskill3_custom_env  # noqa: F401

    env_kwargs = dict(ep_cfg.simulator.env_kwargs or {})
    task_id = str(ep_cfg.simulator.task or "MiniPi0MultiObjectTray-v1")
    env = gym.make(
        task_id,
        num_envs=num_envs,
        obs_mode=env_kwargs.pop("obs_mode", "rgbd"),
        reward_mode=env_kwargs.pop("reward_mode", "dense"),
        control_mode=env_kwargs.pop("control_mode", str(ep_cfg.simulator.controller)),
        render_mode="none",
        render_backend=env_kwargs.pop("render_backend", "gpu"),
        sim_backend=env_kwargs.pop("sim_backend", "auto"),
        robot_uids=env_kwargs.pop("robot_uids", str(ep_cfg.simulator.robot).lower()),
        **env_kwargs,
    )
    raw_obs, _ = env.reset(seed=int(ep_cfg.experiment.seed))

    tray_center = np.asarray(ep_cfg.simulator.env_kwargs.get("tray_center", [0.62, 0.0, 0.0]), dtype=np.float32)
    policies = [ScriptedMultiObjectOracle(tray_center=tray_center) for _ in range(num_envs)]
    for policy in policies:
        policy.reset()

    buffers = [EpisodeBuffer(obs=[], actions=[], rewards=[], dones=[], info_rows=[]) for _ in range(num_envs)]
    finalized: list[tuple[EpisodeBuffer, dict[str, Any]]] = []

    for _ in range(int(max_steps)):
        obs_batch = canonical_obs_batch_from_raw_env(env, image_keys=image_keys, state_keys=state_keys, raw_obs=raw_obs)
        acts = []
        for i in range(num_envs):
            obs_i = obs_batch[i]
            act_i = policies[i].act(obs_i).astype(np.float32)
            buffers[i].obs.append({
                k: np.asarray(obs_i[k])
                for k in set(image_keys + state_keys + [
                    "observation.state.object",
                    "observation.state.object_mask",
                    "observation.state.placed_mask",
                    "observation.state.task_progress",
                ])
                if k in obs_i
            })
            acts.append(act_i)

        actions_np = np.stack(acts, axis=0)
        raw_obs, reward, terminated, truncated, info = env.step(actions_np)
        rew = to_numpy(reward).astype(np.float32).reshape(num_envs)
        ter = to_numpy(terminated).astype(bool).reshape(num_envs)
        tru = to_numpy(truncated).astype(bool).reshape(num_envs)
        info_rows = normalize_info_batched(dict(info), num_envs=num_envs)

        for i in range(num_envs):
            buffers[i].actions.append(actions_np[i])
            buffers[i].rewards.append(float(rew[i]))
            done_i = bool(ter[i] or tru[i] or float(info_rows[i].get("success_fraction", 0.0)) >= 1.0 - 1e-6)
            buffers[i].dones.append(1 if done_i else 0)
            info_rows[i]["success"] = bool(float(info_rows[i].get("success_fraction", 0.0)) >= 1.0 - 1e-6)
            buffers[i].info_rows.append(info_rows[i])

            if done_i:
                fi = dict(info_rows[i])
                fi.setdefault("placed_count", int(np.sum(obs_batch[i]["observation.state.placed_mask"] > 0.5)))
                fi.setdefault("total_objects", int(np.sum(obs_batch[i]["observation.state.object_mask"] > 0.5)))
                if (not only_success) or bool(fi.get("success", False)):
                    finalized.append((buffers[i], fi))
                buffers[i] = EpisodeBuffer(obs=[], actions=[], rewards=[], dones=[], info_rows=[])
                policies[i].reset()
                if len(finalized) >= int(episodes_target):
                    env.close()
                    return finalized

    env.close()
    return finalized


def collect_single_mplib_episode(
    ep_cfg: RootConfig,
    *,
    max_steps: int,
    image_keys: list[str],
    state_keys: list[str],
) -> tuple[EpisodeBuffer, dict[str, Any]]:
    """Collect one episode using mplib motion-planning primitives.

    Args:
        ep_cfg: Runtime config for this episode seed.
        max_steps: Maximum action steps before forced stop.
        image_keys: Canonical image keys to capture.
        state_keys: Canonical state keys to capture.

    Returns:
        `(EpisodeBuffer, final_info)` for one planner-driven trajectory.
    """
    import mplib
    from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb

    env_kwargs = dict(ep_cfg.simulator.env_kwargs or {})
    env_kwargs["control_mode"] = "pd_joint_pos"
    env_kwargs["obs_mode"] = env_kwargs.get("obs_mode", "rgbd")
    env = gym.make(
        str(ep_cfg.simulator.task or "MiniPi0MultiObjectTray-v1"),
        obs_mode=env_kwargs.pop("obs_mode", "rgbd"),
        reward_mode=env_kwargs.pop("reward_mode", "dense"),
        control_mode=env_kwargs.pop("control_mode", "pd_joint_pos"),
        render_mode="rgb_array",
        render_backend=env_kwargs.pop("render_backend", "cpu"),
        sim_backend=env_kwargs.pop("sim_backend", "cpu"),
        robot_uids=env_kwargs.pop("robot_uids", "panda"),
        **env_kwargs,
    )
    raw_obs, _ = env.reset(seed=int(ep_cfg.experiment.seed))
    buf = EpisodeBuffer(obs=[], actions=[], rewards=[], dones=[], info_rows=[])
    final_info: dict[str, Any] = {"success": False, "success_fraction": 0.0, "placed_count": 0, "total_objects": 0}
    step_counter = {"n": 0}

    orig_step = env.step

    def recording_step(action):
        """Step wrapper that records canonical obs/action/reward/info into buffer.

        Args:
            action: Action sent to wrapped env step.

        Returns:
            Original `(obs, reward, terminated, truncated, info)` tuple.
        """
        obs, reward, terminated, truncated, info = orig_step(action)
        step_counter["n"] += 1
        norm = normalize_info(dict(info))
        can_obs = canonical_obs_from_raw_env(env, image_keys=image_keys, state_keys=state_keys, last_raw_obs=obs)
        buf.obs.append(can_obs)
        buf.actions.append(np.asarray(action, dtype=np.float32))
        buf.rewards.append(float(np.asarray(reward).item()))
        done = bool(np.asarray(terminated).item() or np.asarray(truncated).item())
        if float(norm.get("success_fraction", 0.0)) >= 1.0 - 1e-6:
            done = True
        buf.dones.append(1 if done else 0)
        norm["success"] = bool(float(norm.get("success_fraction", 0.0)) >= 1.0 - 1e-6)
        buf.info_rows.append(norm)
        return obs, reward, terminated, truncated, info

    env.step = recording_step

    uw = env.unwrapped
    tray_center = np.asarray(uw.tray_center_np, dtype=np.float32)
    urdf = ".venv/lib/python3.11/site-packages/mani_skill/assets/robots/panda/panda_v2.urdf"
    srdf = ".venv/lib/python3.11/site-packages/mani_skill/assets/robots/panda/panda_v2.srdf"
    planner = mplib.Planner(
        urdf=urdf,
        srdf=srdf,
        user_link_names=[lnk.get_name() for lnk in uw.agent.robot.get_links()],
        user_joint_names=[j.get_name() for j in uw.agent.robot.get_active_joints()],
        move_group="panda_hand_tcp",
    )
    planner.set_base_pose(mplib.Pose(to_numpy(uw.agent.robot.pose.p)[0].astype(np.float64), to_numpy(uw.agent.robot.pose.q)[0].astype(np.float64)))

    lo = np.asarray(env.action_space.low, dtype=np.float32)
    hi = np.asarray(env.action_space.high, dtype=np.float32)
    gripper_open = float(hi[-1])
    gripper_closed = float(lo[-1])

    def to_pose(sp_pose: sapien.Pose):
        """Convert SAPIEN pose to mplib pose.

        Args:
            sp_pose: Pose object in SAPIEN format.

        Returns:
            Equivalent pose in mplib format.
        """
        return mplib.Pose(np.asarray(sp_pose.p, dtype=np.float64), np.asarray(sp_pose.q, dtype=np.float64))

    def exec_path(result: dict[str, Any], grip: float):
        """Execute a planned joint-space trajectory with fixed gripper command.

        Args:
            result: Planner output dictionary containing joint trajectory.
            grip: Gripper command value applied at every executed point.
        """
        if result.get("status") != "Success":
            return
        traj = np.asarray(result.get("position"), dtype=np.float32)
        if traj.ndim != 2 or traj.shape[0] == 0:
            return
        for q7 in traj:
            if step_counter["n"] >= int(max_steps):
                break
            action = np.zeros((env.action_space.shape[0],), dtype=np.float32)
            action[:7] = q7[:7]
            action[-1] = grip
            env.step(np.clip(action, lo, hi))

    def move_pose(sp_pose: sapien.Pose, grip: float) -> bool:
        """Plan and execute a motion to target pose; retry screw->pose planner.

        Args:
            sp_pose: Target end-effector pose.
            grip: Gripper command to apply while executing plan.

        Returns:
            True if planning and execution succeeded, else False.
        """
        q_now = to_numpy(uw.agent.robot.get_qpos())[0].astype(np.float64)
        goal = to_pose(sp_pose)
        res = planner.plan_screw(goal, q_now, time_step=1 / 20)
        if res.get("status") != "Success":
            res = planner.plan_pose(goal, q_now, time_step=1 / 20, planning_time=0.25)
        if res.get("status") != "Success":
            return False
        exec_path(res, grip)
        return True

    def hold_gripper(grip: float, n: int):
        """Hold current arm joints while applying repeated gripper command.

        Args:
            grip: Gripper command to hold.
            n: Number of control steps to repeat.
        """
        for _ in range(n):
            if step_counter["n"] >= int(max_steps):
                break
            q_now = to_numpy(uw.agent.robot.get_qpos())[0].astype(np.float32)
            action = np.zeros((env.action_space.shape[0],), dtype=np.float32)
            action[:7] = q_now[:7]
            action[-1] = grip
            env.step(np.clip(action, lo, hi))

    start_wall = time.time()
    try:
        while step_counter["n"] < int(max_steps):
            if (time.time() - start_wall) > 45.0:
                break
            info = uw.evaluate()
            if float(info["success_fraction"][0].item()) >= 1.0 - 1e-6:
                break

            placed = to_numpy(info["placed_mask"])[0]
            active = to_numpy(uw._active_object_mask)[0]
            candidates = [i for i in range(min(len(active), len(placed))) if active[i] > 0 and placed[i] < 0.5]
            if not candidates:
                break
            target_idx = int(candidates[0])
            actor = uw.objects[target_idx]

            approaching = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            target_closing = uw.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].detach().cpu().numpy()
            obb = get_actor_obb(actor)
            grasp_info = compute_grasp_info_by_obb(obb, approaching=approaching, target_closing=target_closing, depth=0.025)
            base_grasp = uw.agent.build_grasp_pose(approaching, grasp_info["closing"], grasp_info["center"])

            grasp_pose = None
            pre_grasp = None
            for yaw in [0.0, np.pi / 4, -np.pi / 4, np.pi / 2, -np.pi / 2]:
                dq = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=np.float64)
                cand = base_grasp * sapien.Pose(q=dq)
                cand_pre = cand * sapien.Pose([0.0, 0.0, -0.07])
                q_now = to_numpy(uw.agent.robot.get_qpos())[0].astype(np.float64)
                res = planner.plan_screw(to_pose(cand_pre), q_now, time_step=1 / 20)
                if res.get("status") != "Success":
                    res = planner.plan_pose(to_pose(cand_pre), q_now, time_step=1 / 20, planning_time=0.2)
                if res.get("status") == "Success":
                    grasp_pose = cand
                    pre_grasp = cand_pre
                    break
            if grasp_pose is None or pre_grasp is None:
                continue

            actor_p = to_numpy(actor.pose.p)[0]
            lift_pose = sapien.Pose([actor_p[0], actor_p[1], max(actor_p[2] + 0.14, 0.18)], grasp_pose.q)
            tray_above = sapien.Pose([tray_center[0], tray_center[1], 0.20], grasp_pose.q)
            tray_place = sapien.Pose([tray_center[0], tray_center[1], 0.095], grasp_pose.q)
            retreat = sapien.Pose([tray_center[0], tray_center[1], 0.22], grasp_pose.q)

            if not move_pose(pre_grasp, gripper_open):
                continue
            if not move_pose(grasp_pose, gripper_open):
                continue
            hold_gripper(gripper_closed, n=10)
            if not move_pose(lift_pose, gripper_closed):
                continue
            if not move_pose(tray_above, gripper_closed):
                continue
            if not move_pose(tray_place, gripper_closed):
                continue
            hold_gripper(gripper_open, n=10)
            move_pose(retreat, gripper_open)
            hold_gripper(gripper_open, n=6)
    finally:
        env.step = orig_step
        fin = normalize_info(dict(uw.evaluate()))
        final_info = {
            "success": bool(float(fin.get("success_fraction", 0.0)) >= 1.0 - 1e-6),
            "success_fraction": float(fin.get("success_fraction", 0.0)),
            "placed_count": int(float(fin.get("placed_count", 0.0))),
            "total_objects": int(float(fin.get("total_objects", 0.0))),
        }
        env.close()

    return buf, final_info
