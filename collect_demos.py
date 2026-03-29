import argparse
import json
import os

import h5py
import numpy as np
import robosuite as suite


def _load_controller_configs(robot="Panda"):
    # robosuite <=1.4
    if hasattr(suite, "load_controller_config"):
        return suite.load_controller_config(default_controller="OSC_POSE")
    # robosuite >=1.5
    if hasattr(suite, "load_composite_controller_config"):
        cfg = suite.load_composite_controller_config(controller="BASIC", robot=robot)
        body_parts = cfg.get("body_parts", None)
        if isinstance(body_parts, dict):
            preferred = [k for k in ("right", "arm", "single") if k in body_parts]
            if preferred:
                keep = set(preferred)
                cfg["body_parts"] = {k: v for k, v in body_parts.items() if k in keep}
        return cfg
    raise RuntimeError("Unsupported robosuite controller API")


def make_collect_env(task="Lift", render=False, seed=0):
    env = suite.make(
        env_name=task,
        robots="Panda",
        controller_configs=_load_controller_configs(robot="Panda"),
        has_renderer=render,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview"],
        camera_heights=84,
        camera_widths=84,
        control_freq=20,
        horizon=400,
        reward_shaping=True,
        ignore_done=False,
        seed=seed,
    )
    return env


class ScriptedLiftPolicy:
    """
    Simple finite-state policy for Lift.
    Falls back to tiny random actions if object keys are unavailable.
    """

    def __init__(self):
        self.phase = 0
        self.close_count = 0

    def reset(self):
        self.phase = 0
        self.close_count = 0

    def _delta_action(self, eef_pos, target_xyz):
        dpos = (target_xyz - eef_pos) * 8.0
        dpos = np.clip(dpos, -1.0, 1.0)
        # Keep orientation deltas near zero for stability.
        rot = np.zeros(3, dtype=np.float32)
        return np.concatenate([dpos.astype(np.float32), rot], axis=0)

    def act(self, obs):
        eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)

        # Prefer relative key if available.
        if "gripper_to_cube_pos" in obs:
            cube_pos = eef_pos - np.asarray(obs["gripper_to_cube_pos"], dtype=np.float32)
        elif "cube_pos" in obs:
            cube_pos = np.asarray(obs["cube_pos"], dtype=np.float32)
        else:
            noise = np.random.uniform(-0.1, 0.1, size=(7,)).astype(np.float32)
            noise[6] = np.random.uniform(-1.0, 1.0)
            return noise

        above_cube = cube_pos + np.array([0.0, 0.0, 0.10], dtype=np.float32)
        at_cube = cube_pos + np.array([0.0, 0.0, 0.015], dtype=np.float32)
        lift_target = cube_pos + np.array([0.0, 0.0, 0.20], dtype=np.float32)

        # phase 0: move above cube
        if self.phase == 0:
            base = self._delta_action(eef_pos, above_cube)
            grip = -1.0  # open
            if np.linalg.norm(above_cube - eef_pos) < 0.02:
                self.phase = 1

        # phase 1: descend to grasp height
        elif self.phase == 1:
            base = self._delta_action(eef_pos, at_cube)
            grip = -1.0
            if np.linalg.norm(at_cube - eef_pos) < 0.012:
                self.phase = 2

        # phase 2: close gripper
        elif self.phase == 2:
            base = np.zeros(6, dtype=np.float32)
            grip = 1.0
            self.close_count += 1
            if self.close_count > 8:
                self.phase = 3

        # phase 3: lift up
        else:
            base = self._delta_action(eef_pos, lift_target)
            grip = 1.0

        return np.concatenate([base, np.array([grip], dtype=np.float32)], axis=0)


def _extract_step(obs):
    return {
        "agentview_image": np.asarray(obs["agentview_image"], dtype=np.uint8),
        "robot0_eef_pos": np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
        "robot0_eef_quat": np.asarray(obs["robot0_eef_quat"], dtype=np.float32),
        "robot0_gripper_qpos": np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
    }


def _write_demo_to_hdf5(data_group, demo_idx, steps, actions, rewards, success):
    demo_key = f"demo_{demo_idx}"
    if demo_key in data_group:
        del data_group[demo_key]
    demo = data_group.create_group(demo_key)

    t = int(len(actions))
    demo.attrs["num_samples"] = t
    demo.create_dataset("actions", data=np.asarray(actions, dtype=np.float32))
    demo.create_dataset("rewards", data=np.asarray(rewards, dtype=np.float32))

    dones = np.zeros((t,), dtype=np.int32)
    if t > 0:
        dones[-1] = int(success)
    demo.create_dataset("dones", data=dones)

    obs = demo.create_group("obs")
    obs.create_dataset("agentview_image", data=np.stack([s["agentview_image"] for s in steps], axis=0))
    obs.create_dataset("robot0_eef_pos", data=np.stack([s["robot0_eef_pos"] for s in steps], axis=0))
    obs.create_dataset("robot0_eef_quat", data=np.stack([s["robot0_eef_quat"] for s in steps], axis=0))
    obs.create_dataset("robot0_gripper_qpos", data=np.stack([s["robot0_gripper_qpos"] for s in steps], axis=0))
    demo.attrs["success"] = int(success)

    total = int(data_group.attrs.get("total", 0))
    data_group.attrs["total"] = total + t


def collect(args):
    out_dir = os.path.dirname(args.out_hdf5)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if args.overwrite and os.path.exists(args.out_hdf5):
        os.remove(args.out_hdf5)

    policy = ScriptedLiftPolicy() if args.policy == "scripted_lift" else None

    with h5py.File(args.out_hdf5, "a") as h5:
        data_group = h5.require_group("data")
        if "total" not in data_group.attrs:
            data_group.attrs["total"] = 0
        if "env_args" not in data_group.attrs:
            data_group.attrs["env_args"] = json.dumps(
                {
                    "env_name": args.task,
                    "env_type": "robosuite",
                    "env_kwargs": {"robots": "Panda"},
                }
            )

        saved = 0
        tried = 0
        while saved < args.num_episodes and tried < args.max_trials:
            env = make_collect_env(task=args.task, render=args.render, seed=args.seed + tried)
            obs = env.reset()
            if policy is not None:
                policy.reset()

            steps = []
            actions = []
            rewards = []
            success = False

            for _ in range(args.max_steps):
                if args.policy == "random":
                    lo, hi = env.action_spec
                    action = np.random.uniform(lo, hi).astype(np.float32)
                else:
                    action = policy.act(obs)
                    lo, hi = env.action_spec
                    action = np.clip(action, lo, hi).astype(np.float32)

                steps.append(_extract_step(obs))
                actions.append(action)
                obs, reward, done, info = env.step(action)
                rewards.append(float(reward))

                step_success = bool(info.get("success", False))
                if not step_success and hasattr(env, "_check_success"):
                    try:
                        step_success = bool(env._check_success())
                    except Exception:
                        step_success = False

                if step_success:
                    success = True
                    done = True
                if done:
                    break

            env.close()
            tried += 1

            if args.only_success and not success:
                print(f"[skip] trial={tried:03d} len={len(actions):03d} success=0")
                continue

            _write_demo_to_hdf5(
                data_group=data_group,
                demo_idx=saved,
                steps=steps,
                actions=actions,
                rewards=rewards,
                success=success,
            )
            saved += 1
            print(f"[save] demo_{saved - 1:04d} | len={len(actions):03d} | success={int(success)}")

    print(f"Collected {saved} episodes (trials={tried}).")
    print(f"Saved dataset: {args.out_hdf5}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="Lift")
    p.add_argument("--policy", choices=["scripted_lift", "random"], default="scripted_lift")
    p.add_argument("--num_episodes", type=int, default=20)
    p.add_argument("--max_trials", type=int, default=200)
    p.add_argument("--max_steps", type=int, default=120)
    p.add_argument("--out_hdf5", default="data/robomimic/custom/lift/ph/low_dim_v15.hdf5")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only_success", action="store_true")
    p.add_argument("--render", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    collect(parse_args())
