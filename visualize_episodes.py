import argparse
import os
import platform
import shutil
import sys
import time

import h5py
import mujoco
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


def make_view_env(task="Lift", seed=0, camera="frontview"):
    return suite.make(
        env_name=task,
        robots="Panda",
        controller_configs=_load_controller_configs(robot="Panda"),
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        render_camera=camera,
        control_freq=20,
        horizon=1000,
        reward_shaping=True,
        ignore_done=False,
        seed=seed,
    )


def get_available_cameras(task="Lift", seed=0):
    env = suite.make(
        env_name=task,
        robots="Panda",
        controller_configs=_load_controller_configs(robot="Panda"),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        seed=seed,
    )
    model = env.sim.model._model
    cams = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)]
    env.close()
    return cams


def _force_viewer_camera(env, camera_name):
    try:
        cam_id = env.sim.model.camera_name2id(camera_name)
    except Exception:
        return
    if getattr(env, "viewer", None) is None:
        return
    try:
        env.viewer.set_camera(cam_id)
    except Exception:
        pass
    try:
        if getattr(env.viewer, "viewer", None) is not None:
            env.viewer.viewer.cam.type = 2
            env.viewer.viewer.cam.fixedcamid = cam_id
    except Exception:
        pass


def _sync_viewer(env):
    if getattr(env, "viewer", None) is not None and hasattr(env.viewer, "update"):
        env.viewer.update()
    else:
        env.render()


def replay_episode(
    env,
    demo_key,
    demo_group,
    camera_name,
    replay_mode="action",
    fps=20.0,
    max_steps=None,
    log_actions=False,
    init_from_demo_state=True,
):
    actions = np.asarray(demo_group["actions"], dtype=np.float32)
    original_actions = actions.copy()
    states = np.asarray(demo_group["states"]) if "states" in demo_group else None
    recorded_success = None
    if "success" in demo_group.attrs:
        recorded_success = int(demo_group.attrs["success"])
    elif "dones" in demo_group:
        recorded_success = int(np.asarray(demo_group["dones"]).any())

    if replay_mode == "state":
        if states is None:
            print("  states not found in HDF5 demo; falling back to action replay.")
            replay_mode = "action"
        elif max_steps is not None:
            states = states[:max_steps]
    if replay_mode == "action" and max_steps is not None:
        actions = actions[:max_steps]

    env.reset()
    if replay_mode == "action" and bool(init_from_demo_state) and states is not None and len(states) > 0:
        # Align action replay with the recorded initial simulator state.
        # Without this, tiny reset differences often make grasping fail.
        env.sim.set_state_from_flattened(states[0])
        env.sim.forward()

    lo, hi = env.action_spec
    dt = 1.0 / max(1e-6, fps)
    success = False

    _sync_viewer(env)
    _force_viewer_camera(env, camera_name)

    n_steps = len(states) if replay_mode == "state" else len(actions)
    print(f"Replaying {demo_key} | steps={n_steps} | mode={replay_mode}")

    if replay_mode == "state":
        for i, s in enumerate(states):
            t0 = time.perf_counter()
            env.sim.set_state_from_flattened(s)
            env.sim.forward()
            _sync_viewer(env)
            _force_viewer_camera(env, camera_name)
            if log_actions and i < len(original_actions):
                a = original_actions[i]
                print(
                    f"  step={i:03d} action=[{a[0]:+.4f}, {a[1]:+.4f}, {a[2]:+.4f}, "
                    f"{a[3]:+.4f}, {a[4]:+.4f}, {a[5]:+.4f}, {a[6]:+.4f}]"
                )
            step_success = False
            if hasattr(env, "_check_success"):
                try:
                    step_success = bool(env._check_success())
                except Exception:
                    step_success = False
            success = success or step_success
            elapsed = time.perf_counter() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)
    else:
        for i, action in enumerate(actions):
            t0 = time.perf_counter()
            action = np.clip(action, lo, hi)
            if log_actions:
                print(
                    f"  step={i:03d} action=[{action[0]:+.4f}, {action[1]:+.4f}, {action[2]:+.4f}, "
                    f"{action[3]:+.4f}, {action[4]:+.4f}, {action[5]:+.4f}, {action[6]:+.4f}]"
                )
            _, reward, done, info = env.step(action)
            _sync_viewer(env)
            _force_viewer_camera(env, camera_name)

            step_success = bool(info.get("success", False))
            if not step_success and hasattr(env, "_check_success"):
                try:
                    step_success = bool(env._check_success())
                except Exception:
                    step_success = False
            success = success or step_success

            elapsed = time.perf_counter() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)
            if done:
                print(f"  finished early at step={i+1} | reward={reward:.3f}")
                break

    if recorded_success is None:
        print(f"  replay_success={int(success)}")
    else:
        print(f"  replay_success={int(success)} | recorded_success={recorded_success}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="Lift")
    p.add_argument("--input_hdf5", default="data/robomimic/lift/ph/low_dim_v15.hdf5")
    p.add_argument("--data_group", default="data")
    p.add_argument("--camera", default="frontview")
    p.add_argument("--replay_mode", choices=["action", "state"], default="action")
    p.add_argument("--list_cameras", action="store_true")
    p.add_argument("--fps", type=float, default=20.0)
    p.add_argument("--max_episodes", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hold_end", action="store_true")
    p.add_argument("--hold_seconds", type=float, default=0.0)
    p.add_argument("--log_actions", action="store_true")
    p.add_argument(
        "--init_from_demo_state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In action replay, initialize env to first recorded demo state if available.",
    )
    return p.parse_args()


def _maybe_reexec_with_mjpython():
    if platform.system() != "Darwin":
        return
    exe_name = os.path.basename(sys.executable).lower()
    if "mjpython" in exe_name:
        return
    if os.environ.get("MINI_PI0_MJPYTHON_REEXEC") == "1":
        return
    mjpython = shutil.which("mjpython")
    if not mjpython:
        raise RuntimeError(
            "macOS detected and `mjpython` not found in PATH. "
            "Activate your venv and run: mjpython visualize_episodes.py ..."
        )
    env = os.environ.copy()
    env["MINI_PI0_MJPYTHON_REEXEC"] = "1"
    os.execvpe(mjpython, [mjpython, *sys.argv], env)


def main():
    _maybe_reexec_with_mjpython()
    args = parse_args()

    if args.list_cameras:
        cameras = get_available_cameras(task=args.task, seed=args.seed)
        print("Available cameras:")
        for c in cameras:
            print(f" - {c}")
        return

    cameras = get_available_cameras(task=args.task, seed=args.seed)
    if args.camera not in cameras:
        raise ValueError(f"Unknown camera '{args.camera}'. Available: {', '.join(cameras)}")
    if not os.path.exists(args.input_hdf5):
        raise FileNotFoundError(f"HDF5 dataset not found: {args.input_hdf5}")

    with h5py.File(args.input_hdf5, "r") as f:
        if args.data_group not in f:
            raise KeyError(f"Group '{args.data_group}' not found in {args.input_hdf5}")
        demo_keys = sorted(list(f[args.data_group].keys()))[: args.max_episodes]
        if not demo_keys:
            raise FileNotFoundError(f"No demos found in group '{args.data_group}' for {args.input_hdf5}")

        env = make_view_env(task=args.task, seed=args.seed, camera=args.camera)
        try:
            print("Opening MuJoCo viewer...")
            print(f"Using render_camera={args.camera}")
            for demo_key in demo_keys:
                replay_episode(
                    env=env,
                    demo_key=demo_key,
                    demo_group=f[args.data_group][demo_key],
                    camera_name=args.camera,
                    replay_mode=args.replay_mode,
                    fps=args.fps,
                    max_steps=args.max_steps,
                    log_actions=args.log_actions,
                    init_from_demo_state=args.init_from_demo_state,
                )
                time.sleep(0.5)

            if args.hold_seconds > 0:
                print(f"Holding viewer for {args.hold_seconds:.1f}s...")
                t0 = time.perf_counter()
                while time.perf_counter() - t0 < args.hold_seconds:
                    _sync_viewer(env)
                    _force_viewer_camera(env, args.camera)
                    time.sleep(1.0 / 30.0)

            if args.hold_end:
                print("Holding viewer open. Press Ctrl+C to close.")
                try:
                    while True:
                        _sync_viewer(env)
                        _force_viewer_camera(env, args.camera)
                        time.sleep(1.0 / 30.0)
                except KeyboardInterrupt:
                    pass
        finally:
            env.close()


if __name__ == "__main__":
    main()
