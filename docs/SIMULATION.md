# MuJoCo and robosuite Simulation Environment

This document explains the simulation setup used in this repository and points to the exact implementation files.

## Stack Used in This Repo

- Physics engine: `mujoco`
- Task + robot API: `robosuite`
- Repo integration pattern: simulator adapter interface under `mini_pi0/sim/`

Backend status in this codebase:
- `robosuite`: implemented and used end-to-end
- `maniskill3`: scaffolded adapter
- `isaaclab`: scaffolded adapter

## Adapter Contract

All simulator backends follow one common interface in `mini_pi0/sim/base.py`:

- `reset(seed)`
- `step(action) -> StepOutput`
- `action_spec() -> (low, high)`
- `render(camera, width, height)`
- `check_success(info, obs)`
- optional `set_object_pose(...)`
- `close()`

Registry wiring is in `mini_pi0/sim/registry.py`.

## robosuite Runtime Path

Main implementation: `mini_pi0/sim/robosuite_adapter.py`

What it does:

1. Creates env via `robosuite.make(...)` using config fields from `simulator.*`.
2. Handles controller API compatibility:
   - `load_controller_config(...)` path for older API
   - `load_composite_controller_config(...)` path for newer API
3. Maps repo canonical keys to robosuite observation keys.
4. Clips actions to env limits before stepping.
5. Uses `info["success"]` and `_check_success()` for task success.
6. Supports render for recordings / eval artifacts.

## Observation Key Aliases Used

Implemented in `_resolve_obs_key(...)` in `mini_pi0/sim/robosuite_adapter.py`:

- `observation.images.base_0_rgb` -> `agentview_image`
- `observation.images.right_wrist_0_rgb` -> `robot0_eye_in_hand_image`
- `observation.images.wrist_0_rgb` -> `robot0_eye_in_hand_image`
- `observation.state.eef_pos` -> `robot0_eef_pos`
- `observation.state.eef_quat` -> `robot0_eef_quat`
- `observation.state.tool` -> `robot0_gripper_qpos`
- `observation.state.object` -> `object-state`

## Important Config Fields

Defined in `mini_pi0/config/schema.py` under `SimulatorConfig`:

- `simulator.backend`
- `simulator.task`
- `simulator.robot`
- `simulator.controller`
- `simulator.control_freq`
- `simulator.horizon`
- `simulator.reward_shaping`
- `simulator.has_renderer`
- `simulator.has_offscreen_renderer`
- `simulator.use_camera_obs`
- `simulator.camera_names`
- `simulator.camera_width`
- `simulator.camera_height`
- `simulator.env_kwargs`

Reference config examples:

- `examples/configs/robosuite_can_vision.yaml`
- `examples/configs/robosuite_lift.yaml`

## Where Simulation Is Used

- Evaluation:
  - `mini_pi0/eval/runner.py`
  - `mini_pi0/eval/core.py`
- Deployment:
  - `mini_pi0/deploy/sim_runner.py`
- Manual replay / visualization:
  - `visualize_episodes.py`

## macOS Note (MuJoCo Viewer)

For live interactive viewer windows on macOS, `mjpython` may be required by MuJoCo passive viewer.

`visualize_episodes.py` includes `_maybe_reexec_with_mjpython()` to re-exec under `mjpython` when needed.

## Useful Commands

Check backend availability:

```bash
python -m mini_pi0 backends
```

List local robosuite robots and mapped datasets:

```bash
python -m mini_pi0 robot-dataset-map
```

Manual demo replay:

```bash
python visualize_episodes.py --task PickPlaceCan --input_hdf5 <path_to_hdf5>
```

If viewer errors on macOS:

```bash
mjpython visualize_episodes.py --task PickPlaceCan --input_hdf5 <path_to_hdf5>
```

## External References

- MuJoCo docs: https://mujoco.readthedocs.io/
- robosuite docs: https://robosuite.ai/docs/

