# Simulation

The active simulator path is ManiSkill through `simulator.backend=maniskill3`.
IsaacLab remains as a scaffolded adapter for future work.

Check backend status:

```bash
mini-pi0 backends
```

Common ManiSkill config fields:

```yaml
simulator:
  backend: maniskill3
  task: StackCube-v1
  robot: panda_wristcam
  controller: pd_joint_pos
  control_freq: 20
  horizon: 500
  use_camera_obs: true
  camera_names: [base_camera, hand_camera]
  env_kwargs:
    obs_mode: rgbd
    render_mode: rgb_array
    control_mode: pd_joint_pos
```

## Custom Environments

Custom ManiSkill tasks should be normal registered ManiSkill environments. The
current tray task lives in `mini_pi0/sim/maniskill3_custom_env.py` and is
registered as `MiniPi0MultiObjectTray-v1`.

Minimum pattern:

```python
from mani_skill.utils.registration import register_env
from mani_skill.envs.sapien_env import BaseEnv


@register_env("YourTask-v1", max_episode_steps=500, override=True)
class YourTaskEnv(BaseEnv):
    ...
```

After adding a task:

1. Import the module from `mini_pi0/sim/maniskill3_adapter.py` so registration
   happens before `gym.make(...)`.
2. Add or copy a config under `examples/configs/` with
   `simulator.task: YourTask-v1`.
3. Add a collector plugin under
   `mini_pi0/dataset/maniskill_collectors/plugins/` if you want to generate
   demonstrations with `collect-maniskill-demos`.
4. Import the plugin from `mini_pi0/dataset/maniskill_collectors/__init__.py`
   for side-effect registration.

The collector plugin should own task-specific expert behavior. The shared
collection command owns HDF5 writing, episode filtering, and summary stats.

## PegInsertion Hole Camera

PegInsertionSide is sensitive to small peg-hole alignment errors. The built-in
`base_camera` often sees the hole too small, and the wrist camera can be
occluded by the gripper during insertion. The repo registers a local variant:

```text
MiniPi0PegInsertionSide-v1
```

It keeps ManiSkill's `PegInsertionSide-v1` dynamics and adds two fixed
insertion cameras:

```text
hole_left_camera
hole_right_camera
```

Both cameras sit on the peg approach side of the box and look obliquely at the
hole. They are symmetric in `x`, so when the gripper or peg occludes one view,
the opposite side still tends to see the peg-hole contact.

Quick camera check from a recorded episode:

```bash
.venv/bin/python tools/visualize_peginsertion_cameras.py \
  --traj-path demos/maniskill/PegInsertionSide-v1/motionplanning/trajectory.rgbd.pd_ee_delta_pose.physx_cpu.h5 \
  --out-dir tmp/peginsertion_hole_camera_debug \
  --traj-key traj_0 \
  --frames 0,89,178 \
  --sensor-width 224 \
  --sensor-height 224
```

The combined sheet is written to:

```text
tmp/peginsertion_hole_camera_debug/traj_0_camera_sheet.png
```

To replay ManiSkill demos through the local env and save RGBD observations with
the extra camera, use the local replay wrapper. The wrapper stages a JSON copy
with `env_info.env_id=MiniPi0PegInsertionSide-v1`, imports the repo-local env,
then delegates to ManiSkill replay:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/replay_maniskill_local_env.py \
  --env-id MiniPi0PegInsertionSide-v1 \
  --work-dir demos/maniskill/PegInsertionSide-v1/motionplanning_holecam \
  --traj-path demos/maniskill/PegInsertionSide-v1/motionplanning/trajectory.h5 \
  --obs-mode rgbd \
  --target-control-mode pd_ee_delta_pose \
  --save-traj \
  --reward-mode dense \
  --sim-backend physx_cpu \
  --num-envs 1
```

Then convert with an explicit third-camera mapping:

```bash
mini-pi0 convert-maniskill-trajectory \
  --input_hdf5 demos/maniskill/PegInsertionSide-v1/motionplanning_holecam/trajectory.rgbd.pd_ee_delta_pose.physx_cpu.h5 \
  --output_hdf5 data/robomimic/maniskill/peginsertionside/mp/rgbd_pd_ee_delta_pose_holecam.hdf5 \
  --image_camera_map agentview_image=base_camera,robot0_eye_in_hand_image=hand_camera,hole_left_image=hole_left_camera,hole_right_image=hole_right_camera \
  --overwrite
```

Train with:

```bash
mini-pi0 train \
  --config examples/configs/maniskill3_peginsertion_motionplanning_transformer_vit_hist2_medium_holecam.yaml
```
