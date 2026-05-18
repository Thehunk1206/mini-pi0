# Datasets

This repo supports two training dataset formats:

- `robomimic_hdf5`: existing robomimic-style HDF5 files.
- `lerobot_v3`: LeRobot v3 datasets loaded lazily for scalable training.

ManiSkill collection and robomimic conversion commands write this HDF5 schema:

```text
data/demo_*/actions
data/demo_*/obs/<image keys>
data/demo_*/obs/<proprio keys>
```

Typical image keys:

- `agentview_image`
- `robot0_eye_in_hand_image`

Typical proprio keys:

- `robot0_eef_pos`
- `robot0_eef_quat`
- `robot0_gripper_qpos`
- optional task state keys when explicitly configured

## Collection

The current custom tray dataset is collected directly from the registered
custom ManiSkill environment:

```bash
mini-pi0 collect-maniskill-demos \
  --config examples/configs/maniskill3_tray_transformer_vit_hist2_chunk16.yaml \
  --out_hdf5 data/robomimic/custom/tray/dataset.hdf5 \
  --num_episodes 1000 \
  --collector_backend mplib
```

Use `--append` to add more episodes to an existing file, or `--overwrite` to
replace it.

```bash
mini-pi0 collect-maniskill-demos \
  --config examples/configs/maniskill3_tray_transformer_vit_hist2_chunk16.yaml \
  --out_hdf5 data/robomimic/custom/tray/dataset.hdf5 \
  --num_episodes 500 \
  --append
```

## Built-In ManiSkill Demos

For built-in ManiSkill tasks, download demos, replay them with RGBD
observations and the target controller, then convert the replayed trajectory.

```bash
python -m mani_skill.utils.download_demo StackCube-v1 \
  --output_dir demos/maniskill
```

Replay with observations:

```bash
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path demos/maniskill/rigid_body/StackCube-v1/motionplanning/trajectory.h5 \
  --obs-mode rgbd \
  --target-control-mode pd_joint_pos \
  --save-traj \
  --allow-failure
```

Convert the replayed file:

```bash
mini-pi0 convert-maniskill-trajectory \
  --input_hdf5 demos/maniskill/rigid_body/StackCube-v1/motionplanning/trajectory.rgbd.pd_joint_pos.physx_cpu.h5 \
  --output_hdf5 data/robomimic/maniskill/stackcube/mp/rgbd_pd_joint_pos.hdf5 \
  --overwrite
```

You can also use ManiSkill's official LeRobot v3 converter directly on a
replayed trajectory:

```bash
python -m mani_skill.trajectory.convert_to_lerobot \
  --traj-path demos/maniskill/rigid_body/StackCube-v1/motionplanning/trajectory.rgbd.pd_joint_pos.physx_cpu.h5 \
  --output-dir data/lerobot/stackcube-rgbd-pd-joint-pos \
  --task-name "Stack cube" \
  --fps 20 \
  --image-size 128x128 \
  --robot-type panda
```

For another built-in task, change:

- ManiSkill task id passed to `download_demo`
- replay `--traj-path`
- replay `--target-control-mode`
- conversion `--output_hdf5`
- config `simulator.task`, `simulator.controller`, `data.robomimic_hdf5`,
  and `data.action_stats_path`

## New Custom Task Dataset

Add a registered ManiSkill environment and a collection plugin before collecting
data.

1. Register the environment with ManiSkill in a module under `mini_pi0/sim/`.
   The tray task is the reference implementation:

   ```python
   @register_env("MiniPi0MultiObjectTray-v1", max_episode_steps=1000, override=True)
   class MiniPi0MultiObjectTrayEnv(BaseEnv):
       ...
   ```

2. Import that module from `mini_pi0/sim/maniskill3_adapter.py` so the task is
   registered before the adapter calls `gym.make(...)`.

3. Add a collector plugin under
   `mini_pi0/dataset/maniskill_collectors/plugins/`. Copy
   `new_task_template.py`, implement task-specific rollout logic, and register
   the collector with `register_collector(...)`.

4. Import the plugin in `mini_pi0/dataset/maniskill_collectors/__init__.py`.

5. Create a config:

   ```yaml
   simulator:
     backend: maniskill3
     task: YourTask-v1
     controller: pd_joint_pos
     env_kwargs:
       obs_mode: rgbd
       render_mode: rgb_array
       control_mode: pd_joint_pos

   data:
     format: robomimic_hdf5
     robomimic_hdf5: data/robomimic/maniskill/your_task/dataset.hdf5
     action_stats_path: data/robomimic/maniskill/your_task/action_stats.json
   ```

6. Collect:

   ```bash
   mini-pi0 collect-maniskill-demos \
     --config examples/configs/maniskill3_your_task.yaml \
     --out_hdf5 data/robomimic/maniskill/your_task/dataset.hdf5 \
     --num_episodes 1000 \
     --collector_name your_collector_name \
     --collector_backend scripted \
     --overwrite
   ```

7. Train by pointing the config at the collected HDF5:

   ```bash
   mini-pi0 train --config examples/configs/maniskill3_your_task.yaml
   ```

## Conversion

`convert-maniskill-trajectory` expects a replayed ManiSkill HDF5 containing an
`obs` group. Raw action-only trajectories should be replayed first with
`--save-traj -o rgbd` or `--obs-mode rgbd`.

```bash
mini-pi0 convert-maniskill-trajectory \
  --input_hdf5 data/source/trajectory.rgbd.pd_joint_pos.physx_cpu.h5 \
  --output_hdf5 data/robomimic/custom/converted.hdf5 \
  --overwrite
```

## LeRobot v3

Use LeRobot v3 for larger datasets when you do not want to load all HDF5
episodes into memory. The lazy loader reads only the observation history and
action chunk needed by the current training sample.

Install with `uv` and constraints when ManiSkill and LeRobot share one environment:

```bash
uv venv --python 3.11 .venv
. .venv/bin/activate
uv pip install -e ".[maniskill3,lerobot,vision,dev]" \
  -c constraints-maniskill-lerobot.txt
```

Pip fallback:

```bash
pip install -e ".[maniskill3,lerobot,vision,dev]" \
  -c constraints-maniskill-lerobot.txt
```

If Hugging Face cache directories on the machine are read-only, point `HF_HOME`
to a writable path before converting or training:

```bash
export HF_HOME=/tmp/minipi_hf_cache
```

Convert an existing robomimic HDF5 file:

```bash
mini-pi0 convert-robomimic-to-lerobot \
  --input_hdf5 data/robomimic/maniskill/stackcube/mp/rgbd_pd_joint_pos.hdf5 \
  --output_dir data/lerobot/stackcube-rgbd-pd-joint-pos \
  --repo_id local/stackcube-rgbd-pd-joint-pos \
  --task_name "Stack cube" \
  --fps 20 \
  --image_keys agentview_image,robot0_eye_in_hand_image \
  --state_keys robot0_eef_pos,robot0_eef_quat,robot0_gripper_qpos \
  --overwrite
```

Train from the converted dataset:

```bash
mini-pi0 train \
  --config examples/configs/maniskill3_stackcube_motionplanning_transformer_vit_hist2_medium.yaml \
  --set data.format=lerobot_v3 \
  --set data.lerobot_repo_id=local/stackcube-rgbd-pd-joint-pos \
  --set data.lerobot_root=data/lerobot/stackcube-rgbd-pd-joint-pos \
  --set data.lerobot_image_keys='["observation.images.agentview_image","observation.images.robot0_eye_in_hand_image"]' \
  --set data.lerobot_state_key=observation.state \
  --set data.lerobot_action_key=action
```

Evaluation and deployment remain simulator-based. They use the checkpoint and
`artifacts/action_stats.json` produced during training, not the LeRobot dataset
itself.

## Action Stats

Training writes action stats under each run's `artifacts/` directory. Evaluation
and diagnostics should use the same stats file as the checkpoint:

```bash
--action_stats runs/<experiment>/run1/artifacts/action_stats.json
```
