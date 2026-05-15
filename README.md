# mini-pi0

`mini-pi0` is a compact flow-matching policy stack for ManiSkill research.
The supported model is `mini_pi0_fm`, with image observations, proprioception,
and action chunk prediction through transformer, CNN1D, or UNet1D denoisers.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[maniskill3,vision,dev]"
```

## Main Workflows

Collect demonstrations from the current custom tray environment:

```bash
mini-pi0 collect-maniskill-demos \
  --config examples/configs/maniskill3_tray_transformer_vit_hist2_chunk16.yaml \
  --out_hdf5 data/robomimic/custom/tray/dataset.hdf5 \
  --num_episodes 1000 \
  --collector_backend mplib
```

Download built-in ManiSkill demonstrations, replay them with RGBD observations,
then convert them into the robomimic-style HDF5 schema used by training:

```bash
python -m mani_skill.utils.download_demo StackCube-v1 \
  --output_dir demos/maniskill

python -m mani_skill.trajectory.replay_trajectory \
  --traj-path demos/maniskill/rigid_body/StackCube-v1/motionplanning/trajectory.h5 \
  --obs-mode rgbd \
  --target-control-mode pd_joint_pos \
  --save-traj \
  --allow-failure
```

The replay command writes a new trajectory file with observations. Convert that
replayed file:

```bash
mini-pi0 convert-maniskill-trajectory \
  --input_hdf5 demos/maniskill/rigid_body/StackCube-v1/motionplanning/trajectory.rgbd.pd_joint_pos.physx_cpu.h5 \
  --output_hdf5 data/robomimic/maniskill/stackcube/mp/rgbd_pd_joint_pos.hdf5 \
  --overwrite
```

For a new built-in task, repeat the same flow with its ManiSkill task id,
controller, and output path, then create or copy a config under
`examples/configs/` with `simulator.task`, `data.robomimic_hdf5`, and
`data.action_stats_path` pointed at that dataset.

Train an FM policy:

```bash
mini-pi0 train \
  --config examples/configs/maniskill3_stackcube_motionplanning_transformer_vit_hist2_medium.yaml
```

Evaluate a checkpoint:

```bash
mini-pi0 eval \
  --config examples/configs/maniskill3_stackcube_motionplanning_transformer_vit_hist2_medium.yaml \
  --checkpoint runs/<experiment>/run1/checkpoints/best.pt \
  --action_stats runs/<experiment>/run1/artifacts/action_stats.json
```

Run offline action diagnostics:

```bash
python -m mini_pi0.eval.action_diagnostics \
  --config examples/configs/maniskill3_stackcube_motionplanning_transformer_vit_hist2_medium.yaml \
  --checkpoint runs/<experiment>/run1/checkpoints/best.pt \
  --action_stats runs/<experiment>/run1/artifacts/action_stats.json \
  --flow_steps 4,6,8
```

## Configs

ManiSkill configs live in `examples/configs/`. The actively useful starting
points are:

- `maniskill3_stackcube_motionplanning_transformer_vit_hist2_medium.yaml`
- `maniskill3_tray_transformer_vit_hist2_chunk16.yaml`
- `maniskill3_tray_unet1d_resnet18_hist2.yaml`

The model section controls the FM denoiser:

- `action_backbone`: `transformer`, `cnn1d`, or `unet1d`
- `vision_backbone`: `resnet18` or `timm`
- `conditioning_mode`: `cross_attention` or `global`
- `obs_horizon`: number of observation frames to condition on

## Adding A Custom ManiSkill Task

A custom task needs two pieces: a registered ManiSkill environment and a
collector plugin that can generate successful demonstrations for it.

1. Add or extend an environment module under `mini_pi0/sim/`.
   The current tray task is registered in
   `mini_pi0/sim/maniskill3_custom_env.py` with:

   ```python
   @register_env("MiniPi0MultiObjectTray-v1", max_episode_steps=1000, override=True)
   class MiniPi0MultiObjectTrayEnv(BaseEnv):
       ...
   ```

2. Make sure the environment module is imported before `gym.make(...)`.
   `mini_pi0/sim/maniskill3_adapter.py` already imports
   `mini_pi0.sim.maniskill3_custom_env` for side-effect registration. If a new
   custom task lives in another module, import it there as well.

3. Add a collector plugin under
   `mini_pi0/dataset/maniskill_collectors/plugins/`. Start from
   `new_task_template.py`, implement `supports`, `collect_episode`, and
   optionally `collect_vectorized`, then call `register_collector(...)`.

4. Import the plugin in `mini_pi0/dataset/maniskill_collectors/__init__.py`
   so collection registers it at startup.

5. Add a config in `examples/configs/`:

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

6. Collect data:

   ```bash
   mini-pi0 collect-maniskill-demos \
     --config examples/configs/maniskill3_your_task.yaml \
     --out_hdf5 data/robomimic/maniskill/your_task/dataset.hdf5 \
     --num_episodes 1000 \
     --collector_name your_collector_name \
     --collector_backend scripted \
     --overwrite
   ```

## Validation

Run focused checks during development:

```bash
python -m pytest tests/test_config.py tests/test_model_registry.py tests/test_fm_architecture.py \
  tests/test_training_stability_controls.py tests/test_eval_weight_source.py -q
```

Run the full suite:

```bash
python -m pytest -q
```
