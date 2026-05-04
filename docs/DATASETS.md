# Dataset Guide

This repository supports two training dataset formats:

- `robomimic_hdf5`
- `lerobot_hf` (native Hugging Face LeRobot loader)

It also supports two observation-conditioning workflows:

- `image`: train directly from image observations.
- `precomputed`: precompute vision embeddings once, then train policy on cached features.

## Official References

- robomimic v0.1 datasets:
  - https://robomimic.github.io/docs/datasets/robomimic_v0.1.html
- robomimic HDF5 structure and conventions:
  - https://robomimic.github.io/docs/v0.4/datasets/overview.html
- LeRobot dataset format:
  - https://huggingface.co/docs/lerobot/lerobot-dataset-v3

## 1) robomimic HDF5 Format

Config example:

```yaml
data:
  format: robomimic_hdf5
  robomimic_hdf5: data/robomimic/lift/ph/low_dim_v15.hdf5
  robomimic_data_group: data
  fallback_image_hw: [84, 84]
```

Download from CLI:

```bash
python -m mini_pi0 download-robomimic \
  --task lift \
  --dataset_type ph \
  --hdf5_type low_dim \
  --download_dir data/robomimic
```

Expected minimum layout:

```text
/data/demo_x/
  actions
  obs/
    <proprio keys>
    <image key>   # optional for low_dim datasets
```

## 2) LeRobot Native HF Format

Config example:

```yaml
data:
  format: lerobot_hf
  lerobot_repo_id: robotgeneralist/robosuite_lift_ph
  lerobot_action_key: action
  lerobot_episode_index_key: episode_index
  lerobot_local_files_only: false
  lerobot_video_backend: pyav
  fallback_image_hw: [84, 84]

robot:
  image_key: observation.images.right_wrist_0_rgb
  state_keys:
    - observation.state.eef_pos
    - observation.state.eef_quat
    - observation.state.tool
    - observation.state.object
```

Train directly from HF LeRobot:

```bash
python -m mini_pi0 train \
  --config examples/configs/robosuite_lift.yaml \
  --set data.format=lerobot_hf \
  --set data.lerobot_repo_id='robotgeneralist/robosuite_lift_ph' \
  --set data.lerobot_action_key='action' \
  --set data.lerobot_episode_index_key='episode_index' \
  --set data.lerobot_video_backend='pyav' \
  --set robot.image_key='observation.images.right_wrist_0_rgb' \
  --set robot.state_keys="['observation.state.eef_pos','observation.state.eef_quat','observation.state.tool','observation.state.object']"
```

Notes:

- `lerobot` Python package is required for `lerobot_hf` format.
- Keys are dataset-specific. Use the right `image_key`, `proprio_keys`, and `action_key`.
- `robot.state_keys` is the preferred way to define policy state inputs.
  If omitted, the pipeline falls back to `robot.proprio_keys`.
- For `robotgeneralist/robosuite_can_ph`, prefer wrist camera conditioning:
  - `robot.image_key: observation.images.right_wrist_0_rgb`
- No conversion to HDF5 is required for native LeRobot loading.
- On macOS, use `lerobot_video_backend: pyav` to avoid torchcodec / FFmpeg dylib errors.

## 3) CLI Flags for Training

You can set format-specific flags directly:

```bash
python -m mini_pi0 train --config examples/configs/robosuite_lift.yaml \
  --data_format lerobot_hf \
  --lerobot_repo_id robotgeneralist/robosuite_lift_ph \
  --lerobot_action_key action \
  --lerobot_episode_index_key episode_index \
  --lerobot_video_backend pyav
```

Or for robomimic:

```bash
python -m mini_pi0 train --config examples/configs/robosuite_lift.yaml \
  --data_format robomimic_hdf5 \
  --robomimic_hdf5 data/robomimic/lift/ph/low_dim_v15.hdf5
```

## 4) Pre-Training Checklist

- `data.format` is set correctly.
- Dataset path/repo exists and is accessible.
- `robot.proprio_keys` and `robot.image_key` match dataset observation keys.
- Action dimension in data matches robot/model expectation.
- For offline/cached LeRobot use, set `data.lerobot_local_files_only=true`.
- If you hit `libtorchcodec` / `libavutil` errors on macOS, set `data.lerobot_video_backend=pyav`.

## 5) Robot <-> Dataset Mapping

```bash
python -m mini_pi0 robot-dataset-map
```

Detailed mapping guide:

- [docs/ROBOT_DATASET_MAPPING.md](/Users/tauhidkhan/Desktop/projects/VLA/mini-pi0/docs/ROBOT_DATASET_MAPPING.md)

## 6) Precomputed Vision Features (DINO / SSL)

List model options:

```bash
python -m mini_pi0 vision-models
python -m mini_pi0 vision-models --backend timm
python -m mini_pi0 vision-models --backend torchvision
```

Supported torchvision options:
- `resnet18`
- `resnet34`
- `mobilenet_v3_small`
- `efficientnet_b0`

Recommended timm options (good starting points on limited VRAM):
- `vit_tiny_patch16_224`
- `vit_small_patch16_224`
- `convnext_tiny`
- `efficientnet_b0`
- `vit_small_patch14_dinov2.lvd142m`

Precompute embeddings:

```bash
python -m mini_pi0 precompute-vision \
  --config examples/configs/robosuite_lift_robomimic.yaml \
  --vision_backend timm \
  --vision_model_name vit_small_patch14_dinov2.lvd142m \
  --vision_batch_size 64 \
  --vision_image_size 224 \
  --precomputed_features_path data/features/lift_dinov2_vits14.npz
```

DINOv3 example (timm):

```bash
python -m mini_pi0 precompute-vision \
  --config examples/configs/robosuite_can_vision.yaml \
  --vision_backend timm \
  --vision_model_name 'timm/vit_base_patch16_dinov3.lvd1689m' \
  --vision_pretrained \
  --precomputed_features_path data/features/can_dinov3_timm_vitb16.npz
```

Train from cached features:

```bash
python -m mini_pi0 train \
  --config examples/configs/robosuite_lift_robomimic.yaml \
  --observation_mode precomputed \
  --precomputed_features_path data/features/lift_dinov2_vits14.npz \
  --set model.obs_mode=feature
```

Notes:
- Feature `.npz` keys are `ep_000000`, `ep_000001`, ... and must match demo ordering used in training.
- For eval/deploy with feature-conditioned policy, runtime feature extraction is used when `vision.use_runtime_extractor=true`.
- Some official HF DINOv3 repos are gated; if access is denied, use the timm DINOv3 model id above.
- For `lerobot_hf`, precompute runs in streaming mode (episode-by-episode) to keep memory usage low.

---

# ManiSkill Oracle Mixture Data Collection

This guide documents the ManiSkill3 data collection workflow for the custom
multi-object tray task. The main collection target is a robomimic-style HDF5
file containing success-only scripted oracle demonstrations with mixed behavior
profiles.

## Task And Dataset

Primary config:

```text
examples/configs/maniskill3_multiobject_tray.yaml
```

Task id:

```text
MiniPi0MultiObjectTray-v1
```

Dataset output is controlled by the YAML block:

```yaml
dataset_collection:
  output_hdf5: data/robomimic/custom/maniskill3_multiobject/ph/dataset_600.hdf5
  total_episodes: 600
  num_envs: 128
  max_steps: 1000
  only_success: true
  reject_long_episodes: true
  mix:
    core: 0.5
    recovery: 0.25
    suboptimal: 0.25
  difficulty: balanced
```

Training reads the same file through:

```yaml
data:
  format: robomimic_hdf5
  robomimic_hdf5: data/robomimic/custom/maniskill3_multiobject/ph/dataset_600.hdf5
  robomimic_data_group: data
  fallback_image_hw: [256, 256]
  n_demos: 600
  chunk_size: 16
  action_stats_path: data/robomimic/custom/maniskill3_multiobject/ph/dataset_600_action_stats.json
```

Keep `data.robomimic_hdf5` and `dataset_collection.output_hdf5` aligned unless
you intentionally want to train from a different dataset.

## Collection Command

Use the oracle mixture collector:

```bash
.venv/bin/python -m mini_pi0.cli.main collect-maniskill-oracle-mixture \
  --config examples/configs/maniskill3_multiobject_tray.yaml \
  --overwrite
```

This command uses the YAML values directly. Do not pass `--collector_backend
scripted`; `collect-maniskill-oracle-mixture` is already a scripted oracle
collector and does not use MPLib.

## Smoke Test Command

Before launching a long run, use a small override-based smoke test:

```bash
.venv/bin/python -m mini_pi0.cli.main collect-maniskill-oracle-mixture \
  --config examples/configs/maniskill3_multiobject_tray.yaml \
  --output_hdf5 artifacts/oracle_mix_smoke_10eps.hdf5 \
  --total_episodes 10 \
  --num_envs 10 \
  --max_steps 1000 \
  --overwrite
```

Expected result:

- `episodes_saved == 10`
- `success_rate == 1.0`
- HDF5 contains `/data/demo_0`, `/data/demo_1`, ...
- each demo contains `obs/base_image` and `obs/hand_image`

## Behavior Profiles

The mixture collector expands one simple YAML block into three profile types.

### Core

Clean scripted demonstrations:

- randomized object count
- randomized object positions and yaw
- randomized object colors, lighting, cameras, and physics when domain
  randomization is enabled
- randomized tray placement targets
- no explicit mid-episode perturbation

### Recovery

Successful demonstrations with one recovery event:

- object displacement
- object nudging
- bowl escape recovery
- noisy grasp / slip-like recovery

Only successful episodes are retained when `only_success: true`.

### Suboptimal

Successful but imperfect demonstrations:

- action noise
- slower policy execution
- grasp-angle jitter

These demos increase coverage without keeping failed or thrashing trajectories.

## Difficulty Presets

`dataset_collection.difficulty` controls internal perturbation magnitudes:

- `safe`: lower diversity, higher acceptance rate
- `balanced`: recommended default for production collection
- `aggressive`: wider perturbations, more rejection risk

If collection success drops, reduce difficulty before changing the task geometry.

## Cameras And Observations

The current training camera set is:

```yaml
simulator:
  camera_names: [base_camera, hand_camera]

robot:
  image_key: base_image
  image_keys: [base_image, hand_image]
```

Mapping:

```text
base_camera -> obs/base_image
hand_camera -> obs/hand_image
```

`base_camera` is the fixed external view. `hand_camera` is mounted to the Panda
hand link and follows the gripper.

State keys used for policy input:

```yaml
robot:
  state_keys:
    - robot0_eef_pos
    - robot0_eef_quat
    - robot0_gripper_qpos
    - observation.state.object
    - observation.state.place_targets
```

The dataset also stores additional canonical fields when present, including
object masks, placed masks, and task progress.

## Domain Randomization

Domain randomization is configured under:

```yaml
simulator:
  env_kwargs:
    domain_randomization:
```

Current randomization includes:

- camera pose and FOV jitter
- lighting intensity and direction jitter
- object/tray/bowl/table visual color jitter
- active object slot randomization
- spawn yaw randomization
- spawn radius jitter
- randomized tray placement targets
- object mass, friction, and restitution ranges

The placement target is randomized per object and stored in:

```text
obs/observation.state.place_targets
```

The oracle uses this target for placement, but success is still based on being
inside the tray volume. This prevents the robot from retrying already-dropped
objects just because a cylinder or sphere rolls slightly after release.

## Bowl Containment

The source bowl visual mesh collision is disabled:

```yaml
source_bowl_collision: false
```

Objects are kept inside the bowl using a simple collision-only containment ring:

```yaml
source_bowl_containment: true
source_bowl_wall_segments: 16
source_bowl_wall_thickness: 0.008
source_bowl_wall_height: 0.04
```

This avoids nonconvex bowl mesh gripper collisions while preventing cylinders or
spheres from escaping the source area during reset and rollout.

## Resolution And Parallel Envs

GPU camera memory scales roughly with:

```text
num_envs * number_of_cameras * width * height
```

If ManiSkill raises `RuntimeError: cannot create buffer`, reduce in this order:

1. `dataset_collection.num_envs`
2. `simulator.camera_width`, `simulator.camera_height`
3. `simulator.env_kwargs.sensor_width`, `sensor_height`

Do not blindly set `num_envs=256` with two high-resolution cameras. Validate the
chosen resolution/env count with a 10-demo smoke test first.

## Resume / Interrupted Runs

Current behavior:

- `--overwrite` deletes and restarts the HDF5.
- Without `--overwrite`, collection fails if the output file already exists.
- The oracle mixture collector does not yet provide a safe `--append` resume
  command.

If a run is interrupted, inspect the partial file before deciding what to do:

```bash
.venv/bin/python - <<'PY'
import h5py
p = 'data/robomimic/custom/maniskill3_multiobject/ph/dataset_600.hdf5'
with h5py.File(p, 'r') as f:
    data = f['data']
    demos = sorted(data.keys(), key=lambda x: int(x.split('_')[1]))
    counts = {}
    for demo in demos:
        profile = str(data[demo].attrs.get('profile_type', 'unknown'))
        counts[profile] = counts.get(profile, 0) + 1
    print('episodes:', len(demos))
    print('profiles:', counts)
PY
```

For now, the safest options are:

- keep the partial file and collect the remaining demos into a second HDF5, then
  merge intentionally later
- restart with `--overwrite`
- implement proper append/resume support before continuing in-place

## HDF5 Layout

Each accepted demo is written under:

```text
/data/demo_<n>/
```

Required datasets:

```text
actions
rewards
dones
obs/base_image
obs/hand_image
obs/<state keys>
info/success_fraction
info/reward_total
info/reward_progress
info/reward_place
info/reward_terminal
info/reward_shaping
info/reward_penalties
info/reward_step
```

Important attributes:

```text
num_samples
success_bool
final_success_fraction
placed_count
total_objects
collector_type
profile_type
difficulty
seed
perturbation_type
perturbation_magnitude
oracle_retry_count
oracle_phase_timeout_count
oracle_target_switch_count
oracle_max_phase_steps
```

## Visualizing Collected Episodes

The demos already contain RGB frames. Render side-by-side previews directly from
HDF5:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import h5py
import imageio.v2 as imageio
import numpy as np

h5_path = Path('data/robomimic/custom/maniskill3_multiobject/ph/dataset_600.hdf5')
out_dir = Path('artifacts')
out_dir.mkdir(parents=True, exist_ok=True)
stem = h5_path.stem

with h5py.File(h5_path, 'r') as f:
    demos = sorted(f['data'].keys(), key=lambda x: int(x.split('_')[1]))[:5]
    for demo in demos:
        obs = f['data'][demo]['obs']
        base = np.asarray(obs['base_image'], dtype=np.uint8)
        hand = np.asarray(obs['hand_image'], dtype=np.uint8)
        out_path = out_dir / f'{stem}_{demo}_side_by_side.mp4'
        with imageio.get_writer(out_path, fps=20, codec='libx264', quality=8, macro_block_size=16) as writer:
            for left, right in zip(base, hand):
                writer.append_data(np.concatenate([left, right], axis=1))
        print(out_path)
PY
```

The left panel is `base_image`; the right panel is `hand_image`.

## Training From The Collected Dataset

After collection, train using the same YAML:

```bash
.venv/bin/python -m mini_pi0.cli.main train \
  --config examples/configs/maniskill3_multiobject_tray.yaml
```

The training code reads:

```yaml
data.robomimic_hdf5
data.robomimic_data_group
data.n_demos
data.action_stats_path
robot.image_keys
robot.state_keys
```

Make sure `data.n_demos` does not exceed the number of demos actually present in
the HDF5.

## Evaluation After Training

Evaluate with:

```bash
.venv/bin/python -m mini_pi0.cli.main eval \
  --config examples/configs/maniskill3_multiobject_tray.yaml
```

Key eval fields:

```yaml
eval:
  checkpoint: checkpoints/best.pt
  action_stats_path: data/robomimic/custom/maniskill3_multiobject/ph/dataset_600_action_stats.json
  n_episodes: 50
  execute_steps: 6
  n_flow_steps: 10
  max_steps: 1000
  strict_parity: true
```

`strict_parity: true` is useful because it catches mismatches between dataset
state/image keys and checkpoint expectations.
