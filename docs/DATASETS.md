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
