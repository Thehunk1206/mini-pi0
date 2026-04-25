# Dataloader Guide

This document explains how dataset loading works in `mini-pi0` from config to model-ready episodes.

## What The Loader Returns

All dataset formats are converted to one canonical in-memory structure:

```python
EpisodeData(
    obs: list[dict[str, np.ndarray]],
    actions: np.ndarray,  # [T, action_dim], float32
)
```

This means training / eval code can ignore original source format differences.

## Public Loader API

The stable entrypoint is [`mini_pi0/dataset/episodes.py`](/Users/tauhidkhan/Desktop/projects/VLA/mini-pi0/mini_pi0/dataset/episodes.py):

- `list_supported_dataset_formats()`
- `load_episodes_robomimic(...)`
- `load_episodes_lerobot(...)`
- `iter_lerobot_episode_images(...)`
- `load_episodes_from_config(cfg)`

## Internal Structure

The implementation is split by responsibility:

- [`mini_pi0/dataset/episodes.py`](/Users/tauhidkhan/Desktop/projects/VLA/mini-pi0/mini_pi0/dataset/episodes.py): thin dispatcher/facade
- [`mini_pi0/dataset/_robomimic_loader.py`](/Users/tauhidkhan/Desktop/projects/VLA/mini-pi0/mini_pi0/dataset/_robomimic_loader.py): HDF5 loading
- [`mini_pi0/dataset/_lerobot_loader.py`](/Users/tauhidkhan/Desktop/projects/VLA/mini-pi0/mini_pi0/dataset/_lerobot_loader.py): HF LeRobot loading and image streaming
- [`mini_pi0/dataset/_feature_attach.py`](/Users/tauhidkhan/Desktop/projects/VLA/mini-pi0/mini_pi0/dataset/_feature_attach.py): optional precomputed feature attachment
- [`mini_pi0/dataset/_loader_utils.py`](/Users/tauhidkhan/Desktop/projects/VLA/mini-pi0/mini_pi0/dataset/_loader_utils.py): aliasing, nested key extraction, image normalization
- [`mini_pi0/dataset/types.py`](/Users/tauhidkhan/Desktop/projects/VLA/mini-pi0/mini_pi0/dataset/types.py): `EpisodeData` type

## Loading Flow

`load_episodes_from_config(cfg)` does:

1. Resolve key lists:
- `state_keys = effective_state_keys(cfg.robot)`
- `image_keys = effective_image_keys(cfg.robot)`

2. Validate image fallback shape:
- `data.fallback_image_hw` must be `[H, W]`

3. Dispatch by format:
- `data.format=robomimic_hdf5` -> robomimic loader
- `data.format=lerobot_hf` -> lerobot loader

4. Optionally attach precomputed features when:
- `data.observation_mode in {precomputed, feature, features}`
- source path is `data.precomputed_features_path`

## Key Aliasing

The loader supports alias resolution to reduce config friction across datasets.

Examples:
- `observation.images.base_0_rgb` <-> `agentview_image`
- `observation.images.right_wrist_0_rgb` <-> `robot0_eye_in_hand_image`
- `observation.state.eef_pos` <-> `robot0_eef_pos`
- `observation.state.object` <-> `object-state`

So user configs can stay consistent even if source datasets use alternative names.

## Robomimic Loader Behavior

Expected HDF5 layout:

- `/<data_group>/<demo_k>/actions`
- `/<data_group>/<demo_k>/obs/<key>`

Important behavior:

- trims episode length to the minimum shared length across actions/proprio/images
- missing image keys are replaced with zero frames of shape `fallback_image_hw`
- missing required proprio keys raise an explicit error

## LeRobot Loader Behavior

Expected sample keys:

- `action` (or custom `data.lerobot_action_key`)
- `episode_index` (or custom `data.lerobot_episode_index_key`)
- configured `robot.image_keys`
- configured state keys

Important behavior:

- groups rows by `episode_index`
- supports nested dotted key extraction (for example `observation.images.base_0_rgb`)
- supports `load_images=False` fast path for feature-only training (no video decode when possible)
- supports video backend override (`data.lerobot_video_backend`, `pyav` recommended on macOS)

## Streaming API For Vision Precompute

`iter_lerobot_episode_images(...)` yields episode-grouped image frames:

- output: `(episode_seq_idx, frames_by_key)`
- `frames_by_key` maps each image key to a list of `uint8 HxWx3` frames

This keeps memory bounded during precompute and avoids loading all episodes at once.

## Precomputed Feature Attachment

Two storage layouts are supported:

1. Directory mode:
- `<dir>/ep_000000.npy`
- `<dir>/ep_000001.npy`

2. Archive mode:
- `<file>.npz` with arrays named `ep_000000`, `ep_000001`, ...

Validation checks:

- feature arrays must be 2D
- per-episode timesteps must match `len(episode.obs)`
- missing episode keys/files raise explicit errors

Feature vectors are attached to each timestep under:
- `data.precomputed_feature_key` (default `vision_feat`)

## Config Fields You Will Use Most

- `data.format`
- `data.robomimic_hdf5`
- `data.robomimic_data_group`
- `data.lerobot_repo_id`
- `data.lerobot_action_key`
- `data.lerobot_episode_index_key`
- `data.lerobot_video_backend`
- `data.observation_mode`
- `data.precomputed_features_path`
- `data.precomputed_feature_key`
- `robot.image_key` / `robot.image_keys`
- `robot.state_keys`

## Debugging Checklist

If loading fails:

1. Verify format + source path/repo:
- `data.format`
- `data.robomimic_hdf5` or `data.lerobot_repo_id`

2. Verify observation keys:
- `robot.image_keys`
- `robot.state_keys`

3. For LeRobot on macOS:
- use `data.lerobot_video_backend='pyav'`

4. For precomputed mode:
- verify `data.precomputed_features_path`
- verify episode ordering and `ep_XXXXXX` naming
