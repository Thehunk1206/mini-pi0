# Robot <-> Dataset Mapping

This document maps:

1. Robots available in your local robosuite installation.
2. Equivalent Hugging Face datasets for those robosuite tasks.
3. Compatibility status with current `mini_pi0` pipeline.

## Generate Mapping From CLI

```bash
python -m mini_pi0 robot-dataset-map
```

This prints:

- detected local robosuite robots
- robomimic dataset matrix (task/type/variant)
- LeRobot equivalent dataset suggestions
- compatibility notes

## Current Compatibility Summary

- Fully compatible now (single-arm Panda, 7D action):
  - Lift
  - PickPlaceCan
  - NutAssemblySquare
  - ToolHang
- Needs additional two-arm mapping:
  - TwoArmTransport (`action_dim=14`)

## Hugging Face Sources

- Official robomimic datasets:
  - `robomimic/robomimic_datasets`
- LeRobot equivalents (robosuite-origin examples):
  - `robotgeneralist/robosuite_lift_ph`
  - `robotgeneralist/robosuite_can_ph`
  - `robotgeneralist/robosuite_square_ph`

## Why Some Mappings Are Not "Direct"

LeRobot datasets use LeRobot feature naming and storage conventions
(Parquet + optional video). `mini_pi0` now supports native LeRobot loading
through `data.format=lerobot_hf`, but keys must be mapped correctly:

- `data.lerobot_repo_id`
- `data.lerobot_action_key`
- `data.lerobot_episode_index_key`
- `data.lerobot_video_backend` (`pyav` recommended on macOS)
- `robot.image_key`
- `robot.proprio_keys`
