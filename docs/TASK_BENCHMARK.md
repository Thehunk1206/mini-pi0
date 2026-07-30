# Task Benchmark

This document tracks policy benchmark results across ManiSkill tasks. It is meant
as the compact source of truth for model comparisons; README keeps only the most
visible highlights.

## Current Results

| Task | Dataset / Controller | Policy | Obs | Episodes | Success | CI95 | Mean Len | Checkpoint | Best-success checkpoint |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| StackCube-v1 | motion planning / `pd_joint_pos` | FM transformer + ViT medium | hist2, base + wrist cameras | 200 | 95.5% | 92.5%-98.0% | 166.2 | [best.pt](../assets/checkpoints/stackcube-vit-medium-hist2-pd-joint-pos/best.pt) | [best_success.pt](../assets/checkpoints/stackcube-vit-medium-hist2-pd-joint-pos/best_success.pt) |
| PegInsertionSide-v1 | motion planning / `pd_ee_delta_pose` | FM transformer + ViT medium + hole cameras + contact state | hist3, base + wrist + hole cameras | 100 | 10.0% | 5.0%-16.0% | 474.8 | [best.pt](../assets/checkpoints/peginsertion-vit-medium-hist3-holecam-contacts/best.pt) | Not produced |
| StackPyramid-v1 | motion planning / `pd_ee_delta_pose` | FM transformer + ViT medium | hist2, base + wrist cameras | 50 | 26.0% | 14.0%-38.0% | 444.9 | [best.pt](../assets/checkpoints/stackpyramid-vit-medium-hist2-pd-ee-delta-pose/best.pt) | [best_success.pt](../assets/checkpoints/stackpyramid-vit-medium-hist2-pd-ee-delta-pose/best_success.pt) |
| PullCubeTool-v1 | motion planning / `pd_ee_delta_pose` | FM transformer + ViT small | hist2, base + wrist cameras | 100 | 31.0% | 23.0%-41.0% | 408.4 | [best.pt](../assets/checkpoints/pullcubetool-vit-small-hist2-pd-ee-delta-pose/best.pt) | Not produced |

The reported benchmark metrics use `best.pt`. A `best_success.pt` link is
included only when that source training run saved a distinct checkpoint.

The complete release index, matching action statistics, resolved configs,
evaluation evidence, download instructions, and SHA-256 checksums are in
[`assets/checkpoints`](../assets/checkpoints/README.md).

## Reporting Rules

- Report simulator success from ManiSkill native `info["success"]` when available.
- Record the controller and observation cameras with every result.
- Prefer at least 100 episodes for hard tasks and 200 episodes for stable headline results.

## ReinFlow Status

PickCube scratch and PegInsertion checkpoint fine-tuning have passed short
engineering updates with native vector environments. They are not benchmark
results and are excluded from the table above. PegInsertion RL will be added
only after three training seeds and paired deterministic evaluation on episode
seeds `10000-10099`.
