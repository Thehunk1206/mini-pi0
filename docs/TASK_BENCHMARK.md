# Task Benchmark

This document tracks policy benchmark results across ManiSkill tasks. It is meant
as the compact source of truth for model comparisons; README keeps only the most
visible highlights.

## Current Results

| Task | Dataset / Controller | Policy | Obs | Episodes | Success | CI95 | Mean Len |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| StackCube-v1 | motion planning / `pd_joint_pos` | FM transformer + ViT medium | hist2, base + wrist cameras | 200 | 95.5% | 92.5%-98.0% | 166.2 |
| PegInsertionSide-v1 | motion planning / `pd_ee_delta_pose` | FM transformer + ViT medium + hole cameras + contact state | hist3, base + wrist + hole cameras | 100 | 10.0% | 5.0%-16.0% | 474.8 |
| StackPyramid-v1 | motion planning / `pd_ee_delta_pose` | FM transformer + ViT medium | hist2, base + wrist cameras | 50 | 26.0% | 14.0%-38.0% | 444.9 |
| PullCubeTool-v1 | motion planning / `pd_ee_delta_pose` | FM transformer + ViT small | hist2, base + wrist cameras | 100 | 31.0% | 23.0%-41.0% | 408.4 |

## Reporting Rules

- Report simulator success from ManiSkill native `info["success"]` when available.
- Record the controller and observation cameras with every result.
- Prefer at least 100 episodes for hard tasks and 200 episodes for stable headline results.
