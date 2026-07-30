# Benchmark Checkpoints

These are the trained checkpoints behind the results in
[`docs/TASK_BENCHMARK.md`](../../docs/TASK_BENCHMARK.md). Checkpoint binaries
are stored with Git LFS. Each directory also includes the matching action
statistics, resolved training configuration, evaluation summary, and evaluation
provenance.

| Task | Policy | Success | `best.pt` size | Additional checkpoint | Directory |
| --- | --- | ---: | ---: | --- | --- |
| StackCube-v1 | FM transformer + ViT medium, hist2, `pd_joint_pos` | 95.5% | 1,222,775,822 bytes | [`best_success.pt`](stackcube-vit-medium-hist2-pd-joint-pos/best_success.pt) | [`stackcube-vit-medium-hist2-pd-joint-pos`](stackcube-vit-medium-hist2-pd-joint-pos) |
| PegInsertionSide-v1 | FM transformer + ViT medium, hist3, hole cameras and contacts, `pd_ee_delta_pose` | 10.0% | 1,222,750,350 bytes | Not produced | [`peginsertion-vit-medium-hist3-holecam-contacts`](peginsertion-vit-medium-hist3-holecam-contacts) |
| StackPyramid-v1 | FM transformer + ViT medium, hist2, `pd_ee_delta_pose` | 26.0% | 1,222,754,702 bytes | [`best_success.pt`](stackpyramid-vit-medium-hist2-pd-ee-delta-pose/best_success.pt) | [`stackpyramid-vit-medium-hist2-pd-ee-delta-pose`](stackpyramid-vit-medium-hist2-pd-ee-delta-pose) |
| PullCubeTool-v1 | FM transformer + ViT small, hist2, `pd_ee_delta_pose` | 31.0% | 783,486,254 bytes | Not produced | [`pullcubetool-vit-small-hist2-pd-ee-delta-pose`](pullcubetool-vit-small-hist2-pd-ee-delta-pose) |

## Download

Install Git LFS before cloning:

```bash
git lfs install
git clone git@github.com:Thehunk1206/mini-pi0.git
```

To clone without downloading every checkpoint and then fetch one task:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone git@github.com:Thehunk1206/mini-pi0.git
cd mini-pi0
git lfs pull \
  --include="assets/checkpoints/pullcubetool-vit-small-hist2-pd-ee-delta-pose/best.pt"
```

Use `best.pt` together with the `action_stats.json` in the same directory. The
checkpoint and action statistics must not be mixed across tasks or runs.
The benchmark success rates were measured with the linked `best.pt` files.
`best_success.pt` is an additional training-selected checkpoint and is included
only when the source run produced it.

## Integrity

```text
37808c4331361d8fb85d81dfa495b09c6844ff8d69d1359bcfe6daacc051f8cf  stackcube-vit-medium-hist2-pd-joint-pos/best.pt
3165ad7a6978d35720b701f940ad55c48bd3dc29a7ac9961dbf43c9d201eae00  stackcube-vit-medium-hist2-pd-joint-pos/best_success.pt
d139a410695efe83d48ecc48015959510812768ec41db985afc44612138e41f0  peginsertion-vit-medium-hist3-holecam-contacts/best.pt
5dee69548e6831e34a120a5006441fcfa537e14c5cf9ec9169c653003ad44c19  stackpyramid-vit-medium-hist2-pd-ee-delta-pose/best.pt
81a035b1f6693de863e502f263096f7c3ad20cf6f940be7ea0d03a295788e0fe  stackpyramid-vit-medium-hist2-pd-ee-delta-pose/best_success.pt
069ee65e42cce6b484efeeb49ea2b6f1bffdf11e2b146ac7547ca6a1d3f14a64  pullcubetool-vit-small-hist2-pd-ee-delta-pose/best.pt
```
