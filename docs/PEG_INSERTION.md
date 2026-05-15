# PegInsertionSide Notes

PegInsertionSide is harder task because visual alignment and contact
resolution dominate the final phase. The policy can often pick the peg and move
toward the hole, but reliable insertion needs better hole visibility and richer
failure diagnostics.

## Current Environment

The repo adds a local ManiSkill task wrapper:

```text
MiniPi0PegInsertionSide-v1
```

Implementation:

```text
mini_pi0/sim/maniskill3_peginsertion_env.py
```

It extends the ManiSkill `PegInsertionSide-v1` environment with two close
hole-facing cameras:

```text
hole_left_camera
hole_right_camera
```

The training image keys are:

```yaml
robot:
  image_keys:
    - agentview_image
    - robot0_eye_in_hand_image
    - hole_left_image
    - hole_right_image
```

The conversion mapping is:

```text
agentview_image=base_camera
robot0_eye_in_hand_image=hand_camera
hole_left_image=hole_left_camera
hole_right_image=hole_right_camera
```

## Dataset Preparation

The full replay/contact/conversion flow is wrapped by:

```bash
NUM_ENVS=16 tools/prepare_peginsertion_holecam_contacts.sh
```

This does three things:

1. Replays ManiSkill motion-planning demonstrations through
   `MiniPi0PegInsertionSide-v1`.
2. Adds contact and passive-force observations.
3. Converts the result into Robomimic-style HDF5.

Default training output:

```text
data/robomimic/maniskill/peginsertionside/mp/rgbd_pd_ee_delta_pose_holecam_contacts.hdf5
```

For a short smoke run:

```bash
COUNT=5 NUM_ENVS=1 tools/prepare_peginsertion_holecam_contacts.sh
```

## Contact Features

The contact-aware config currently uses this state set:

```yaml
robot:
  state_keys:
    - robot0_eef_pos
    - robot0_eef_quat
    - robot0_gripper_qpos
    - robot_arm_passive_qf
    - leftfinger_force_norm
    - rightfinger_force_norm
    - pair_leftfinger_peg_force_norm
    - pair_rightfinger_peg_force_norm
    - pair_peg_box_force_norm
    - pair_leftfinger_peg_contact
    - pair_rightfinger_peg_contact
    - pair_peg_box_contact

model:
  prop_dim: 24
```

The key diagnostic signals are:

- `pair_leftfinger_peg_force_norm`
- `pair_rightfinger_peg_force_norm`
- `pair_peg_box_force_norm`
- `pair_leftfinger_peg_contact`
- `pair_rightfinger_peg_contact`
- `pair_peg_box_contact`

`robot_arm_passive_qf` is passive dynamics compensation, not a direct measured
contact torque. Treat it as proprio/dynamics context, not as the main contact
signal.

## Contact Overlay Example

The following video overlays synchronized contact-force traces on the base
camera for `demo_886`, which was selected because it has strong peg-box contact:

<video src="../assets/peginsertion_contact_overlay_demo_886_h264.mp4" controls muted loop width="720"></video>

The overlay plots:

- peg-box force norm
- left finger-peg force norm
- right finger-peg force norm
- total peg force norm
- contact activation bands for peg-box and finger-peg contacts

This is useful for checking that contact timing lines up with the visual
trajectory: finger contact should appear during grasp, and peg-box force should
spike during the insertion/contact phase.

## Training

Current config:

```text
examples/configs/maniskill3_peginsertion_motionplanning_transformer_vit_hist3_medium_holecam_contacts.yaml
```

Train:

```bash
mini-pi0 train \
  --config examples/configs/maniskill3_peginsertion_motionplanning_transformer_vit_hist3_medium_holecam_contacts.yaml
```

If memory is tight with four cameras and ViT:

```bash
mini-pi0 train \
  --config examples/configs/maniskill3_peginsertion_motionplanning_transformer_vit_hist3_medium_holecam_contacts.yaml \
  --set train.batch_size=32
```

## Evaluation

Contact features during live eval are available through the ManiSkill adapter.
For contact-enabled policies, use sequential CPU simulation:

```yaml
simulator:
  env_kwargs:
    sim_backend: physx_cpu

eval:
  vectorized: false
  num_envs: 1
```

Run eval with close-camera grid videos:

```bash
mini-pi0 eval \
  --config examples/configs/maniskill3_peginsertion_motionplanning_transformer_vit_hist3_medium_holecam_contacts.yaml \
  --set eval.checkpoint=runs/<experiment>/run1/checkpoints/best.pt \
  --set eval.action_stats_path=runs/<experiment>/run1/artifacts/action_stats.json \
  --set eval.record_grid=true \
  --set eval.grid_cameras='["base_camera","hole_left_camera","hole_right_camera"]'
```

This saves one success/failure grid per camera:

```text
success_grid_base_camera_3x3.mp4
failure_grid_base_camera_3x3.mp4
success_grid_hole_left_camera_3x3.mp4
failure_grid_hole_left_camera_3x3.mp4
success_grid_hole_right_camera_3x3.mp4
failure_grid_hole_right_camera_3x3.mp4
```

## Current Result

Current reference run:

```text
runs/maniskill3-peginsertion-motionplanning-transformer-vit-hist3-medium-holecam-contacts/run1/final_eval_best_seed24
```

Summary:

| Metric | Value |
| --- | ---: |
| Success rate | 0.0% |
| Episodes | 10 |
| Mean episode length | 460.9 steps |
| Mean reward | 1245.4 |
| Mean inference speed | 49.0 ms/chunk |
| Mean action clipping | 0.12% |
| Failure modes | 9 timeout after progress, 1 drop/unstable |

The low clipping rate is good. The failures are not action saturation failures;
they are mostly alignment/insertion failures after partial progress.

## Debugging Direction

The current failure labels are still coarse. For PegInsertion, better rollout
diagnostics should be phase-based:

- peg grasped
- peg dropped
- peg tip to hole distance
- peg-hole axis angle error
- insertion depth
- peg-box contact force
- sustained high force without insertion progress

These can separate:

- no grasp
- grasped but dropped
- never reached hole
- reached hole but misaligned
- jammed at contact
- partial insertion timeout

The next improvement should compute those task-specific metrics directly from
peg and box poses plus contact features during eval.
