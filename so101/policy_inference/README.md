# SO-101 mini-pi0 Inference

This runtime deploys the trained 16M or 25M dual-camera flow-matching policy
at a 30 Hz robot-control rate. It accepts wrist and base RGB frames plus the
six measured joint positions and predicts 32 absolute joint-position commands.
There is no IK or Ruckig in this learned-policy path.

LeRobot Real-Time Chunking (RTC) generates overlapping chunks on a background
thread while the control loop consumes the current queue. RTC guidance is
performed in normalized model space. Commands are then denormalized with the
statistics embedded in the checkpoint, validated against `handy_bot.json`, and
slew-limited before they can reach the follower.

## Hardware-free checks

On Apple Silicon, `auto` selects MPS with FP32. FP16/BF16 are available as
explicit experiments but are slower for these small models on the tested Mac.

```bash
# Full synthetic model -> RTC -> safety soak test with Rerun
.venv/bin/python -m so101.policy_inference \
  --variant 16m --device mps --duration 30

# Benchmark both checkpoints, including guided RTC
.venv/bin/python -m so101.policy_inference \
  --benchmark-both --device mps --flow-steps 8

# Replay real recorded camera/state observations without opening the servo bus
.venv/bin/python -m so101.policy_inference \
  --variant 16m --device mps \
  --replay-dataset data/lerobot/so101_pick_place_blocks_dual_cam \
  --replay-repo-id local/so101-pick-place-blocks-dual-cam \
  --replay-episode 0 --replay-max-frames 300
```

You can validate the live cameras through the complete inference stack while
the arm remains untouched. Camera arguments must preserve the training order:
wrist first, then base.

```bash
.venv/bin/python -m so101.policy_inference \
  --variant 16m --device mps --duration 30 \
  --camera wrist=1:180 --camera base=0:0 \
  --camera-native-size wrist=640x480 \
  --camera-native-size base=1920x1080 \
  --camera-output-size wrist=480x480 \
  --camera-output-size base=640x360
```

None of these commands opens the servo bus because they omit
`--enable-motors`.

## Hardware execution

Only proceed after the camera dry-run is stable and the arm is supported and
unloaded. Replace the port and camera IDs with the discovered devices:

```bash
.venv/bin/python -m so101.policy_inference \
  --variant 16m --device mps --enable-motors \
  --robot-port /dev/cu.usbmodem5B610338651 \
  --camera wrist=1:180 --camera base=0:0 \
  --camera-native-size wrist=640x480 \
  --camera-native-size base=1920x1080 \
  --camera-output-size wrist=480x480 \
  --camera-output-size base=640x360
```

Runtime keys are `p` to pause/resume, `b` for a bounded return to the saved
base pose, and `q` to quit. Base return invalidates the policy queue and remains
paused until `p` is pressed.

Hardware mode requires the six-joint calibration at:

```text
~/.cache/huggingface/lerobot/calibration/robots/so_follower/handy_bot.json
```

The runtime saturates small (at most 2-degree/2-percent) learned overshoots at
the exact calibrated endpoint and rejects larger or non-finite chunks atomically.
It also limits per-cycle joint motion, holds measured position on camera staleness or
queue starvation, and pauses after a 15-degree arm following error persists
for three cycles (10 percent for the gripper). Rerun is decimated and uses
explicit viewer/server memory limits so visualization cannot grow without
bound or block the motor loop.

## Performance defaults

- Control: 30 Hz
- Flow integration: 8 Euler steps
- RTC execution horizon: 10
- Replan interval: 6 control frames
- RTC guidance cap: 5
- Fixed sampling noise: enabled (seed 42)
- MPS `auto`: FP32
- CUDA `auto`: BF16 when supported, otherwise FP16

On the development M1 Pro, guided RTC p95 was approximately 61 ms for the 16M
checkpoint and 69 ms for the 25M checkpoint. Because inference is asynchronous,
both fit inside a 32-action queue while the output loop continues at 30 Hz.
