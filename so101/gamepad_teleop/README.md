# SO-101 gamepad teleoperation

Control the physical SO-101 with an Xbox-style controller while retaining the
shared calibrated IK, joint-limit validation, telemetry, flight recorder,
base return, and visualization helpers in [`../teleop/`](../teleop/).

The controller uses a reference-style planar IK layout:

```text
pygame/SDL axes
    -> rescaled deadzone + cubic response
    -> 250 ms stick-velocity ramp
    -> integrated wrist-pivot reach/height target
       + direct shoulder-pan/wrist-flex/wrist-roll targets
    -> model-derived reachable-workspace projection
    -> position-only shoulder/elbow IK on the active folding branch
    -> calibrated limit/jump validation
    -> direct arm position commands

LT/RT -> direct gripper position increments -> gripper servo
```

There is deliberately no Ruckig or other online trajectory generator in the
gamepad path. The bounded stick velocities create small target increments and
invalid finite/limit/jump targets are discarded atomically, similar to the
reference controller. At the reach boundary, only the IK-driven shoulder/elbow
pair is held; independent pan, wrist, and gripper commands remain available.
Phone teleoperation continues to use Ruckig separately.

References:

- [LeRobot gamepad teleoperator](https://github.com/huggingface/lerobot/blob/main/src/lerobot/teleoperators/gamepad/teleop_gamepad.py)
- [LeRobot HIL-SERL gamepad controls](https://github.com/huggingface/lerobot/blob/main/docs/source/hilserl.mdx)
- [SO-101 Bench Xbox simulation teleoperation](https://github.com/5hadytru/so101_bench#teleoperation-with-an-xbox-controller)

## Controls

| Input | Command |
| --- | --- |
| Left stick horizontal | Shoulder pan (`velocity` or calibrated full-span `absolute`) |
| Left stick vertical | Planar reach forward/backward |
| D-pad up / down | Wrist-pivot height up/down |
| Right stick vertical | Wrist-flex down/up |
| Right stick horizontal | Wrist-roll velocity |
| LT / RT | LT opens and RT closes at 200%/s |
| B | Return to the saved base pose |
| Y / X | Log episode success / failure |
| Back / View | Log episode rerecord request |
| A / Start / RB | Currently unused; stick motion is immediately active |

The dataset recorder assigns the otherwise-unused buttons to episode control:
A starts an episode, Y saves it, X and Back discard it, and Start/Menu
finalizes the dataset and exits. B only returns to base and does not save,
discard, start, or stop an episode. If an episode is active, every base-return
camera/state/action frame remains part of that episode.

The default input settings are a `0.12` deadzone, `0.65` cubic expo blend, and
a `250 ms` full-scale velocity ramp to remove abrupt starts and stops without
using Ruckig. Maximum reach/height speeds are `0.12/0.12 m/s`; shoulder pan is
`45 deg/s`, wrist flex/roll are `50/70 deg/s`, and the direct gripper is `200%/s`
(about `0.5 s` for full calibrated travel). Planar offsets are bounded to
`-0.10/+0.40 m` from the latched reference and height to `-0.12/+0.15 m`.
Position-only IK controls shoulder lift and elbow flex at the wrist pivot;
shoulder pan, wrist flex, and wrist roll are direct bounded commands. No EEF
orientation target is applied. Outward motion stops just before the ambiguous straight-elbow
singularity instead of folding onto the opposite branch. Every joint command
is clipped to the physical motor ranges loaded from LeRobot's
`~/.cache/huggingface/lerobot/calibration/robots/so_follower/handy_bot.json`,
and discontinuous IK solutions are rejected. The calibrated limits are also
installed into the Placo solver, rather than leaving its narrower nominal URDF
inequalities active. The URDF remains the link-geometry and visualization
model. Gamepad direct mode has no motion profile. Distinct controller
vibrations indicate a calibrated joint endpoint, rejected IK jump, or
Cartesian workspace/extension boundary.

The straight-elbow coordinate is derived at startup from the loaded kinematic
model within the calibrated elbow interval; calibrated motor `0°` is not
assumed to be a straight arm. The current model derives approximately `-73.8°`
and keeps a `2°` margin on the active folding branch. Near that boundary,
unreachable XYZ targets are projected onto a safe reach sphere so vertical and
tangential motion can slide along the workspace edge. A final branch guard
holds only shoulder lift/elbow if IK still attempts to cross the singularity,
while pan, wrist, and gripper continue normally.

For physical direct control, startup overrides LeRobot's maximum `254` servo
acceleration with gamepad-only RAM values (`90` for the arm and `254` for the
faster gripper) and verifies each readback. These settings are volatile: they
do not modify `handy_bot.json`, EEPROM calibration, or the phone teleoperator's
next connection.

## Install

From the repository root:

```bash
.venv/bin/python -m pip install -r so101/gamepad_teleop/requirements.txt
```

## Verify the controller without hardware

This command never constructs a robot or opens the serial/servo bus:

```bash
.venv/bin/python -m so101.gamepad_teleop.teleoperate \
  --diagnose-controller --diagnose-seconds 10
```

Move each stick and press each control. For the current controller, pygame
reports `Xbox 360 Controller`, 6 axes, and 15 buttons. Confirm LT/RT appear as
axes 4/5 and B appears as button 1. These are SDL's standardized
GameController indices, independent of the device's raw HID map.

Test vibration independently without opening the servo bus:

```bash
.venv/bin/python -m so101.gamepad_teleop.teleoperate --test-vibration
```

Boundary feedback emits one strong pulse when a commanded axis first tries to
move farther into a Cartesian workspace boundary or the final `5°` of its
calibrated joint range. Static near-limit joints—including joints in the saved
base pose—and motion away from an endpoint stay silent. Feedback re-arms after
`0.5 s` of safe motion instead of buzzing continuously. This package intentionally retains
pygame 2.6.1 for the controller's working live input stream. SDL's rumble
packet is rejected by Apple's current `XboxGamepad` driver, so this exact
`045e:028e` macOS path uses a direct, prefixed XUSB output report instead. The
haptic endpoint is opened only for each short write and immediately closed;
keeping it open would starve pygame and freeze all buttons and axes. Other
controllers fall back to SDL GameController/joystick rumble.

## Control the kinematic robot in Rerun

This command opens the controller plots and simulated SO-101 in Rerun without
constructing a robot or opening the servo bus:

```bash
.venv/bin/python -m so101.gamepad_teleop.simulator
```

Move the sticks to exercise the same planar IK and direct joint validation used
by physical teleoperation. The controlled Cartesian point is the wrist pivot;
pan, wrist flex, and wrist roll remain independent. LT/RT drives the simulated
gripper, and B resets the simulated robot to home. Rerun renders all 17 articulated
visual parts from the official SO-101 URDF and its 13 cached STL assets. The
opaque robot is the current simulated state and the translucent orange robot is
the next command. This is a kinematic control-stack check; it does not model
gravity, contacts, or grasp physics.

The simulator starts from a bent central pose with usable motion in every
direction. It loads `handy_bot.json` when available, derives the same active IK
branch and reach boundary as hardware, and rejects implausible joint jumps while
retaining the last safe target.

Stop with `Ctrl+C`.

To test full-span pan, map the full left-stick range to the exact shoulder-pan
endpoints in `handy_bot.json`. The speed argument limits how quickly the target
can travel across that span; it does not shrink the available range:

```bash
.venv/bin/python -m so101.gamepad_teleop.simulator \
  --pan-mode absolute --pan-speed-deg-s 35
```

In this mode, releasing the spring-centered stick returns pan to the calibrated
midpoint. The default `velocity` mode instead holds the current pan when the
stick is centered.

## Start the physical teleoperator

Support the unloaded arm, center both sticks, clear the workspace, make sure no
other program owns the servo port, and run:

```bash
.venv/bin/python -m so101.gamepad_teleop.teleoperate
```

After validating clearance in the simulator, enable calibrated full-span pan
on hardware with:

```bash
.venv/bin/python -m so101.gamepad_teleop.teleoperate \
  --pan-mode absolute --pan-speed-deg-s 35
```

Gamepad teleoperation opens Rerun by default and does not start a web-control
server. All motion, gripper, base-return, and episode-label commands come from
the controller. Rerun uses the same official articulated SO-101 mesh as the
hardware-free simulator and overlays measured and commanded poses. Startup
prints the loaded calibration path and normalized command range for every
motor. To run without any graphical interface, use:

```bash
.venv/bin/python -m so101.gamepad_teleop.teleoperate --no-rerun
```

For another servo device, pass `--robot-port /dev/your-device`. A controller
read failure stops the loop and runs the existing robot disconnect/torque-off
cleanup path.

## Record a LeRobot dataset

Install the dataset-enabled dependencies, then launch the dedicated Rerun-only
recorder (it never starts the web UI):

```bash
.venv/bin/python -m pip install -r so101/gamepad_teleop/requirements.txt

.venv/bin/python -m so101.gamepad_teleop.record \
  --task "pick up the cube" \
  --dataset-repo-id local/so101-pick-cube \
  --dataset-root data/lerobot/so101_pick_cube \
  --camera wrist=0:180
```

The default camera is also `wrist=0:180`, which corrects the currently
upside-down wrist camera. Add as many named OpenCV cameras as needed by
repeating the option and manually assigning each ID:

```bash
.venv/bin/python -m so101.gamepad_teleop.record \
  --task "pick up the cube" \
  --dataset-repo-id local/so101-pick-cube \
  --dataset-root data/lerobot/so101_pick_cube \
  --camera wrist=0:180 \
  --camera overview=1:0 \
  --camera side=2:90
```

Rotation is clockwise and accepts `0`, `90`, `180`, or `270`. Camera IDs,
rotations, and actual output shapes are validated before the servo bus opens
and shown in Rerun. If the output already exists, choose one explicit mode:

```bash
# Append episodes to a compatible dataset.
.venv/bin/python -m so101.gamepad_teleop.record \
  --task "pick up the cube" \
  --dataset-repo-id local/so101-pick-cube \
  --dataset-root data/lerobot/so101_pick_cube \
  --camera wrist=0:180 \
  --append

# Start fresh. The previous directory is moved to a timestamped .backup path.
.venv/bin/python -m so101.gamepad_teleop.record \
  --task "pick up the cube" \
  --dataset-repo-id local/so101-pick-cube \
  --dataset-root data/lerobot/so101_pick_cube \
  --camera wrist=0:180 \
  --overwrite
```

Append requires the same camera names/shapes, video mode, and state schema.
Overwrite is recoverable: it archives rather than deletes the previous data.

Rerun places a small articulated measured/commanded SO-101 view in the
upper-left, a measured/action joint-position plot to its right, and all live
camera previews across the bottom. Previews are JPEG-compressed and decimated
to 5 Hz to bound Rerun memory; the dataset still stores every 30 Hz frame as
MP4 by default. The status panel keeps the live waiting/recording state, save
and discard events, episode counts, task and output path visible together with
the gamepad recording commands.

Each LeRobotDataset frame contains:

- `observation.state`: the six measured joint positions only;
- `action`: the six position commands actually accepted by the robot;
- one `observation.images.<name>` stream per configured camera; and
- the natural-language task.

Electrical telemetry remains available to the separate safety/incident flight
recorder, but is deliberately excluded from the learning dataset. Camera
videos are encoded sequentially to keep peak memory predictable. An unfinished
episode is discarded on Start/Menu, Ctrl+C, an exception, or shutdown; only Y
creates a dataset episode. The recorder writes locally and does not push to
Hugging Face.

Replay a finalized episode in the same hardware-free Rerun layout:

```bash
.venv/bin/python -m so101.gamepad_teleop.replay \
  --dataset-repo-id local/so101-pick-cube \
  --dataset-root data/lerobot/so101_pick_cube \
  --episode-index 0
```

The replay view keeps the articulated measured/commanded mesh and saved camera
streams, but the time-series panel contains only the six measured
`state/<joint>.pos` and six commanded `action/<joint>.pos` channels. It never
opens the controller, cameras, or servo bus. Do not replay an active recording;
save the episode with Y and finalize it with Start/Menu first.

JSONL flight-recorder sessions are written under `logs/gamepad_teleop/` with
controller input, measured/requested/commanded joints, Cartesian error,
calibration/workspace events, episode markers, and sampled electrical telemetry.
The simulator never constructs a robot or opens a serial bus. Neither path
models floor contact, gravity, or collision avoidance, so keep the physical arm
supported and clear of the ground and obstacles.

## Hardware-free tests

```bash
.venv/bin/python -m pytest -q so101/gamepad_teleop/tests
```
