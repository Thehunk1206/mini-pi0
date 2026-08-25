# SO-101 Android phone teleoperation

Control an SO-101 follower using Android WebXR phone motion. The application
maps the calibrated phone pose to end-effector targets, solves joint commands
through inverse kinematics, and provides:

- Android WebXR over USB with ADB port forwarding
- a localhost desktop console for articulated URDF tracking and telemetry
- Rerun 3D actual/commanded robot tracking and motor telemetry
- a terminal `B` shortcut for return-to-base and phone recalibration
- per-motor voltage, current, load, and temperature telemetry
- 30 Hz session logs and automatic pre-failure incident captures
- explicit cleanup warnings when the servo bus cannot receive torque-disable

## Project layout

```text
so101/phone_teleop/
├── teleoperate.py              # Phone, IK, robot control, and base return
├── flight_recorder.py          # Electrical telemetry and incident capture
├── control_ui.py               # Thread-safe localhost desktop console
├── visualization.py            # Rerun robot/end-effector 3D view and trail
├── urdf_model.py               # Lightweight URDF forward kinematics
├── dashboard/                  # Desktop UI HTML, CSS, and JavaScript
├── kinematics/
│   └── so101_kinematics.urdf   # Self-contained kinematics model
├── tests/                      # Hardware-independent safety/logging tests
└── README.md
```

## Current robot configuration

| Setting | Value |
| --- | --- |
| Robot | SO-101 follower |
| LeRobot calibration ID | `handy_bot` |
| Servo port | `/dev/cu.usbmodem5B610338651` |
| Calibration | `~/.cache/huggingface/lerobot/calibration/robots/so_follower/handy_bot.json` |
| Base position | `~/.cache/huggingface/lerobot/base_positions/robots/so_follower/handy_bot.json` |
| Phone | Android WebXR |
| Control rate | 30 Hz |
| Phone-controlled axes | XYZ translation and gripper; roll/pitch/yaw disabled by the global switch |
| Translation axis map | robot X ← mapped phone Y, robot Y ← mapped phone X, Z unchanged |
| Translation gain | `0.5` per axis |
| Cartesian step limit | `0.10 m` per control cycle |
| Joint target clamp | disabled, matching the LeRobot example |
| Gripper speed factor | `20.0` |
| STS3215 acceleration | LeRobot default (`254`) |
| Desktop console | `http://127.0.0.1:8001` |
| LeRobot version | `0.6.1` |

The motion values reproduce the previously working LeRobot phone example. The
single global `ENABLE_PHONE_ORIENTATION` near the top of `teleoperate.py`
selects the orientation behavior. It defaults to `False` for XYZ-only control;
set it to `True` to restore roll, pitch, and yaw. The desktop console cannot
change the mapping or reset its reference.

Two additional globals define the translation frame without embedding it in
the processor implementation:

```python
PHONE_TRANSLATION_AXIS_MAP = {"x": "y", "y": "x", "z": "z"}
PHONE_TRANSLATION_AXIS_SIGNS = {"x": 1.0, "y": 1.0, "z": 1.0}
```

The map fixes the observed lateral/forward exchange while preserving vertical
motion. Each map key is a robot output axis and its value is the
LeRobot-mapped phone input axis. If an axis is correct but moves in the opposite
direction, change only that output's sign from `1.0` to `-1.0`.

## Installation

Use the repository's existing `.venv` and install the pinned phone, Rerun, and
desktop-console dependencies:

```bash
.venv/bin/python -m pip install -r so101/phone_teleop/requirements.txt
```

### Install ADB

ADB is provided by the Android SDK Platform Tools. Use the installation method
for the host operating system.

#### macOS

```bash
brew install --cask android-platform-tools
```

#### Ubuntu or Debian

```bash
sudo apt update
sudo apt install adb android-sdk-platform-tools-common
```

The `android-sdk-platform-tools-common` package installs common Android `udev`
rules. If `adb devices` reports insufficient permissions, add the current user
to `plugdev`, then log out and back in:

```bash
sudo usermod -aG plugdev "$LOGNAME"
```

#### Fedora

```bash
sudo dnf install android-tools
```

#### Arch Linux

```bash
sudo pacman -S android-tools android-udev
```

`android-udev` provides additional device rules and is useful when the phone is
not detected as a regular user.

For other Linux distributions, install the latest standalone Linux archive
from the [official Android SDK Platform Tools page](https://developer.android.com/tools/releases/platform-tools),
extract it, and add its `platform-tools` directory to `PATH`.

On Android, enable Developer options and USB debugging, connect the phone by
USB, unlock it, and accept the computer authorization prompt.

### Linux servo-port setup

The configured `/dev/cu.usbmodem5B610338651` servo path is specific to macOS.
On Linux, connect the Waveshare servo adapter and find its device:

```bash
ls -l /dev/ttyACM* /dev/ttyUSB* 2>/dev/null
```

It commonly appears as `/dev/ttyACM0` or `/dev/ttyUSB0`. Update the `port`
value in `so101/phone_teleop/teleoperate.py` before launching.

If opening the servo port returns `Permission denied`, add the current user to
the serial-device group and then log out and back in:

```bash
sudo usermod -aG dialout "$USER"
```

Confirm access after signing in again:

```bash
id
ls -l /dev/ttyACM0
```

## Before every run

1. Support the arm and clear its workspace.
2. Confirm the motor model and use the matching externally powered servo
   supply. Do not power the servos from USB alone.
3. Close the joint UI or any other process using the servo port.
4. Confirm the Android device is authorized:

   ```bash
   adb devices
   ```

5. Forward the WebXR HTTPS/WebSocket port over USB:

   ```bash
   adb reverse tcp:4443 tcp:4443
   ```

## Start teleoperation

From the repository root, run:

```bash
.venv/bin/python -m so101.phone_teleop.teleoperate
```

The launcher starts the phone server before opening the servo bus. On Android:

1. Open `https://127.0.0.1:4443` in Chrome.
2. Accept the local certificate warning if shown.
3. Hold the phone screen-up with its top edge pointing in the robot's forward
   direction.
4. Touch and move on the WebXR page when the terminal asks for calibration.
5. Release **Hold to move** before the robot connects.
6. Wait for `Starting teleop loop` before commanding motion.

The desktop console opens automatically at <http://127.0.0.1:8001>. It is
available during phone calibration and throughout teleoperation. Rerun opens
automatically after phone calibration. If the browser does not open, visit the
desktop-console address manually.

Phone translation controls the end-effector while roll, pitch, and yaw deltas
are zeroed. The end-effector keeps the orientation captured when **Hold to
move** establishes the latched phone and robot references. To enable rotation,
change this global before starting the process:

```python
ENABLE_PHONE_ORIENTATION = True
```

## Phone controls

| Control | Behavior |
| --- | --- |
| **Hold to move** | Enables phone-pose control while held. Releasing it holds the last end-effector command. |
| **Scale** | Scales phone motion. Keep it at or below `0.25` while power and motion limits are being validated. |
| Phone **A** | Commands positive gripper velocity (opens the configured gripper). |
| Phone **B** | Commands negative gripper velocity (closes the configured gripper). |
| Gripper engage/disengage | Enables or disables gripper interaction in the WebXR interface. |

The phone's **B** button controls the gripper. It is different from pressing
`B` in the terminal.

## Desktop control console

The console runs in a background web-server thread and never talks to the
servo bus directly. The 30 Hz teleoperation thread remains the only owner of
the serial connection and the phone processor.

It provides:

- an orbitable and zoomable articulated skeleton generated from the checked-in
  SO-101 URDF
- green measured joint geometry, orange commanded geometry, and a cyan
  measured end-effector trail
- actual/commanded joint position and live voltage, current, load, and
  temperature for every servo
- loop time, Cartesian tracking error, minimum bus voltage, and summed current
- a read-only summary of the active XYZ-only or six-DoF control mapping
- **Return to base + recalibrate**, equivalent to terminal `B`

There are no live motion sliders. This prevents the dashboard from changing or
resetting the phone reference during a teleoperation session.

The checked-in URDF is intentionally kinematics-only and contains no visual or
collision meshes. Consequently, both desktop and Rerun views render the full
articulated link/joint skeleton rather than a textured CAD model.

## Return to base and recalibrate

Focus the launching terminal and press `B`; Enter is not required. One keypress
runs the complete reset sequence:

1. Phone motion is ignored while all six joints return to the saved base pose.
2. Release **Hold to move**.
3. The phone calibration prompt restarts automatically.
4. Hold the phone in the new screen-up/forward reference pose and press **Hold
   to move** once to capture it.
5. Release **Hold to move** once more.
6. The next press starts motion from a fresh IK reference at base.

Base return uses a 30°/s arm-joint command trajectory and a 25%/s gripper
trajectory. Completion tolerances are 2° for arm joints and 3% for the gripper.
A 15° following-error limit stops a blocked base return.

Before exiting, use terminal `B` to return to base. Then press `Ctrl+C`. If the
bus has already failed, software may be unable to disable torque; switch off
motor power before touching the arm.

## 3D visualization and flight-recorder logs

Rerun opens with a synchronized 3D robot and end-effector view:

- green/orange link skeletons: measured and commanded URDF poses
- green marker and trail: measured end-effector position from forward kinematics
- RGB arrows: measured end-effector orientation axes
- orange marker: commanded end-effector position
- red line: Cartesian difference between measured and commanded positions
- plots: actual/target XYZ and Cartesian error in millimeters

The measured 3D pose is a forward-kinematics estimate derived from the joint
encoders. It does not measure structural flex, gripper deflection, backlash, or
external displacement with a camera.

The same Rerun session displays phone/action values and these per-motor signals:

- voltage in volts
- estimated current in milliamps
- load in percent
- temperature in °C

Every run writes:

```text
logs/phone_teleop/session_<timestamp>.jsonl
```

The session log contains measured positions, requested IK commands, commands
sent to the robot, actual/target Cartesian positions, Cartesian error, phone
state, control-loop timing, and the latest electrical readings at 30 Hz.
Voltage and current are sampled from the servo bus at 5 Hz; load and temperature
are sampled at 1 Hz.

When an exception occurs, the final approximately 20 seconds are copied into:

```text
logs/phone_teleop/incident_<timestamp>_<number>.json
```

The incident includes the exact control phase, exception, traceback, requested
and sent commands, joint and Cartesian positions, phone inputs, and last valid
electrical samples. A second incident is written when torque-disable also fails
during shutdown.

The terminal prints the generated paths and a once-per-second summary:

```text
TELEMETRY Vmin=5.2V Itotal=850mA Ipeak=elbow_flex:320mA Tmax=36C
```

The servo telemetry cannot measure the minimum of a sub-200 ms voltage
transient after the bus stops responding. Use an oscilloscope or power analyzer
for the true rail minimum.

## LeRobot control mapping and safe use

The phone processor order and motion values match the previously working
LeRobot example:

1. `MapPhoneActionToRobotAction` maps the Android pose and buttons.
2. `RemapPhoneTranslation` applies `PHONE_TRANSLATION_AXIS_MAP` and
   `PHONE_TRANSLATION_AXIS_SIGNS`; the current configuration swaps X/Y and
   preserves Z.
3. When `ENABLE_PHONE_ORIENTATION` is `False`, one stateless filter zeros only
   `target_wx`, `target_wy`, and `target_wz`.
4. `EEReferenceAndDelta` applies the example's XYZ gain `0.5`.
5. `EEBoundsAndSafety` uses the example's `0.10 m` jump threshold.
6. `GripperVelocityToJoint` uses speed factor `20`.
7. `InverseKinematicsEEToJoints` solves the joint commands.

No live reference-changing settings, joint-target clamp, or custom
acceleration write is inserted in this path. Setting the orientation global to
`True` omits only the orientation filter; the configured robot-frame axis map
remains active.

The phone should still be used as a clutch: hold **Move**, make a small motion,
release it, reposition the phone, and engage it again. Keep phone **Scale** at
`0.25` or below until voltage and following behavior are validated.

Previous flight-recorder analysis identified single-frame target changes above
20°, following errors above 40°, and a servo-rail drop from approximately
5.2 V to 4.5 V before all six motors stopped replying. Because the restored
mapping is more responsive, use it cautiously and validate the power path:

- keep phone **Scale** at `0.25` or below
- move slowly and avoid sudden direction reversals
- do not test with an undersized or mismatched supply
- keep the hardware power switch reachable
- stop testing after any whole-bus communication dropout

Repeated `There is no status packet` errors for all IDs indicate a bus-wide
power or communication loss, not an individual IK failure. After such an error,
do not assume the shutdown torque-disable command succeeded.

The Waveshare Bus Servo Adapter (A) passes the input motor voltage through
without regulation and is specified for a maximum of 5 A. Use a correctly
rated external servo supply and keep the power path short. Do not disable the
servo's voltage or current protection. See the
[Waveshare adapter FAQ](https://docs.waveshare.com/Bus_Servo_Adapter_A/FAQ),
[LeRobot phone example](https://github.com/huggingface/lerobot/blob/main/examples/phone_to_so100/teleoperate.py),
and [SO follower configuration](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/config_so_follower.py).

## Troubleshooting

### Phone page does not open

Verify the USB device and forwarding rule:

```bash
adb devices
adb reverse --list
```

The list should include `tcp:4443 tcp:4443`. Refresh
`https://127.0.0.1:4443` after the Python server starts.

On Linux, if `adb devices` reports `no permissions`, confirm that the user is in
`plugdev`, install the distribution's Android `udev` rules, reconnect the phone,
and restart ADB:

```bash
adb kill-server
adb start-server
adb devices
```

### Servo port is busy

Find the process that owns it:

```bash
lsof /dev/cu.usbmodem5B610338651
```

Stop the previous teleoperation or joint-UI process before starting another.

### All motors report no status packet

Check, in order:

1. External servo power and its voltage/current rating.
2. Driver-board power switch and DC connector.
3. The first servo cable and all daisy-chain connectors.
4. Whether rapid commands caused a voltage drop in the latest incident file.
5. Whether another process attempted to use the bus.

Do not solve a bus-wide power dropout by only increasing serial retry counts.

## Tests

The recorder, desktop API, URDF, and Cartesian-calculation tests do not
require hardware:

```bash
.venv/bin/python -m unittest discover \
  -s so101/phone_teleop/tests \
  -p 'test_*.py'
```
