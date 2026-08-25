# SO-101 Android phone teleoperation

Control an SO-101 follower using Android WebXR phone motion. The application
maps the calibrated phone pose to end-effector targets, solves joint commands
through inverse kinematics, and provides:

- Android WebXR over USB with ADB port forwarding
- Rerun visualization
- a terminal `B` shortcut for return-to-base and phone recalibration
- per-motor voltage, current, load, and temperature telemetry
- 30 Hz session logs and automatic pre-failure incident captures
- explicit cleanup warnings when the servo bus cannot receive torque-disable

## Project layout

```text
phone_to_so101/
├── teleoperate.py              # Phone, IK, robot control, and base return
├── flight_recorder.py          # Electrical telemetry and incident capture
├── SO101/
│   └── so101_kinematics.urdf   # Self-contained kinematics model
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
| LeRobot version | `0.6.1` |

These values are currently defined directly in `teleoperate.py`; the launcher
does not yet expose command-line configuration flags.

## Installation

Use the repository's existing `.venv`. Install the phone and visualization
extras if they are not already present:

```bash
.venv/bin/python -m pip install 'lerobot[phone,viz]==0.6.1'
```

This branch uses `rerun-sdk 0.33.1`, which satisfies LeRobot 0.6.1's supported
Rerun range:

```bash
.venv/bin/python -m pip install 'rerun-sdk==0.33.1'
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
value in `phone_to_so101/teleoperate.py` before launching.

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
.venv/bin/python phone_to_so101/teleoperate.py
```

The launcher starts the phone server before opening the servo bus. On Android:

1. Open `https://127.0.0.1:4443` in Chrome.
2. Accept the local certificate warning if shown.
3. Hold the phone screen-up with its top edge pointing in the robot's forward
   direction.
4. Touch and move on the WebXR page when the terminal asks for calibration.
5. Release **Hold to move** before the robot connects.
6. Wait for `Starting teleop loop` before commanding motion.

Rerun opens automatically after phone calibration.

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

## Visualization and flight-recorder logs

Rerun displays phone/action values and these per-motor signals:

- voltage in volts
- estimated current in milliamps
- load in percent
- temperature in °C

Every run writes:

```text
logs/phone_teleop/session_<timestamp>.jsonl
```

The session log contains measured positions, generated joint commands, phone
state, control-loop timing, and the latest electrical readings at 30 Hz.
Voltage and current are sampled from the servo bus at 5 Hz; load and temperature
are sampled at 1 Hz.

When an exception occurs, the final approximately 20 seconds are copied into:

```text
logs/phone_teleop/incident_<timestamp>_<number>.json
```

The incident includes the exact control phase, exception, traceback, commands,
positions, phone inputs, and last valid electrical samples. A second incident
is written when torque-disable also fails during shutdown.

The terminal prints the generated paths and a once-per-second summary:

```text
TELEMETRY Vmin=5.2V Itotal=850mA Ipeak=elbow_flex:320mA Tmax=36C
```

The servo telemetry cannot measure the minimum of a sub-200 ms voltage
transient after the bus stops responding. Use an oscilloscope or power analyzer
for the true rail minimum.

## Current safety limitation

The retained upstream example still maps IK output directly to joint targets.
It currently uses:

- 0.5 m/m end-effector translation scaling in the Python pipeline
- a 10 cm Cartesian step limit
- no accumulating joint-command slew limiter during phone motion
- LeRobot's default maximum servo acceleration

Flight-recorder analysis identified single-frame target changes above 20°,
following errors above 40°, and a servo-rail drop from approximately 5.2 V to
4.5 V before all six motors stopped replying. Until trajectory limiting and
lower acceleration are implemented:

- keep phone **Scale** at `0.25` or below
- move slowly and avoid sudden direction reversals
- do not test with an undersized or mismatched supply
- keep the hardware power switch reachable
- stop testing after any whole-bus communication dropout

Repeated `There is no status packet` errors for all IDs indicate a bus-wide
power or communication loss, not an individual IK failure. After such an error,
do not assume the shutdown torque-disable command succeeded.

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
