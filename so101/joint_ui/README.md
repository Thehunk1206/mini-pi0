# SO-101 Joint Control UI

A local browser dashboard for inspecting and moving every calibrated joint of
an SO-101 follower. The dashboard provides bounded joint sliders, live motor
telemetry, persistent base-position controls, and explicit torque and emergency
controls.

The server listens only on `127.0.0.1`; it is intentionally unavailable to
other devices on the network.

## Hardware and calibration

Before starting, confirm that:

- the SO-101 is fully assembled and supported;
- the servo bus is connected to the computer and to a suitable external power
  supply;
- no other process is using the servo serial port; and
- the follower calibration contains all six joints with motor IDs 1 through 6.

The expected joint mapping is:

| Motor ID | Joint | UI unit |
| --- | --- | --- |
| 1 | `shoulder_pan` | degrees |
| 2 | `shoulder_lift` | degrees |
| 3 | `elbow_flex` | degrees |
| 4 | `wrist_flex` | degrees |
| 5 | `wrist_roll` | degrees |
| 6 | `gripper` | percent |

The default robot ID is derived from the calibration filename. For example,
`handy_bot.json` selects the `handy_bot` LeRobot calibration.

## Installation

From the repository root, activate the project environment and install the UI
requirements:

```bash
source .venv/bin/activate
uv pip install -r so101/joint_ui/requirements.txt
```

If `uv` is unavailable, use pip:

```bash
source .venv/bin/activate
python -m pip install -r so101/joint_ui/requirements.txt
```

The requirements install FastAPI, Uvicorn, and the Feetech-enabled LeRobot
package used to communicate with the SO-101 servo bus.

## Find the servo port

Make sure the servo bus is powered and connected, then run:

```bash
lerobot-find-port
```

Typical ports are:

- macOS: `/dev/cu.usbmodem...`
- Linux: `/dev/ttyACM0` or `/dev/ttyUSB0`

On Linux, if the port exists but cannot be opened, add the current user to the
serial-port group and then log out and back in:

```bash
sudo usermod -aG dialout "$USER"
```

## Run the dashboard

Use the calibration and serial port for your robot:

```bash
python -m so101.joint_ui.app \
  --serial-port /dev/cu.usbmodem5B610338651 \
  --calibration ~/.cache/huggingface/lerobot/calibration/robots/so_follower/handy_bot.json
```

On Linux, replace the serial port with the detected device, for example:

```bash
python -m so101.joint_ui.app \
  --serial-port /dev/ttyACM0 \
  --calibration ~/.cache/huggingface/lerobot/calibration/robots/so_follower/handy_bot.json
```

Open <http://127.0.0.1:8000> in a browser on the same computer. The serial port
may also be changed in the dashboard before selecting **Connect**.

Use another web port if port 8000 is occupied:

```bash
python -m so101.joint_ui.app --web-port 8001
```

## Recommended operating sequence

1. Support the arm and clear its full workspace.
2. Power the servo bus and start the dashboard.
3. Confirm the serial port, then select **Connect**. The controller reads the
   measured pose and uses it as the initial goal before enabling torque.
4. Move one slider at a time while watching the actual position, command,
   temperature, voltage, current, load, and safety log.
5. To define a parked pose, release torque, position and support the arm by
   hand, select **Capture base position**, and then enable hold.
6. Use **Return to base position** for a controlled, rate-limited move to the
   saved pose.
7. Select **Disconnect** before unplugging the servo bus.

## Controls

- **Connect** validates the calibration stored in the motors, reads the current
  pose, aligns the motor goals to it, and enables holding torque.
- **Release torque** makes the arm back-drivable. Support the arm first because
  gravity can make it fall.
- **Enable hold** reads the current pose and enables torque without commanding a
  jump to an old target.
- **Capture base position** saves the freshly measured six-joint pose.
- **Return to base position** moves all joints to the saved pose using the same
  motion and following-error limits as the sliders. Torque remains enabled at
  the destination.
- **Emergency stop** immediately releases torque while keeping the serial
  connection open. The `Escape` key performs the same action.
- **Disconnect** releases torque and closes the serial connection.

Each joint card displays three positions:

- **Actual** is the latest measured position.
- **Target** is the position requested by the slider.
- **Command** is the rate-limited intermediate setpoint sent to the motor.

## Base-position storage

By default, the captured pose is stored at:

```text
~/.cache/huggingface/lerobot/base_positions/robots/so_follower/handy_bot.json
```

It is loaded when the server starts. Capturing a new base position replaces the
stored pose. A manual slider command cancels an active return-to-base motion.

Returning to base is not an emergency stop: it deliberately moves the robot.
Use **Emergency stop** or `Escape` when motion must stop immediately.

## Telemetry

The dashboard polls joint positions at 20 Hz and samples motor telemetry once
per second. Each joint card shows:

- temperature in degrees Celsius;
- input voltage;
- estimated current draw in milliamps; and
- signed servo load in percent.

The current/load panel keeps the most recent 120 samples, equivalent to about
two minutes. **Clear view** hides existing samples in the browser; it does not
write or delete a log file.

Current is converted from the STS3215 `Present_Current` register at 6.5 mA per
count. Signed load describes motor output effort and direction. It is not a
direct measurement of gripping force, weight, or grams; force estimation
requires calibration against a known force or load sensor.

## Safety behavior

The controller applies the following limits:

- slider values are clamped to the saved calibration range;
- arm joints are rate-limited to 45 degrees per second;
- the gripper is rate-limited to 35 percent per second;
- a 15-degree arm-joint or 20-percent gripper following error releases torque;
- a 55°C motor temperature releases torque; and
- a communication error stops the control worker and attempts to release
  torque.

The gripper also uses a reduced torque/current configuration. These software
safeguards do not replace supervision or access to the hardware power switch.

## Configuration

Command-line options:

| Option | Purpose | Default |
| --- | --- | --- |
| `--serial-port` | Servo bus device | `SO101_SERIAL_PORT` or the built-in macOS port |
| `--calibration` | LeRobot calibration JSON | `SO101_CALIBRATION_FILE` or `handy_bot.json` in the LeRobot cache |
| `--base-position` | Persistent base-pose JSON | `SO101_BASE_POSITION_FILE` or the LeRobot cache path shown above |
| `--web-port` | Local dashboard port | `8000` |

The corresponding environment variables can be used instead:

```bash
export SO101_SERIAL_PORT=/dev/ttyACM0
export SO101_CALIBRATION_FILE="$HOME/.cache/huggingface/lerobot/calibration/robots/so_follower/handy_bot.json"
export SO101_BASE_POSITION_FILE="$HOME/.cache/huggingface/lerobot/base_positions/robots/so_follower/handy_bot.json"
python -m so101.joint_ui.app
```

## Troubleshooting

### `There is no status packet`

This means the servo bus did not receive a response from one or more motors.
Check that the external servo power is on, the USB and servo cables are secure,
the selected port is correct, and no calibration, teleoperation, or other UI
process is using the same port. Stop the server, power-cycle the bus if safe,
and reconnect.

### Calibration mismatch or missing joint

The UI requires all six named joints and the exact ID mapping shown above. Run
the LeRobot follower calibration again and pass the resulting JSON path with
`--calibration`. In particular, an absent `wrist_roll` entry prevents startup.

### Torque is enabled but a joint does not move

Check the safety log for a following-error, temperature, or communication stop.
Also verify adequate servo-bus power and inspect voltage/current while moving a
single unloaded joint. Do not increase torque or bypass safety limits to force a
mechanically blocked joint.

### Dashboard opens but cannot connect

Run `lerobot-find-port` again and enter the reported port. On Linux, verify port
ownership with `ls -l /dev/ttyACM0` and confirm that the current login session
has the `dialout` group with `id`.

### Another computer cannot open the dashboard

This is expected. Both the HTTP server and request checks restrict access to
localhost as a safety measure.

## Tests

The controller helper and base-position tests do not require connected hardware:

```bash
python -m unittest so101.joint_ui.tests.test_controller
```
