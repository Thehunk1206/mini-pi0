# SO-101 Joint Control UI

A localhost-only browser dashboard for moving the six calibrated joints of the
`handy_bot` SO-101 follower.

## Safety

- Assemble and support the arm before enabling torque.
- Keep the workspace clear and keep a hand near the controller power switch.
- Connect aligns each goal to the measured pose before enabling holding torque.
- Slider targets are clamped to the saved calibration ranges.
- Joint motion is rate-limited to 45 degrees/second; the gripper is limited to
  35 percent/second.
- The rate-limited commanded trajectory advances independently of measured
  position so loaded joints receive enough position error to overcome static
  friction.
- A 15-degree following-error watchdog releases torque if an arm joint cannot
  track the commanded trajectory.
- The UI releases torque on emergency stop, disconnect, server shutdown,
  over-temperature, or a motor communication error.
- Press `Escape` in the browser to trigger the emergency stop.
- A captured base position is stored persistently and can be used for a
  coordinated, rate-limited return that keeps holding torque enabled.
- **Return to base** is a controlled motion, not an emergency stop. The true
  emergency stop still releases torque immediately so a trapped or colliding
  arm does not initiate additional movement.

## Run

From the repository root:

```bash
source .venv/bin/activate
python -m so101_joint_ui.app \
  --serial-port /dev/cu.usbmodem5B610338651 \
  --calibration /Users/tauhidkhan/.cache/huggingface/lerobot/calibration/robots/so_follower/handy_bot.json
```

Open <http://127.0.0.1:8000>. The server binds only to localhost.

The serial port can also be entered or changed in the dashboard before
connecting. If it changes, find it with:

```bash
lerobot-find-port
```

## Controls

- **Connect** verifies that the calibration stored in all six motors matches
  `handy_bot.json`, reads the measured pose, and enables holding torque at that
  pose.
- **Release torque** makes the arm back-drivable. Support the arm before using
  it.
- **Enable hold** captures the newly measured pose before re-enabling torque.
- **Capture base position** saves a freshly measured six-joint pose. Place the
  arm in a stable parked pose before capturing it.
- **Return to base position** moves all six targets to the saved pose together
  using the existing speed and following-error limits, then continues holding
  at base.
- **Emergency stop** immediately releases torque while leaving the serial
  connection open.
- **Disconnect** releases torque and closes the serial connection.
- Each joint card shows measured position, slider target, and the intermediate
  rate-limited command sent to the motor.
- Each joint card also shows live temperature, voltage, current draw in mA,
  and signed servo load in percent. A rolling two-minute current/load history
  is kept in the motor telemetry log at one sample per second.
- Current is converted from the STS3215 register using 6.5 mA per count. Load
  is signed motor output effort, not a direct force measurement.

## Configuration

The defaults can be supplied as environment variables:

```bash
export SO101_SERIAL_PORT=/dev/cu.usbmodem5B610338651
export SO101_CALIBRATION_FILE=/path/to/handy_bot.json
export SO101_BASE_POSITION_FILE=/path/to/handy_bot_base_position.json
python -m so101_joint_ui.app
```

By default, the base pose is saved to
`~/.cache/huggingface/lerobot/base_positions/robots/so_follower/handy_bot.json`.
It is reloaded whenever the dashboard starts.

Use `--web-port` if port 8000 is already occupied.
