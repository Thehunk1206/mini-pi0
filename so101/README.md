# SO-101 Hardware Tools

This directory contains the repository's physical SO-101 utilities. They share
the same LeRobot calibration and servo bus, but each application owns the bus
exclusively while it is running.

Phone and gamepad teleoperation share calibration limits, Cartesian/IK helpers,
base return, visualization, and flight recording in [`teleop/`](teleop/).
Phone control uses Ruckig; gamepad control uses validated direct joint updates.

## Applications

| Directory | Purpose | Entry point |
| --- | --- | --- |
| [`joint_ui/`](joint_ui/README.md) | Local six-joint dashboard, telemetry, base pose, and safety controls | `python -m so101.joint_ui.app` |
| [`phone_teleop/`](phone_teleop/README.md) | Android WebXR control, telemetry/URDF console, optional Rerun, and incident capture | `python -m so101.phone_teleop.teleoperate` |
| [`gamepad_teleop/`](gamepad_teleop/README.md) | Xbox-style articulated velocity control through wrist-pivot IK and direct joint commands | `python -m so101.gamepad_teleop.teleoperate` |
| [`deployment/`](deployment/README.md) | Legacy physical-policy deployment prototype | No supported SO-101 entry point |

Start with the hardware-free gamepad/Rerun path before opening the servo bus:

```bash
.venv/bin/python -m so101.gamepad_teleop.simulator --pan-mode velocity
```

The matching hardware entry point is:

```bash
.venv/bin/python -m so101.gamepad_teleop.teleoperate --pan-mode velocity
```

See the [gamepad guide](gamepad_teleop/README.md) for the complete controls,
absolute-pan option, calibration behavior, logging, and safety checks.

## Shared robot data

The calibrated tools currently use the `handy_bot` SO-101 follower data:

```text
~/.cache/huggingface/lerobot/calibration/robots/so_follower/handy_bot.json
~/.cache/huggingface/lerobot/base_positions/robots/so_follower/handy_bot.json
```

The calibration must contain all six joints (`shoulder_pan`, `shoulder_lift`,
`elbow_flex`, `wrist_flex`, `wrist_roll`, and `gripper`) with motor IDs 1
through 6.

## Safe use

- Support the arm before releasing torque and keep the power switch reachable.
- Use an external servo supply matched to the installed motors; USB is for
  communication, not servo power.
- Run only one hardware controller at a time. The joint UI, phone teleoperation,
  calibration tools, and deployment code cannot share the serial port.
- Treat `There is no status packet` for every motor as a bus power or
  communication failure. Software may be unable to release torque afterward.
- Read the application-specific safety section before commanding motion.

The deployment prototype predates the calibrated LeRobot integrations and
uses raw register access with placeholder action mapping. It is retained for
reference only and must not be used on the SO-101 without a separate hardware
review and calibration-aware rewrite.
