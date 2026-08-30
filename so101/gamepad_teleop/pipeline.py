"""Gamepad-specific configuration for the shared Cartesian/IK pipeline."""

import math

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotProcessorPipeline

from so101.teleop.cartesian import build_cartesian_ik_processor


FPS = 30
MAX_EE_STEP_M = 0.025
# Full calibrated gripper travel now takes about 0.5 s while a trigger is held.
# This remains a direct bounded position increment with no trajectory generator.
GRIPPER_SPEED_PERCENT_S = 200.0
GRIPPER_STEP_PERCENT = GRIPPER_SPEED_PERCENT_S / FPS
MAX_IK_TARGET_STEP_DEG = (12.0, 15.0, 18.0, 18.0, 25.0)


def apply_calibrated_ik_limits(
    kinematics_solver: RobotKinematics,
    joint_limits_deg: dict[str, tuple[float, float]],
) -> None:
    """Make Placo solve inside the same calibrated envelope as the motors.

    ``RobotKinematics`` normally takes its inequalities from the URDF. The
    physical motor coordinates are normalized from ``handy_bot.json`` and can
    have wider recorded travel, so leaving the original URDF inequalities in
    place silently stops Cartesian motion before the calibrated endpoint.
    """

    for name in kinematics_solver.joint_names:
        if name == "gripper":
            # The gripper bus uses 0..100 percent, not a revolute angle.
            continue
        if name not in joint_limits_deg:
            raise ValueError(f"missing calibrated IK limit for {name}")
        lower, upper = (float(value) for value in joint_limits_deg[name])
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError(f"invalid calibrated IK limit for {name}")
        kinematics_solver.robot.set_joint_limits(
            name,
            math.radians(lower),
            math.radians(upper),
        )
    kinematics_solver.solver.enable_joint_limits(True)


def build_gamepad_processor(
    kinematics_solver: RobotKinematics,
    joint_names: list[str],
) -> RobotProcessorPipeline:
    """Build wrist-pivot XYZ delta -> position-only IK joint processing."""

    return build_cartesian_ik_processor(
        kinematics_solver,
        joint_names,
        max_ee_step_m=MAX_EE_STEP_M,
        gripper_speed_factor=GRIPPER_STEP_PERCENT,
        raise_on_ee_jump=False,
        # Seed every solve from measured joints. A rejected command never
        # reaches the hardware, so advancing an internal previous-solution seed
        # would make IK diverge from the actual arm and eventually select the
        # opposite folded branch.
        initial_guess_current_joints=True,
        orientation_weight=0.0,
    )
