"""Shared Cartesian target -> SO-101 joint-target processor construction."""

from __future__ import annotations

from collections.abc import Sequence

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    RobotActionProcessorStep,
    RobotProcessorPipeline,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)


def build_cartesian_ik_processor(
    kinematics_solver: RobotKinematics,
    joint_names: list[str],
    *,
    input_steps: Sequence[RobotActionProcessorStep] = (),
    end_effector_step_sizes: dict[str, float] | None = None,
    max_ee_step_m: float,
    gripper_speed_factor: float,
    raise_on_ee_jump: bool = True,
    initial_guess_current_joints: bool = True,
    orientation_weight: float = 0.01,
) -> RobotProcessorPipeline:
    """Build the one authoritative live Cartesian-to-joint pipeline.

    Input adapters must produce ``enabled``, cumulative ``target_x/y/z``,
    rotation-vector deltas, and ``gripper_vel``. Phone mapping and gamepad
    integration differ before this boundary; EE latching, safety, gripper
    conversion, and IK are identical after it.
    """

    step_sizes = end_effector_step_sizes or {"x": 1.0, "y": 1.0, "z": 1.0}
    steps: list[RobotActionProcessorStep] = [*input_steps]
    steps.extend(
        [
            EEReferenceAndDelta(
                kinematics=kinematics_solver,
                end_effector_step_sizes=step_sizes,
                motor_names=joint_names,
                use_latched_reference=True,
            ),
            EEBoundsAndSafety(
                end_effector_bounds={
                    "min": [-1.0, -1.0, -1.0],
                    "max": [1.0, 1.0, 1.0],
                },
                max_ee_step_m=max_ee_step_m,
                raise_on_jump=raise_on_ee_jump,
            ),
            GripperVelocityToJoint(speed_factor=gripper_speed_factor),
            InverseKinematicsEEToJoints(
                kinematics=kinematics_solver,
                motor_names=joint_names,
                initial_guess_current_joints=initial_guess_current_joints,
                orientation_weight=orientation_weight,
            ),
        ]
    )
    return RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=steps,
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
