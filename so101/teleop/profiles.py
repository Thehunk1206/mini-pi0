"""Motion-profile definitions shared without importing the Ruckig runtime."""

PROFILE_LIMIT_SCALES = {
    "Smooth": {
        "arm_velocity": 2.0 / 3.0,
        "arm_acceleration": 2.0 / 3.0,
        "arm_jerk": 0.4,
        "gripper_velocity": 0.6,
        "gripper_acceleration": 0.6,
        "gripper_jerk": 0.48,
    },
    "Safe": dict.fromkeys(
        (
            "arm_velocity",
            "arm_acceleration",
            "arm_jerk",
            "gripper_velocity",
            "gripper_acceleration",
            "gripper_jerk",
        ),
        1.0,
    ),
    "Balanced": dict.fromkeys(
        (
            "arm_velocity",
            "arm_acceleration",
            "arm_jerk",
            "gripper_velocity",
            "gripper_acceleration",
            "gripper_jerk",
        ),
        1.5,
    ),
    "Responsive": dict.fromkeys(
        (
            "arm_velocity",
            "arm_acceleration",
            "arm_jerk",
            "gripper_velocity",
            "gripper_acceleration",
            "gripper_jerk",
        ),
        2.5,
    ),
}

PROFILE_NAMES = tuple(PROFILE_LIMIT_SCALES)
