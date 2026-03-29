from __future__ import annotations

from typing import Any


def list_robosuite_robots() -> list[str]:
    """Return available robosuite robot names from local installation.

    Returns:
        Sorted robot list. Returns empty list when robosuite is unavailable.
    """

    try:
        from robosuite.robots import ROBOT_CLASS_MAPPING

        return sorted(list(ROBOT_CLASS_MAPPING.keys()))
    except Exception:
        return []


def _robomimic_variants_for(dataset_type: str) -> list[str]:
    if dataset_type in {"ph", "mh"}:
        return ["low_dim"]
    if dataset_type == "mg":
        return ["low_dim_sparse", "low_dim_dense"]
    return []


def _robomimic_task_to_robosuite_env(task: str) -> str:
    mapping = {
        "lift": "Lift",
        "can": "PickPlaceCan",
        "square": "NutAssemblySquare",
        "transport": "TwoArmTransport",
        "tool_hang": "ToolHang",
    }
    return mapping.get(task, task)


def _build_robomimic_entries(version: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    tasks_by_type: dict[str, list[str]] = {
        "ph": ["lift", "can", "square", "transport", "tool_hang"],
        "mh": ["lift", "can", "square", "transport"],
        "mg": ["lift", "can"],
    }

    for dataset_type, tasks in tasks_by_type.items():
        for task in tasks:
            for hdf5_type in _robomimic_variants_for(dataset_type):
                filename = f"{hdf5_type}_v15.hdf5"
                hf_path = f"{version}/{task}/{dataset_type}/{filename}"
                download_url = (
                    "https://huggingface.co/datasets/robomimic/robomimic_datasets/resolve/main/"
                    f"{hf_path}"
                )
                local_path = f"data/robomimic/{task}/{dataset_type}/{filename}"

                action_dim = 14 if task == "transport" else 7
                requires_two_arm = task == "transport"
                compatible_now = not requires_two_arm
                compat_note = (
                    "requires two-arm observation/action mapping (not wired in current pipeline)"
                    if requires_two_arm
                    else "directly compatible with current single-arm Panda pipeline"
                )

                entries.append(
                    {
                        "source": "robomimic",
                        "repo_id": "robomimic/robomimic_datasets",
                        "version": version,
                        "task": task,
                        "dataset_type": dataset_type,
                        "hdf5_type": hdf5_type,
                        "robosuite_env": _robomimic_task_to_robosuite_env(task),
                        "expected_robot": "Panda",
                        "action_dim": action_dim,
                        "requires_two_arm": requires_two_arm,
                        "compatible_now": compatible_now,
                        "compat_note": compat_note,
                        "hf_path": hf_path,
                        "download_url": download_url,
                        "local_path": local_path,
                    }
                )

    return entries


def _build_lerobot_equivalents() -> list[dict[str, Any]]:
    # Equivalent robosuite-origin LeRobot datasets discovered on HF.
    return [
        {
            "source": "lerobot",
            "repo_id": "robotgeneralist/robosuite_lift_ph",
            "task": "lift",
            "robosuite_env": "Lift",
            "expected_robot": "Panda",
            "action_key": "action",
            "episode_index_key": "episode_index",
            "action_dim": 7,
            "image_key": "observation.images.base_0_rgb",
            "proprio_keys": [
                "observation.state.eef_pos",
                "observation.state.eef_quat",
                "observation.state.tool",
            ],
            "compatible_now": True,
            "compat_note": (
                "compatible via native LeRobotDataset loader with "
                "data.format=lerobot_hf and matching robot/data keys"
            ),
        },
        {
            "source": "lerobot",
            "repo_id": "robotgeneralist/robosuite_can_ph",
            "task": "can",
            "robosuite_env": "PickPlaceCan",
            "expected_robot": "Panda",
            "action_key": "action",
            "episode_index_key": "episode_index",
            "action_dim": 7,
            "image_key": "observation.images.right_wrist_0_rgb",
            "proprio_keys": [
                "observation.state.eef_pos",
                "observation.state.eef_quat",
                "observation.state.tool",
            ],
            "compatible_now": True,
            "compat_note": (
                "compatible via native LeRobotDataset loader with "
                "data.format=lerobot_hf and matching robot/data keys"
            ),
        },
        {
            "source": "lerobot",
            "repo_id": "robotgeneralist/robosuite_square_ph",
            "task": "square",
            "robosuite_env": "NutAssemblySquare",
            "expected_robot": "Panda",
            "action_key": "action",
            "episode_index_key": "episode_index",
            "action_dim": 7,
            "image_key": "observation.images.base_0_rgb",
            "proprio_keys": [
                "observation.state.eef_pos",
                "observation.state.eef_quat",
                "observation.state.tool",
            ],
            "compatible_now": True,
            "compat_note": (
                "compatible via native LeRobotDataset loader with "
                "data.format=lerobot_hf and matching robot/data keys"
            ),
        },
    ]


def build_robot_dataset_mapping(version: str = "v1.5", include_lerobot: bool = True) -> dict[str, Any]:
    """Build mapping between robosuite robots and HF datasets for this repo.

    Args:
        version: robomimic dataset version segment used in download URLs.
        include_lerobot: Include LeRobot equivalent dataset suggestions.

    Returns:
        Structured mapping dictionary for CLI printing and docs.
    """

    robots = list_robosuite_robots()
    robomimic = _build_robomimic_entries(version=version)
    lerobot = _build_lerobot_equivalents() if include_lerobot else []

    return {
        "local_robosuite": {
            "supported_robots_detected": robots,
            "default_robot_in_repo": "Panda",
            "note": "Current training/eval pipeline is single-arm focused with 7D actions by default.",
        },
        "hf_datasets": {
            "robomimic": robomimic,
            "lerobot_equivalents": lerobot,
        },
    }
