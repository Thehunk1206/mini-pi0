"""Record SO-101 gamepad demonstrations in LeRobotDataset format.

The gamepad owns motion, episode transitions, base return, and shutdown.  This
entrypoint opens only Rerun for visualization; it never starts the browser UI.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from so101.teleop.runtime import DEFAULT_ROBOT_PORT, FPS

from .gamepad import PAN_CONTROL_MODES
from .recording import RecordingConfig, parse_camera_specs
from .teleoperate import main as run_hardware_teleoperation


DEFAULT_DATASET_ROOT = Path("data/lerobot/so101_gamepad")
DEFAULT_REPO_ID = "local/so101-gamepad"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        required=True,
        help='natural-language task stored in every frame, e.g. "pick up the cube"',
    )
    parser.add_argument("--dataset-repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    existing_group = parser.add_mutually_exclusive_group()
    existing_group.add_argument(
        "--resume",
        "--append",
        dest="resume",
        action="store_true",
        help="append episodes to an existing compatible local dataset",
    )
    existing_group.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "archive an existing dataset beside its current path and create "
            "a fresh dataset"
        ),
    )
    parser.add_argument(
        "--camera",
        action="append",
        metavar="NAME=ID[:ROTATION]",
        help=(
            "repeat for every camera; rotation is 0/90/180/270 degrees. "
            "Default: wrist=0:180"
        ),
    )
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument(
        "--rerun-camera-hz",
        type=float,
        default=5.0,
        help="Rerun preview rate; dataset cameras remain at 30 Hz",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="store individual images instead of MP4 (mainly for debugging)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=0,
        help="finalize automatically after N successful episodes; zero is unlimited",
    )
    parser.add_argument(
        "--max-episode-seconds",
        type=float,
        default=0.0,
        help="discard an overlong episode; zero disables the limit",
    )
    parser.add_argument("--controller-index", type=int, default=0)
    parser.add_argument("--robot-port", default=DEFAULT_ROBOT_PORT)
    parser.add_argument(
        "--pan-mode",
        choices=PAN_CONTROL_MODES,
        default="velocity",
    )
    parser.add_argument("--pan-speed-deg-s", type=float, default=45.0)
    return parser


def cli() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        camera_specs = parse_camera_specs(args.camera)
        dataset_root = args.dataset_root.expanduser().resolve()
        if args.resume and not dataset_root.exists():
            parser.error(
                f"cannot append: dataset does not exist at {dataset_root}"
            )
        if dataset_root.exists() and not args.resume and not args.overwrite:
            parser.error(
                f"dataset already exists at {dataset_root}; use --append or "
                "--overwrite"
            )
        config = RecordingConfig(
            repo_id=args.dataset_repo_id,
            root=dataset_root,
            task=args.task,
            camera_specs=camera_specs,
            fps=FPS,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            use_videos=not args.no_video,
            resume=args.resume,
            overwrite=args.overwrite,
            num_episodes=args.num_episodes,
            max_episode_seconds=args.max_episode_seconds,
            image_writer_threads=max(4, 4 * len(camera_specs)),
            rerun_camera_hz=args.rerun_camera_hz,
        )
    except ValueError as exc:
        parser.error(str(exc))

    run_hardware_teleoperation(
        enable_rerun=True,
        controller_index=args.controller_index,
        robot_port=args.robot_port,
        pan_control_mode=args.pan_mode,
        pan_speed_deg_s=args.pan_speed_deg_s,
        recording_config=config,
    )


if __name__ == "__main__":
    cli()
