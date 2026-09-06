"""Run mini-pi0 SO-101 policy inference or a hardware-free benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import benchmark_variants, print_benchmark
from .config import InferenceConfig, RTCInferenceConfig, SafetyConfig
from .policy_bundle import checkpoint_for_variant
from .runner import run_camera_dry, run_dataset_replay, run_hardware, run_synthetic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("16m", "25m"), default="16m")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--device", choices=("auto", "mps", "cuda", "cpu"), default="auto"
    )
    parser.add_argument(
        "--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto"
    )
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--no-rtc", action="store_true")
    parser.add_argument("--execution-horizon", type=int, default=10)
    parser.add_argument("--replan-interval", type=int, default=6)
    parser.add_argument("--rtc-guidance", type=float, default=5.0)
    parser.add_argument(
        "--rtc-schedule", choices=("ZEROS", "ONES", "LINEAR", "EXP"), default="EXP"
    )
    parser.add_argument(
        "--random-noise-each-chunk",
        action="store_true",
        help="disable the default fixed sampling noise (less repeatable and usually less smooth)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--duration", type=float, default=10.0, help="dry-run duration in seconds"
    )
    parser.add_argument("--no-rerun", action="store_true")
    parser.add_argument("--no-spawn-rerun", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-both", action="store_true")
    parser.add_argument("--benchmark-repeats", type=int, default=10)
    parser.add_argument("--benchmark-guided-repeats", type=int, default=3)
    parser.add_argument("--replay-dataset", type=Path)
    parser.add_argument(
        "--replay-repo-id", default="local/so101-pick-place-blocks-dual-cam"
    )
    parser.add_argument("--replay-episode", type=int, default=0)
    parser.add_argument("--replay-max-frames", type=int, default=300)
    parser.add_argument(
        "--enable-motors",
        action="store_true",
        help="explicitly open the servo bus and execute policy actions; absent means hardware-free dry-run",
    )
    parser.add_argument("--robot-port", default="/dev/cu.usbmodem5B610338651")
    parser.add_argument(
        "--camera",
        action="append",
        metavar="NAME=ID[:ROTATION][@FPS]",
        help="hardware mode: repeat in trained order, wrist first then base",
    )
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument(
        "--camera-native-size", action="append", metavar="NAME=WIDTHxHEIGHT"
    )
    parser.add_argument(
        "--camera-output-size", action="append", metavar="NAME=WIDTHxHEIGHT"
    )
    return parser


def cli() -> None:
    parser = build_parser()
    args = parser.parse_args()
    checkpoint = (
        (args.checkpoint or checkpoint_for_variant(args.variant)).expanduser().resolve()
    )
    rtc = RTCInferenceConfig(
        enabled=not args.no_rtc,
        flow_steps=args.flow_steps,
        execution_horizon=args.execution_horizon,
        replan_interval=args.replan_interval,
        max_guidance_weight=args.rtc_guidance,
        prefix_attention_schedule=args.rtc_schedule,
        fixed_noise=not args.random_noise_each_chunk,
        seed=args.seed,
    )
    config = InferenceConfig(
        checkpoint=checkpoint,
        device=args.device,
        precision=args.precision,
        control_hz=args.control_hz,
        rtc=rtc,
        safety=SafetyConfig(),
    )
    if args.benchmark or args.benchmark_both:
        variants = ("16m", "25m") if args.benchmark_both else (args.variant,)
        print_benchmark(
            benchmark_variants(
                variants=variants,
                device=args.device,
                precision=args.precision,
                flow_steps=args.flow_steps,
                repeats=args.benchmark_repeats,
                guided_repeats=args.benchmark_guided_repeats,
            )
        )
        return

    if args.enable_motors:
        if not args.camera:
            parser.error(
                "--enable-motors requires --camera wrist=... and --camera base=..."
            )
        from so101.gamepad_teleop.recording import (
            apply_camera_capture_sizes,
            apply_camera_output_sizes,
            parse_camera_specs,
        )

        try:
            camera_specs = apply_camera_output_sizes(
                apply_camera_capture_sizes(
                    parse_camera_specs(args.camera), args.camera_native_size
                ),
                args.camera_output_size,
            )
        except ValueError as exc:
            parser.error(str(exc))
        run_hardware(
            config,
            camera_specs=camera_specs,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            robot_port=args.robot_port,
            enable_rerun=not args.no_rerun,
        )
        return

    if args.replay_dataset is not None:
        summary = run_dataset_replay(
            config,
            dataset_root=args.replay_dataset,
            repo_id=args.replay_repo_id,
            episode_index=args.replay_episode,
            max_frames=args.replay_max_frames,
            enable_rerun=not args.no_rerun,
            spawn_rerun=not args.no_spawn_rerun,
        )
        print(json.dumps(summary, indent=2))
        return

    if args.camera:
        from so101.gamepad_teleop.recording import (
            apply_camera_capture_sizes,
            apply_camera_output_sizes,
            parse_camera_specs,
        )

        try:
            camera_specs = apply_camera_output_sizes(
                apply_camera_capture_sizes(
                    parse_camera_specs(args.camera), args.camera_native_size
                ),
                args.camera_output_size,
            )
        except ValueError as exc:
            parser.error(str(exc))
        summary = run_camera_dry(
            config,
            camera_specs=camera_specs,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            duration_s=args.duration,
            enable_rerun=not args.no_rerun,
            spawn_rerun=not args.no_spawn_rerun,
        )
        print(json.dumps(summary, indent=2))
        return

    summary = run_synthetic(
        config,
        duration_s=args.duration,
        enable_rerun=not args.no_rerun,
        spawn_rerun=not args.no_spawn_rerun,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    cli()
