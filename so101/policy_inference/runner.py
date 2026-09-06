"""Dry-run and explicitly enabled hardware runtimes for mini-pi0."""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from so101.teleop.calibration import (
    calibrated_joint_limits,
    load_motor_calibration,
    positions_outside_limits,
)
from so101.teleop.control_stack import DEFAULT_JOINT_LIMITS_DEG
from so101.teleop.flight_recorder import ElectricalTelemetrySampler, FlightRecorder
from so101.teleop.model_assets import DEFAULT_MODEL_CACHE, ensure_model_cache
from so101.teleop.runtime import (
    BASE_GRIPPER_TOLERANCE_PERCENT,
    BASE_JOINT_TOLERANCE_DEG,
    BASE_MOVE_TIMEOUT_S,
    BASE_POSITION_PATH,
    DEFAULT_ROBOT_PORT,
    GRIPPER_SPEED_PERCENT_S,
    JOINT_SPEED_DEG_S,
    MAX_FOLLOWING_ERROR,
    MOTOR_CALIBRATION_PATH,
    TerminalKeyReader,
    bounded_step,
    load_base_position,
)
from so101.teleop.urdf_model import URDFKinematicModel

from .config import InferenceConfig
from .inference_engine import AsyncInferenceEngine
from .policy_bundle import JOINT_NAMES, PolicyBundle
from .rerun_ui import PolicyRerunLogger
from .safety import PolicySafetyGate

LOG_DIR = Path(__file__).resolve().parents[2] / "logs" / "policy_inference"


def _load_safety(
    config: InferenceConfig,
    *,
    require_calibration: bool = False,
) -> tuple[dict[str, tuple[float, float]], PolicySafetyGate]:
    if MOTOR_CALIBRATION_PATH.is_file():
        calibration = load_motor_calibration(MOTOR_CALIBRATION_PATH, list(JOINT_NAMES))
        limits = calibrated_joint_limits(calibration)
    elif require_calibration:
        raise FileNotFoundError(
            "Hardware policy inference requires the follower calibration at "
            f"{MOTOR_CALIBRATION_PATH}"
        )
    else:
        # Simulation and recorded-data replay must remain portable and must not
        # imply that these generic URDF limits are safe for a physical arm.
        limits = {
            name: tuple(map(float, DEFAULT_JOINT_LIMITS_DEG[name]))
            for name in JOINT_NAMES
        }
    safety = PolicySafetyGate(
        limits, control_hz=config.control_hz, config=config.safety
    )
    return limits, safety


def _load_visual_model() -> URDFKinematicModel:
    metadata = ensure_model_cache()
    return URDFKinematicModel.from_file(DEFAULT_MODEL_CACHE / metadata["urdf"])


def _synthetic_cameras(bundle: PolicyBundle, frame_index: int) -> dict[str, np.ndarray]:
    """Generate deterministic moving RGB inputs for hardware-free soak tests."""

    phase = frame_index % 16
    wrist = np.full((480, 480, 3), 48 + phase, dtype=np.uint8)
    base = np.full((360, 640, 3), 48 + phase, dtype=np.uint8)
    wrist[:, frame_index % 480 : (frame_index % 480) + 2, 0] = 80
    base[frame_index % 360 : (frame_index % 360) + 2, :, 1] = 80
    frames = {"wrist": wrist, "base": base}
    return {name: frames[name] for name in bundle.camera_names}


def _return_to_base(
    *,
    robot: object,
    camera_rig: object,
    engine: AsyncInferenceEngine,
    base: dict[str, float],
    recorder: FlightRecorder,
    telemetry: ElectricalTelemetrySampler,
    rerun: PolicyRerunLogger | None,
    frame_index: int,
    control_hz: float,
) -> tuple[np.ndarray, int]:
    """Execute a bounded base return while policy inference remains invalidated."""

    observation = robot.get_observation()  # type: ignore[attr-defined]
    measured = np.asarray(
        [observation[f"{name}.pos"] for name in JOINT_NAMES], dtype=np.float32
    )
    engine.pause(measured, reason="return_to_base")
    command = measured.copy()
    target = np.asarray([base[name] for name in JOINT_NAMES], dtype=np.float32)
    maximum_step = (
        np.asarray(
            (*([JOINT_SPEED_DEG_S] * 5), GRIPPER_SPEED_PERCENT_S),
            dtype=np.float32,
        )
        / control_hz
    )
    started = time.monotonic()
    period = 1.0 / control_hz
    print("Returning to the saved base pose. Policy remains paused afterward.")
    while time.monotonic() - started < BASE_MOVE_TIMEOUT_S:
        cycle_started = time.monotonic()
        command = np.asarray(
            [
                bounded_step(float(current), float(goal), float(step))
                for current, goal, step in zip(
                    command, target, maximum_step, strict=True
                )
            ],
            dtype=np.float32,
        )
        sent = robot.send_action(  # type: ignore[attr-defined]
            {
                f"{name}.pos": float(command[index])
                for index, name in enumerate(JOINT_NAMES)
            }
        )
        time.sleep(max(0.0, period - (time.monotonic() - cycle_started)))
        observation = robot.get_observation()  # type: ignore[attr-defined]
        measured = np.asarray(
            [observation[f"{name}.pos"] for name in JOINT_NAMES], dtype=np.float32
        )
        command = np.asarray(
            [sent[f"{name}.pos"] for name in JOINT_NAMES], dtype=np.float32
        )
        errors = np.abs(command - measured)
        worst_index = int(np.argmax(errors))
        if float(errors[worst_index]) > MAX_FOLLOWING_ERROR:
            raise RuntimeError(
                f"Base return stopped: {JOINT_NAMES[worst_index]} following error "
                f"{errors[worst_index]:.1f} exceeds {MAX_FOLLOWING_ERROR:.1f}"
            )
        cameras = camera_rig.read()  # type: ignore[attr-defined]
        electrical = telemetry.maybe_read(robot.bus)  # type: ignore[attr-defined]
        recorder.record(
            observation=observation,
            action=sent,
            electrical=electrical,
            event="return_to_base",
        )
        if rerun is not None:
            rerun.submit(
                frame_index=frame_index,
                cameras=cameras,
                measured=measured,
                command=command,
                raw_policy_action=None,
                engine=engine.status,
                tracking=engine.latest_tracking,
                electrical=electrical,
            )
        frame_index += 1
        reached = all(
            abs(float(measured[index] - target[index]))
            <= (
                BASE_GRIPPER_TOLERANCE_PERCENT
                if name == "gripper"
                else BASE_JOINT_TOLERANCE_DEG
            )
            for index, name in enumerate(JOINT_NAMES)
        )
        if reached:
            engine.pause(measured, reason="base_reached_requires_resume")
            print("Base reached. Press p when you are ready to resume the policy.")
            return measured, frame_index
    raise TimeoutError(
        f"Base return did not finish within {BASE_MOVE_TIMEOUT_S:.0f} seconds"
    )


def run_synthetic(
    config: InferenceConfig,
    *,
    duration_s: float,
    enable_rerun: bool,
    spawn_rerun: bool = True,
) -> dict[str, float | int | str | bool | None]:
    """Exercise model, RTC queue, safety, and optional Rerun with no hardware."""

    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    bundle = PolicyBundle.load(
        config.checkpoint, device=config.device, precision=config.precision
    )
    limits, safety = _load_safety(config)
    measured = np.clip(
        bundle.normalization.state_mean.copy(),
        [limits[name][0] for name in JOINT_NAMES],
        [limits[name][1] for name in JOINT_NAMES],
    ).astype(np.float32)
    safety.reset(measured)
    engine = AsyncInferenceEngine(bundle, config, safety)
    rerun = (
        PolicyRerunLogger(bundle, _load_visual_model(), control_hz=config.control_hz)
        if enable_rerun
        else None
    )
    if rerun is not None:
        rerun.start(spawn=spawn_rerun)
    engine.start()
    started = time.monotonic()
    period = 1.0 / config.control_hz
    frame_index = 0
    actions_executed = 0
    maximum_loop_ms = 0.0
    try:
        while time.monotonic() - started < duration_s:
            loop_started = time.monotonic()
            cameras = _synthetic_cameras(bundle, frame_index)
            engine.publish_observation(cameras, measured, timestamp=loop_started)
            action = engine.get_action(measured, now=loop_started)
            command = measured.copy() if action is None else action
            if action is not None:
                measured = command.copy()
                actions_executed += 1
            if rerun is not None:
                rerun.submit(
                    frame_index=frame_index,
                    cameras=cameras,
                    measured=measured,
                    command=command,
                    raw_policy_action=engine.latest_raw_action,
                    engine=engine.status,
                    tracking=engine.latest_tracking,
                )
            frame_index += 1
            elapsed = time.monotonic() - loop_started
            maximum_loop_ms = max(maximum_loop_ms, elapsed * 1000.0)
            time.sleep(max(0.0, period - elapsed))
    finally:
        engine.stop()
        if rerun is not None:
            rerun.stop()
    status = engine.status
    return {
        "mode": "synthetic_dry_run",
        "hardware_opened": False,
        "device": str(bundle.device),
        "precision": bundle.precision_name,
        "parameters": bundle.parameter_count,
        "frames": frame_index,
        "actions_executed": actions_executed,
        "inferences": status.inference_count,
        "underflows": status.underflow_count,
        "rejected_chunks": status.rejected_chunks,
        "rtc_fallbacks": status.rtc_fallbacks,
        "stale_results_dropped": status.stale_results_dropped,
        "last_inference_ms": status.last_latency_ms,
        "maximum_control_loop_ms": maximum_loop_ms,
        "fault": status.fault,
        "last_inference_error": status.last_inference_error,
        "last_rejection_reason": status.last_rejection_reason,
        "rerun_dropped_frames": rerun.dropped_frames if rerun is not None else 0,
    }


def run_camera_dry(
    config: InferenceConfig,
    *,
    camera_specs: Iterable[object],
    camera_width: int,
    camera_height: int,
    duration_s: float,
    enable_rerun: bool,
    spawn_rerun: bool = True,
) -> dict[str, float | int | str | bool | None]:
    """Run real cameras through the full policy stack without opening a robot."""

    if duration_s <= 0:
        raise ValueError("duration_s must be positive")

    from so101.gamepad_teleop.recording import CameraRig

    bundle = PolicyBundle.load(
        config.checkpoint, device=config.device, precision=config.precision
    )
    specs = tuple(camera_specs)
    if tuple(getattr(spec, "name", None) for spec in specs) != bundle.camera_names:
        raise ValueError(
            f"Cameras must be supplied in trained order {bundle.camera_names}; "
            f"got {tuple(getattr(spec, 'name', None) for spec in specs)}"
        )
    limits, safety = _load_safety(config)
    measured = np.clip(
        bundle.normalization.state_mean.copy(),
        [limits[name][0] for name in JOINT_NAMES],
        [limits[name][1] for name in JOINT_NAMES],
    ).astype(np.float32)
    safety.reset(measured)
    engine = AsyncInferenceEngine(bundle, config, safety)
    camera_rig = CameraRig(
        specs,
        fps=round(config.control_hz),
        width=camera_width,
        height=camera_height,
    )
    rerun = (
        PolicyRerunLogger(bundle, _load_visual_model(), control_hz=config.control_hz)
        if enable_rerun
        else None
    )
    frame_index = 0
    actions_executed = 0
    period = 1.0 / config.control_hz
    try:
        camera_rig.connect()
        bundle.warmup(flow_steps=min(2, config.rtc.flow_steps))
        if rerun is not None:
            rerun.start(spawn=spawn_rerun)
        engine.start()
        started = time.monotonic()
        while time.monotonic() - started < duration_s:
            cycle_started = time.monotonic()
            cameras = camera_rig.read()
            engine.publish_observation(cameras, measured, timestamp=cycle_started)
            action = engine.get_action(measured, now=time.monotonic())
            command = measured.copy() if action is None else action
            if action is not None:
                measured = command.copy()
                actions_executed += 1
            if rerun is not None:
                rerun.submit(
                    frame_index=frame_index,
                    cameras=cameras,
                    measured=measured,
                    command=command,
                    raw_policy_action=engine.latest_raw_action,
                    engine=engine.status,
                    tracking=engine.latest_tracking,
                )
            frame_index += 1
            time.sleep(max(0.0, period - (time.monotonic() - cycle_started)))
    finally:
        engine.stop()
        camera_rig.disconnect()
        if rerun is not None:
            rerun.stop()
    if frame_index == 0:
        raise RuntimeError("Live-camera dry-run completed without processing a frame")
    status = engine.status
    return {
        "mode": "live_camera_dry_run",
        "hardware_opened": False,
        "device": str(bundle.device),
        "precision": bundle.precision_name,
        "frames": frame_index,
        "actions_executed": actions_executed,
        "inferences": status.inference_count,
        "underflows": status.underflow_count,
        "rejected_chunks": status.rejected_chunks,
        "rtc_fallbacks": status.rtc_fallbacks,
        "fault": status.fault,
        "last_inference_error": status.last_inference_error,
        "last_rejection_reason": status.last_rejection_reason,
        "rerun_dropped_frames": rerun.dropped_frames if rerun is not None else 0,
    }


def run_dataset_replay(
    config: InferenceConfig,
    *,
    dataset_root: Path,
    repo_id: str,
    episode_index: int,
    max_frames: int,
    enable_rerun: bool,
    spawn_rerun: bool = True,
) -> dict[str, float | int | str | bool | None]:
    """Feed a recorded dual-camera episode through online RTC without hardware."""

    import json

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from so101.gamepad_teleop.replay import to_hwc_uint8

    root = dataset_root.expanduser().resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Dataset metadata not found: {info_path}")
    info = json.loads(info_path.read_text())
    features = info.get("features", {})
    state_names = tuple(features.get("observation.state", {}).get("names", ()))
    action_names = tuple(features.get("action", {}).get("names", ()))
    expected_names = tuple(f"{name}.pos" for name in JOINT_NAMES)
    if state_names != expected_names or action_names != expected_names:
        raise ValueError(
            "Dataset state/action joint order does not match the trained SO-101 schema"
        )
    expected_images = tuple(f"observation.images.{name}" for name in ("wrist", "base"))
    if not all(name in features for name in expected_images):
        raise ValueError(
            f"Dataset must contain both trained camera streams {expected_images}"
        )

    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        episodes=[episode_index],
        video_backend="pyav",
        return_uint8=True,
    )
    frame_count = len(dataset) if max_frames <= 0 else min(len(dataset), max_frames)
    if frame_count <= 0:
        raise ValueError("Selected dataset episode contains no frames")
    bundle = PolicyBundle.load(
        config.checkpoint, device=config.device, precision=config.precision
    )
    _limits, safety = _load_safety(config)
    first_state = np.asarray(dataset[0]["observation.state"], dtype=np.float32).reshape(
        -1
    )
    safety.reset(first_state)
    engine = AsyncInferenceEngine(bundle, config, safety)
    rerun = (
        PolicyRerunLogger(bundle, _load_visual_model(), control_hz=config.control_hz)
        if enable_rerun
        else None
    )
    if rerun is not None:
        rerun.start(spawn=spawn_rerun)
    engine.start()
    measured = first_state.copy()
    actions_executed = 0
    absolute_errors: list[np.ndarray] = []
    period = 1.0 / config.control_hz
    started = time.monotonic()
    try:
        for frame_index in range(frame_count):
            cycle_started = time.monotonic()
            frame = dataset[frame_index]
            recorded_state = np.asarray(
                frame["observation.state"], dtype=np.float32
            ).reshape(-1)
            expert_action = np.asarray(frame["action"], dtype=np.float32).reshape(-1)
            cameras = {
                name: to_hwc_uint8(frame[f"observation.images.{name}"])
                for name in bundle.camera_names
            }
            engine.publish_observation(cameras, recorded_state, timestamp=cycle_started)
            action = engine.get_action(measured, now=cycle_started)
            command = measured.copy() if action is None else action
            if action is not None:
                measured = command.copy()
                actions_executed += 1
                absolute_errors.append(np.abs(command - expert_action))
            if rerun is not None:
                rerun.submit(
                    frame_index=frame_index,
                    cameras=cameras,
                    measured=measured,
                    command=command,
                    raw_policy_action=engine.latest_raw_action,
                    engine=engine.status,
                    tracking=engine.latest_tracking,
                )
            time.sleep(max(0.0, period - (time.monotonic() - cycle_started)))
    finally:
        engine.stop()
        if rerun is not None:
            rerun.stop()
    status = engine.status
    mae = (
        np.mean(np.stack(absolute_errors), axis=0)
        if absolute_errors
        else np.full(6, np.nan)
    )
    return {
        "mode": "dataset_replay",
        "hardware_opened": False,
        "dataset": str(root),
        "episode": episode_index,
        "frames": frame_count,
        "elapsed_s": time.monotonic() - started,
        "actions_executed": actions_executed,
        "inferences": status.inference_count,
        "underflows": status.underflow_count,
        "rejected_chunks": status.rejected_chunks,
        "rtc_fallbacks": status.rtc_fallbacks,
        "fault": status.fault,
        "last_inference_error": status.last_inference_error,
        "last_rejection_reason": status.last_rejection_reason,
        "mean_absolute_action_error": {
            name: float(mae[index]) for index, name in enumerate(JOINT_NAMES)
        },
    }


def run_hardware(
    config: InferenceConfig,
    *,
    camera_specs: Iterable[object],
    camera_width: int,
    camera_height: int,
    robot_port: str = DEFAULT_ROBOT_PORT,
    enable_rerun: bool = True,
) -> None:
    """Run the policy on motors; the CLI guards this behind ``--enable-motors``."""

    from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
    from lerobot.utils.robot_utils import precise_sleep

    from so101.gamepad_teleop.recording import CameraRig

    bundle = PolicyBundle.load(
        config.checkpoint, device=config.device, precision=config.precision
    )
    limits, safety = _load_safety(config, require_calibration=True)
    specs = tuple(camera_specs)
    if tuple(getattr(spec, "name", None) for spec in specs) != bundle.camera_names:
        raise ValueError(
            f"Hardware cameras must be supplied in trained order {bundle.camera_names}; "
            f"got {tuple(getattr(spec, 'name', None) for spec in specs)}"
        )
    camera_rig = CameraRig(
        specs, fps=round(config.control_hz), width=camera_width, height=camera_height
    )
    visual_model = _load_visual_model()
    rerun = (
        PolicyRerunLogger(bundle, visual_model, control_hz=config.control_hz)
        if enable_rerun
        else None
    )
    recorder = FlightRecorder(LOG_DIR, list(JOINT_NAMES), fps=round(config.control_hz))
    telemetry = ElectricalTelemetrySampler(
        list(JOINT_NAMES), frequency_hz=5.0, fast_load=True
    )
    robot = SO100Follower(
        SO100FollowerConfig(
            port=robot_port,
            id="handy_bot",
            use_degrees=True,
            num_read_retries=2,
        )
    )
    engine: AsyncInferenceEngine | None = None
    last_observation: dict | None = None
    last_command: np.ndarray | None = None
    try:
        recorder.set_phase("cameras.connect")
        camera_rig.connect()
        if tuple(camera_rig.frame_shapes) != bundle.camera_names:
            raise RuntimeError("Camera rig did not return the trained camera set")
        bundle.warmup(flow_steps=min(2, config.rtc.flow_steps))
        if rerun is not None:
            rerun.start()

        # No code above this line can open the servo bus.
        recorder.set_phase("robot.connect")
        robot.connect()
        telemetry.maybe_read(robot.bus, force=True)
        last_observation = robot.get_observation()
        measured = np.asarray(
            [last_observation[f"{name}.pos"] for name in JOINT_NAMES], dtype=np.float32
        )
        outside = positions_outside_limits(
            {name: float(measured[index]) for index, name in enumerate(JOINT_NAMES)},
            limits,
        )
        if outside:
            raise ValueError(f"Measured startup pose is outside calibration: {outside}")
        safety.reset(measured)
        engine = AsyncInferenceEngine(bundle, config, safety)
        engine.start()
        last_command = measured.copy()
        print("Policy active. Keys: p=pause/resume, b=return to base, q=quit.")
        print("The arm holds measured position while the first chunk is generated.")

        period = 1.0 / config.control_hz
        frame_index = 0
        paused_by_operator = False
        with TerminalKeyReader() as keys:
            while True:
                loop_started = time.monotonic()
                key = keys.poll()
                if key == "q":
                    break
                recorder.set_phase("policy.get_observation")
                last_observation = robot.get_observation()
                measured = np.asarray(
                    [last_observation[f"{name}.pos"] for name in JOINT_NAMES],
                    dtype=np.float32,
                )
                if key == "p":
                    if engine.status.paused:
                        engine.resume(measured)
                        paused_by_operator = False
                        print("Policy resumed; holding until a fresh chunk is ready.")
                    else:
                        engine.pause(measured, reason="operator_pause")
                        paused_by_operator = True
                        print("Policy paused.")
                if key == "b":
                    engine.pause(measured, reason="base_return_requested")
                    if not BASE_POSITION_PATH.is_file():
                        print(
                            f"Base return unavailable: {BASE_POSITION_PATH} does not exist"
                        )
                    else:
                        base = load_base_position(BASE_POSITION_PATH, list(JOINT_NAMES))
                        base_outside = positions_outside_limits(base, limits)
                        if base_outside:
                            print(f"Base return blocked by calibration: {base_outside}")
                        else:
                            measured, frame_index = _return_to_base(
                                robot=robot,
                                camera_rig=camera_rig,
                                engine=engine,
                                base=base,
                                recorder=recorder,
                                telemetry=telemetry,
                                rerun=rerun,
                                frame_index=frame_index,
                                control_hz=config.control_hz,
                            )
                    paused_by_operator = True

                recorder.set_phase("policy.read_cameras")
                cameras = camera_rig.read()
                engine.publish_observation(cameras, measured, timestamp=loop_started)
                action = engine.get_action(measured, now=time.monotonic())
                last_command = measured.copy() if action is None else action
                recorder.set_phase("policy.send_action")
                sent = robot.send_action(
                    {
                        f"{name}.pos": float(last_command[index])
                        for index, name in enumerate(JOINT_NAMES)
                    }
                )
                last_command = np.asarray(
                    [sent[f"{name}.pos"] for name in JOINT_NAMES], dtype=np.float32
                )
                electrical = telemetry.maybe_read(robot.bus)
                if rerun is not None:
                    rerun.submit(
                        frame_index=frame_index,
                        cameras=cameras,
                        measured=measured,
                        command=last_command,
                        raw_policy_action=engine.latest_raw_action,
                        engine=engine.status,
                        tracking=engine.latest_tracking,
                        electrical=electrical,
                    )
                recorder.record(
                    observation=last_observation,
                    action=sent,
                    electrical=electrical,
                    control={
                        **engine.status.__dict__,
                        "operator_paused": paused_by_operator,
                    },
                    loop_ms=(time.monotonic() - loop_started) * 1000.0,
                )
                recorder.record_electrical_summary(telemetry)
                frame_index += 1
                precise_sleep(max(0.0, period - (time.monotonic() - loop_started)))
    except Exception as exc:
        incident = recorder.capture_incident(
            exc,
            observation=last_observation,
            last_command=last_command,
            engine=(engine.status.__dict__ if engine is not None else None),
        )
        print(f"INCIDENT CAPTURED: {incident}")
        raise
    finally:
        cleanup_steps = (
            ("inference engine", lambda: engine.stop() if engine is not None else None),
            ("robot", lambda: robot.disconnect() if robot.is_connected else None),
            ("camera rig", camera_rig.disconnect),
            ("Rerun", lambda: rerun.stop() if rerun is not None else None),
            ("flight recorder", recorder.close),
        )
        for label, cleanup in cleanup_steps:
            try:
                cleanup()
            except Exception as cleanup_error:  # noqa: BLE001 - finish every cleanup step
                print(f"WARNING: failed to close {label}: {cleanup_error}")
