"""Hardware-free SO-101 trajectory lab and browser simulator.

Run with ``python -m so101.phone_teleop.simulator``. This module never imports
or constructs a robot, motor bus, serial port, or phone teleoperator.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .control_stack import (
    CommissioningLimits,
    DEFAULT_JOINT_LIMITS_DEG,
    PROFILE_NAMES,
    PhoneControlStack,
)
from .filtering import (
    DEFAULT_PHONE_FILTER_SETTINGS,
    ConstantVelocityKalmanXYZ,
    OneEuroXYZFilter,
    validated_phone_filter_settings,
)
from .model_assets import (
    DEFAULT_MODEL_CACHE,
    MODEL_FILENAME,
    ensure_model_cache,
    verify_kinematic_urdf,
)
from .trajectory import OnlineQuinticRetargeter


JOINT_NAMES = (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"
)
FPS = 30
DEFAULT_TARGET = np.array([55.0, -45.0, 50.0, 35.0, 75.0, 85.0])
HOME = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 35.0])
PHONE_TO_JOINT = np.array(
    [
        [500.0, 0.0, 0.0],
        [0.0, 600.0, 0.0],
        [0.0, 0.0, 700.0],
        [150.0, -200.0, 0.0],
        [0.0, 200.0, 300.0],
        [0.0, 0.0, 700.0],
    ]
)


def _clip_targets(values: np.ndarray) -> np.ndarray:
    clipped = np.asarray(values, dtype=float).copy()
    for index, name in enumerate(JOINT_NAMES):
        clipped[..., index] = np.clip(clipped[..., index], *DEFAULT_JOINT_LIMITS_DEG[name])
    return clipped


def _derivative_stream(position: np.ndarray, dt: float) -> dict[str, np.ndarray]:
    if len(position) < 2:
        zeros = np.zeros_like(position)
        return {"position": position, "velocity": zeros, "acceleration": zeros, "jerk": zeros}
    velocity = np.gradient(position, dt, axis=0, edge_order=1)
    acceleration = np.gradient(velocity, dt, axis=0, edge_order=1)
    jerk = np.gradient(acceleration, dt, axis=0, edge_order=1)
    return {"position": position, "velocity": velocity, "acceleration": acceleration, "jerk": jerk}


def _phone_scenario(name: str, times: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    true = np.zeros((len(times), 3), dtype=float)
    if name == "continuous_retarget":
        true[:, 0] = 0.065 * np.sin(2 * math.pi * 0.22 * times)
        true[:, 1] = 0.055 * np.sin(2 * math.pi * 0.31 * times + 0.7)
        true[:, 2] = 0.045 * np.sin(2 * math.pi * 0.17 * times + 1.2)
    elif name in {"synchronized_joints", "point_to_point"}:
        desired = np.linalg.lstsq(PHONE_TO_JOINT, target - HOME, rcond=None)[0]
        true[times >= 1.0] = desired
    else:
        true[times >= 1.0] = [0.08, -0.05, 0.06]
    rng = np.random.default_rng(1701)
    noise = rng.normal(0.0, 0.0025, size=true.shape)
    # A deterministic tracking spike makes the filter comparison easy to see.
    if len(times) > FPS * 3:
        noise[FPS * 3] += [0.012, -0.010, 0.008]
    return true, true + noise


def _joint_dict(values: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(JOINT_NAMES, values, strict=True)}


class SimulationEngine:
    """Thread-safe timeline state and deterministic scenario generator."""

    scenario_names = ("phone_step", "synchronized_joints", "continuous_retarget", "point_to_point")

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.profile = "Safe"
        self.filter_settings = dict(DEFAULT_PHONE_FILTER_SETTINGS)
        self.selected_stream = "ruckig"
        self.visible_streams = ["raw_ik", "quintic", "ruckig"]
        self.speed = 1.0
        self.playing = False
        self._started_at = 0.0
        self._paused_time_s = 0.0
        self.scenario = "phone_step"
        self.target = DEFAULT_TARGET.copy()
        self.samples: list[dict[str, Any]] = []
        self.duration_s = 0.0
        self.load_scenario(self.scenario)

    def _filters(self) -> tuple[OneEuroXYZFilter, ConstantVelocityKalmanXYZ]:
        return OneEuroXYZFilter(**self.filter_settings), ConstantVelocityKalmanXYZ()

    def _generate(self, name: str, target: np.ndarray) -> list[dict[str, Any]]:
        duration = 8.0 if name == "continuous_retarget" else 6.0
        times = np.arange(0.0, duration + 0.5 / FPS, 1.0 / FPS)
        true_phone, raw_phone = _phone_scenario(name, times, target)
        one_euro, kalman = self._filters()
        filtered_phone = np.asarray(
            [one_euro.update(value, float(t)).deadband_position_m for t, value in zip(times, raw_phone, strict=True)]
        )
        kalman_phone = np.asarray(
            [kalman.update(value, float(t))[:3] for t, value in zip(times, raw_phone, strict=True)]
        )

        raw_targets = _clip_targets(HOME + raw_phone @ PHONE_TO_JOINT.T)
        one_euro_targets = _clip_targets(HOME + filtered_phone @ PHONE_TO_JOINT.T)
        kalman_targets = _clip_targets(HOME + kalman_phone @ PHONE_TO_JOINT.T)
        if name in {"synchronized_joints", "point_to_point"}:
            direct = np.repeat(HOME[None, :], len(times), axis=0)
            direct[times >= 1.0] = target
            raw_targets = direct
            one_euro_targets = direct.copy()
            kalman_targets = direct.copy()
        clutch_released = times < 0.5
        raw_targets[clutch_released] = HOME
        one_euro_targets[clutch_released] = HOME
        kalman_targets[clutch_released] = HOME

        commissioning = CommissioningLimits().scaled(self.profile)
        vmax = np.r_[commissioning["arm_velocity"], commissioning["gripper_velocity"]]
        amax = np.r_[commissioning["arm_acceleration"], commissioning["gripper_acceleration"]]
        jmax = np.r_[commissioning["arm_jerk"], commissioning["gripper_jerk"]]

        quintic = OnlineQuinticRetargeter(HOME, vmax, amax, jmax)
        quintic_state = np.zeros((len(times), 4, len(JOINT_NAMES)))
        previous_target = HOME.copy()
        for index, (time_s, requested) in enumerate(zip(times, one_euro_targets, strict=True)):
            if not np.allclose(requested, previous_target, atol=1e-12):
                quintic.retarget(requested, float(time_s))
                previous_target = requested.copy()
            state = quintic.sample(float(time_s))
            quintic_state[index] = [state.position, state.velocity, state.acceleration, state.jerk]

        ruckig_stack = PhoneControlStack(JOINT_NAMES)
        ruckig_stack.set_profile(self.profile)
        measured = _joint_dict(HOME)
        ruckig_state = np.zeros((len(times), 4, len(JOINT_NAMES)))
        measured_positions = np.zeros((len(times), len(JOINT_NAMES)))
        simulated_measured = HOME.copy()
        ruckig_results: list[str] = []
        for index, requested in enumerate(one_euro_targets):
            control_active = bool(times[index] >= 0.5)
            gripper_direction = int(np.sign(requested[5] - simulated_measured[5]))
            command = ruckig_stack.step(
                measured,
                {f"{joint}.pos": float(value) for joint, value in zip(JOINT_NAMES, requested, strict=True)},
                hold_active=control_active,
                gripper_active=control_active,
                gripper_direction=gripper_direction,
            )
            snapshot = ruckig_stack.latest["ruckig"]
            command_vector = np.asarray(
                [command[f"{joint}.pos"] for joint in JOINT_NAMES], dtype=float
            )
            simulated_measured += 0.72 * (command_vector - simulated_measured)
            measured_positions[index] = simulated_measured
            measured = _joint_dict(simulated_measured)
            ruckig_state[index, 0] = command_vector
            for derivative_index, field in enumerate(
                ("velocity", "acceleration", "jerk"), start=1
            ):
                ruckig_state[index, derivative_index, :5] = list(
                    snapshot[field].values()
                )
            ruckig_results.append(snapshot["result"])

        # The sixth axis is deliberately direct rather than Ruckig-generated;
        # numerical derivatives make its discontinuities visible in the lab.
        for derivative_index in range(1, 4):
            ruckig_state[:, derivative_index, 5] = np.gradient(
                ruckig_state[:, derivative_index - 1, 5], 1.0 / FPS
            )

        streams = {
            "raw_ik": _derivative_stream(raw_targets, 1.0 / FPS),
            "one_euro": _derivative_stream(one_euro_targets, 1.0 / FPS),
            "kalman": _derivative_stream(kalman_targets, 1.0 / FPS),
            "quintic": {
                field: quintic_state[:, index, :]
                for index, field in enumerate(("position", "velocity", "acceleration", "jerk"))
            },
            "ruckig": {
                field: ruckig_state[:, index, :]
                for index, field in enumerate(("position", "velocity", "acceleration", "jerk"))
            },
            "measured": _derivative_stream(measured_positions, 1.0 / FPS),
        }

        samples: list[dict[str, Any]] = []
        for index, time_s in enumerate(times):
            speed_fraction = np.clip(
                np.abs(ruckig_state[index, 1]) / np.maximum(vmax, 1e-9),
                0.0,
                1.0,
            )
            electrical = {
                joint: {
                    "voltage_v": 7.4 - 0.08 * float(speed_fraction[joint_index]),
                    "current_ma": 110.0 + 520.0 * float(speed_fraction[joint_index]),
                    "load_percent": 4.0 + 35.0 * float(speed_fraction[joint_index]),
                    "temperature_c": 28,
                }
                for joint_index, joint in enumerate(JOINT_NAMES)
            }
            sample_streams = {
                stream_name: {
                    field: _joint_dict(stream[field][index])
                    for field in ("position", "velocity", "acceleration", "jerk")
                }
                for stream_name, stream in streams.items()
            }
            tracking = np.abs(measured_positions[index] - ruckig_state[index, 0])
            samples.append(
                {
                    "time_s": float(time_s),
                    "streams": sample_streams,
                    "phone": {
                        "true_xyz_m": true_phone[index].tolist(),
                        "raw_xyz_m": raw_phone[index].tolist(),
                        "filtered_xyz_m": filtered_phone[index].tolist(),
                        "kalman_xyz_m": kalman_phone[index].tolist(),
                    },
                    "electrical": electrical,
                    "tracking_error": _joint_dict(tracking),
                    "otg_result": ruckig_results[index],
                    "clutch_active": bool(time_s >= 0.5),
                }
            )
        return samples

    def _load_recording(self, path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("positions"):
                    rows.append(row)
        if not rows:
            raise ValueError(f"Recording contains no joint samples: {path}")
        stride = max(1, math.ceil(len(rows) / 900))
        rows = rows[::stride]
        started = float(rows[0].get("monotonic_s", 0.0))
        samples: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            measured = np.asarray([row.get("positions", {}).get(name, 0.0) for name in JOINT_NAMES], dtype=float)
            commanded = np.asarray([row.get("commands", {}).get(name, measured[i]) for i, name in enumerate(JOINT_NAMES)], dtype=float)
            requested = np.asarray([row.get("requested_commands", {}).get(name, commanded[i]) for i, name in enumerate(JOINT_NAMES)], dtype=float)
            control = row.get("control", {})
            ruckig = control.get("ruckig", {})
            def control_vector(field: str, fallback: np.ndarray) -> np.ndarray:
                values = ruckig.get(field, {})
                return np.asarray([values.get(name, fallback[i]) for i, name in enumerate(JOINT_NAMES)], dtype=float)
            zero = np.zeros(6)
            raw_phone = row.get("phone", {}).get("phone.pos", [0.0, 0.0, 0.0])
            phone_filter = control.get("phone_filter", {})
            streams = {
                "raw_ik": {"position": requested, "velocity": zero, "acceleration": zero, "jerk": zero},
                "one_euro": {"position": requested, "velocity": zero, "acceleration": zero, "jerk": zero},
                "kalman": {"position": requested, "velocity": zero, "acceleration": zero, "jerk": zero},
                "quintic": {"position": commanded, "velocity": zero, "acceleration": zero, "jerk": zero},
                "ruckig": {
                    "position": control_vector("position", commanded),
                    "velocity": control_vector("velocity", zero),
                    "acceleration": control_vector("acceleration", zero),
                    "jerk": control_vector("jerk", zero),
                },
                "measured": {"position": measured, "velocity": zero, "acceleration": zero, "jerk": zero},
            }
            samples.append(
                {
                    "time_s": float(row.get("monotonic_s", started + index / FPS)) - started,
                    "streams": {
                        name: {field: _joint_dict(values) for field, values in stream.items()}
                        for name, stream in streams.items()
                    },
                    "phone": {
                        "raw_xyz_m": raw_phone,
                        "filtered_xyz_m": phone_filter.get("filtered_position_m", raw_phone),
                        "kalman_xyz_m": raw_phone,
                    },
                    "electrical": row.get("electrical", {}),
                    "tracking_error": control.get("tracking", {}).get("errors", _joint_dict(np.abs(commanded - measured))),
                    "otg_result": ruckig.get("result", "recorded"),
                    "clutch_active": bool(row.get("phone", {}).get("phone.enabled", False)),
                }
            )
        return samples

    def load_scenario(
        self,
        name: str,
        *,
        target: list[float] | None = None,
        recording: str | None = None,
    ) -> None:
        with self._lock:
            if name == "recorded_session":
                if recording is None:
                    raise ValueError("recorded_session requires a JSONL path")
                path = Path(recording).expanduser().resolve()
                if path.suffix != ".jsonl" or not path.is_file():
                    raise ValueError("Recording must be an existing JSONL file")
                samples = self._load_recording(path)
            else:
                if name not in self.scenario_names:
                    raise ValueError(f"Unknown scenario: {name}")
                requested_target = self.target if target is None else np.asarray(target, dtype=float)
                if requested_target.shape != (6,) or not np.all(np.isfinite(requested_target)):
                    raise ValueError("Point-to-point target must contain six finite values")
                clipped = _clip_targets(requested_target)
                if not np.allclose(clipped, requested_target):
                    raise ValueError("Point-to-point target is outside SO-101 joint limits")
                self.target = requested_target.copy()
                samples = self._generate(name, requested_target)
            self.scenario = name
            self.samples = samples
            self.duration_s = samples[-1]["time_s"]
            self.playing = False
            self._paused_time_s = 0.0

    def _time_s(self) -> float:
        if not self.playing:
            return self._paused_time_s
        elapsed = (time.monotonic() - self._started_at) * self.speed
        if elapsed >= self.duration_s:
            self.playing = False
            self._paused_time_s = self.duration_s
            return self.duration_s
        return elapsed

    def _index(self, time_s: float) -> int:
        times = [sample["time_s"] for sample in self.samples]
        return min(len(times) - 1, int(np.searchsorted(times, time_s, side="right") - 1))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            time_s = self._time_s()
            index = max(0, self._index(time_s))
            sample = copy.deepcopy(self.samples[index])
            selected = sample["streams"][self.selected_stream]
            sample.update(
                {
                    "mode": "simulation",
                    "phase": "playing" if self.playing else "paused",
                    "playing": self.playing,
                    "scenario": self.scenario,
                    "duration_s": self.duration_s,
                    "playback_time_s": time_s,
                    "playback_speed": self.speed,
                    "sample_index": index,
                    "sample_count": len(self.samples),
                    "selected_stream": self.selected_stream,
                    "visible_streams": list(self.visible_streams),
                    "positions": sample["streams"]["measured"]["position"],
                    "commands": selected["position"],
                    "phone_enabled": sample["clutch_active"],
                    "active_profile": self.profile,
                    "filter_settings": dict(self.filter_settings),
                    "constraints": CommissioningLimits().scaled(self.profile),
                    "model_url": f"/model/{MODEL_FILENAME}",
                }
            )
            return sample

    def playback(self, action: str, **values: Any) -> dict[str, Any]:
        with self._lock:
            now_s = self._time_s()
            if "speed" in values:
                speed = float(values["speed"])
                if not 0.1 <= speed <= 4.0:
                    raise ValueError("Playback speed must be between 0.1 and 4.0")
                self.speed = speed
            if action == "play":
                if now_s >= self.duration_s:
                    now_s = 0.0
                self._paused_time_s = now_s
                self._started_at = time.monotonic() - now_s / self.speed
                self.playing = True
            elif action == "pause":
                self._paused_time_s = now_s
                self.playing = False
            elif action == "restart":
                self._paused_time_s = 0.0
                self.playing = False
            elif action == "step":
                self._paused_time_s = min(self.duration_s, now_s + 1.0 / FPS)
                self.playing = False
            elif action == "scrub":
                target_time = float(values.get("time_s", 0.0))
                self._paused_time_s = float(np.clip(target_time, 0.0, self.duration_s))
                self.playing = False
            else:
                raise ValueError(f"Unknown playback action: {action}")
            return self.snapshot()

    def select_streams(self, selected: str, visible: list[str]) -> dict[str, Any]:
        valid = set(self.samples[0]["streams"])
        if selected not in valid or not visible or not set(visible) <= valid:
            raise ValueError(f"Streams must be selected from {sorted(valid)}")
        with self._lock:
            self.selected_stream = selected
            self.visible_streams = list(dict.fromkeys(visible))
            return self.snapshot()

    def change_settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self.snapshot()["clutch_active"]:
                raise RuntimeError("Release the simulated Hold before changing settings")
            profile = settings.get("profile", self.profile)
            if profile not in PROFILE_NAMES:
                raise ValueError(f"Unknown motion profile: {profile}")
            next_filter = validated_phone_filter_settings(
                {name: value for name, value in settings.items() if name != "profile"},
                base=self.filter_settings,
            )
            self.profile = profile
            self.filter_settings = next_filter
            current_scenario = self.scenario
            if current_scenario != "recorded_session":
                self.load_scenario(current_scenario, target=self.target.tolist())
            return {"profile": self.profile, **self.filter_settings}

    def history(self) -> dict[str, Any]:
        with self._lock:
            return {
                "scenario": self.scenario,
                "joint_names": list(JOINT_NAMES),
                "samples": copy.deepcopy(self.samples),
                "constraints": CommissioningLimits().scaled(self.profile),
            }


def create_app(
    engine: SimulationEngine | None = None,
    *,
    model_cache: Path = DEFAULT_MODEL_CACHE,
    ready_event: threading.Event | None = None,
) -> FastAPI:
    engine = engine or SimulationEngine()
    ready_event = ready_event or threading.Event()
    static_dir = Path(__file__).parent / "dashboard"

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        ready_event.set()
        yield

    app = FastAPI(title="SO-101 Trajectory Lab", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.middleware("http")
    async def localhost_only(request, call_next):
        host = request.client.host if request.client else ""
        if host not in {"127.0.0.1", "::1", "testclient"}:
            return JSONResponse(status_code=403, content={"detail": "SO-101 simulator is restricted to localhost"})
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        return {"mode": "simulation", "model_url": f"/model/{MODEL_FILENAME}", "joint_names": list(JOINT_NAMES)}

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        return engine.snapshot()

    @app.get("/api/scenarios")
    def scenarios() -> dict[str, Any]:
        return {"scenarios": [*engine.scenario_names, "recorded_session"], "default_target": DEFAULT_TARGET.tolist()}

    @app.post("/api/scenario")
    def scenario(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            engine.load_scenario(payload.get("name", ""), target=payload.get("target"), recording=payload.get("recording"))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return engine.snapshot()

    @app.post("/api/playback")
    def playback(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return engine.playback(payload.get("action", ""), **{key: value for key, value in payload.items() if key != "action"})
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.put("/api/streams")
    def streams(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return engine.select_streams(payload.get("selected", ""), payload.get("visible", []))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.put("/api/settings")
    @app.post("/api/settings")
    def settings(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return engine.change_settings(payload)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/history")
    def history() -> dict[str, Any]:
        return engine.history()

    @app.get("/model/{asset_path:path}", include_in_schema=False)
    def model_asset(asset_path: str) -> FileResponse:
        root = Path(model_cache).resolve()
        requested = (root / asset_path).resolve()
        if not requested.is_relative_to(root) or not requested.is_file():
            raise HTTPException(status_code=404, detail="SO-101 model asset not found")
        return FileResponse(requested, media_type="application/xml" if requested.suffix == ".urdf" else "model/stl")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="SO-101 hardware-free trajectory lab")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("The SO-101 simulator may only bind to localhost")

    metadata = ensure_model_cache()
    kinematic = Path(__file__).parent / "kinematics" / "so101_kinematics.urdf"
    verify_kinematic_urdf(kinematic, DEFAULT_MODEL_CACHE / metadata["urdf"])
    app = create_app(model_cache=DEFAULT_MODEL_CACHE)
    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        import webbrowser

        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    print(f"SO-101 trajectory lab: {url}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
