"""Low-overhead telemetry and incident capture for SO-101 phone teleoperation."""

from __future__ import annotations

import json
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any


CURRENT_MA_PER_COUNT = 6.5
VOLTAGE_V_PER_COUNT = 0.1
LOAD_PERCENT_PER_COUNT = 0.1


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


class ElectricalTelemetrySampler:
    """Poll slower electrical registers without loading the bus every control frame."""

    def __init__(self, joint_names: list[str], frequency_hz: float = 5.0) -> None:
        self.joint_names = joint_names
        self.interval_s = 1.0 / frequency_hz
        self.next_sample_at = 0.0
        self.next_slow_sample_at = 0.0
        self._load_raw: dict[str, int] = {}
        self._temperature_c: dict[str, int] = {}
        self.latest: dict[str, dict[str, float | int]] = {}

    def maybe_read(self, bus, *, force: bool = False) -> dict[str, dict[str, float | int]]:
        now = time.monotonic()
        if not force and now < self.next_sample_at:
            return self.latest

        voltage_raw = bus.sync_read("Present_Voltage", normalize=False, num_retry=2)
        current_raw = bus.sync_read("Present_Current", normalize=False, num_retry=2)
        if force or now >= self.next_slow_sample_at or not self._load_raw:
            self._load_raw = {
                joint: int(value)
                for joint, value in bus.sync_read(
                    "Present_Load", normalize=False, num_retry=2
                ).items()
            }
            self._temperature_c = {
                joint: int(value)
                for joint, value in bus.sync_read(
                    "Present_Temperature", normalize=False, num_retry=2
                ).items()
            }
            self.next_slow_sample_at = now + 1.0

        self.latest = {
            joint: {
                "voltage_raw": int(voltage_raw[joint]),
                "voltage_v": float(voltage_raw[joint]) * VOLTAGE_V_PER_COUNT,
                "current_raw": int(current_raw[joint]),
                "current_ma": abs(float(current_raw[joint])) * CURRENT_MA_PER_COUNT,
                "load_raw": self._load_raw[joint],
                "load_percent": float(self._load_raw[joint]) * LOAD_PERCENT_PER_COUNT,
                "temperature_c": self._temperature_c[joint],
            }
            for joint in self.joint_names
        }
        self.next_sample_at = now + self.interval_s
        return self.latest

    def rerun_scalars(self) -> dict[str, float]:
        scalars: dict[str, float] = {}
        for joint, values in self.latest.items():
            for name in ("voltage_v", "current_ma", "load_percent", "temperature_c"):
                scalars[f"telemetry.{joint}.{name}"] = float(values[name])
        return scalars

    def summary(self) -> str:
        if not self.latest:
            return "electrical telemetry unavailable"
        min_voltage = min(float(values["voltage_v"]) for values in self.latest.values())
        total_current = sum(float(values["current_ma"]) for values in self.latest.values())
        peak_joint = max(self.latest, key=lambda joint: float(self.latest[joint]["current_ma"]))
        peak_current = float(self.latest[peak_joint]["current_ma"])
        hottest = max(int(values["temperature_c"]) for values in self.latest.values())
        return (
            f"Vmin={min_voltage:.1f}V Itotal={total_current:.0f}mA "
            f"Ipeak={peak_joint}:{peak_current:.0f}mA Tmax={hottest}C"
        )


class FlightRecorder:
    """Write a session log and retain a high-rate pre-failure ring buffer."""

    def __init__(
        self,
        log_dir: Path,
        joint_names: list[str],
        *,
        fps: int,
        history_seconds: int = 20,
    ) -> None:
        self.log_dir = log_dir
        self.joint_names = joint_names
        self.phase = "startup"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        self.session_path = self.log_dir / f"session_{stamp}.jsonl"
        self._stream = self.session_path.open("a", buffering=1)
        self._ring: deque[dict[str, Any]] = deque(maxlen=fps * history_seconds)
        self._incident_counter = 0
        self._last_console_summary = 0.0

    def set_phase(self, phase: str) -> None:
        self.phase = phase

    def record(
        self,
        *,
        observation: dict[str, Any] | None = None,
        action: dict[str, Any] | None = None,
        requested_action: dict[str, Any] | None = None,
        phone_action: dict[str, Any] | None = None,
        electrical: dict[str, dict[str, float | int]] | None = None,
        cartesian: dict[str, Any] | None = None,
        loop_ms: float | None = None,
        event: str | None = None,
    ) -> dict[str, Any]:
        sample = {
            "timestamp": _timestamp(),
            "monotonic_s": time.monotonic(),
            "phase": self.phase,
            "event": event,
            "loop_ms": loop_ms,
            "positions": {
                joint: float(observation[f"{joint}.pos"])
                for joint in self.joint_names
                if observation is not None and f"{joint}.pos" in observation
            },
            "commands": {
                joint: float(action[f"{joint}.pos"])
                for joint in self.joint_names
                if action is not None and f"{joint}.pos" in action
            },
            "requested_commands": {
                joint: float(requested_action[f"{joint}.pos"])
                for joint in self.joint_names
                if requested_action is not None and f"{joint}.pos" in requested_action
            },
            "cartesian": _json_value(cartesian or {}),
            "phone": _json_value(phone_action or {}),
            "electrical": _json_value(electrical or {}),
        }
        self._ring.append(sample)
        self._stream.write(json.dumps(sample, separators=(",", ":")) + "\n")
        return sample

    def record_electrical_summary(self, sampler: ElectricalTelemetrySampler) -> None:
        now = time.monotonic()
        if now - self._last_console_summary >= 1.0:
            print(f"TELEMETRY {sampler.summary()}")
            self._last_console_summary = now

    def capture_incident(self, exc: BaseException, **details: Any) -> Path:
        self._incident_counter += 1
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        incident_path = self.log_dir / f"incident_{stamp}_{self._incident_counter}.json"
        payload = {
            "captured_at": _timestamp(),
            "phase": self.phase,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "details": _json_value(details),
            "session_log": str(self.session_path),
            "history": list(self._ring),
        }
        incident_path.write_text(json.dumps(payload, indent=2) + "\n")
        self._stream.flush()
        return incident_path

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.flush()
            self._stream.close()
