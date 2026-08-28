"""Phone-pose filters used by live teleoperation and offline comparisons."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


DEFAULT_PHONE_FILTER_SETTINGS = {
    "min_cutoff_hz": 1.0,
    "beta": 2.0,
    "derivative_cutoff_hz": 1.0,
    "deadband_m": 0.00025,
}

PHONE_FILTER_SETTING_BOUNDS = {
    "min_cutoff_hz": (0.1, 10.0),
    "beta": (0.0, 10.0),
    "derivative_cutoff_hz": (0.1, 10.0),
    "deadband_m": (0.0, 0.01),
}


def validated_phone_filter_settings(
    settings: dict[str, Any],
    *,
    base: dict[str, float] | None = None,
) -> dict[str, float]:
    """Merge and validate user-adjustable One-Euro settings."""
    unknown = set(settings) - set(PHONE_FILTER_SETTING_BOUNDS)
    if unknown:
        raise ValueError(f"Unknown phone filter setting(s): {', '.join(sorted(unknown))}")
    merged = dict(DEFAULT_PHONE_FILTER_SETTINGS if base is None else base)
    for name, value in settings.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
        lower, upper = PHONE_FILTER_SETTING_BOUNDS[name]
        if not math.isfinite(numeric) or not lower <= numeric <= upper:
            raise ValueError(f"{name} must be between {lower} and {upper}")
        merged[name] = numeric
    return merged


def _alpha(cutoff_hz: np.ndarray | float, dt: float) -> np.ndarray:
    cutoff = np.asarray(cutoff_hz, dtype=float)
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


@dataclass(frozen=True)
class PhoneFilterSample:
    """Complete diagnostic state for one phone-position sample."""

    timestamp_s: float
    raw_position_m: list[float]
    filtered_position_m: list[float]
    estimated_velocity_m_s: list[float]
    cutoff_hz: list[float]
    deadband_position_m: list[float]
    dt_s: float | None
    reset: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OneEuroXYZFilter:
    """Adaptive One-Euro filter with a radial output deadband.

    The derivative and signal low-pass filters are evaluated independently per
    Cartesian axis. The deadband is radial: the output changes only when the
    Euclidean displacement from the last emitted point reaches ``deadband_m``.
    """

    def __init__(
        self,
        *,
        min_cutoff_hz: float = DEFAULT_PHONE_FILTER_SETTINGS["min_cutoff_hz"],
        beta: float = DEFAULT_PHONE_FILTER_SETTINGS["beta"],
        derivative_cutoff_hz: float = DEFAULT_PHONE_FILTER_SETTINGS[
            "derivative_cutoff_hz"
        ],
        deadband_m: float = DEFAULT_PHONE_FILTER_SETTINGS["deadband_m"],
        min_dt_s: float = 1.0 / 120.0,
        max_dt_s: float = 0.1,
        reset_gap_s: float = 0.2,
    ) -> None:
        if min_cutoff_hz <= 0 or derivative_cutoff_hz <= 0:
            raise ValueError("One-Euro cutoffs must be positive")
        if beta < 0 or deadband_m < 0:
            raise ValueError("One-Euro beta and deadband must be non-negative")
        if not (0 < min_dt_s <= max_dt_s < reset_gap_s):
            raise ValueError("Expected 0 < min_dt <= max_dt < reset_gap")
        self.min_cutoff_hz = float(min_cutoff_hz)
        self.beta = float(beta)
        self.derivative_cutoff_hz = float(derivative_cutoff_hz)
        self.deadband_m = float(deadband_m)
        self.min_dt_s = float(min_dt_s)
        self.max_dt_s = float(max_dt_s)
        self.reset_gap_s = float(reset_gap_s)
        self.reset()

    def reset(self) -> None:
        self._timestamp_s: float | None = None
        self._raw: np.ndarray | None = None
        self._filtered: np.ndarray | None = None
        self._velocity: np.ndarray | None = None
        self._deadband_output: np.ndarray | None = None
        self.latest: PhoneFilterSample | None = None

    def _initialize(self, position: np.ndarray, timestamp_s: float) -> PhoneFilterSample:
        self._timestamp_s = timestamp_s
        self._raw = position.copy()
        self._filtered = position.copy()
        self._velocity = np.zeros(3, dtype=float)
        self._deadband_output = position.copy()
        self.latest = PhoneFilterSample(
            timestamp_s=timestamp_s,
            raw_position_m=position.tolist(),
            filtered_position_m=position.tolist(),
            estimated_velocity_m_s=[0.0, 0.0, 0.0],
            cutoff_hz=[self.min_cutoff_hz] * 3,
            deadband_position_m=position.tolist(),
            dt_s=None,
            reset=True,
        )
        return self.latest

    def update(self, position_m: Any, timestamp_s: float) -> PhoneFilterSample:
        position = np.asarray(position_m, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("Phone position must contain three finite XYZ values")
        timestamp = float(timestamp_s)
        if not math.isfinite(timestamp):
            raise ValueError("Phone sample timestamp must be finite")

        if self._timestamp_s is None:
            return self._initialize(position, timestamp)
        elapsed = timestamp - self._timestamp_s
        if elapsed <= 0.0 or elapsed > self.reset_gap_s:
            return self._initialize(position, timestamp)

        dt = float(np.clip(elapsed, self.min_dt_s, self.max_dt_s))
        assert self._raw is not None
        assert self._filtered is not None
        assert self._velocity is not None
        assert self._deadband_output is not None

        raw_velocity = (position - self._raw) / dt
        derivative_alpha = _alpha(self.derivative_cutoff_hz, dt)
        velocity = derivative_alpha * raw_velocity + (1.0 - derivative_alpha) * self._velocity
        cutoff = self.min_cutoff_hz + self.beta * np.abs(velocity)
        signal_alpha = _alpha(cutoff, dt)
        filtered = signal_alpha * position + (1.0 - signal_alpha) * self._filtered

        deadband_output = self._deadband_output
        if float(np.linalg.norm(filtered - deadband_output)) >= self.deadband_m:
            deadband_output = filtered.copy()

        self._timestamp_s = timestamp
        self._raw = position.copy()
        self._filtered = filtered
        self._velocity = velocity
        self._deadband_output = deadband_output
        self.latest = PhoneFilterSample(
            timestamp_s=timestamp,
            raw_position_m=position.tolist(),
            filtered_position_m=filtered.tolist(),
            estimated_velocity_m_s=velocity.tolist(),
            cutoff_hz=cutoff.tolist(),
            deadband_position_m=deadband_output.tolist(),
            dt_s=dt,
            reset=False,
        )
        return self.latest


class ConstantVelocityKalmanXYZ:
    """Offline constant-velocity Kalman filter for educational comparison."""

    def __init__(
        self,
        *,
        process_acceleration_std: float = 0.8,
        measurement_std: float = 0.003,
        initial_velocity_std: float = 1.0,
    ) -> None:
        if min(process_acceleration_std, measurement_std, initial_velocity_std) <= 0:
            raise ValueError("Kalman standard deviations must be positive")
        self.process_acceleration_std = float(process_acceleration_std)
        self.measurement_std = float(measurement_std)
        self.initial_velocity_std = float(initial_velocity_std)
        self.reset()

    def reset(self) -> None:
        self.state = np.zeros(6, dtype=float)
        self.covariance = np.eye(6, dtype=float)
        self._timestamp_s: float | None = None

    def update(self, position_m: Any, timestamp_s: float) -> np.ndarray:
        measurement = np.asarray(position_m, dtype=float)
        if measurement.shape != (3,) or not np.all(np.isfinite(measurement)):
            raise ValueError("Kalman measurement must contain three finite XYZ values")
        timestamp = float(timestamp_s)
        if self._timestamp_s is None or timestamp <= self._timestamp_s:
            self.state = np.r_[measurement, np.zeros(3, dtype=float)]
            variances = [self.measurement_std**2] * 3 + [self.initial_velocity_std**2] * 3
            self.covariance = np.diag(variances)
            self._timestamp_s = timestamp
            return self.state.copy()

        dt = timestamp - self._timestamp_s
        transition = np.eye(6, dtype=float)
        transition[:3, 3:] = np.eye(3) * dt
        noise_axis = np.array(
            [[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]],
            dtype=float,
        ) * self.process_acceleration_std**2
        process_noise = np.zeros((6, 6), dtype=float)
        for axis in range(3):
            indices = np.ix_([axis, axis + 3], [axis, axis + 3])
            process_noise[indices] = noise_axis

        predicted = transition @ self.state
        predicted_covariance = transition @ self.covariance @ transition.T + process_noise
        observation = np.c_[np.eye(3), np.zeros((3, 3))]
        residual_covariance = (
            observation @ predicted_covariance @ observation.T
            + np.eye(3) * self.measurement_std**2
        )
        gain = predicted_covariance @ observation.T @ np.linalg.inv(residual_covariance)
        self.state = predicted + gain @ (measurement - observation @ predicted)
        identity = np.eye(6)
        self.covariance = (identity - gain @ observation) @ predicted_covariance
        self._timestamp_s = timestamp
        return self.state.copy()
