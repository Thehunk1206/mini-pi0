"""Educational, simulation-only synchronized quintic trajectory generation.

The live arm uses Ruckig. This module intentionally exposes the mathematics of
online retargeting: each replacement polynomial begins at the exact sampled
position, velocity, and acceleration of the previous polynomial, giving C2
continuity. Jerk is bounded inside every segment but may jump at a retarget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MotionState:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray


@dataclass(frozen=True)
class QuinticSegment:
    """One or more fifth-order polynomials sharing a duration."""

    coefficients: np.ndarray
    duration_s: float

    @classmethod
    def from_boundary_conditions(
        cls,
        position: Any,
        velocity: Any,
        acceleration: Any,
        target_position: Any,
        duration_s: float,
    ) -> "QuinticSegment":
        p0 = np.atleast_1d(np.asarray(position, dtype=float))
        v0 = np.atleast_1d(np.asarray(velocity, dtype=float))
        a0 = np.atleast_1d(np.asarray(acceleration, dtype=float))
        p1 = np.atleast_1d(np.asarray(target_position, dtype=float))
        if not (p0.shape == v0.shape == a0.shape == p1.shape):
            raise ValueError("Quintic boundary arrays must have identical shapes")
        duration = float(duration_s)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Quintic duration must be finite and positive")
        if not all(np.all(np.isfinite(values)) for values in (p0, v0, a0, p1)):
            raise ValueError("Quintic boundary conditions must be finite")

        coefficients = np.empty((p0.size, 6), dtype=float)
        coefficients[:, 0] = p0
        coefficients[:, 1] = v0
        coefficients[:, 2] = a0 / 2.0
        t = duration
        matrix = np.array(
            [
                [t**3, t**4, t**5],
                [3 * t**2, 4 * t**3, 5 * t**4],
                [6 * t, 12 * t**2, 20 * t**3],
            ],
            dtype=float,
        )
        known_end_position = coefficients[:, 0] + coefficients[:, 1] * t + coefficients[:, 2] * t**2
        rhs = np.column_stack(
            [
                p1 - known_end_position,
                -coefficients[:, 1] - 2 * coefficients[:, 2] * t,
                -2 * coefficients[:, 2],
            ]
        )
        coefficients[:, 3:] = np.linalg.solve(matrix, rhs.T).T
        return cls(coefficients=coefficients, duration_s=duration)

    @property
    def degrees_of_freedom(self) -> int:
        return int(self.coefficients.shape[0])

    def sample(self, time_s: float) -> MotionState:
        t = float(np.clip(time_s, 0.0, self.duration_s))
        c = self.coefficients
        position = c[:, 0] + c[:, 1] * t + c[:, 2] * t**2 + c[:, 3] * t**3 + c[:, 4] * t**4 + c[:, 5] * t**5
        velocity = c[:, 1] + 2 * c[:, 2] * t + 3 * c[:, 3] * t**2 + 4 * c[:, 4] * t**3 + 5 * c[:, 5] * t**4
        acceleration = 2 * c[:, 2] + 6 * c[:, 3] * t + 12 * c[:, 4] * t**2 + 20 * c[:, 5] * t**3
        jerk = 6 * c[:, 3] + 24 * c[:, 4] * t + 60 * c[:, 5] * t**2
        return MotionState(position, velocity, acceleration, jerk)

    @staticmethod
    def _real_times(polynomial_descending: np.ndarray, duration_s: float) -> list[float]:
        coefficients = np.trim_zeros(polynomial_descending, "f")
        if coefficients.size <= 1:
            return []
        roots = np.roots(coefficients)
        return [
            float(root.real)
            for root in roots
            if abs(float(root.imag)) <= 1e-9 and 0.0 < float(root.real) < duration_s
        ]

    def peak_derivatives(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return exact per-axis maxima of |velocity|, |acceleration|, and |jerk|."""
        velocity_peaks = np.zeros(self.degrees_of_freedom)
        acceleration_peaks = np.zeros(self.degrees_of_freedom)
        jerk_peaks = np.zeros(self.degrees_of_freedom)
        for axis, c in enumerate(self.coefficients):
            velocity_times = [0.0, self.duration_s] + self._real_times(
                np.array([20 * c[5], 12 * c[4], 6 * c[3], 2 * c[2]]), self.duration_s
            )
            acceleration_times = [0.0, self.duration_s] + self._real_times(
                np.array([60 * c[5], 24 * c[4], 6 * c[3]]), self.duration_s
            )
            jerk_times = [0.0, self.duration_s] + self._real_times(
                np.array([120 * c[5], 24 * c[4]]), self.duration_s
            )
            velocity_peaks[axis] = max(abs(self.sample(t).velocity[axis]) for t in velocity_times)
            acceleration_peaks[axis] = max(abs(self.sample(t).acceleration[axis]) for t in acceleration_times)
            jerk_peaks[axis] = max(abs(self.sample(t).jerk[axis]) for t in jerk_times)
        return velocity_peaks, acceleration_peaks, jerk_peaks

    def within_limits(
        self,
        max_velocity: Any,
        max_acceleration: Any,
        max_jerk: Any,
        *,
        tolerance: float = 1e-9,
    ) -> bool:
        limits = [np.atleast_1d(np.asarray(value, dtype=float)) for value in (max_velocity, max_acceleration, max_jerk)]
        if any(value.shape != (self.degrees_of_freedom,) for value in limits):
            raise ValueError("Trajectory limit arrays must match degrees of freedom")
        peaks = self.peak_derivatives()
        return all(np.all(peak <= limit * (1.0 + tolerance)) for peak, limit in zip(peaks, limits, strict=True))

    def numerically_within_limits(
        self,
        max_velocity: Any,
        max_acceleration: Any,
        max_jerk: Any,
        *,
        samples: int = 257,
        tolerance: float = 1e-9,
    ) -> bool:
        """Densely sample the complete segment as an independent validation."""
        if samples < 3:
            raise ValueError("Numerical trajectory validation needs at least three samples")
        limits = [np.atleast_1d(np.asarray(value, dtype=float)) for value in (max_velocity, max_acceleration, max_jerk)]
        if any(value.shape != (self.degrees_of_freedom,) for value in limits):
            raise ValueError("Trajectory limit arrays must match degrees of freedom")
        t = np.linspace(0.0, self.duration_s, samples)[:, None]
        c = self.coefficients[None, :, :]
        velocity = (
            c[:, :, 1]
            + 2 * c[:, :, 2] * t
            + 3 * c[:, :, 3] * t**2
            + 4 * c[:, :, 4] * t**3
            + 5 * c[:, :, 5] * t**4
        )
        acceleration = (
            2 * c[:, :, 2]
            + 6 * c[:, :, 3] * t
            + 12 * c[:, :, 4] * t**2
            + 20 * c[:, :, 5] * t**3
        )
        jerk = 6 * c[:, :, 3] + 24 * c[:, :, 4] * t + 60 * c[:, :, 5] * t**2
        return all(
            np.all(np.max(np.abs(values), axis=0) <= limit * (1.0 + tolerance))
            for values, limit in zip((velocity, acceleration, jerk), limits, strict=True)
        )


def _initial_duration(
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    target: np.ndarray,
    max_velocity: np.ndarray,
    max_acceleration: np.ndarray,
    max_jerk: np.ndarray,
    minimum_duration_s: float,
) -> float:
    distance = np.abs(target - position)
    estimates = np.r_[
        distance / max_velocity,
        np.sqrt(distance / max_acceleration),
        np.cbrt(distance / max_jerk),
        np.abs(velocity) / max_acceleration,
        np.sqrt(np.abs(velocity) / max_jerk),
        np.abs(acceleration) / max_jerk,
        [minimum_duration_s],
    ]
    return float(max(np.max(estimates), minimum_duration_s))


def synchronized_quintic(
    position: Any,
    velocity: Any,
    acceleration: Any,
    target_position: Any,
    max_velocity: Any,
    max_acceleration: Any,
    max_jerk: Any,
    *,
    minimum_duration_s: float = 1.0 / 30.0,
    expansion: float = 1.2,
    max_iterations: int = 200,
) -> QuinticSegment:
    """Build the shortest numerically validated, synchronized quintic segment."""
    arrays = [np.atleast_1d(np.asarray(value, dtype=float)) for value in (position, velocity, acceleration, target_position, max_velocity, max_acceleration, max_jerk)]
    if any(value.shape != arrays[0].shape for value in arrays[1:]):
        raise ValueError("Quintic state, target, and limit arrays must have identical shapes")
    p0, v0, a0, p1, vmax, amax, jmax = arrays
    if np.any(vmax <= 0) or np.any(amax <= 0) or np.any(jmax <= 0):
        raise ValueError("Quintic limits must be positive")
    if expansion <= 1.0:
        raise ValueError("Duration expansion factor must exceed one")

    # Find a feasible time for each axis, then use the longest as the shared
    # duration. The final multi-axis validation also covers nonzero start state.
    durations: list[float] = []
    for axis in range(p0.size):
        duration = _initial_duration(
            p0[axis : axis + 1], v0[axis : axis + 1], a0[axis : axis + 1], p1[axis : axis + 1],
            vmax[axis : axis + 1], amax[axis : axis + 1], jmax[axis : axis + 1], minimum_duration_s,
        )
        for _ in range(max_iterations):
            segment = QuinticSegment.from_boundary_conditions(
                p0[axis : axis + 1], v0[axis : axis + 1], a0[axis : axis + 1], p1[axis : axis + 1], duration
            )
            axis_limits = (
                vmax[axis : axis + 1],
                amax[axis : axis + 1],
                jmax[axis : axis + 1],
            )
            if segment.within_limits(*axis_limits) and segment.numerically_within_limits(
                *axis_limits
            ):
                durations.append(duration)
                break
            duration *= expansion
        else:
            raise RuntimeError("Could not find a feasible quintic duration")

    duration = max(durations)
    for _ in range(max_iterations):
        segment = QuinticSegment.from_boundary_conditions(p0, v0, a0, p1, duration)
        if segment.within_limits(vmax, amax, jmax) and segment.numerically_within_limits(
            vmax, amax, jmax
        ):
            return segment
        duration *= expansion
    raise RuntimeError("Could not synchronize a feasible quintic trajectory")


class OnlineQuinticRetargeter:
    """Simulation-only online retargeter with C2 segment boundaries."""

    def __init__(
        self,
        position: Any,
        max_velocity: Any,
        max_acceleration: Any,
        max_jerk: Any,
        *,
        timestamp_s: float = 0.0,
    ) -> None:
        self.max_velocity = np.atleast_1d(np.asarray(max_velocity, dtype=float))
        self.max_acceleration = np.atleast_1d(np.asarray(max_acceleration, dtype=float))
        self.max_jerk = np.atleast_1d(np.asarray(max_jerk, dtype=float))
        initial = np.atleast_1d(np.asarray(position, dtype=float))
        if any(limit.shape != initial.shape for limit in (self.max_velocity, self.max_acceleration, self.max_jerk)):
            raise ValueError("Online quintic limits must match the initial position")
        self._state = MotionState(initial.copy(), np.zeros_like(initial), np.zeros_like(initial), np.zeros_like(initial))
        self._segment: QuinticSegment | None = None
        self._segment_started_s = float(timestamp_s)
        self.target_position = initial.copy()

    def sample(self, timestamp_s: float) -> MotionState:
        if self._segment is None:
            return MotionState(*(value.copy() for value in (self._state.position, self._state.velocity, self._state.acceleration, self._state.jerk)))
        elapsed = max(0.0, float(timestamp_s) - self._segment_started_s)
        self._state = self._segment.sample(elapsed)
        return self._state

    def retarget(self, target_position: Any, timestamp_s: float) -> QuinticSegment:
        start = self.sample(timestamp_s)
        target = np.atleast_1d(np.asarray(target_position, dtype=float))
        if target.shape != start.position.shape:
            raise ValueError("Online quintic target shape does not match its state")
        self._segment = synchronized_quintic(
            start.position,
            start.velocity,
            start.acceleration,
            target,
            self.max_velocity,
            self.max_acceleration,
            self.max_jerk,
        )
        self._segment_started_s = float(timestamp_s)
        self.target_position = target.copy()
        return self._segment
