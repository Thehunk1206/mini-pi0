"""Domain exceptions raised by ReinFlow training infrastructure."""

from __future__ import annotations


class ReinFlowError(RuntimeError):
    """Base exception for ReinFlow runtime failures."""


class ReinFlowConfigError(ReinFlowError, ValueError):
    """Raised when an RL configuration violates an algorithm contract."""


class ReinFlowCheckpointError(ReinFlowError):
    """Raised when an RL checkpoint is missing or incompatible."""


class ReinFlowNumericalError(ReinFlowError):
    """Raised when a policy or optimizer tensor becomes non-finite."""
