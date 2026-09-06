from __future__ import annotations

"""Runtime precision helpers for train/eval forward passes.

This module centralizes dtype parsing and autocast creation so training and
evaluation use the same config contract.
"""

from contextlib import AbstractContextManager, nullcontext

import torch

_BF16_ALIASES = {"bf16", "bfloat16", "torch.bfloat16"}
_FP16_ALIASES = {"fp16", "float16", "half", "torch.float16"}
_FP32_ALIASES = {"fp32", "float32", "full", "none", "torch.float32"}
_AUTO_ALIASES = {"", "auto", "default", "null"}
_AUTOCAST_DEVICE_TYPES = {"cuda", "cpu", "mps"}


def resolve_runtime_dtype(
    *,
    runtime_dtype: str | None,
    model_dtype: str | None,
) -> torch.dtype | None:
    """Resolve the autocast dtype requested by config.

    Args:
        runtime_dtype: Command-specific dtype such as ``eval.dtype`` or
            ``train.dtype``. When null/auto, falls back to ``model.dtype``.
        model_dtype: Model-level dtype fallback.

    Returns:
        ``torch.bfloat16`` / ``torch.float16`` when mixed precision is enabled,
        otherwise ``None`` for fp32/no autocast.

    Raises:
        ValueError: If either dtype string is unknown.

    Example:
        >>> resolve_runtime_dtype(runtime_dtype="bf16", model_dtype=None)
        torch.bfloat16
    """

    selected = _select_dtype_text(runtime_dtype=runtime_dtype, model_dtype=model_dtype)
    if selected in _BF16_ALIASES:
        return torch.bfloat16
    if selected in _FP16_ALIASES:
        return torch.float16
    if selected in _FP32_ALIASES or selected in _AUTO_ALIASES:
        return None
    raise ValueError(
        f"Unsupported runtime dtype '{selected}'. "
        "Use one of: auto, fp32, float32, bf16, bfloat16, fp16, float16."
    )


def autocast_context(
    *,
    device: torch.device,
    dtype: torch.dtype | None,
) -> AbstractContextManager[None]:
    """Create a torch autocast context for the requested dtype.

    Args:
        device: Target torch device.
        dtype: Autocast dtype, or ``None`` to disable autocast.

    Returns:
        A context manager suitable for wrapping model forward calls.

    Raises:
        ValueError: If mixed precision is requested on an unsupported device.
    """

    if dtype is None:
        return nullcontext()
    if device.type not in _AUTOCAST_DEVICE_TYPES:
        raise ValueError(
            f"Runtime dtype {dtype} requires autocast, but device '{device.type}' is unsupported. "
            "Use cuda/cpu or set dtype to fp32."
        )
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)


def resolve_inference_dtype(
    *,
    device: torch.device,
    requested: str | None = "auto",
) -> torch.dtype | None:
    """Resolve a backend-aware inference precision.

    MPS defaults to fp32 because the small SO-101 policies benchmark faster in
    fp32 on Apple Silicon. CUDA prefers bf16 when the GPU supports it and fp16
    otherwise. CPU and unknown backends default to fp32. Explicit requests are
    still honored and validated by :func:`autocast_context`.
    """

    normalized = _normalize_dtype_text(requested)
    if normalized not in _AUTO_ALIASES:
        return resolve_runtime_dtype(runtime_dtype=normalized, model_dtype=None)
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return None


def describe_runtime_dtype(
    *,
    runtime_dtype: str | None,
    model_dtype: str | None,
) -> str:
    """Return a compact display string for the resolved runtime precision."""

    dtype = resolve_runtime_dtype(runtime_dtype=runtime_dtype, model_dtype=model_dtype)
    if dtype is torch.bfloat16:
        return "bfloat16"
    if dtype is torch.float16:
        return "float16"
    return "float32"


def _select_dtype_text(*, runtime_dtype: str | None, model_dtype: str | None) -> str:
    """Pick command dtype first, falling back to model dtype."""

    runtime = _normalize_dtype_text(runtime_dtype)
    if runtime not in _AUTO_ALIASES:
        return runtime
    return _normalize_dtype_text(model_dtype)


def _normalize_dtype_text(value: str | None) -> str:
    """Normalize nullable dtype config values."""

    if value is None:
        return "auto"
    return str(value).strip().lower()
