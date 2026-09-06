from __future__ import annotations

import pytest
import torch

from mini_pi0.utils.precision import (
    autocast_context,
    describe_runtime_dtype,
    resolve_inference_dtype,
    resolve_runtime_dtype,
)


@pytest.mark.parametrize("value", ["bf16", "bfloat16", "torch.bfloat16"])
def test_resolve_runtime_dtype_accepts_bfloat16_aliases(value: str) -> None:
    # Arrange / Act
    dtype = resolve_runtime_dtype(runtime_dtype=value, model_dtype=None)

    # Assert
    assert dtype is torch.bfloat16


def test_resolve_runtime_dtype_prefers_runtime_override_over_model_dtype() -> None:
    # Arrange / Act
    dtype = resolve_runtime_dtype(runtime_dtype="fp32", model_dtype="bf16")

    # Assert
    assert dtype is None


def test_resolve_runtime_dtype_falls_back_to_model_dtype() -> None:
    # Arrange / Act
    dtype = resolve_runtime_dtype(runtime_dtype=None, model_dtype="bfloat16")

    # Assert
    assert dtype is torch.bfloat16


def test_describe_runtime_dtype_reports_float32_when_autocast_disabled() -> None:
    # Arrange / Act
    description = describe_runtime_dtype(runtime_dtype="auto", model_dtype=None)

    # Assert
    assert description == "float32"


def test_resolve_runtime_dtype_rejects_unknown_dtype() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="Unsupported runtime dtype"):
        resolve_runtime_dtype(runtime_dtype="int8", model_dtype=None)


def test_autocast_context_rejects_mixed_precision_on_unsupported_device() -> None:
    device = torch.device("meta")

    # Act / Assert
    with pytest.raises(ValueError, match="unsupported"):
        autocast_context(device=device, dtype=torch.bfloat16)


def test_inference_auto_precision_uses_fp32_on_mps_and_cpu() -> None:
    assert resolve_inference_dtype(device=torch.device("mps"), requested="auto") is None
    assert resolve_inference_dtype(device=torch.device("cpu"), requested="auto") is None


def test_inference_precision_honors_explicit_mps_dtype() -> None:
    dtype = resolve_inference_dtype(device=torch.device("mps"), requested="fp16")

    assert dtype is torch.float16
    with autocast_context(device=torch.device("mps"), dtype=dtype):
        pass


def test_inference_auto_precision_prefers_bfloat16_on_supported_cuda(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    dtype = resolve_inference_dtype(device=torch.device("cuda"), requested="auto")

    assert dtype is torch.bfloat16


def test_inference_auto_precision_falls_back_to_float16_on_older_cuda(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    dtype = resolve_inference_dtype(device=torch.device("cuda"), requested="auto")

    assert dtype is torch.float16
