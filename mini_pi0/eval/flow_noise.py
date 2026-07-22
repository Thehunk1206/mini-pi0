"""Deterministic base-noise streams shared by FM evaluators."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def seeded_flow_generator(seed: int) -> torch.Generator:
    """Create a device-independent CPU generator for one evaluation episode."""

    return torch.Generator(device="cpu").manual_seed(int(seed))


def sample_flow_initial_noise(
    generators: Sequence[torch.Generator],
    active: Sequence[bool],
    *,
    chunk_size: int,
    action_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Draw one independent FM base-noise sample for each active environment."""

    if len(generators) != len(active):
        raise ValueError("Flow-noise generators and active mask must have the same length.")
    shape = (int(chunk_size), int(action_dim))
    rows = [
        torch.randn(shape, generator=generator, dtype=torch.float32)
        if is_active
        else torch.zeros(shape, dtype=torch.float32)
        for generator, is_active in zip(generators, active, strict=True)
    ]
    return torch.stack(rows).to(device=device, dtype=dtype)
