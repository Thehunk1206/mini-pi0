"""Tests for reproducible flow-matching evaluation noise."""

from __future__ import annotations

import pytest
import torch

from mini_pi0.eval.flow_noise import sample_flow_initial_noise, seeded_flow_generator


def test_flow_noise_repeats_for_the_same_episode_seeds() -> None:
    first = sample_flow_initial_noise(
        [seeded_flow_generator(10), seeded_flow_generator(11)],
        [True, True],
        chunk_size=3,
        action_dim=2,
        device=torch.device("cpu"),
    )
    repeated = sample_flow_initial_noise(
        [seeded_flow_generator(10), seeded_flow_generator(11)],
        [True, True],
        chunk_size=3,
        action_dim=2,
        device=torch.device("cpu"),
    )

    assert torch.equal(first, repeated)
    assert not torch.equal(first[0], first[1])


def test_flow_noise_zeros_inactive_environment_rows() -> None:
    noise = sample_flow_initial_noise(
        [seeded_flow_generator(5), seeded_flow_generator(6)],
        [True, False],
        chunk_size=2,
        action_dim=3,
        device=torch.device("cpu"),
    )

    assert torch.count_nonzero(noise[0]) > 0
    assert torch.count_nonzero(noise[1]) == 0


def test_flow_noise_rejects_mismatched_generator_count() -> None:
    with pytest.raises(ValueError, match="same length"):
        sample_flow_initial_noise(
            [seeded_flow_generator(5)],
            [True, False],
            chunk_size=2,
            action_dim=3,
            device=torch.device("cpu"),
        )
