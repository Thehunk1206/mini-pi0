from __future__ import annotations

import pytest
import torch

from so101.policy_inference.config import RTCInferenceConfig
from so101.policy_inference.policy_bundle import PolicyBundle
from so101.policy_inference.rtc_policy import MiniPi0RTCPolicy


def test_first_chunk_matches_ordinary_sampler_exactly(
    real_policy_bundle: PolicyBundle,
) -> None:
    config = RTCInferenceConfig(flow_steps=2)
    rtc = MiniPi0RTCPolicy(real_policy_bundle, config)
    images = torch.zeros(1, 2, 3, 224, 224)
    state = torch.zeros(1, 6)
    noise = torch.randn(1, 32, 6)

    ordinary = real_policy_bundle.sample(
        images, state, flow_steps=2, initial_noise=noise
    )
    first_rtc = rtc.sample_normalized(
        images,
        state,
        previous_leftover=None,
        inference_delay=0,
        initial_noise=noise,
    )

    assert torch.equal(first_rtc, ordinary)


@torch.no_grad()
def _ordinary_second(
    bundle: PolicyBundle, images: torch.Tensor, state: torch.Tensor, noise: torch.Tensor
) -> torch.Tensor:
    return bundle.sample(images, state, flow_steps=1, initial_noise=noise)


def test_guidance_reduces_overlap_error(real_policy_bundle: PolicyBundle) -> None:
    config = RTCInferenceConfig(
        flow_steps=1, execution_horizon=8, max_guidance_weight=2.0
    )
    rtc = MiniPi0RTCPolicy(real_policy_bundle, config)
    images = torch.zeros(1, 2, 3, 224, 224)
    state = torch.zeros(1, 6)
    first_noise = torch.randn(1, 32, 6)
    second_noise = torch.randn(1, 32, 6)
    first = real_policy_bundle.sample(
        images, state, flow_steps=1, initial_noise=first_noise
    )
    leftover = first[0, 3:]

    ordinary = _ordinary_second(real_policy_bundle, images, state, second_noise)
    guided = rtc.sample_normalized(
        images,
        state,
        previous_leftover=leftover,
        inference_delay=1,
        initial_noise=second_noise,
    )
    prefix = leftover[:8]
    ordinary_error = torch.mean(torch.abs(ordinary[0, :8] - prefix))
    guided_error = torch.mean(torch.abs(guided[0, :8] - prefix))

    assert guided_error < ordinary_error


@pytest.fixture(scope="module")
def real_policy_bundle() -> PolicyBundle:
    return PolicyBundle.load(
        "runs/so101-pick-place-dual-cam-resnet18-chunk32/run3/checkpoints/final.pt",
        device="cpu",
        precision="fp32",
    )
