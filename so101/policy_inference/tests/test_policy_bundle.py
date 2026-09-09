from __future__ import annotations

import numpy as np
import pytest
import torch

from so101.policy_inference.policy_bundle import PolicyBundle


def test_preprocess_preserves_trained_camera_order(
    real_policy_bundle: PolicyBundle,
) -> None:
    cameras = {
        "wrist": np.zeros((480, 480, 3), dtype=np.uint8),
        "base": np.zeros((360, 640, 3), dtype=np.uint8),
    }

    images, state = real_policy_bundle.preprocess(
        cameras, np.zeros(6, dtype=np.float32)
    )

    assert tuple(images.shape) == (1, 2, 3, 224, 224)
    assert tuple(state.shape) == (1, 6)
    assert images.device.type == "cpu"


def test_preprocess_rejects_reversed_camera_mapping(
    real_policy_bundle: PolicyBundle,
) -> None:
    cameras = {
        "base": np.zeros((360, 640, 3), dtype=np.uint8),
        "wrist": np.zeros((480, 480, 3), dtype=np.uint8),
    }

    with pytest.raises(ValueError, match="preserve trained order"):
        real_policy_bundle.preprocess(cameras, np.zeros(6, dtype=np.float32))


def test_embedded_normalization_round_trip(real_policy_bundle: PolicyBundle) -> None:
    actions = torch.randn(2, 4, 6)

    restored = real_policy_bundle.denormalize_actions(
        real_policy_bundle.normalize_actions(actions)
    )

    assert torch.allclose(restored, actions, atol=1e-5, rtol=1e-5)


@pytest.fixture(scope="module")
def real_policy_bundle() -> PolicyBundle:
    return PolicyBundle.load(
        "runs/so101-pick-place-dual-cam-resnet18-chunk32/run3/checkpoints/final.pt",
        device="cpu",
        precision="fp32",
    )
