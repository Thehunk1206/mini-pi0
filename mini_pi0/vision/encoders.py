from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_TORCHVISION_SUPPORTED_MODELS = [
    "resnet18",
    "resnet34",
    "mobilenet_v3_small",
    "efficientnet_b0",
]

_TIMM_RECOMMENDED_MODELS = [
    "vit_tiny_patch16_224",
    "vit_small_patch16_224",
    "convnext_tiny",
    "efficientnet_b0",
    "vit_small_patch14_dinov2.lvd142m",
]


@dataclass
class VisionExtractorSpec:
    """Runtime descriptor for a constructed vision feature extractor.

    Attributes:
        backend: Encoder backend key.
        model_name: Underlying model identifier.
        image_size: Input image size expected by preprocessing.
        feature_dim: Produced embedding dimension.
    """

    backend: str
    model_name: str
    image_size: int
    feature_dim: int


class VisionFeatureExtractor(nn.Module):
    """Unified wrapper around torchvision/timm/HF encoders.

    The forward contract is always:
    - input: ``[B, 3, H, W]`` in float range ``[0, 1]``
    - output: ``[B, D]`` float embeddings
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        backend: str,
        model_name: str,
        image_size: int,
        feature_dim: int,
    ):
        super().__init__()
        self.module = module
        self.spec = VisionExtractorSpec(
            backend=backend,
            model_name=model_name,
            image_size=int(image_size),
            feature_dim=int(feature_dim),
        )

    @property
    def feature_dim(self) -> int:
        """Return output embedding dimension."""

        return int(self.spec.feature_dim)

    @property
    def image_size(self) -> int:
        """Return required square input size."""

        return int(self.spec.image_size)

    @property
    def backend(self) -> str:
        """Return backend identifier."""

        return str(self.spec.backend)

    @property
    def model_name(self) -> str:
        """Return model identifier."""

        return str(self.spec.model_name)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images into dense embeddings.

        Args:
            images: Tensor shaped ``[B, 3, H, W]`` in ``[0, 1]``.

        Returns:
            Embeddings shaped ``[B, D]``.
        """

        x = F.interpolate(images, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        feat = self.module(x)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        if feat.ndim > 2:
            feat = feat.flatten(1)
        return feat


class _TorchvisionResnet18Features(nn.Module):
    """Frozen ResNet18 feature trunk returning pooled 512-D embeddings."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        import torchvision.models as models

        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.resnet18(weights=weights)
        except Exception:
            net = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)

        self.backbone = nn.Sequential(*list(net.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).flatten(1)


class _TorchvisionResnet34Features(nn.Module):
    """Frozen ResNet34 feature trunk returning pooled 512-D embeddings."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        import torchvision.models as models

        try:
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.resnet34(weights=weights)
        except Exception:
            net = models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
        self.backbone = nn.Sequential(*list(net.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).flatten(1)


class _TorchvisionMobileNetV3SmallFeatures(nn.Module):
    """Frozen MobileNetV3-Small feature trunk returning pooled 576-D embeddings."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        import torchvision.models as models

        try:
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.mobilenet_v3_small(weights=weights)
        except Exception:
            net = models.mobilenet_v3_small(weights="IMAGENET1K_V1" if pretrained else None)
        self.features = net.features
        self.pool = net.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(x)).flatten(1)


class _TorchvisionEfficientNetB0Features(nn.Module):
    """Frozen EfficientNet-B0 feature trunk returning pooled 1280-D embeddings."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        import torchvision.models as models

        try:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.efficientnet_b0(weights=weights)
        except Exception:
            net = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        self.features = net.features
        self.pool = net.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(x)).flatten(1)


class _HFVisionFeatures(nn.Module):
    """Transformers-backed vision feature extractor wrapper."""

    def __init__(self, model_id: str, local_files_only: bool = False):
        super().__init__()
        try:
            from transformers import AutoModel
        except Exception as e:
            raise RuntimeError(
                "HF backend requires `transformers`. Install with `uv pip install transformers`."
            ) from e

        self.model = AutoModel.from_pretrained(model_id, local_files_only=local_files_only)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            return out.last_hidden_state[:, 0]
        if isinstance(out, tuple) and len(out) > 0:
            t = out[0]
            if t.ndim >= 3:
                return t[:, 0]
            return t
        raise RuntimeError("Could not derive feature tensor from HF model output.")


def _build_timm_model(model_name: str, pretrained: bool = True) -> tuple[nn.Module, int, int]:
    try:
        import timm
    except Exception as e:
        raise RuntimeError("timm backend requires `timm`. Install with `uv pip install timm`.") from e

    # Some timm + checkpoint combos (notably certain DINO variants) can fail
    # when forcing `global_pool="avg"` due state-dict key differences
    # (e.g. fc_norm vs norm). Try robust fallbacks.
    attempts = [
        {"num_classes": 0, "global_pool": "avg"},
        {"num_classes": 0},
    ]
    model = None
    errors: list[str] = []
    for kwargs in attempts:
        try:
            model = timm.create_model(model_name, pretrained=pretrained, **kwargs)
            break
        except Exception as e:
            errors.append(f"{kwargs}: {type(e).__name__}: {e}")
    if model is None:
        raise RuntimeError(
            f"Failed to build timm model '{model_name}' with pretrained={pretrained}. "
            f"Attempts: {' | '.join(errors)}"
        )

    feat_dim = int(getattr(model, "num_features", 0) or 0)
    input_size = 224
    cfg = getattr(model, "pretrained_cfg", None) or getattr(model, "default_cfg", None) or {}
    if isinstance(cfg, dict):
        ishape = cfg.get("input_size")
        if isinstance(ishape, (tuple, list)) and len(ishape) >= 3:
            input_size = int(ishape[-1])
    if feat_dim <= 0:
        # Best-effort probing
        with torch.no_grad():
            y = model(torch.zeros(1, 3, 224, 224))
        feat_dim = int(y.reshape(1, -1).shape[-1])
    return model, feat_dim, input_size


def list_torchvision_model_options() -> list[str]:
    """Return supported torchvision model names for feature extraction."""

    return list(_TORCHVISION_SUPPORTED_MODELS)


def list_timm_model_options(only_recommended: bool = True) -> list[str]:
    """List timm model choices.

    Args:
        only_recommended: If ``True``, return a curated low-memory shortlist.
            If ``False``, return all discoverable timm model names.

    Returns:
        List of timm model identifiers.
    """

    if only_recommended:
        return list(_TIMM_RECOMMENDED_MODELS)
    try:
        import timm
    except Exception as e:
        raise RuntimeError("timm is not installed. Install with `uv pip install timm`.") from e
    return sorted(timm.list_models(pretrained=False))


def build_vision_extractor(
    *,
    backend: str,
    model_name: str,
    pretrained: bool = True,
    image_size: int = 224,
    hf_model_id: str | None = None,
    local_files_only: bool = False,
    device: torch.device | str = "cpu",
) -> VisionFeatureExtractor:
    """Construct a vision feature extractor from configuration.

    Args:
        backend: One of ``torchvision``, ``timm``, or ``hf``.
        model_name: Backend model identifier.
        pretrained: Use pretrained weights when supported.
        image_size: Input square size used during preprocessing.
        hf_model_id: Optional explicit HF model id for ``hf`` backend.
        local_files_only: Restrict HF model loading to local cache.
        device: Torch device.

    Returns:
        Ready-to-use feature extractor module in eval mode on target device.
    """

    key = str(backend).strip().lower()
    if key == "torchvision":
        m = str(model_name).lower()
        if m == "resnet18":
            module = _TorchvisionResnet18Features(pretrained=bool(pretrained))
            feature_dim = 512
        elif m == "resnet34":
            module = _TorchvisionResnet34Features(pretrained=bool(pretrained))
            feature_dim = 512
        elif m == "mobilenet_v3_small":
            module = _TorchvisionMobileNetV3SmallFeatures(pretrained=bool(pretrained))
            feature_dim = 576
        elif m == "efficientnet_b0":
            module = _TorchvisionEfficientNetB0Features(pretrained=bool(pretrained))
            feature_dim = 1280
        else:
            supported = ", ".join(_TORCHVISION_SUPPORTED_MODELS)
            raise ValueError(
                f"Unsupported torchvision model '{model_name}'. Supported: {supported}"
            )
    elif key == "timm":
        module, feature_dim, required_image_size = _build_timm_model(
            model_name=str(model_name), pretrained=bool(pretrained)
        )
        image_size = int(required_image_size)
    elif key in {"hf", "huggingface", "transformers"}:
        mid = hf_model_id or model_name
        module = _HFVisionFeatures(model_id=str(mid), local_files_only=bool(local_files_only))
        with torch.no_grad():
            y = module(torch.zeros(1, 3, int(image_size), int(image_size)))
        feature_dim = int(y.reshape(1, -1).shape[-1])
        model_name = str(mid)
        key = "hf"
    else:
        raise ValueError(
            f"Unsupported vision backend '{backend}'. Supported: torchvision, timm, hf."
        )

    extractor = VisionFeatureExtractor(
        module=module,
        backend=key,
        model_name=str(model_name),
        image_size=int(image_size),
        feature_dim=int(feature_dim),
    )
    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad = False
    return extractor.to(device)


def images_to_tensor(images: list[np.ndarray], device: torch.device | str) -> torch.Tensor:
    """Convert list of uint8 HWC images to normalized BCHW tensor."""

    arr = np.asarray(images, dtype=np.uint8)
    if arr.ndim != 4:
        raise ValueError(f"Expected image batch with shape [B,H,W,C], got {arr.shape}")
    x = torch.from_numpy(arr).float().permute(0, 3, 1, 2) / 255.0
    return x.to(device)
