from __future__ import annotations

"""Flow-matching visuomotor policy modules.

The module keeps the original FM objective intact while supporting richer
observation conditioning for robot policies: spatial visual tokens,
observation history, cross-attention action denoisers, and token-to-FiLM
conditioning for Conv1D/UNet action experts.
"""

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


ConditioningMode = Literal["global", "cross_attention"]
FlowSolver = Literal["euler", "heun"]


def _valid_group_count(channels: int, max_groups: int = 8) -> int:
    """Return a GroupNorm group count that divides ``channels``."""

    groups = min(int(max_groups), int(channels))
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


def _normalize_conditioning_mode(mode: str) -> ConditioningMode:
    """Normalize and validate the FM conditioning mode."""

    key = str(mode or "global").strip().lower()
    if key not in {"global", "cross_attention"}:
        raise ValueError("model.conditioning_mode must be 'global' or 'cross_attention'.")
    return key  # type: ignore[return-value]


def sample_tau_beta(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    s: float = 0.999,
    alpha: float = 1.5,
    beta: float = 1.0,
) -> torch.Tensor:
    """Pi0-style timestep sampling biased toward lower/noisier timesteps."""

    u = torch.distributions.Beta(
        float(max(1e-4, alpha)),
        float(max(1e-4, beta)),
    ).sample((batch_size,)).to(device=device, dtype=dtype)
    return (1.0 - u) * s


class SinusoidalTimestep(nn.Module):
    """Sinusoidal embedding module for scalar flow timesteps."""

    def __init__(self, dim: int):
        """Initialize timestep embedding module."""

        super().__init__()
        self.dim = int(dim)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """Encode timesteps shaped ``[B]`` into sinusoidal features."""

        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=tau.device, dtype=tau.dtype) / half)
        args = tau[:, None] * freqs[None]
        out = torch.cat([args.sin(), args.cos()], dim=-1)
        if out.shape[-1] < self.dim:
            out = F.pad(out, (0, self.dim - out.shape[-1]))
        return out


class _TorchvisionResNet18TokenBackbone(nn.Module):
    """ResNet18 trunk that returns spatial feature maps instead of pooled vectors."""

    def __init__(self, freeze: bool):
        """Create a torchvision ResNet18 spatial trunk."""

        super().__init__()
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            transforms = weights.transforms()
            mean = tuple(float(v) for v in transforms.mean)
            std = tuple(float(v) for v in transforms.std)
        except AttributeError:
            weights = "IMAGENET1K_V1"
        net = models.resnet18(weights=weights)
        self.trunk = nn.Sequential(*list(net.children())[:-2])
        self.register_buffer("image_mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))
        for p in self.trunk.parameters():
            p.requires_grad = not bool(freeze)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return feature maps shaped ``[B, 512, h, w]``."""

        return self.trunk((x - self.image_mean) / self.image_std)


class _TimmTokenBackbone(nn.Module):
    """Best-effort timm wrapper that exposes patch/spatial tokens."""

    def __init__(self, model_name: str | None, freeze: bool, pretrained: bool = True):
        """Build a timm model and infer its token dimension.

        Raises:
            RuntimeError: If timm is unavailable or the model cannot return
                token-like features.
        """

        super().__init__()
        try:
            import timm
        except Exception as e:
            raise RuntimeError("model.vision_backbone=timm requires `timm` to be installed.") from e

        if model_name is None or not str(model_name).strip():
            raise ValueError("model.vision_model_name is required when model.vision_backbone='timm'.")
        try:
            self.model = timm.create_model(str(model_name), pretrained=bool(pretrained), num_classes=0)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create timm model '{model_name}' with pretrained={bool(pretrained)}. "
                "If pretrained weights are unavailable in this environment, set model.vision_pretrained=false."
            ) from e
        cfg = getattr(self.model, "pretrained_cfg", None) or getattr(self.model, "default_cfg", None) or {}
        input_size = cfg.get("input_size") if isinstance(cfg, dict) else None
        self.image_size = int(input_size[-1]) if isinstance(input_size, (tuple, list)) and len(input_size) >= 3 else 224
        mean = cfg.get("mean", (0.485, 0.456, 0.406)) if isinstance(cfg, dict) else (0.485, 0.456, 0.406)
        std = cfg.get("std", (0.229, 0.224, 0.225)) if isinstance(cfg, dict) else (0.229, 0.224, 0.225)
        self.register_buffer("image_mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))
        for p in self.model.parameters():
            p.requires_grad = not bool(freeze)
        with torch.no_grad():
            tokens = self._forward_tokens(torch.zeros(1, 3, self.image_size, self.image_size))
        if tokens.ndim != 3:
            raise RuntimeError(
                f"timm model '{model_name}' did not expose patch tokens; got shape {tuple(tokens.shape)}."
            )
        self.out_dim = int(tokens.shape[-1])

    def _forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw token-like features from a timm model."""

        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        x = (x - self.image_mean) / self.image_std
        if hasattr(self.model, "forward_features"):
            feat = self.model.forward_features(x)
        else:
            feat = self.model(x)
        if isinstance(feat, dict):
            for key in ("x_norm_patchtokens", "patch_tokens", "tokens", "last_hidden_state"):
                val = feat.get(key)
                if isinstance(val, torch.Tensor):
                    feat = val
                    break
            else:
                raise RuntimeError(f"Unsupported timm feature dict keys: {sorted(feat.keys())}")
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        if feat.ndim == 4:
            feat = feat.flatten(2).transpose(1, 2)
        if feat.ndim == 3 and feat.shape[1] > 1:
            # Drop CLS token for common ViT outputs when present.
            return feat[:, 1:, :] if feat.shape[1] % 2 == 1 else feat
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return patch tokens shaped ``[B, N, C]``."""

        return self._forward_tokens(x)


class ObservationEncoder(nn.Module):
    """Encode image/feature observations and proprioception into vectors/tokens."""

    def __init__(
        self,
        prop_dim: int = 9,
        cond_dim: int = 256,
        obs_mode: str = "image",
        vision_dim: int = 0,
        freeze_vision_backbone: bool = True,
        obs_horizon: int = 1,
        vision_backbone: str = "resnet18",
        vision_model_name: str | None = None,
        vision_pretrained: bool = True,
        vision_token_grid_size: int = 4,
        max_cameras: int = 8,
    ):
        """Initialize observation encoding branches.

        Args:
            prop_dim: Dimension of one proprioception timestep.
            cond_dim: Output conditioning dimension.
            obs_mode: ``image`` or ``feature``/``precomputed``.
            vision_dim: Feature dimension for precomputed feature mode.
            freeze_vision_backbone: Freeze image/token backbone parameters.
            obs_horizon: Number of observation history timesteps.
            vision_backbone: ``resnet18`` or ``timm``.
            vision_model_name: timm model name when using timm tokens. Ignored
                for ``vision_backbone=resnet18``.
            vision_pretrained: Load pretrained timm weights when
                ``vision_backbone=timm``.
            vision_token_grid_size: ResNet adaptive grid size per side.
            max_cameras: Maximum number of camera embeddings.
        """

        super().__init__()
        self.obs_mode = str(obs_mode).strip().lower()
        self.obs_horizon = int(max(1, obs_horizon))
        self.cond_dim = int(cond_dim)
        self.prop_dim = int(prop_dim)
        self.vision_token_grid_size = int(max(1, vision_token_grid_size))
        self.max_cameras = int(max(1, max_cameras))
        self.vision_backbone_name = str(vision_backbone or "resnet18").strip().lower()

        self.img_backbone: nn.Module | None
        self.img_proj: nn.Module
        self.img_token_proj: nn.Module
        self.spatial_tokens = self.vision_token_grid_size * self.vision_token_grid_size

        if self.obs_mode in {"feature", "precomputed", "features"}:
            if int(vision_dim) <= 0:
                raise ValueError("vision_dim must be > 0 when obs_mode=feature.")
            self.img_backbone = None
            self.img_proj = nn.Linear(int(vision_dim), cond_dim)
            self.img_token_proj = nn.Linear(int(vision_dim), cond_dim)
            self._timm_token_backbone = None
        elif self.vision_backbone_name == "timm":
            timm_backbone = _TimmTokenBackbone(
                vision_model_name,
                freeze=bool(freeze_vision_backbone),
                pretrained=bool(vision_pretrained),
            )
            self.img_backbone = timm_backbone
            self._timm_token_backbone = timm_backbone
            self.img_token_proj = nn.Linear(timm_backbone.out_dim, cond_dim)
            self.img_proj = nn.Linear(timm_backbone.out_dim, cond_dim)
            self.spatial_tokens = 1024
        else:
            if self.vision_backbone_name != "resnet18":
                raise ValueError("model.vision_backbone must be 'resnet18' or 'timm'.")
            resnet = _TorchvisionResNet18TokenBackbone(freeze=bool(freeze_vision_backbone))
            self.img_backbone = resnet
            self._timm_token_backbone = None
            self.img_token_proj = nn.Linear(512, cond_dim)
            self.img_proj = nn.Linear(512, cond_dim)

        self.prop_proj = nn.Linear(prop_dim, cond_dim)
        self.global_fusion = nn.Sequential(
            nn.Linear(cond_dim * 2, cond_dim),
            nn.LayerNorm(cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.camera_embed = nn.Embedding(self.max_cameras, cond_dim)
        self.history_embed = nn.Embedding(max(16, self.obs_horizon), cond_dim)
        self.spatial_embed = nn.Embedding(max(1, self.spatial_tokens), cond_dim)
        self.prop_type = nn.Parameter(torch.zeros(1, 1, cond_dim))

    def _flatten_image_input(self, img: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """Convert image input into ``[B*T*Cams, 3, H, W]``."""

        if img.ndim == 4:
            b = img.shape[0]
            return img, 1, 1
        if img.ndim == 5:
            b, axis1 = img.shape[:2]
            if img.shape[2] in {1, 3, 4}:
                # [B, T, 3, H, W] when history is enabled for one camera;
                # otherwise [B, N_cam, 3, H, W].
                if self.obs_horizon > 1 and int(axis1) == self.obs_horizon:
                    return img.reshape(b * axis1, *img.shape[2:]), int(axis1), 1
                return img.reshape(b * axis1, *img.shape[2:]), 1, int(axis1)
            # [B, T, 3, H, W]
            if img.shape[2] not in {1, 3, 4}:
                raise ValueError(f"Unsupported image history shape: {tuple(img.shape)}")
        if img.ndim == 6:
            b, t, n = img.shape[:3]
            return img.reshape(b * t * n, *img.shape[3:]), int(t), int(n)
        raise ValueError(f"Unsupported image tensor shape: {tuple(img.shape)}")

    def _feature_tokens(self, img: torch.Tensor) -> torch.Tensor:
        """Project feature-mode tensors into visual tokens."""

        if img.ndim == 2:
            return self.img_token_proj(img).unsqueeze(1)
        if img.ndim == 3:
            b, t, d = img.shape
            tok = self.img_token_proj(img.reshape(b * t, d)).reshape(b, t, self.cond_dim)
            hist = self.history_embed(torch.arange(t, device=img.device)).unsqueeze(0)
            return tok + hist
        if img.ndim == 4:
            b, t, n, d = img.shape
            tok = self.img_token_proj(img.reshape(b * t * n, d)).reshape(b, t, n, self.cond_dim)
            hist = self.history_embed(torch.arange(t, device=img.device)).view(1, t, 1, -1)
            cam = self.camera_embed(torch.arange(n, device=img.device)).view(1, 1, n, -1)
            return (tok + hist + cam).reshape(b, t * n, self.cond_dim)
        raise ValueError(f"Unsupported feature tensor shape: {tuple(img.shape)}")

    def _image_tokens(self, img: torch.Tensor) -> torch.Tensor:
        """Encode image tensors into spatial visual tokens."""

        flat, t, n_cam = self._flatten_image_input(img)
        if n_cam > self.max_cameras:
            raise ValueError(f"Number of cameras {n_cam} exceeds max supported cameras {self.max_cameras}.")
        if self.img_backbone is None:
            raise RuntimeError("img_backbone is not initialized for image mode.")

        if self._timm_token_backbone is not None:
            raw_tokens = self._timm_token_backbone(flat)
            tokens_per_view = int(raw_tokens.shape[1])
            if tokens_per_view > self.spatial_embed.num_embeddings:
                raise ValueError(
                    f"timm model produced {tokens_per_view} tokens, but spatial embedding capacity is "
                    f"{self.spatial_embed.num_embeddings}."
                )
            tok = self.img_token_proj(raw_tokens)
        else:
            fmap = self.img_backbone(flat)
            pooled = F.adaptive_avg_pool2d(fmap, (self.vision_token_grid_size, self.vision_token_grid_size))
            raw_tokens = pooled.flatten(2).transpose(1, 2)
            tokens_per_view = int(raw_tokens.shape[1])
            tok = self.img_token_proj(raw_tokens)

        b = img.shape[0]
        tok = tok.reshape(b, t, n_cam, tokens_per_view, self.cond_dim)
        hist = self.history_embed(torch.arange(t, device=img.device)).view(1, t, 1, 1, -1)
        cam = self.camera_embed(torch.arange(n_cam, device=img.device)).view(1, 1, n_cam, 1, -1)
        spatial = self.spatial_embed(torch.arange(tokens_per_view, device=img.device)).view(1, 1, 1, tokens_per_view, -1)
        return (tok + hist + cam + spatial).reshape(b, t * n_cam * tokens_per_view, self.cond_dim)

    def _prop_tokens(self, prop: torch.Tensor) -> torch.Tensor:
        """Project proprioception history into state tokens."""

        if prop.ndim == 2:
            return self.prop_proj(prop).unsqueeze(1) + self.prop_type
        if prop.ndim == 3:
            b, t, p = prop.shape
            if p != self.prop_dim:
                raise ValueError(f"Expected proprio dim {self.prop_dim}, got {p}.")
            tok = self.prop_proj(prop.reshape(b * t, p)).reshape(b, t, self.cond_dim)
            hist = self.history_embed(torch.arange(t, device=prop.device)).unsqueeze(0)
            return tok + hist + self.prop_type
        raise ValueError(f"Unsupported proprio tensor shape: {tuple(prop.shape)}")

    def forward_tokens(self, img: torch.Tensor, prop: torch.Tensor) -> torch.Tensor:
        """Encode observations into memory tokens shaped ``[B, N, cond_dim]``."""

        if self.obs_mode in {"feature", "precomputed", "features"}:
            visual = self._feature_tokens(img)
        else:
            visual = self._image_tokens(img)
        proprio = self._prop_tokens(prop)
        return torch.cat([visual, proprio], dim=1)

    def forward(self, img: torch.Tensor, prop: torch.Tensor) -> torch.Tensor:
        """Encode observations into one legacy/global conditioning vector."""

        tokens = self.forward_tokens(img, prop)
        visual = tokens[:, :-1].mean(dim=1) if tokens.shape[1] > 1 else tokens.mean(dim=1)
        proprio = self._prop_tokens(prop).mean(dim=1)
        return self.global_fusion(torch.cat([visual, proprio], dim=-1))


class ActionTransformer(nn.Module):
    """Legacy pooled-context transformer denoiser over action chunks."""

    def __init__(
        self,
        action_dim: int = 7,
        chunk_size: int = 16,
        cond_dim: int = 256,
        d_model: int = 256,
        nhead: int = 4,
        nlayers: int = 4,
        add_action_pos_embed: bool = True,
        causal: bool = False,
        dropout: float = 0.0,
    ):
        """Initialize the pooled-context action transformer."""

        super().__init__()
        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)
        self.add_action_pos_embed = bool(add_action_pos_embed)
        self.causal = bool(causal)

        self.tau_emb = SinusoidalTimestep(d_model)
        self.action_in = nn.Sequential(
            nn.Linear(action_dim + d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.cond_proj = nn.Linear(cond_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.chunk_size + 1, d_model))

        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=float(max(0.0, min(1.0, dropout))),
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=nlayers)
        self.out_proj = nn.Linear(d_model, action_dim)

    def _mask(self, length: int, device: torch.device) -> torch.Tensor | None:
        """Create optional causal mask that leaves context token visible."""

        if not self.causal:
            return None
        mask = torch.zeros(length, length, device=device)
        action_mask = torch.triu(torch.full((length - 1, length - 1), float("-inf"), device=device), diagonal=1)
        mask[1:, 1:] = action_mask
        return mask

    def forward(self, noisy_actions: torch.Tensor, tau: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Predict flow velocity for noisy actions at timestep ``tau``."""

        _, h, _ = noisy_actions.shape
        if h > self.chunk_size:
            raise ValueError(f"Action horizon {h} exceeds configured chunk_size {self.chunk_size}.")
        tau_feat = self.tau_emb(tau).unsqueeze(1).expand(-1, h, -1)
        tokens = self.action_in(torch.cat([noisy_actions, tau_feat], dim=-1))
        ctx = self.cond_proj(cond).unsqueeze(1)
        seq = torch.cat([ctx, tokens], dim=1)
        if self.add_action_pos_embed:
            seq = seq + self.pos_embed[:, : h + 1, :]
        out = self.transformer(seq, mask=self._mask(h + 1, noisy_actions.device))
        return self.out_proj(out[:, 1:, :])


class CrossAttentionDecoderLayer(nn.Module):
    """Pre-norm action self-attention, observation cross-attention, and MLP."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        """Initialize one cross-attention action decoder layer."""

        super().__init__()
        p = float(max(0.0, min(1.0, dropout)))
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=p, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=p, batch_first=True)
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(p),
        )
        self.resid_dropout = nn.Dropout(p)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Apply one decoder block."""

        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + self.resid_dropout(h)
        h = self.norm2(x)
        h, _ = self.cross_attn(h, memory, memory, need_weights=False)
        x = x + self.resid_dropout(h)
        return x + self.ff(self.norm3(x))


class CrossAttentionActionTransformer(nn.Module):
    """Action denoiser whose action tokens repeatedly attend observation tokens."""

    def __init__(
        self,
        action_dim: int = 7,
        chunk_size: int = 16,
        cond_dim: int = 256,
        d_model: int = 256,
        nhead: int = 4,
        nlayers: int = 4,
        add_action_pos_embed: bool = True,
        causal: bool = False,
        dropout: float = 0.0,
    ):
        """Initialize cross-attention action transformer."""

        super().__init__()
        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)
        self.add_action_pos_embed = bool(add_action_pos_embed)
        self.causal = bool(causal)
        self.tau_emb = SinusoidalTimestep(d_model)
        self.action_in = nn.Sequential(
            nn.Linear(action_dim + d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.memory_proj = nn.Sequential(nn.LayerNorm(cond_dim), nn.Linear(cond_dim, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.chunk_size, d_model))
        self.layers = nn.ModuleList([CrossAttentionDecoderLayer(d_model, nhead, dropout=dropout) for _ in range(int(nlayers))])
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, action_dim)

    def _mask(self, horizon: int, device: torch.device) -> torch.Tensor | None:
        """Create optional action-only causal mask."""

        if not self.causal:
            return None
        return torch.triu(torch.full((horizon, horizon), float("-inf"), device=device), diagonal=1)

    def forward(self, noisy_actions: torch.Tensor, tau: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Predict velocity using repeated cross-attention to observation memory."""

        _, h, _ = noisy_actions.shape
        if h > self.chunk_size:
            raise ValueError(f"Action horizon {h} exceeds configured chunk_size {self.chunk_size}.")
        if cond.ndim != 3:
            raise ValueError("CrossAttentionActionTransformer requires token memory shaped [B, N, C].")
        tau_feat = self.tau_emb(tau).unsqueeze(1).expand(-1, h, -1)
        x = self.action_in(torch.cat([noisy_actions, tau_feat], dim=-1))
        if self.add_action_pos_embed:
            x = x + self.pos_embed[:, :h, :]
        memory = self.memory_proj(cond)
        mask = self._mask(h, noisy_actions.device)
        for layer in self.layers:
            x = layer(x, memory, mask)
        return self.out_proj(self.out_norm(x))


class AttentionFiLMPooler(nn.Module):
    """Pool observation tokens into a dynamic FiLM vector for Conv/UNet experts."""

    def __init__(self, action_dim: int, cond_dim: int, d_model: int, nhead: int = 4, dropout: float = 0.0):
        """Initialize dynamic query attention pooling."""

        super().__init__()
        p = float(max(0.0, min(1.0, dropout)))
        self.tau_emb = SinusoidalTimestep(d_model)
        self.action_summary = nn.Linear(action_dim, d_model)
        self.query = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.memory_proj = nn.Sequential(nn.LayerNorm(cond_dim), nn.Linear(cond_dim, d_model))
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=p, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(p), nn.Linear(d_model, d_model))

    def forward(self, tokens: torch.Tensor, noisy_actions: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """Return one FiLM vector shaped ``[B, d_model]``."""

        if tokens.ndim != 3:
            raise ValueError("AttentionFiLMPooler expects tokens shaped [B, N, C].")
        action_feat = self.action_summary(noisy_actions.mean(dim=1))
        tau_feat = self.tau_emb(tau)
        query = self.query(torch.cat([action_feat, tau_feat], dim=-1)).unsqueeze(1)
        memory = self.memory_proj(tokens)
        pooled, _ = self.attn(query, memory, memory, need_weights=False)
        return self.out(pooled.squeeze(1))


class FiLMConv1DBlock(nn.Module):
    """Residual temporal Conv1D block with global FiLM conditioning."""

    def __init__(self, channels: int, cond_dim: int, kernel_size: int = 5):
        """Initialize a FiLM-conditioned residual Conv1D block."""

        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("action_cnn_kernel_size must be odd for length-preserving Conv1D.")
        padding = kernel_size // 2
        groups = _valid_group_count(channels)
        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=channels)
        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.film = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, channels * 2))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply residual temporal convolution with FiLM modulation."""

        shift, scale = self.film(cond).chunk(2, dim=-1)
        y = self.conv1(F.silu(self.norm1(x)))
        y = y * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        y = self.conv2(F.silu(self.norm2(y)))
        return x + y


class ActionCNN1D(nn.Module):
    """Temporal Conv1D action denoiser with pooled or token-to-FiLM context."""

    def __init__(
        self,
        action_dim: int = 7,
        chunk_size: int = 16,
        cond_dim: int = 256,
        d_model: int = 256,
        nlayers: int = 4,
        kernel_size: int = 5,
        nhead: int = 4,
        add_action_pos_embed: bool = True,
        token_conditioning: bool = False,
        dropout: float = 0.0,
    ):
        """Initialize FiLM-conditioned Conv1D action denoiser."""

        super().__init__()
        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)
        self.add_action_pos_embed = bool(add_action_pos_embed)
        self.token_conditioning = bool(token_conditioning)
        self.tau_emb = SinusoidalTimestep(d_model)
        self.obs_cond = nn.Sequential(nn.Linear(cond_dim, d_model), nn.LayerNorm(d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.tau_cond = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.token_pooler = AttentionFiLMPooler(action_dim, cond_dim, d_model, nhead=nhead, dropout=dropout) if token_conditioning else None
        self.action_in = nn.Conv1d(self.action_dim, d_model, kernel_size=1)
        self.pos_embed = nn.Parameter(torch.zeros(1, d_model, self.chunk_size))
        self.blocks = nn.ModuleList([FiLMConv1DBlock(d_model, d_model, int(kernel_size)) for _ in range(int(nlayers))])
        self.out_norm = nn.GroupNorm(num_groups=_valid_group_count(d_model), num_channels=d_model)
        self.out_proj = nn.Conv1d(d_model, self.action_dim, kernel_size=1)

    def _conditioning(self, noisy_actions: torch.Tensor, tau: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Resolve pooled or token-derived FiLM conditioning."""

        if cond.ndim == 3:
            if self.token_pooler is None:
                cond = cond.mean(dim=1)
            else:
                return self.token_pooler(cond, noisy_actions, tau)
        tau_feat = self.tau_emb(tau)
        return self.obs_cond(cond) + self.tau_cond(tau_feat)

    def forward(self, noisy_actions: torch.Tensor, tau: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Predict flow velocity with FiLM-conditioned temporal Conv1D blocks."""

        _, h, _ = noisy_actions.shape
        if h > self.chunk_size:
            raise ValueError(f"Action horizon {h} exceeds configured chunk_size {self.chunk_size}.")
        global_cond = self._conditioning(noisy_actions, tau, cond)
        x = self.action_in(noisy_actions.transpose(1, 2))
        if self.add_action_pos_embed:
            x = x + self.pos_embed[:, :, :h]
        for block in self.blocks:
            x = block(x, global_cond)
        x = self.out_proj(F.silu(self.out_norm(x)))
        return x.transpose(1, 2)


class FiLMUNet1DBlock(nn.Module):
    """Residual Conv1D block with separate observation and timestep FiLM."""

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, kernel_size: int = 5):
        """Initialize a UNet block."""

        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("action_cnn_kernel_size must be odd for length-preserving Conv1D.")
        padding = kernel_size // 2
        self.norm1 = nn.GroupNorm(num_groups=_valid_group_count(in_channels), num_channels=in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(num_groups=_valid_group_count(out_channels), num_channels=out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.film_obs = nn.Linear(cond_dim, out_channels * 2)
        self.film_tau = nn.Linear(cond_dim, out_channels * 2)
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, obs_cond: torch.Tensor, tau_cond: torch.Tensor) -> torch.Tensor:
        """Apply FiLM-conditioned residual temporal convolution."""

        y = self.conv1(F.silu(self.norm1(x)))
        obs_scale, obs_shift = self.film_obs(obs_cond).chunk(2, dim=-1)
        tau_scale, tau_shift = self.film_tau(tau_cond).chunk(2, dim=-1)
        y = y * (1.0 + (obs_scale + tau_scale).unsqueeze(-1)) + (obs_shift + tau_shift).unsqueeze(-1)
        y = self.conv2(F.silu(self.norm2(y)))
        return F.silu(y + self.residual(x))


class ActionUNet1D(nn.Module):
    """Diffusion Policy-style UNet1D action denoiser."""

    def __init__(
        self,
        action_dim: int = 7,
        chunk_size: int = 16,
        cond_dim: int = 256,
        d_model: int = 256,
        nlayers: int = 4,
        kernel_size: int = 5,
        nhead: int = 4,
        add_action_pos_embed: bool = True,
        token_conditioning: bool = False,
        dropout: float = 0.0,
    ):
        """Initialize UNet1D action denoiser."""

        super().__init__()
        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)
        self.add_action_pos_embed = bool(add_action_pos_embed)
        self.token_conditioning = bool(token_conditioning)
        levels = int(max(1, nlayers))
        self.tau_emb = SinusoidalTimestep(cond_dim)
        self.tau_cond = nn.Sequential(nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.obs_cond = nn.Sequential(nn.LayerNorm(cond_dim), nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.token_pooler = AttentionFiLMPooler(action_dim, cond_dim, cond_dim, nhead=nhead, dropout=dropout) if token_conditioning else None
        self.action_in = nn.Conv1d(self.action_dim, d_model, kernel_size=1)
        self.pos_embed = nn.Parameter(torch.zeros(1, d_model, self.chunk_size))

        channels = [int(d_model)] * (levels + 1)
        self.enc_blocks = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        for i in range(levels):
            self.enc_blocks.append(FiLMUNet1DBlock(channels[i], channels[i + 1], cond_dim, kernel_size))
            self.down_convs.append(nn.Conv1d(channels[i + 1], channels[i + 1], kernel_size=3, stride=2, padding=1))
        self.bottleneck = FiLMUNet1DBlock(channels[-1], channels[-1], cond_dim, kernel_size)
        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in reversed(range(levels)):
            self.up_convs.append(nn.ConvTranspose1d(channels[i + 1], channels[i], kernel_size=2, stride=2))
            self.dec_blocks.append(FiLMUNet1DBlock(channels[i] * 2, channels[i], cond_dim, kernel_size))
        self.out_norm = nn.GroupNorm(num_groups=_valid_group_count(channels[0]), num_channels=channels[0])
        self.out_proj = nn.Conv1d(channels[0], self.action_dim, kernel_size=1)

    def _conditioning(self, noisy_actions: torch.Tensor, tau: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve observation and timestep FiLM conditioning."""

        if cond.ndim == 3:
            if self.token_pooler is None:
                cond = cond.mean(dim=1)
            else:
                return self.token_pooler(cond, noisy_actions, tau), self.tau_cond(self.tau_emb(tau))
        return self.obs_cond(cond), self.tau_cond(self.tau_emb(tau))

    def forward(self, noisy_actions: torch.Tensor, tau: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Predict flow velocity with UNet1D temporal denoising."""

        _, h, _ = noisy_actions.shape
        if h > self.chunk_size:
            raise ValueError(f"Action horizon {h} exceeds configured chunk_size {self.chunk_size}.")
        obs_cond, tau_cond = self._conditioning(noisy_actions, tau, cond)
        x = self.action_in(noisy_actions.transpose(1, 2))
        if self.add_action_pos_embed:
            x = x + self.pos_embed[:, :, :h]

        skips: list[torch.Tensor] = []
        for enc, down in zip(self.enc_blocks, self.down_convs, strict=True):
            x = enc(x, obs_cond, tau_cond)
            skips.append(x)
            x = down(x)
        x = self.bottleneck(x, obs_cond, tau_cond)
        for up, dec, skip in zip(self.up_convs, self.dec_blocks, reversed(skips), strict=True):
            x = up(x)
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1])
            x = dec(torch.cat([x, skip], dim=1), obs_cond, tau_cond)
        x = self.out_proj(F.silu(self.out_norm(x)))
        return x.transpose(1, 2)


class MiniPi0FlowMatching(nn.Module):
    """Flow-matching action chunk model used across train/eval/deploy."""

    def __init__(
        self,
        action_dim: int = 7,
        prop_dim: int = 9,
        obs_mode: str = "image",
        vision_dim: int = 0,
        chunk_size: int = 16,
        cond_dim: int = 256,
        d_model: int = 256,
        nhead: int = 4,
        nlayers: int = 4,
        action_backbone: str = "transformer",
        conditioning_mode: str = "cross_attention",
        action_attention_causal: bool = False,
        obs_horizon: int = 1,
        vision_backbone: str = "resnet18",
        vision_model_name: str | None = None,
        vision_pretrained: bool = True,
        action_cnn_kernel_size: int = 5,
        freeze_vision_backbone: bool = True,
        noise_beta_alpha: float = 1.5,
        noise_beta_beta: float = 1.0,
        noise_s: float = 0.999,
        add_action_pos_embed: bool = True,
        vision_token_grid_size: int = 4,
        dropout: float = 0.0,
    ):
        """Initialize observation encoder and selected action denoiser."""

        super().__init__()
        self.noise_beta_alpha = float(max(1e-4, noise_beta_alpha))
        self.noise_beta_beta = float(max(1e-4, noise_beta_beta))
        self.noise_s = float(max(0.0, min(1.0, noise_s)))
        self.dropout = float(max(0.0, min(1.0, dropout)))
        self.conditioning_mode = _normalize_conditioning_mode(conditioning_mode)
        self.obs_encoder = ObservationEncoder(
            prop_dim=prop_dim,
            cond_dim=cond_dim,
            obs_mode=obs_mode,
            vision_dim=vision_dim,
            freeze_vision_backbone=freeze_vision_backbone,
            obs_horizon=obs_horizon,
            vision_backbone=vision_backbone,
            vision_model_name=vision_model_name,
            vision_pretrained=vision_pretrained,
            vision_token_grid_size=vision_token_grid_size,
        )

        backbone = str(action_backbone).strip().lower()
        use_tokens = self.conditioning_mode == "cross_attention"
        if backbone == "transformer":
            if use_tokens:
                self.action_transformer = CrossAttentionActionTransformer(
                    action_dim=action_dim,
                    chunk_size=chunk_size,
                    cond_dim=cond_dim,
                    d_model=d_model,
                    nhead=nhead,
                    nlayers=nlayers,
                    add_action_pos_embed=add_action_pos_embed,
                    causal=action_attention_causal,
                    dropout=self.dropout,
                )
            else:
                self.action_transformer = ActionTransformer(
                    action_dim=action_dim,
                    chunk_size=chunk_size,
                    cond_dim=cond_dim,
                    d_model=d_model,
                    nhead=nhead,
                    nlayers=nlayers,
                    add_action_pos_embed=add_action_pos_embed,
                    causal=action_attention_causal,
                    dropout=self.dropout,
                )
        elif backbone in {"cnn1d", "conv1d"}:
            self.action_transformer = ActionCNN1D(
                action_dim=action_dim,
                chunk_size=chunk_size,
                cond_dim=cond_dim,
                d_model=d_model,
                nlayers=nlayers,
                kernel_size=action_cnn_kernel_size,
                nhead=nhead,
                add_action_pos_embed=add_action_pos_embed,
                token_conditioning=use_tokens,
                dropout=self.dropout,
            )
        elif backbone == "unet1d":
            self.action_transformer = ActionUNet1D(
                action_dim=action_dim,
                chunk_size=chunk_size,
                cond_dim=cond_dim,
                d_model=d_model,
                nlayers=nlayers,
                kernel_size=action_cnn_kernel_size,
                nhead=nhead,
                add_action_pos_embed=add_action_pos_embed,
                token_conditioning=use_tokens,
                dropout=self.dropout,
            )
        else:
            raise ValueError("model.action_backbone must be 'transformer', 'cnn1d', or 'unet1d'.")

    def _encode_conditioning(self, img: torch.Tensor, prop: torch.Tensor) -> torch.Tensor:
        """Encode observations for the configured conditioning path."""

        if self.conditioning_mode == "cross_attention":
            return self.obs_encoder.forward_tokens(img, prop)
        return self.obs_encoder(img, prop)

    def _flow_loss_components(
        self,
        img: torch.Tensor,
        prop: torch.Tensor,
        clean_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return FM velocity loss and predicted clean-action estimate."""

        batch = img.shape[0]
        cond = self._encode_conditioning(img, prop)
        tau = sample_tau_beta(
            batch,
            clean_actions.device,
            dtype=clean_actions.dtype,
            s=self.noise_s,
            alpha=self.noise_beta_alpha,
            beta=self.noise_beta_beta,
        )
        eps = torch.randn_like(clean_actions)
        t = tau.view(batch, 1, 1)
        noisy = t * clean_actions + (1 - t) * eps
        v_pred = self.action_transformer(noisy, tau, cond)
        target = clean_actions - eps
        clean_pred = noisy + (1.0 - t) * v_pred
        return F.mse_loss(v_pred, target), clean_pred

    def compute_loss(
        self,
        img: torch.Tensor,
        prop: torch.Tensor,
        clean_actions: torch.Tensor,
        *,
        smoothness_weight: float = 0.0,
        jerk_weight: float = 0.0,
    ) -> torch.Tensor:
        """Compute FM loss plus optional smoothness/jerk regularization."""

        loss, clean_pred = self._flow_loss_components(img, prop, clean_actions)
        if smoothness_weight > 0.0 and clean_pred.shape[1] > 1:
            smooth = (clean_pred[:, 1:] - clean_pred[:, :-1]).square().mean()
            loss = loss + float(smoothness_weight) * smooth
        if jerk_weight > 0.0 and clean_pred.shape[1] > 2:
            jerk = (clean_pred[:, 2:] - 2.0 * clean_pred[:, 1:-1] + clean_pred[:, :-2]).square().mean()
            loss = loss + float(jerk_weight) * jerk
        return loss

    def forward(self, img: torch.Tensor, prop: torch.Tensor, clean_actions: torch.Tensor) -> torch.Tensor:
        """Compute the base flow-matching training loss for one minibatch."""

        return self.compute_loss(img, prop, clean_actions)

    def _velocity(self, actions: torch.Tensor, tau_value: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Evaluate the learned velocity field."""

        tau = tau_value.expand(actions.shape[0])
        return self.action_transformer(actions, tau, cond)

    @torch.no_grad()
    def sample(
        self,
        img: torch.Tensor,
        prop: torch.Tensor,
        n_steps: int = 10,
        solver: str = "euler",
    ) -> torch.Tensor:
        """Generate one normalized action chunk by integrating the flow field."""

        batch = img.shape[0]
        cond = self._encode_conditioning(img, prop)
        h = self.action_transformer.chunk_size
        a_dim = self.action_transformer.action_dim
        actions = torch.randn(batch, h, a_dim, device=img.device, dtype=img.dtype)
        steps = max(1, int(n_steps))
        solver_key = str(solver or "euler").strip().lower()
        if solver_key not in {"euler", "heun"}:
            raise ValueError("flow_solver must be 'euler' or 'heun'.")
        time_grid = torch.linspace(0.0, 1.0, steps + 1, device=img.device, dtype=actions.dtype)
        for i in range(steps):
            t0 = time_grid[i]
            t1 = time_grid[i + 1]
            dt = t1 - t0
            v0 = self._velocity(actions, t0, cond)
            if solver_key == "heun":
                proposal = actions + dt * v0
                v1 = self._velocity(proposal, t1, cond)
                actions = actions + 0.5 * dt * (v0 + v1)
            else:
                actions = actions + dt * v0
        return actions
