from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def sample_tau_beta(batch_size: int, device: torch.device, s: float = 0.999, alpha: float = 1.5) -> torch.Tensor:
    """Pi0-style timestep sampling biased toward lower (noisier) timesteps.

    With the current interpolation convention:
    ``noisy = tau * clean + (1 - tau) * eps``,
    smaller ``tau`` corresponds to noisier training points.

    We therefore mirror the Beta(alpha, 1) sample to emphasize small tau.
    """

    u = torch.distributions.Beta(alpha, 1.0).sample((batch_size,)).to(device)
    return (1.0 - u) * s


class SinusoidalTimestep(nn.Module):
    """Sinusoidal embedding module for scalar flow timesteps."""

    def __init__(self, dim: int):
        """Initialize timestep embedding module.

        Args:
            dim: Output embedding dimension.
        """

        super().__init__()
        self.dim = dim

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """Encode timesteps into sinusoidal features.

        Args:
            tau: Timestep tensor shaped ``[B]``.

        Returns:
            Sin/cos embedding tensor shaped ``[B, dim]``.
        """

        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=tau.device, dtype=tau.dtype) / half)
        args = tau[:, None] * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ObservationEncoder(nn.Module):
    """Encode image + proprio observations into a single conditioning vector."""

    def __init__(self, prop_dim: int = 9, cond_dim: int = 256, obs_mode: str = "image", vision_dim: int = 0):
        """Initialize frozen image backbone and fusion MLP.

        Args:
            prop_dim: Concatenated proprio feature dimension.
            cond_dim: Output conditioning feature dimension.
            obs_mode: Observation input mode (``image`` or ``feature``).
            vision_dim: Input feature size when ``obs_mode=feature``.
        """

        super().__init__()
        self.obs_mode = str(obs_mode).strip().lower()

        if self.obs_mode in {"feature", "precomputed", "features"}:
            if int(vision_dim) <= 0:
                raise ValueError("vision_dim must be > 0 when obs_mode=feature.")
            self.img_backbone = None
            self.img_proj = nn.Linear(int(vision_dim), cond_dim)
        else:
            try:
                weights = models.ResNet18_Weights.IMAGENET1K_V1
            except AttributeError:
                weights = "IMAGENET1K_V1"

            resnet = models.resnet18(weights=weights)
            self.img_backbone = nn.Sequential(*list(resnet.children())[:-1])
            for p in self.img_backbone.parameters():
                p.requires_grad = False
            self.img_proj = nn.Linear(512, cond_dim)

        self.prop_proj = nn.Linear(prop_dim, cond_dim)
        self.fusion = nn.Sequential(
            nn.Linear(cond_dim * 2, cond_dim),
            nn.LayerNorm(cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, img: torch.Tensor, prop: torch.Tensor) -> torch.Tensor:
        """Encode observation batch.

        Args:
            img: Image tensor shaped ``[B, 3, H, W]``.
            prop: Proprio tensor shaped ``[B, prop_dim]``.

        Returns:
            Conditioning embedding shaped ``[B, cond_dim]``.
        """

        if self.obs_mode in {"feature", "precomputed", "features"}:
            img_feat = img.reshape(img.shape[0], -1)
        else:
            if self.img_backbone is None:
                raise RuntimeError("img_backbone is not initialized for image mode.")
            img_feat = self.img_backbone(img).flatten(1)
        return self.fusion(torch.cat([self.img_proj(img_feat), self.prop_proj(prop)], dim=-1))


class ActionTransformer(nn.Module):
    """Transformer denoiser/predictor over fixed-horizon action chunks."""

    def __init__(
        self,
        action_dim: int = 7,
        chunk_size: int = 16,
        cond_dim: int = 256,
        d_model: int = 256,
        nhead: int = 4,
        nlayers: int = 4,
    ):
        """Initialize action transformer.

        Args:
            action_dim: Action vector dimension.
            chunk_size: Number of action tokens in one chunk.
            cond_dim: Conditioning vector dimension from observation encoder.
            d_model: Transformer hidden dimension.
            nhead: Number of attention heads.
            nlayers: Number of transformer encoder layers.
        """

        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        self.tau_emb = SinusoidalTimestep(d_model)
        self.action_in = nn.Sequential(
            nn.Linear(action_dim + d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.cond_proj = nn.Linear(cond_dim, d_model)

        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=nlayers)
        self.out_proj = nn.Linear(d_model, action_dim)

    def forward(self, noisy_actions: torch.Tensor, tau: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Predict flow velocity for noisy actions at timestep ``tau``.

        Args:
            noisy_actions: Noisy action tensor shaped ``[B, H, action_dim]``.
            tau: Scalar timesteps shaped ``[B]``.
            cond: Observation conditioning vectors shaped ``[B, cond_dim]``.

        Returns:
            Predicted velocity tensor shaped ``[B, H, action_dim]``.
        """

        _, h, _ = noisy_actions.shape
        tau_feat = self.tau_emb(tau).unsqueeze(1).expand(-1, h, -1)
        tokens = self.action_in(torch.cat([noisy_actions, tau_feat], dim=-1))
        ctx = self.cond_proj(cond).unsqueeze(1)
        seq = torch.cat([ctx, tokens], dim=1)
        out = self.transformer(seq)
        return self.out_proj(out[:, 1:, :])


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
    ):
        """Initialize observation encoder + action transformer stack.

        Args:
            action_dim: Action vector dimension.
            prop_dim: Proprioceptive vector dimension.
            obs_mode: Observation input mode (``image`` or ``feature``).
            vision_dim: Input feature dimension when ``obs_mode=feature``.
            chunk_size: Predicted horizon length.
            cond_dim: Observation conditioning dimension.
            d_model: Transformer hidden dimension.
            nhead: Number of transformer attention heads.
            nlayers: Number of transformer encoder layers.
        """

        super().__init__()
        self.obs_encoder = ObservationEncoder(
            prop_dim=prop_dim,
            cond_dim=cond_dim,
            obs_mode=obs_mode,
            vision_dim=vision_dim,
        )
        self.action_transformer = ActionTransformer(
            action_dim=action_dim,
            chunk_size=chunk_size,
            cond_dim=cond_dim,
            d_model=d_model,
            nhead=nhead,
            nlayers=nlayers,
        )

    def forward(self, img: torch.Tensor, prop: torch.Tensor, clean_actions: torch.Tensor) -> torch.Tensor:
        """Compute flow-matching training loss for one minibatch.

        Args:
            img: Image batch shaped ``[B, 3, H, W]``.
            prop: Proprio batch shaped ``[B, prop_dim]``.
            clean_actions: Target action chunks shaped ``[B, H, action_dim]``.

        Returns:
            Scalar MSE loss tensor.
        """

        batch = img.shape[0]
        cond = self.obs_encoder(img, prop)

        tau = sample_tau_beta(batch, img.device)
        eps = torch.randn_like(clean_actions)
        t = tau.view(batch, 1, 1)
        noisy = t * clean_actions + (1 - t) * eps

        v_pred = self.action_transformer(noisy, tau, cond)
        target = clean_actions - eps
        return F.mse_loss(v_pred, target)

    @torch.no_grad()
    def sample(self, img: torch.Tensor, prop: torch.Tensor, n_steps: int = 10) -> torch.Tensor:
        """Generate one action chunk by integrating learned flow field.

        Args:
            img: Image batch shaped ``[B, 3, H, W]``.
            prop: Proprio batch shaped ``[B, prop_dim]``.
            n_steps: Number of Euler integration steps.

        Returns:
            Generated normalized actions shaped ``[B, H, action_dim]``.
        """

        batch = img.shape[0]
        cond = self.obs_encoder(img, prop)
        h = self.action_transformer.chunk_size
        a_dim = self.action_transformer.action_dim

        actions = torch.randn(batch, h, a_dim, device=img.device)
        delta = 1.0 / max(1, int(n_steps))
        for i in range(max(1, int(n_steps))):
            tau = torch.full((batch,), i * delta, device=img.device)
            actions = actions + delta * self.action_transformer(actions, tau, cond)
        return actions
