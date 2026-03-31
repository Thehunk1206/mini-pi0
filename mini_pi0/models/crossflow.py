from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ObservationContextEncoder(nn.Module):
    """Encode visual + proprio observations into context tokens.

    Design inspiration:
    - separate state and action pathways
    - optional state dropout/noise during training
    - context normalization before cross-attention denoising
    """

    def __init__(
        self,
        *,
        prop_dim: int,
        d_model: int,
        obs_mode: str,
        vision_dim: int,
        state_dropout_prob: float = 0.0,
        state_additive_noise_scale: float = 0.0,
        use_context_layernorm: bool = True,
    ):
        super().__init__()
        self.obs_mode = str(obs_mode).strip().lower()
        self.state_dropout_prob = float(max(0.0, min(1.0, state_dropout_prob)))
        self.state_additive_noise_scale = float(max(0.0, state_additive_noise_scale))
        self.context_norm = nn.LayerNorm(d_model) if use_context_layernorm else nn.Identity()

        if self.obs_mode in {"feature", "precomputed", "features"}:
            if int(vision_dim) <= 0:
                raise ValueError("vision_dim must be > 0 when obs_mode=feature.")
            self.img_backbone = None
            self.vision_proj = nn.Linear(int(vision_dim), d_model)
        else:
            try:
                weights = models.ResNet18_Weights.IMAGENET1K_V1
            except AttributeError:
                weights = "IMAGENET1K_V1"
            resnet = models.resnet18(weights=weights)
            self.img_backbone = nn.Sequential(*list(resnet.children())[:-1])
            for p in self.img_backbone.parameters():
                p.requires_grad = False
            self.vision_proj = nn.Linear(512, d_model)

        self.state_proj = nn.Sequential(
            nn.Linear(prop_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.state_mask_token = (
            nn.Parameter(0.02 * torch.randn(1, 1, d_model))
            if self.state_dropout_prob > 0.0
            else None
        )

    def forward(self, img: torch.Tensor, prop: torch.Tensor) -> torch.Tensor:
        """Return context tokens shaped ``[B, 2, d_model]`` (vision + state)."""

        if self.obs_mode in {"feature", "precomputed", "features"}:
            vision_feat = img.reshape(img.shape[0], -1)
        else:
            if self.img_backbone is None:
                raise RuntimeError("img_backbone is not initialized for image mode.")
            vision_feat = self.img_backbone(img).flatten(1)
        vision_token = self.vision_proj(vision_feat).unsqueeze(1)

        state_token = self.state_proj(prop).unsqueeze(1)
        if self.training and self.state_additive_noise_scale > 0.0:
            state_token = state_token + (
                torch.randn_like(state_token) * self.state_additive_noise_scale
            )
        if self.training and self.state_dropout_prob > 0.0 and self.state_mask_token is not None:
            drop = (
                torch.rand((state_token.shape[0], 1, 1), device=state_token.device)
                < self.state_dropout_prob
            )
            state_token = torch.where(drop, self.state_mask_token.expand_as(state_token), state_token)

        return self.context_norm(torch.cat([vision_token, state_token], dim=1))


class ActionFlowDecoder(nn.Module):
    """Cross-attention denoiser over noised action tokens."""

    def __init__(
        self,
        *,
        action_dim: int,
        chunk_size: int,
        d_model: int,
        nhead: int,
        nlayers: int,
        num_timestep_buckets: int = 1000,
        add_action_pos_embed: bool = True,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.num_timestep_buckets = int(max(2, num_timestep_buckets))

        self.action_in = nn.Sequential(
            nn.Linear(self.action_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.timestep_embedding = nn.Embedding(self.num_timestep_buckets, d_model)
        self.timestep_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.add_action_pos_embed = bool(add_action_pos_embed)
        if self.add_action_pos_embed:
            self.position_embedding = nn.Embedding(self.chunk_size, d_model)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=nlayers)
        self.out_proj = nn.Linear(d_model, self.action_dim)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep_buckets: torch.Tensor,
        context_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Predict action velocity from noised action tokens + context."""

        b, h, _ = noisy_actions.shape
        x = self.action_in(noisy_actions)

        t_embed = self.timestep_embedding(timestep_buckets)
        t_embed = self.timestep_mlp(t_embed).unsqueeze(1).expand(-1, h, -1)
        x = x + t_embed

        if self.add_action_pos_embed:
            pos_ids = torch.arange(h, device=noisy_actions.device, dtype=torch.long)
            pos = self.position_embedding(pos_ids).unsqueeze(0).expand(b, -1, -1)
            x = x + pos

        y = self.decoder(tgt=x, memory=context_tokens)
        return self.out_proj(y)


class CrossFlowActionModel(nn.Module):
    """Lightweight GR00T-inspired flow matching policy for robot actions.

    Key ideas (adapted for this repo):
    - dedicated state token encoder with optional dropout/noise
    - action token denoiser with context cross-attention
    - discretized timestep buckets for the denoiser
    - Beta-based low-timestep-biased flow training
    """

    def __init__(
        self,
        *,
        action_dim: int = 7,
        prop_dim: int = 9,
        obs_mode: str = "image",
        vision_dim: int = 0,
        chunk_size: int = 16,
        cond_dim: int = 256,  # kept for config compatibility
        d_model: int = 256,
        nhead: int = 4,
        nlayers: int = 4,
        num_timestep_buckets: int = 1000,
        noise_beta_alpha: float = 1.5,
        noise_beta_beta: float = 1.0,
        noise_s: float = 0.999,
        state_dropout_prob: float = 0.0,
        state_additive_noise_scale: float = 0.0,
        add_action_pos_embed: bool = True,
        use_context_layernorm: bool = True,
    ):
        super().__init__()
        _ = cond_dim  # not used; retained for uniform config surface.
        self.noise_s = float(max(0.0, min(1.0, noise_s)))
        self.num_timestep_buckets = int(max(2, num_timestep_buckets))
        self.beta_dist = torch.distributions.Beta(
            float(max(1e-4, noise_beta_alpha)),
            float(max(1e-4, noise_beta_beta)),
        )

        self.context_encoder = ObservationContextEncoder(
            prop_dim=prop_dim,
            d_model=d_model,
            obs_mode=obs_mode,
            vision_dim=vision_dim,
            state_dropout_prob=state_dropout_prob,
            state_additive_noise_scale=state_additive_noise_scale,
            use_context_layernorm=use_context_layernorm,
        )
        self.denoiser = ActionFlowDecoder(
            action_dim=action_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            nhead=nhead,
            nlayers=nlayers,
            num_timestep_buckets=self.num_timestep_buckets,
            add_action_pos_embed=add_action_pos_embed,
        )

    def _sample_tau(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        sample = self.beta_dist.sample((batch_size,)).to(device=device, dtype=dtype)
        return (1.0 - sample) * self.noise_s

    def _to_buckets(self, tau: torch.Tensor) -> torch.Tensor:
        b = (tau * self.num_timestep_buckets).long()
        return torch.clamp(b, 0, self.num_timestep_buckets - 1)

    def forward(self, img: torch.Tensor, prop: torch.Tensor, clean_actions: torch.Tensor) -> torch.Tensor:
        """Compute flow matching objective for one batch."""

        batch = clean_actions.shape[0]
        context = self.context_encoder(img, prop)
        tau = self._sample_tau(batch, clean_actions.device, clean_actions.dtype)
        eps = torch.randn_like(clean_actions)
        t = tau[:, None, None]

        noisy = (1.0 - t) * eps + t * clean_actions
        velocity_target = clean_actions - eps

        t_bucket = self._to_buckets(tau)
        velocity_pred = self.denoiser(noisy_actions=noisy, timestep_buckets=t_bucket, context_tokens=context)
        return F.mse_loss(velocity_pred, velocity_target)

    @torch.no_grad()
    def sample(self, img: torch.Tensor, prop: torch.Tensor, n_steps: int = 10) -> torch.Tensor:
        """Sample action chunk with Euler integration of the learned flow."""

        batch = img.shape[0]
        context = self.context_encoder(img, prop)
        horizon = self.denoiser.chunk_size
        action_dim = self.denoiser.action_dim

        actions = torch.randn(batch, horizon, action_dim, device=img.device, dtype=img.dtype)
        steps = max(1, int(n_steps))
        delta = 1.0 / float(steps)
        for i in range(steps):
            tau = torch.full((batch,), i * delta, device=img.device, dtype=img.dtype)
            t_bucket = self._to_buckets(tau)
            vel = self.denoiser(noisy_actions=actions, timestep_buckets=t_bucket, context_tokens=context)
            actions = actions + delta * vel
        return actions

