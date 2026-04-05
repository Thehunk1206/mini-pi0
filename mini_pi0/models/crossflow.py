from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply DiT-style affine modulation to normalized tokens."""

    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class ObservationContextEncoder(nn.Module):
    """Encode visual + proprio observations into context tokens.

    For image observations, this encoder returns multiple spatial vision tokens
    (adaptive grid pooled), which improves local grasp/reach conditioning versus
    a single globally pooled token.
    """

    def __init__(
        self,
        *,
        prop_dim: int,
        d_model: int,
        obs_mode: str,
        vision_dim: int,
        vision_token_grid_size: int = 4,
        state_dropout_prob: float = 0.0,
        state_additive_noise_scale: float = 0.0,
        use_context_layernorm: bool = True,
    ):
        super().__init__()
        self.obs_mode = str(obs_mode).strip().lower()
        self.state_dropout_prob = float(max(0.0, min(1.0, state_dropout_prob)))
        self.state_additive_noise_scale = float(max(0.0, state_additive_noise_scale))
        self.context_norm = nn.LayerNorm(d_model) if use_context_layernorm else nn.Identity()

        self.vision_token_grid_size = int(max(1, vision_token_grid_size))
        self.image_mode = self.obs_mode not in {"feature", "precomputed", "features"}
        if not self.image_mode:
            if int(vision_dim) <= 0:
                raise ValueError("vision_dim must be > 0 when obs_mode=feature.")
            self.img_backbone = None
            self.vision_proj = nn.Linear(int(vision_dim), d_model)
            self.vision_pos_embed = None
            self.n_vision_tokens = 1
        else:
            try:
                weights = models.ResNet18_Weights.IMAGENET1K_V1
            except AttributeError:
                weights = "IMAGENET1K_V1"
            try:
                resnet = models.resnet18(weights=weights)
            except Exception:
                # Keep model construction robust in offline / no-cache environments.
                resnet = models.resnet18(weights=None)
            # Keep feature map before global avg pool to preserve spatial tokens.
            self.img_backbone = nn.Sequential(*list(resnet.children())[:-2])
            for p in self.img_backbone.parameters():
                p.requires_grad = False
            self.vision_proj = nn.Linear(512, d_model)
            self.n_vision_tokens = self.vision_token_grid_size * self.vision_token_grid_size
            self.vision_pos_embed = nn.Parameter(0.02 * torch.randn(1, self.n_vision_tokens, d_model))

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

    def forward(self, img: torch.Tensor, prop: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(vision_tokens, state_token)``.

        Shapes:
        - ``vision_tokens``: ``[B, Nv, d_model]``
        - ``state_token``: ``[B, 1, d_model]``
        """

        if not self.image_mode:
            vision_feat = img.reshape(img.shape[0], -1)
            vision_tokens = self.vision_proj(vision_feat).unsqueeze(1)
        else:
            if self.img_backbone is None:
                raise RuntimeError("img_backbone is not initialized for image mode.")
            feat = self.img_backbone(img)
            # MPS currently requires adaptive pool output sizes to evenly divide
            # input sizes. Interpolate for robust behavior on Apple Silicon.
            gh, gw = feat.shape[-2], feat.shape[-1]
            if gh == self.vision_token_grid_size and gw == self.vision_token_grid_size:
                pooled = feat
            elif feat.device.type == "mps":
                pooled = F.interpolate(
                    feat,
                    size=(self.vision_token_grid_size, self.vision_token_grid_size),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                pooled = F.adaptive_avg_pool2d(
                    feat,
                    output_size=(self.vision_token_grid_size, self.vision_token_grid_size),
                )
            # [B, C, Gh, Gw] -> [B, Gh*Gw, C]
            vision_tokens = pooled.flatten(2).transpose(1, 2)
            vision_tokens = self.vision_proj(vision_tokens)
            if self.vision_pos_embed is not None:
                vision_tokens = vision_tokens + self.vision_pos_embed[:, : vision_tokens.shape[1], :]

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

        full = self.context_norm(torch.cat([vision_tokens, state_token], dim=1))
        v_count = vision_tokens.shape[1]
        return full[:, :v_count, :], full[:, v_count:, :]


class _DiTSelfAttentionBlock(nn.Module):
    """Self-attention + MLP block with optional timestep-conditioned AdaLN."""

    def __init__(self, d_model: int, nhead: int, use_dit_adaln: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.0, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.use_dit_adaln = bool(use_dit_adaln)
        self.cond_proj = (
            nn.Sequential(nn.SiLU(), nn.Linear(d_model, d_model * 4))
            if self.use_dit_adaln
            else None
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if self.use_dit_adaln and self.cond_proj is not None:
            s1, sc1, s2, sc2 = self.cond_proj(cond).chunk(4, dim=-1)
            x1 = _modulate(self.norm1(x), s1, sc1)
        else:
            x1 = self.norm1(x)
        attn_out, _ = self.self_attn(x1, x1, x1, need_weights=False)
        x = x + attn_out

        if self.use_dit_adaln and self.cond_proj is not None:
            s1, sc1, s2, sc2 = self.cond_proj(cond).chunk(4, dim=-1)
            x2 = _modulate(self.norm2(x), s2, sc2)
        else:
            x2 = self.norm2(x)
        x = x + self.ff(x2)
        return x


class _DiTCrossAttentionBlock(nn.Module):
    """Cross-attention + MLP block with optional timestep-conditioned AdaLN."""

    def __init__(self, d_model: int, nhead: int, use_dit_adaln: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.0, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.use_dit_adaln = bool(use_dit_adaln)
        self.cond_proj = (
            nn.Sequential(nn.SiLU(), nn.Linear(d_model, d_model * 4))
            if self.use_dit_adaln
            else None
        )

    def forward(self, x: torch.Tensor, context_tokens: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if self.use_dit_adaln and self.cond_proj is not None:
            s1, sc1, s2, sc2 = self.cond_proj(cond).chunk(4, dim=-1)
            q = _modulate(self.norm1(x), s1, sc1)
        else:
            q = self.norm1(x)

        if context_tokens is not None and context_tokens.shape[1] > 0:
            cross_out, _ = self.cross_attn(q, context_tokens, context_tokens, need_weights=False)
            x = x + cross_out

        if self.use_dit_adaln and self.cond_proj is not None:
            s1, sc1, s2, sc2 = self.cond_proj(cond).chunk(4, dim=-1)
            x2 = _modulate(self.norm2(x), s2, sc2)
        else:
            x2 = self.norm2(x)
        x = x + self.ff(x2)
        return x


class ActionFlowDecoder(nn.Module):
    """DiT-like action denoiser with alternating self/cross attention blocks."""

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
        use_dit_adaln: bool = True,
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

        self.self_blocks = nn.ModuleList(
            [_DiTSelfAttentionBlock(d_model=d_model, nhead=nhead, use_dit_adaln=use_dit_adaln) for _ in range(nlayers)]
        )
        self.cross_blocks = nn.ModuleList(
            [_DiTCrossAttentionBlock(d_model=d_model, nhead=nhead, use_dit_adaln=use_dit_adaln) for _ in range(nlayers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, self.action_dim)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep_buckets: torch.Tensor,
        context_tokens: torch.Tensor,
        state_token: torch.Tensor,
    ) -> torch.Tensor:
        """Predict flow velocity from noised actions conditioned on vision + state."""

        b, h, _ = noisy_actions.shape
        action_tokens = self.action_in(noisy_actions)

        t_cond = self.timestep_embedding(timestep_buckets)
        t_cond = self.timestep_mlp(t_cond)
        action_tokens = action_tokens + t_cond.unsqueeze(1)

        if self.add_action_pos_embed:
            pos_ids = torch.arange(h, device=noisy_actions.device, dtype=torch.long)
            pos = self.position_embedding(pos_ids).unsqueeze(0).expand(b, -1, -1)
            action_tokens = action_tokens + pos

        # GR00T-style conditioning pattern:
        # self-attention over (state + action tokens), then cross-attention to vision-language tokens.
        tokens = torch.cat([state_token, action_tokens], dim=1)
        for self_block, cross_block in zip(self.self_blocks, self.cross_blocks, strict=True):
            tokens = self_block(tokens, t_cond)
            tokens = cross_block(tokens, context_tokens, t_cond)

        tokens = self.final_norm(tokens)
        action_tokens = tokens[:, 1:, :]
        return self.out_proj(action_tokens)


class CrossFlowActionModel(nn.Module):
    """Lightweight GR00T-inspired flow matching policy for robot actions.

    Key implemented ideas:
    - multi-token visual context for spatial manipulation cues
    - alternating self/cross attention DiT blocks
    - dedicated state token with optional dropout/noise regularization
    - low-timestep-biased Beta flow timestep sampling
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
        vision_token_grid_size: int = 4,
        use_dit_adaln: bool = True,
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
            vision_token_grid_size=vision_token_grid_size,
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
            use_dit_adaln=use_dit_adaln,
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
        vision_tokens, state_token = self.context_encoder(img, prop)
        tau = self._sample_tau(batch, clean_actions.device, clean_actions.dtype)
        eps = torch.randn_like(clean_actions)
        t = tau[:, None, None]

        noisy = (1.0 - t) * eps + t * clean_actions
        velocity_target = clean_actions - eps

        t_bucket = self._to_buckets(tau)
        velocity_pred = self.denoiser(
            noisy_actions=noisy,
            timestep_buckets=t_bucket,
            context_tokens=vision_tokens,
            state_token=state_token,
        )
        return F.mse_loss(velocity_pred, velocity_target)

    @torch.no_grad()
    def sample(self, img: torch.Tensor, prop: torch.Tensor, n_steps: int = 10) -> torch.Tensor:
        """Sample action chunk with Euler integration of the learned flow."""

        batch = img.shape[0]
        vision_tokens, state_token = self.context_encoder(img, prop)
        horizon = self.denoiser.chunk_size
        action_dim = self.denoiser.action_dim

        actions = torch.randn(batch, horizon, action_dim, device=img.device, dtype=img.dtype)
        steps = max(1, int(n_steps))
        delta = 1.0 / float(steps)
        for i in range(steps):
            tau = torch.full((batch,), i * delta, device=img.device, dtype=img.dtype)
            t_bucket = self._to_buckets(tau)
            vel = self.denoiser(
                noisy_actions=actions,
                timestep_buckets=t_bucket,
                context_tokens=vision_tokens,
                state_token=state_token,
            )
            actions = actions + delta * vel
        return actions
