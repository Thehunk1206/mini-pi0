from __future__ import annotations

"""Reference Diffusion Policy-style flow matching architecture.

This module is intentionally not registered as a production model yet. It
keeps a fuller architecture sketch available for experiments while the main
training/eval stack continues to use ``mini_pi0.models.fm``.
"""

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def _valid_groups(channels: int, max_groups: int = 8) -> int:
    """Return the largest group count that divides ``channels``."""

    groups = min(max_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


def sample_tau(
    batch: int,
    device: torch.device,
    mode: Literal["uniform", "beta"] = "uniform",
    beta_alpha: float = 1.5,
    s: float = 0.999,
) -> torch.Tensor:
    """Sample flow timesteps using uniform or noise-biased beta sampling."""

    if mode == "beta":
        u = torch.distributions.Beta(beta_alpha, 1.0).sample((batch,)).to(device)
        return (1.0 - u) * s
    return torch.rand(batch, device=device) * s


class SinusoidalTimestep(nn.Module):
    """Sinusoidal timestep embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """Encode scalar timesteps shaped ``[B]`` into ``[B, dim]``."""

        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000)
            * torch.arange(half, device=tau.device, dtype=tau.dtype)
            / half
        )
        args = tau[:, None] * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ObservationEncoder(nn.Module):
    """Encode stacked image/proprio history into one conditioning vector."""

    def __init__(
        self,
        prop_dim: int = 9,
        obs_horizon: int = 2,
        cond_dim: int = 256,
        obs_mode: str = "image",
        vision_dim: int = 0,
        unfreeze_last_n_layers: int = 2,
    ):
        super().__init__()
        self.obs_mode = obs_mode.strip().lower()
        self.obs_horizon = int(obs_horizon)

        if self.obs_mode in {"feature", "precomputed", "features"}:
            if vision_dim <= 0:
                raise ValueError("vision_dim must be > 0 when obs_mode='feature'.")
            self.img_backbone = None
            self.img_proj = nn.Linear(vision_dim, cond_dim)
        else:
            in_channels = 3 * self.obs_horizon
            try:
                weights = models.ResNet18_Weights.IMAGENET1K_V1
            except AttributeError:
                weights = "IMAGENET1K_V1"
            resnet = models.resnet18(weights=weights)

            if in_channels != 3:
                old_conv = resnet.conv1
                resnet.conv1 = nn.Conv2d(
                    in_channels,
                    old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=False,
                )
                with torch.no_grad():
                    repeated = old_conv.weight.repeat(1, self.obs_horizon, 1, 1)
                    resnet.conv1.weight.copy_(repeated[:, :in_channels] / self.obs_horizon)

            self.img_backbone = nn.Sequential(*list(resnet.children())[:-1])
            layer_names = ["layer1", "layer2", "layer3", "layer4"]
            unfrozen = set(layer_names[-unfreeze_last_n_layers:]) if unfreeze_last_n_layers > 0 else set()
            for name, param in self.img_backbone.named_parameters():
                param.requires_grad = any(layer_name in name for layer_name in unfrozen)
            self.img_proj = nn.Linear(512, cond_dim)

        self.prop_proj = nn.Linear(prop_dim * self.obs_horizon, cond_dim)
        self.fusion = nn.Sequential(
            nn.Linear(cond_dim * 2, cond_dim * 2),
            nn.LayerNorm(cond_dim * 2),
            nn.SiLU(),
            nn.Linear(cond_dim * 2, cond_dim),
            nn.LayerNorm(cond_dim),
        )

    def forward(self, img: torch.Tensor, prop: torch.Tensor) -> torch.Tensor:
        """Encode image/proprio tensors into ``[B, cond_dim]``."""

        if self.obs_mode in {"feature", "precomputed", "features"}:
            img_feat = img.reshape(img.shape[0], -1)
        else:
            if self.img_backbone is None:
                raise RuntimeError("img_backbone is not initialized for image mode.")
            img_feat = self.img_backbone(img).flatten(1)

        img_enc = self.img_proj(img_feat)
        prop_enc = self.prop_proj(prop)
        return self.fusion(torch.cat([img_enc, prop_enc], dim=-1))


class FiLMConv1DBlock(nn.Module):
    """Residual Conv1D block with separate observation and timestep FiLM."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, kernel_size: int = 5):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        pad = kernel_size // 2

        self.norm1 = nn.GroupNorm(_valid_groups(in_ch), in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=pad)
        self.norm2 = nn.GroupNorm(_valid_groups(out_ch), out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=pad)
        self.film_obs = nn.Linear(cond_dim, out_ch * 2)
        self.film_tau = nn.Linear(cond_dim, out_ch * 2)
        self.residual = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, obs_cond: torch.Tensor, tau_cond: torch.Tensor) -> torch.Tensor:
        """Apply one FiLM-conditioned residual block."""

        y = self.conv1(F.silu(self.norm1(x)))
        scale_obs, shift_obs = self.film_obs(obs_cond).chunk(2, -1)
        scale_tau, shift_tau = self.film_tau(tau_cond).chunk(2, -1)
        scale = (scale_obs + scale_tau).unsqueeze(-1)
        shift = (shift_obs + shift_tau).unsqueeze(-1)
        y = y * (1.0 + scale) + shift
        y = self.conv2(F.silu(self.norm2(y)))
        return F.silu(y + self.residual(x))


class ActionUNet1D(nn.Module):
    """UNet1D denoiser with skip connections and FiLM conditioning."""

    def __init__(
        self,
        action_dim: int = 7,
        chunk_size: int = 16,
        cond_dim: int = 256,
        base_ch: int = 256,
        nlayers: int = 4,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)
        self.tau_emb = SinusoidalTimestep(cond_dim)
        self.tau_proj = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.action_in = nn.Conv1d(action_dim, base_ch, kernel_size=1)

        channels = [base_ch] * (nlayers + 1)
        self.enc_blocks = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        for i in range(nlayers):
            self.enc_blocks.append(FiLMConv1DBlock(channels[i], channels[i + 1], cond_dim, kernel_size))
            self.down_convs.append(
                nn.Conv1d(channels[i + 1], channels[i + 1], kernel_size=3, stride=2, padding=1)
            )

        self.bottleneck = FiLMConv1DBlock(channels[-1], channels[-1], cond_dim, kernel_size)

        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in reversed(range(nlayers)):
            self.up_convs.append(nn.ConvTranspose1d(channels[i + 1], channels[i], kernel_size=2, stride=2))
            self.dec_blocks.append(FiLMConv1DBlock(channels[i] * 2, channels[i], cond_dim, kernel_size))

        self.out_norm = nn.GroupNorm(_valid_groups(channels[0]), channels[0])
        self.out_proj = nn.Conv1d(channels[0], action_dim, kernel_size=1)

    def forward(self, noisy_actions: torch.Tensor, tau: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Predict velocity field shaped ``[B, H, action_dim]``."""

        tau_cond = self.tau_proj(self.tau_emb(tau))
        obs_cond = cond
        x = self.action_in(noisy_actions.transpose(1, 2))

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


class CrossAttentionDecoderLayer(nn.Module):
    """Transformer decoder layer with causal self-attention and cross-attention."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mem: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        """Run one decoder layer."""

        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h, attn_mask=causal_mask)
        x = x + h
        h = self.norm2(x)
        h, _ = self.cross_attn(h, mem, mem)
        x = x + h
        return x + self.ff(self.norm3(x))


class ActionTransformer(nn.Module):
    """Cross-attention Transformer action denoiser."""

    def __init__(
        self,
        action_dim: int = 7,
        chunk_size: int = 16,
        cond_dim: int = 256,
        d_model: int = 256,
        nhead: int = 4,
        nlayers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)
        self.tau_emb = SinusoidalTimestep(d_model)
        self.tau_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.obs_to_mem = nn.Linear(cond_dim, d_model)
        self.action_in = nn.Linear(action_dim, d_model)
        self.pos_embed = nn.Embedding(chunk_size, d_model)
        self.layers = nn.ModuleList(
            [CrossAttentionDecoderLayer(d_model, nhead, dropout) for _ in range(nlayers)]
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, action_dim)

    def forward(self, noisy_actions: torch.Tensor, tau: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Predict velocity field shaped ``[B, H, action_dim]``."""

        _, horizon, _ = noisy_actions.shape
        obs_tok = self.obs_to_mem(cond).unsqueeze(1)
        tau_tok = self.tau_proj(self.tau_emb(tau)).unsqueeze(1)
        memory = torch.cat([obs_tok, tau_tok], dim=1)
        pos = torch.arange(horizon, device=noisy_actions.device)
        x = self.action_in(noisy_actions) + self.pos_embed(pos)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(horizon, device=noisy_actions.device)

        for layer in self.layers:
            x = layer(x, memory, causal_mask)
        return self.out_proj(self.out_norm(x))


class FlowMatchingPolicy(nn.Module):
    """Reference flow-matching visuomotor policy with observation history."""

    def __init__(
        self,
        action_dim: int = 7,
        prop_dim: int = 9,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        exec_horizon: int = 8,
        cond_dim: int = 256,
        d_model: int = 256,
        nhead: int = 4,
        nlayers: int = 4,
        action_backbone: str = "transformer",
        obs_mode: str = "image",
        vision_dim: int = 0,
        unfreeze_last_n: int = 2,
        tau_mode: str = "uniform",
        n_steps: int = 10,
        kernel_size: int = 5,
    ):
        super().__init__()
        if exec_horizon > pred_horizon:
            raise ValueError("exec_horizon must be <= pred_horizon")
        self.pred_horizon = int(pred_horizon)
        self.exec_horizon = int(exec_horizon)
        self.action_dim = int(action_dim)
        self.n_steps = int(n_steps)
        self.tau_mode = str(tau_mode)

        self.obs_encoder = ObservationEncoder(
            prop_dim=prop_dim,
            obs_horizon=obs_horizon,
            cond_dim=cond_dim,
            obs_mode=obs_mode,
            vision_dim=vision_dim,
            unfreeze_last_n_layers=unfreeze_last_n,
        )

        backbone = action_backbone.strip().lower()
        if backbone == "transformer":
            self.denoiser = ActionTransformer(action_dim, pred_horizon, cond_dim, d_model, nhead, nlayers)
        elif backbone in {"unet1d", "cnn1d", "conv1d"}:
            self.denoiser = ActionUNet1D(action_dim, pred_horizon, cond_dim, d_model, nlayers, kernel_size)
        else:
            raise ValueError(f"Unknown action_backbone: {action_backbone!r}")

    def forward(self, img: torch.Tensor, prop: torch.Tensor, clean_actions: torch.Tensor) -> torch.Tensor:
        """Compute flow-matching MSE loss."""

        batch = img.shape[0]
        cond = self.obs_encoder(img, prop)
        tau = sample_tau(batch, img.device, mode=self.tau_mode)
        eps = torch.randn_like(clean_actions)
        t = tau.view(batch, 1, 1)
        noisy = t * clean_actions + (1.0 - t) * eps
        v_pred = self.denoiser(noisy, tau, cond)
        return F.mse_loss(v_pred, clean_actions - eps)

    @torch.no_grad()
    def sample(self, img: torch.Tensor, prop: torch.Tensor, n_steps: int | None = None) -> torch.Tensor:
        """Generate normalized execution-horizon actions."""

        steps = int(n_steps or self.n_steps)
        batch = img.shape[0]
        cond = self.obs_encoder(img, prop)
        x = torch.randn(batch, self.pred_horizon, self.action_dim, device=img.device, dtype=img.dtype)
        dt = 1.0 / float(max(1, steps))
        for i in range(max(1, steps)):
            tau = torch.full((batch,), (i + 1) * dt, device=img.device, dtype=img.dtype)
            x = x + dt * self.denoiser(x, tau, cond)
        return x[:, : self.exec_horizon, :]


class Normalizer(nn.Module):
    """Per-dimension normalizer saved as module buffers."""

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    def fit(self, data: torch.Tensor) -> None:
        """Fit mean/std from ``[N, D]`` or ``[N, H, D]`` data."""

        flat = data.reshape(-1, data.shape[-1])
        self.mean.copy_(flat.mean(0))
        self.std.copy_(flat.std(0).clamp(min=1e-6))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize by fitted mean/std."""

        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Map normalized values back to data scale."""

        return x * self.std + self.mean
