"""
π₀.5 – SmolVLM edition
=======================
A faithful port of the π₀.5 dual-stream flow-matching VLA architecture
using HuggingFace SmolVLM (SigLIP vision + LLaMA-based text) as the VLM
backbone and a smaller LLaMA-config Gemma-style model as the action expert.

Architecture summary
--------------------
- VLM backbone : SmolVLMForConditionalGeneration  (vision_model + connector + LlamaModel)
- Action expert: a second, lighter LlamaModel initialised from scratch
- Joint attention: at every transformer layer, Q/K/V from both streams are
  concatenated along the sequence dimension, a single eager attention pass
  is computed, then the output is split and routed through each stream's own
  o_proj + MLP (identical to the original PaliGemmaWithExpertModel cat-split trick)
- Flow matching: linear-Gaussian path  x_t = t*ε + (1-t)*a,  loss = MSE(ε-a, v_θ)
- AdaRMS       : timestep τ is encoded via sinusoidal pos-emb → 2-layer SiLU MLP
                 and used to modulate the action expert's RMSNorm layers (scale+shift)
- Blockwise causal attention mask (prefix bidirectional, suffix bidirectional,
  suffix can see prefix but prefix cannot see suffix)

Training
--------
  model = PI05SmolVLM(MiniPI05Config())
  loss  = model(observation, actions)          # MSE scalar
  loss.backward()

Inference
---------
  actions = model.sample_actions(observation)  # [B, H, action_dim]

Observation format
------------------
  observation.images       : list[Tensor]   each [B, C, H, W]  (1-3 cameras)
  observation.image_masks  : list[Tensor]   each [B]  bool, True = valid image
  observation.lang_tokens  : Tensor         [B, L]   token ids
  observation.lang_masks   : Tensor         [B, L]   bool
  observation.state        : Tensor         [B, action_dim]
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformers import SmolVLMConfig, SmolVLMForConditionalGeneration
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ExpertConfig:
    """Defines a smaller LLaMA-style action expert."""
    hidden_size: int = 512
    intermediate_size: int = 1536
    num_hidden_layers: int = 18      # must match VLM num_hidden_layers
    # Must match VLM attention head structure for joint cat/split attention.
    num_attention_heads: int = 9
    num_key_value_heads: int = 3
    head_dim: int = 64
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0


@dataclass
class MiniPI05Config:
    # ── VLM backbone (SmolVLM-256M sized) ──────────────────────────────────
    vlm_text_hidden_size: int = 576          # SmolLM2-360M hidden dim
    vlm_text_layers: int = 18               # must equal expert layers
    vlm_text_heads: int = 9
    vlm_text_kv_heads: int = 3
    vlm_text_intermediate: int = 1536
    vlm_text_head_dim: int = 64
    vlm_vision_hidden: int = 384            # SigLIP-small hidden
    vlm_vision_layers: int = 12
    vlm_vision_heads: int = 6
    vlm_vision_image_size: int = 224
    vlm_vision_patch_size: int = 14

    # ── Action expert ───────────────────────────────────────────────────────
    expert: ExpertConfig = field(default_factory=ExpertConfig)

    # ── Task ────────────────────────────────────────────────────────────────
    action_dim: int = 18           # largest robot DoF (zero-pad smaller robots)
    action_horizon: int = 50       # H – chunk length
    num_cameras: int = 2           # 1-3
    state_dim: int | None = None   # if None, defaults to action_dim

    # ── Training ────────────────────────────────────────────────────────────
    dtype: str = "bfloat16"        # "bfloat16" | "float32"

    # ── Flow matching ───────────────────────────────────────────────────────
    fm_min_period: float = 4e-3
    fm_max_period: float = 4.0
    fm_num_steps: int = 10         # Euler integration steps at inference


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class Observation(NamedTuple):
    images: list[Tensor]       # each [B, C, H, W]
    image_masks: list[Tensor]  # each [B] bool – True = real image
    lang_tokens: Tensor        # [B, L]
    lang_masks: Tensor         # [B, L] bool
    state: Tensor              # [B, action_dim]


def sinusoidal_pos_embedding(
    time: Tensor,           # [B]
    dim: int,
    min_period: float,
    max_period: float,
) -> Tensor:
    """Sine-cosine positional encoding for scalar timestep τ ∈ [0,1]."""
    if dim % 2 != 0:
        raise ValueError(f"dim ({dim}) must be even")
    dtype = torch.float32
    frac = torch.linspace(0.0, 1.0, dim // 2, dtype=dtype, device=time.device)
    period = min_period * (max_period / min_period) ** frac          # [dim/2]
    angles = (2 * math.pi / period)[None, :] * time[:, None].float() # [B, dim/2]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)   # [B, dim]


def sample_beta_time(batch_size: int, device: torch.device) -> Tensor:
    """Sample τ from Beta(1.5, 1.0) shifted to [0.001, 0.999] – emphasises low τ."""
    dist = torch.distributions.Beta(
        torch.tensor(1.5, device=device),
        torch.tensor(1.0, device=device),
    )
    t = dist.sample((batch_size,))
    return (t * 0.998 + 0.001).to(dtype=torch.float32, device=device)


def make_blockwise_att_mask(pad_masks: Tensor, att_masks: Tensor) -> Tensor:
    """
    Build a 2-D boolean attention mask from:
      pad_masks : [B, N]  True = valid token
      att_masks : [B, N]  int  1 = start of new causal block, 0 = same block

    Token i can attend to token j iff:
      cumsum(att_masks)[i] >= cumsum(att_masks)[j]   AND both are valid.

    This gives:
      - prefix (all 0s)  → full bidirectional
      - suffix first token (1) → new block; suffix tokens (0s) → bidirectional
        among themselves and can see prefix, but prefix cannot see suffix
    """
    cumsum = torch.cumsum(att_masks, dim=1)                         # [B, N]
    causal = cumsum[:, :, None] >= cumsum[:, None, :]               # [B, N, N]
    pad2d  = pad_masks[:, :, None] & pad_masks[:, None, :]          # [B, N, N]
    return causal & pad2d                                            # [B, N, N]


def att_mask_to_bias(att_2d: Tensor) -> Tensor:
    """Convert bool [B, N, N] → float [B, 1, N, N] additive bias."""
    return torch.where(att_2d[:, None], 0.0, -2.3819763e38)


# ---------------------------------------------------------------------------
# AdaRMS – Adaptive RMSNorm (timestep-conditioned)
# ---------------------------------------------------------------------------

class AdaRMSNorm(nn.Module):
    """
    Replaces LlamaRMSNorm in the action expert.
    Applies RMSNorm then affine-transforms with (scale, shift) predicted from
    an external conditioning vector (the flow-matching timestep embedding).

    forward(x, cond) -> (normed_x * (1 + scale) + shift, gate)
    The gate is returned so the caller can use it in the residual connection
    (identical pattern to the original openpi AdaRMS implementation).
    """

    def __init__(self, hidden_size: int, cond_dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = LlamaRMSNorm(hidden_size, eps=eps)
        # predict scale and shift from cond – zero-init so it starts as identity
        self.proj = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
        """
        x    : [B, L, H]
        cond : [B, H_cond]   (timestep embedding, same for all tokens in chunk)
        """
        normed = self.norm(x)
        # [B, 2H] -> scale [B,1,H], shift [B,1,H]
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        scale = scale[:, None, :]
        shift = shift[:, None, :]
        out = normed * (1.0 + scale) + shift
        # gate for gated residual (mirroring _gated_residual in original code)
        gate = torch.ones(1, dtype=x.dtype, device=x.device)
        return out, gate


# ---------------------------------------------------------------------------
# Dual-stream transformer engine
# ---------------------------------------------------------------------------

class SmolVLMWithExpertModel(nn.Module):
    """
    Houses two LLaMA transformer stacks:
      - VLM backbone  (prefix stream) – weights from SmolVLM text model
      - Action expert (suffix stream) – fresh weights, AdaRMS on layernorms

    Joint attention: at every layer, Q/K/V from both streams are concatenated
    along the sequence dim, a single attention pass is computed, then the
    output is split and each stream runs its own o_proj + MLP.

    forward modes (controlled by which inputs_embeds entry is None):
      [prefix, None]  → prefix-only  (builds KV cache at inference)
      [None, suffix]  → suffix-only  (denoising steps against cached KV)
      [prefix, suffix]→ joint        (training and first inference step)
    """

    def __init__(self, cfg: MiniPI05Config):
        super().__init__()
        self.cfg = cfg

        # ── Build SmolVLM ──────────────────────────────────────────────────
        smolvlm_cfg = SmolVLMConfig(
            text_config={
                "model_type": "llama",
                "hidden_size":        cfg.vlm_text_hidden_size,
                "intermediate_size":  cfg.vlm_text_intermediate,
                "num_hidden_layers":  cfg.vlm_text_layers,
                "num_attention_heads": cfg.vlm_text_heads,
                "num_key_value_heads": cfg.vlm_text_kv_heads,
                "head_dim":           cfg.vlm_text_head_dim,
                "rms_norm_eps":       1e-6,
                "rope_theta":         10000.0,
                "max_position_embeddings": 4096,
                "attention_dropout":  0.0,
                "attention_bias":     False,
            },
            vision_config={
                "hidden_size":        cfg.vlm_vision_hidden,
                "intermediate_size":  cfg.vlm_vision_hidden * 4,
                "num_hidden_layers":  cfg.vlm_vision_layers,
                "num_attention_heads": cfg.vlm_vision_heads,
                "image_size":         cfg.vlm_vision_image_size,
                "patch_size":         cfg.vlm_vision_patch_size,
            },
        )
        self.smolvlm = SmolVLMForConditionalGeneration(smolvlm_cfg)
        # Shortcuts into the text backbone
        self.vlm_layers: nn.ModuleList = self.smolvlm.model.text_model.layers
        self.vlm_norm: LlamaRMSNorm    = self.smolvlm.model.text_model.norm
        self.rotary_emb: LlamaRotaryEmbedding = self.smolvlm.model.text_model.rotary_emb

        # ── Build action expert ────────────────────────────────────────────
        exp = cfg.expert
        assert exp.num_hidden_layers == cfg.vlm_text_layers, (
            "Action expert must have the same number of layers as the VLM text backbone "
            f"(got expert={exp.num_hidden_layers}, vlm={cfg.vlm_text_layers})"
        )
        expert_llama_cfg = LlamaConfig(
            hidden_size=exp.hidden_size,
            intermediate_size=exp.intermediate_size,
            num_hidden_layers=exp.num_hidden_layers,
            num_attention_heads=exp.num_attention_heads,
            num_key_value_heads=exp.num_key_value_heads,
            head_dim=exp.head_dim,
            rms_norm_eps=exp.rms_norm_eps,
            rope_theta=exp.rope_theta,
            max_position_embeddings=4096,
            attention_dropout=0.0,
            attention_bias=False,
        )
        # Build just the layer stack + final norm; no embed_tokens, no lm_head
        self.expert_layers = nn.ModuleList(
            [LlamaDecoderLayer(expert_llama_cfg, layer_idx=i)
             for i in range(exp.num_hidden_layers)]
        )
        self.expert_norm = LlamaRMSNorm(exp.hidden_size, eps=exp.rms_norm_eps)

        # Replace expert layernorms with AdaRMS
        cond_dim = exp.hidden_size  # timestep MLP output dim
        for layer in self.expert_layers:
            layer.input_layernorm = AdaRMSNorm(
                exp.hidden_size, cond_dim, eps=exp.rms_norm_eps
            )
            layer.post_attention_layernorm = AdaRMSNorm(
                exp.hidden_size, cond_dim, eps=exp.rms_norm_eps
            )

        # ── dtype ─────────────────────────────────────────────────────────
        if cfg.dtype == "bfloat16":
            self._cast_to_bfloat16()

    # ── dtype helpers ───────────────────────────────────────────────────────

    def _cast_to_bfloat16(self):
        """Cast to bfloat16 but keep numerically sensitive params in float32."""
        self.to(dtype=torch.bfloat16)
        fp32_selectors = [
            "embeddings.patch_embedding",
            "embeddings.position_embedding",
            "input_layernorm",
            "post_attention_layernorm",
            ".norm.",
            "model.norm",
        ]
        for name, param in self.named_parameters():
            if any(s in name for s in fp32_selectors):
                param.data = param.data.to(torch.float32)

    # ── Image / language embedding (prefix) ─────────────────────────────────

    def embed_images(self, images: list[Tensor], image_masks: list[Tensor]):
        """
        Run each camera image through SmolVLM's vision tower + connector.
        Returns list of [B, num_patches, vlm_hidden] tensors, one per camera.
        """
        feats, pad_masks = [], []
        for img, mask in zip(images, image_masks):
            # img: [B, C, H, W]  →  pixel_values: [B, 1, C, H, W]  (1 image per sample)
            pv = img.unsqueeze(1)
            out = self.smolvlm.model.get_image_features(pv)
            feat = out.pooler_output  # [B, num_patches, vlm_hidden]
            B, P, _ = feat.shape
            feats.append(feat)
            pad_masks.append(mask[:, None].expand(B, P))
        return feats, pad_masks

    def embed_lang(self, lang_tokens: Tensor) -> Tensor:
        """Embed language tokens via SmolVLM's text embedding table."""
        emb = self.smolvlm.model.text_model.embed_tokens(lang_tokens)
        return emb * math.sqrt(emb.shape[-1])

    # ── Single joint transformer layer ──────────────────────────────────────

    def _joint_layer(
        self,
        layer_idx: int,
        prefix_h: Tensor,   # [B, L0, vlm_hidden]
        suffix_h: Tensor,   # [B, L1, exp_hidden]
        position_embeddings_prefix: tuple[Tensor, Tensor],
        position_embeddings_suffix: tuple[Tensor, Tensor],
        attention_mask: Tensor,  # [B, 1, L0+L1, L0+L1] additive bias
        adarms_cond: Tensor,     # [B, exp_hidden]  timestep embedding
    ) -> tuple[Tensor, Tensor]:
        vlm_layer: LlamaDecoderLayer  = self.vlm_layers[layer_idx]
        exp_layer: LlamaDecoderLayer  = self.expert_layers[layer_idx]

        # ── Pre-attention layernorm ─────────────────────────────────────────
        # VLM: standard RMSNorm
        prefix_n = vlm_layer.input_layernorm(prefix_h)           # [B, L0, vlm_h]
        # Expert: AdaRMS returns (normed, gate)
        suffix_n, gate0 = exp_layer.input_layernorm(suffix_h, adarms_cond)

        # ── Q K V projections (separate weights per stream) ─────────────────
        def _qkv(attn, h, pos_emb):
            shape = (*h.shape[:-1], -1, attn.head_dim)
            q = attn.q_proj(h).view(shape).transpose(1, 2)
            k = attn.k_proj(h).view(shape).transpose(1, 2)
            v = attn.v_proj(h).view(shape).transpose(1, 2)
            cos, sin = pos_emb
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            return q, k, v

        q0, k0, v0 = _qkv(vlm_layer.self_attn, prefix_n, position_embeddings_prefix)
        q1, k1, v1 = _qkv(exp_layer.self_attn, suffix_n, position_embeddings_suffix)

        # ── Concatenate along sequence dim → joint attention ────────────────
        # IMPORTANT: eager_attention_forward() internally performs repeat_kv()
        # based on module.num_key_value_groups. We must pass raw K/V with
        # num_key_value_heads here (not pre-expanded), otherwise K/V get
        # expanded twice and heads mismatch at matmul.

        # Pad / project expert Q to VLM head dim if they differ
        vlm_head_dim = vlm_layer.self_attn.head_dim
        exp_head_dim = exp_layer.self_attn.head_dim

        # Cat along seq dim (dim=2 after transpose)
        # Both have shape [B, num_heads, L, head_dim]
        # We cat on dim=2 to get [B, num_heads, L0+L1, head_dim]
        # This requires same num_heads and head_dim – our config ensures this
        # by design (ExpertConfig must match vlm heads for clean joint attn).
        # If they differ, a learned linear projection can be added; for now
        # we assert and document the constraint in MiniPI05Config.

        # NOTE: for simplicity we require expert to match VLM head structure
        # in the joint pass. The expert can have a smaller hidden_size but
        # must have the same num_attention_heads and head_dim.
        assert q0.shape[1] == q1.shape[1] and q0.shape[-1] == q1.shape[-1], (
            "VLM and expert must have identical num_attention_heads and head_dim "
            "for the joint cat-split attention pass. "
            f"Got VLM: heads={q0.shape[1]} head_dim={q0.shape[-1]}, "
            f"expert: heads={q1.shape[1]} head_dim={q1.shape[-1]}"
        )

        Q = torch.cat([q0, q1], dim=2)   # [B, H, L0+L1, head_dim]
        K = torch.cat([k0, k1], dim=2)
        V = torch.cat([v0, v1], dim=2)

        # ── Single full attention pass ───────────────────────────────────────
        # eager_attention_forward(module, Q, K, V, mask, scaling)
        # We use vlm's attention module just for the scaling factor
        att_out, _ = eager_attention_forward(
            vlm_layer.self_attn,
            Q, K, V,
            attention_mask,
            scaling=vlm_layer.self_attn.scaling,
        )
        # transformers eager attention returns [B, L, num_heads, head_dim]
        # for this version; flatten head dims for o_proj.
        if att_out.dim() == 4:
            att_out = att_out.reshape(att_out.shape[0], att_out.shape[1], -1)
        # att_out: [B, L0+L1, H * head_dim]

        L0 = prefix_h.shape[1]
        att_prefix = att_out[:, :L0, :]   # [B, L0, H*head_dim]
        att_suffix = att_out[:, L0:, :]   # [B, L1, H*head_dim]

        # ── o_proj + first residual (separate weights) ───────────────────────
        if att_prefix.dtype != vlm_layer.self_attn.o_proj.weight.dtype:
            att_prefix = att_prefix.to(vlm_layer.self_attn.o_proj.weight.dtype)
        if att_suffix.dtype != exp_layer.self_attn.o_proj.weight.dtype:
            att_suffix = att_suffix.to(exp_layer.self_attn.o_proj.weight.dtype)

        prefix_h = prefix_h + vlm_layer.self_attn.o_proj(att_prefix)
        suffix_h = suffix_h + gate0 * exp_layer.self_attn.o_proj(att_suffix)

        # ── Post-attention layernorm + MLP ────────────────────────────────────
        # VLM
        prefix_h = prefix_h + vlm_layer.mlp(vlm_layer.post_attention_layernorm(prefix_h))

        # Expert (AdaRMS again)
        suffix_pn, gate1 = exp_layer.post_attention_layernorm(suffix_h, adarms_cond)
        if exp_layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            suffix_pn = suffix_pn.to(torch.bfloat16)
        suffix_h = suffix_h + gate1 * exp_layer.mlp(suffix_pn)

        return prefix_h, suffix_h

    # ── Three-mode forward ──────────────────────────────────────────────────

    def forward(
        self,
        inputs_embeds: list[Tensor | None],   # [prefix_emb | None, suffix_emb | None]
        pad_masks: list[Tensor | None],        # matching pad masks
        att_2d_bias: Tensor | None,            # [B,1,N,N] or None
        position_ids: Tensor,                  # [B, N]
        adarms_cond: Tensor | None = None,     # [B, exp_hidden]
        past_key_values=None,
        use_cache: bool = False,
    ) -> tuple[list[Tensor | None], any]:

        prefix_emb, suffix_emb = inputs_embeds

        # ── Prefix-only (KV cache build at inference) ────────────────────────
        if suffix_emb is None:
            out = self.smolvlm.model.text_model(
                inputs_embeds=prefix_emb,
                attention_mask=pad_masks[0].long() if pad_masks[0] is not None else None,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            return [out.last_hidden_state, None], out.past_key_values

        # ── Suffix-only (fast denoising against cached KVs) ──────────────────
        if prefix_emb is None:
            # The suffix attends to cached prefix KVs via past_key_values.
            # We run only the expert layers with the suffix embeddings.
            # The attention mask (att_2d_bias) already has the correct
            # prefix prefix_len x suffix_len layout from the caller.
            h = suffix_emb
            pos_emb = self.rotary_emb(h, position_ids)
            for layer in self.expert_layers:
                # Standard LlamaDecoderLayer forward, but layernorms are AdaRMS
                residual = h
                h_n, gate0 = layer.input_layernorm(h, adarms_cond)
                h_attn, _ = layer.self_attn(
                    hidden_states=h_n,
                    attention_mask=att_2d_bias,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    position_embeddings=pos_emb,
                )
                h = residual + gate0 * h_attn
                residual = h
                h_pn, gate1 = layer.post_attention_layernorm(h, adarms_cond)
                if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                    h_pn = h_pn.to(torch.bfloat16)
                h = residual + gate1 * layer.mlp(h_pn)
            h = self.expert_norm(h)
            return [None, h], None

        # ── Joint forward (training + first inference step) ─────────────────
        B, L0, _ = prefix_emb.shape
        B, L1, _ = suffix_emb.shape
        N = L0 + L1

        # Shared RoPE: compute for the full joint sequence
        dummy = torch.zeros(B, N, self.cfg.vlm_text_head_dim,
                            device=prefix_emb.device, dtype=prefix_emb.dtype)
        cos, sin = self.rotary_emb(dummy, position_ids)
        # Split back into prefix / suffix position embeddings
        pos_emb_prefix = (cos[:, :L0], sin[:, :L0])
        pos_emb_suffix = (cos[:, L0:], sin[:, L0:])

        prefix_h = prefix_emb
        suffix_h = suffix_emb

        use_gc = (
            getattr(self, "gradient_checkpointing", False) and self.training
        )

        for i in range(len(self.vlm_layers)):
            if use_gc:
                def _layer_fn(i, ph, sh, pep, pes, mask, cond):
                    return self._joint_layer(i, ph, sh, pep, pes, mask, cond)
                prefix_h, suffix_h = torch.utils.checkpoint.checkpoint(
                    _layer_fn, i,
                    prefix_h, suffix_h,
                    pos_emb_prefix, pos_emb_suffix,
                    att_2d_bias, adarms_cond,
                    use_reentrant=False, preserve_rng_state=False,
                )
            else:
                prefix_h, suffix_h = self._joint_layer(
                    i, prefix_h, suffix_h,
                    pos_emb_prefix, pos_emb_suffix,
                    att_2d_bias, adarms_cond,
                )

        # Final norms
        prefix_h = self.vlm_norm(prefix_h)
        suffix_h = self.expert_norm(suffix_h)

        return [prefix_h, suffix_h], None


# ---------------------------------------------------------------------------
# Top-level π₀.5 policy
# ---------------------------------------------------------------------------

class PI05SmolVLM(nn.Module):
    """
    Full π₀.5 policy with SmolVLM backbone.

    Training  : loss = model(obs, actions).mean()
    Inference : actions = model.sample_actions(obs)
    """

    def __init__(self, cfg: MiniPI05Config):
        super().__init__()
        self.cfg = cfg

        # Dual-stream transformer
        self.backbone = SmolVLMWithExpertModel(cfg)

        exp_w = cfg.expert.hidden_size

        # Action input projection: action_dim → expert_width (fused in joint pass)
        self.action_in_proj = nn.Linear(cfg.action_dim, exp_w)

        # Action output projection: expert_width → action_dim
        self.action_out_proj = nn.Linear(exp_w, cfg.action_dim)
        self.state_dim = int(cfg.state_dim if cfg.state_dim is not None else cfg.action_dim)
        self.state_to_action = (
            nn.Identity()
            if self.state_dim == cfg.action_dim
            else nn.Linear(self.state_dim, cfg.action_dim)
        )

        # Timestep MLP for AdaRMS conditioning (π₀.5 style)
        # sinusoidal(τ) [B, exp_w] → SiLU → [B, exp_w] → SiLU → [B, exp_w]
        self.time_mlp = nn.Sequential(
            nn.Linear(exp_w, exp_w),
            nn.SiLU(),
            nn.Linear(exp_w, exp_w),
            nn.SiLU(),
        )

        # Cast projections to match backbone dtype
        if cfg.dtype == "bfloat16":
            self.action_in_proj.to(torch.bfloat16)
            self.action_out_proj.to(torch.bfloat16)
            self.time_mlp.to(torch.bfloat16)
            self.state_to_action.to(torch.bfloat16)

    # ── Gradient checkpointing ──────────────────────────────────────────────

    def gradient_checkpointing_enable(self):
        self.backbone.gradient_checkpointing = True
        self.backbone.smolvlm.model.text_model.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.backbone.gradient_checkpointing = False
        self.backbone.smolvlm.model.text_model.gradient_checkpointing = False

    # ── Embedding helpers ────────────────────────────────────────────────────

    def _embed_prefix(self, obs: Observation):
        """Embed images + language tokens → [B, L0, vlm_hidden], pad/att masks."""
        embs, pad_ms, att_ms = [], [], []

        # Camera images
        img_feats, img_pads = self.backbone.embed_images(obs.images, obs.image_masks)
        for feat, pm in zip(img_feats, img_pads):
            embs.append(feat)
            pad_ms.append(pm)
            att_ms.extend([0] * feat.shape[1])   # same block → bidirectional

        # Language tokens
        lang_emb = self.backbone.embed_lang(obs.lang_tokens)  # [B, L, vlm_h]
        embs.append(lang_emb)
        pad_ms.append(obs.lang_masks)
        att_ms.extend([0] * lang_emb.shape[1])

        prefix_emb  = torch.cat(embs,   dim=1)            # [B, L0, vlm_h]
        prefix_pad  = torch.cat(pad_ms, dim=1)            # [B, L0]
        att_t       = torch.tensor(att_ms, dtype=torch.long, device=prefix_pad.device)
        prefix_att  = att_t[None, :].expand(prefix_emb.shape[0], -1)

        return prefix_emb, prefix_pad, prefix_att

    def _observation_from_img_prop(self, img: Tensor, prop: Tensor) -> Observation:
        """Adapt mini-pi0 train/eval tensors into PI05 observation schema.

        Args:
            img: Image tensor shaped [B, C, H, W]. When multiple cameras are
                fused by width-concatenation, this method splits width into
                `cfg.num_cameras` equal segments.
            prop: Proprio tensor shaped [B, P].

        Returns:
            Observation namedtuple consumed by PI05 internals.
        """
        if img.ndim != 4:
            raise ValueError(f"Expected img shape [B,C,H,W], got {tuple(img.shape)}")
        if prop.ndim != 2:
            raise ValueError(f"Expected prop shape [B,P], got {tuple(prop.shape)}")

        b, _, _, w = img.shape
        cam_count = int(max(1, self.cfg.num_cameras))
        if cam_count > 1 and (w % cam_count == 0):
            images = list(torch.chunk(img, cam_count, dim=-1))
        else:
            # Fallback: treat as single camera if split is ambiguous.
            images = [img]
        image_masks = [
            torch.ones((b,), dtype=torch.bool, device=img.device) for _ in images
        ]

        state = prop.to(self.state_to_action.weight.dtype if isinstance(self.state_to_action, nn.Linear) else prop.dtype)
        state = self.state_to_action(state).to(img.dtype)
        lang_tokens = torch.zeros((b, 1), dtype=torch.long, device=img.device)
        lang_masks = torch.ones((b, 1), dtype=torch.bool, device=img.device)
        return Observation(
            images=images,
            image_masks=image_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
        )

    def _embed_suffix(self, state: Tensor, noisy_actions: Tensor, timestep: Tensor):
        """
        Embed robot state + noisy action chunk + timestep for the expert stream.

        Returns
        -------
        suffix_emb  : [B, 1+H, exp_w]   state token + H action tokens
        suffix_pad  : [B, 1+H]
        suffix_att  : [B, 1+H]          block indices
        adarms_cond : [B, exp_w]         timestep embedding for AdaRMS
        """
        exp_w = self.cfg.expert.hidden_size
        B     = state.shape[0]
        dev   = state.device

        # ── Timestep embedding → AdaRMS conditioning ────────────────────────
        tau_emb = sinusoidal_pos_embedding(
            timestep, exp_w,
            self.cfg.fm_min_period, self.cfg.fm_max_period,
        ).to(dtype=self.time_mlp[0].weight.dtype)
        adarms_cond = self.time_mlp(tau_emb)          # [B, exp_w]

        embs, pad_ms = [], []

        # ── State token ─────────────────────────────────────────────────────
        # Project state (action_dim) → exp_w via action_in_proj
        # (reuse action_in_proj; state and actions share the same space)
        state_f = state.to(self.action_in_proj.weight.dtype)
        state_emb = self.action_in_proj(state_f)[:, None, :]  # [B, 1, exp_w]
        embs.append(state_emb)
        pad_ms.append(torch.ones(B, 1, dtype=torch.bool, device=dev))

        # ── Action tokens (noisy) ────────────────────────────────────────────
        act_f = noisy_actions.to(self.action_in_proj.weight.dtype)
        action_emb = self.action_in_proj(act_f)               # [B, H, exp_w]
        embs.append(action_emb)
        pad_ms.append(torch.ones(B, self.cfg.action_horizon, dtype=torch.bool, device=dev))

        suffix_emb = torch.cat(embs,   dim=1)                 # [B, 1+H, exp_w]
        suffix_pad = torch.cat(pad_ms, dim=1)                 # [B, 1+H]

        # Attention block indices:
        #   state token     → 1  (new block, image/lang cannot attend here)
        #   first action    → 1  (new block within suffix)
        #   remaining acts  → 0  (same block, bidirectional among themselves)
        att_vals = [1, 1] + [0] * (self.cfg.action_horizon - 1)
        att_t = torch.tensor(att_vals, dtype=torch.long, device=dev)
        suffix_att = att_t[None, :].expand(B, -1)

        return suffix_emb, suffix_pad, suffix_att, adarms_cond

    # ── Training forward ─────────────────────────────────────────────────────

    def _forward_observation(self, obs: Observation, actions: Tensor) -> Tensor:
        """
        Compute flow-matching MSE loss.

        Parameters
        ----------
        obs     : Observation namedtuple
        actions : [B, H, action_dim]  clean ground-truth action chunk

        Returns
        -------
        loss : [B, H, action_dim]  element-wise MSE (caller should .mean())
        """
        B = actions.shape[0]
        dev = actions.device

        # Sample noise and timestep
        noise = torch.randn_like(actions)
        tau   = sample_beta_time(B, dev)                          # [B]
        t_exp = tau[:, None, None]                                # [B,1,1]
        x_t   = t_exp * noise + (1.0 - t_exp) * actions          # noisy actions
        u_t   = noise - actions                                   # target vector field

        # Embed prefix and suffix
        prefix_emb, prefix_pad, prefix_att = self._embed_prefix(obs)
        suffix_emb, suffix_pad, suffix_att, adarms_cond = self._embed_suffix(
            obs.state, x_t, tau
        )

        # Cast to backbone dtype
        if self.backbone.smolvlm.model.text_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            prefix_emb = prefix_emb.to(torch.bfloat16)
            suffix_emb = suffix_emb.to(torch.bfloat16)

        # Build joint attention mask
        pad_cat = torch.cat([prefix_pad, suffix_pad], dim=1)   # [B, L0+L1]
        att_cat = torch.cat([prefix_att, suffix_att], dim=1)   # [B, L0+L1]
        att_2d  = make_blockwise_att_mask(pad_cat, att_cat)    # [B, L0+L1, L0+L1]
        att_bias = att_mask_to_bias(att_2d)                    # [B, 1, N, N]

        # Position IDs: contiguous within valid tokens
        pos_ids = (torch.cumsum(pad_cat.long(), dim=1) - 1).clamp(min=0)

        # Joint forward
        (_, suffix_out), _ = self.backbone.forward(
            inputs_embeds=[prefix_emb, suffix_emb],
            pad_masks=[prefix_pad, suffix_pad],
            att_2d_bias=att_bias,
            position_ids=pos_ids,
            adarms_cond=adarms_cond,
        )

        # Slice last H tokens (action tokens; state token is first)
        suffix_out = suffix_out[:, -self.cfg.action_horizon:]   # [B, H, exp_w]
        suffix_out = suffix_out.to(torch.float32)
        v_t = self.action_out_proj(suffix_out)                  # [B, H, action_dim]

        return F.mse_loss(u_t, v_t, reduction="none")

    def forward(self, *args) -> Tensor:
        """Run PI05 training forward.

        Supported call signatures:
        1) `forward(obs: Observation, actions: Tensor) -> [B,H,A] loss tensor`
        2) `forward(img: Tensor, prop: Tensor, actions: Tensor) -> scalar loss`
           compatible with mini-pi0 training runner.
        """
        if len(args) == 2 and isinstance(args[0], Observation):
            obs, actions = args
            return self._forward_observation(obs, actions)
        if len(args) == 3 and torch.is_tensor(args[0]) and torch.is_tensor(args[1]) and torch.is_tensor(args[2]):
            img, prop, actions = args
            obs = self._observation_from_img_prop(img, prop)
            return self._forward_observation(obs, actions).mean()
        raise TypeError(
            "PI05SmolVLM.forward expected (Observation, actions) or (img, prop, actions)"
        )

    # ── Inference ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample_actions(
        self,
        obs: Observation,
        noise: Tensor | None = None,
        num_steps: int | None = None,
    ) -> Tensor:
        """
        Denoise from pure noise to a clean action chunk via forward Euler.

        Returns
        -------
        actions : [B, H, action_dim]
        """
        num_steps = num_steps or self.cfg.fm_num_steps
        B   = obs.state.shape[0]
        dev = obs.state.device

        if noise is None:
            noise = torch.randn(B, self.cfg.action_horizon, self.cfg.action_dim, device=dev)

        # ── Encode prefix once → KV cache ────────────────────────────────────
        prefix_emb, prefix_pad, prefix_att = self._embed_prefix(obs)
        if self.backbone.smolvlm.model.text_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            prefix_emb = prefix_emb.to(torch.bfloat16)

        prefix_att_2d  = make_blockwise_att_mask(prefix_pad, prefix_att)
        prefix_att_bias = att_mask_to_bias(prefix_att_2d)
        prefix_pos_ids = (torch.cumsum(prefix_pad.long(), dim=1) - 1).clamp(min=0)

        # prefix-only forward: builds KV cache
        self.backbone.smolvlm.model.text_model.config._attn_implementation = "eager"
        _, past_kv = self.backbone.forward(
            inputs_embeds=[prefix_emb, None],
            pad_masks=[prefix_pad, None],
            att_2d_bias=prefix_att_bias,
            position_ids=prefix_pos_ids,
            past_key_values=None,
            use_cache=True,
        )

        # ── Euler denoising loop ─────────────────────────────────────────────
        dt = -1.0 / num_steps
        x_t = noise
        tau = torch.tensor(1.0, dtype=torch.float32, device=dev)
        prefix_len = prefix_pad.shape[1]

        while tau >= -dt / 2:
            expanded_tau = tau.expand(B)
            suffix_emb, suffix_pad, suffix_att, adarms_cond = self._embed_suffix(
                obs.state, x_t, expanded_tau
            )
            if self.backbone.smolvlm.model.text_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
                suffix_emb = suffix_emb.to(torch.bfloat16)

            # Suffix attends to cached prefix: build combined mask
            suffix_len = suffix_pad.shape[1]
            prefix_pad_2d = prefix_pad[:, None, :].expand(B, suffix_len, prefix_len)
            suffix_att_2d = make_blockwise_att_mask(suffix_pad, suffix_att)
            full_att_2d   = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)
            full_att_bias  = att_mask_to_bias(full_att_2d)

            # Position IDs offset by prefix length
            prefix_offset = prefix_pad.long().sum(-1)[:, None]  # [B,1]
            suffix_pos_ids = (prefix_offset + torch.cumsum(suffix_pad.long(), dim=1) - 1).clamp(0)

            self.backbone.smolvlm.model.text_model.config._attn_implementation = "eager"
            # DynamicCache is updated in-place even when use_cache=False.
            # Clone the prefix cache per denoising step so each step conditions
            # on the same fixed prefix context.
            step_past_kv = copy.deepcopy(past_kv)
            (_, suffix_out), _ = self.backbone.forward(
                inputs_embeds=[None, suffix_emb],
                pad_masks=[None, suffix_pad],
                att_2d_bias=full_att_bias,
                position_ids=suffix_pos_ids,
                adarms_cond=adarms_cond,
                past_key_values=step_past_kv,
                use_cache=False,
            )

            suffix_out = suffix_out[:, -self.cfg.action_horizon:].to(torch.float32)
            v_t = self.action_out_proj(suffix_out)    # [B, H, action_dim]

            x_t = x_t + dt * v_t
            tau = tau + dt

        return x_t

    @torch.no_grad()
    def sample(self, img: Tensor, prop: Tensor, n_steps: int = 10) -> Tensor:
        """mini-pi0 eval/deploy compatible sampling API.

        Args:
            img: [B, C, H, W] image tensor.
            prop: [B, P] proprio tensor.
            n_steps: Number of Euler denoising steps.

        Returns:
            Normalized sampled action chunk [B, H, action_dim].
        """
        obs = self._observation_from_img_prop(img, prop)
        return self.sample_actions(obs, num_steps=int(n_steps))


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("Building π₀.5 – SmolVLM edition ...")

    cfg = MiniPI05Config(
        # Tiny config for fast smoke test
        vlm_text_hidden_size=128,
        vlm_text_layers=4,
        vlm_text_heads=4,
        vlm_text_kv_heads=2,
        vlm_text_intermediate=256,
        vlm_text_head_dim=32,
        vlm_vision_hidden=128,
        vlm_vision_layers=2,
        vlm_vision_heads=4,
        vlm_vision_image_size=64,
        vlm_vision_patch_size=8,
        expert=ExpertConfig(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=4,   # must match vlm_text_layers
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
        ),
        action_dim=8,
        action_horizon=10,
        num_cameras=2,
        dtype="float32",           # float32 for CPU smoke test
    )

    model = PI05SmolVLM(cfg)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total parameters: {total_params:.1f}M")

    B, L, C, H, W = 2, 16, 3, 64, 64
    obs = Observation(
        images      = [torch.randn(B, C, H, W), torch.randn(B, C, H, W)],
        image_masks = [torch.ones(B, dtype=torch.bool),
                       torch.ones(B, dtype=torch.bool)],
        lang_tokens = torch.zeros(B, L, dtype=torch.long),
        lang_masks  = torch.ones(B, L, dtype=torch.bool),
        state       = torch.randn(B, cfg.action_dim),
    )
    actions = torch.randn(B, cfg.action_horizon, cfg.action_dim)

    print("Forward (training) ...")
    loss = model(obs, actions)
    print(f"  loss shape : {loss.shape}   mean = {loss.mean().item():.4f}")
    loss.mean().backward()
    print("  backward OK")

    print("Forward (inference) ...")
    model.eval()
    with torch.no_grad():
        sampled = model.sample_actions(obs, num_steps=3)
    print(f"  sampled shape : {sampled.shape}")

    print("\nSmoke test passed.")
    sys.exit(0)
