"""PPO optimization for stochastic ReinFlow denoising paths."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

import torch

from mini_pi0.config.schema import RootConfig
from mini_pi0.rl.buffers import ReinFlowRolloutBatch, ReinFlowRolloutBuffer
from mini_pi0.rl.exceptions import ReinFlowNumericalError
from mini_pi0.rl.flow_policy import ActionBounds, ReinFlowActorCritic
from mini_pi0.rl.kernels import diagonal_gaussian_kl

_OLD_POLICY_LOG_RATIO_TOLERANCE = 1e-5


@dataclass(frozen=True)
class ReinFlowPPOUpdateStats:
    """Averaged diagnostics from one PPO update."""

    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    ratio_min: float
    ratio_max: float
    reference_w2: float
    transition_kl: float
    velocity_anchor: float
    actor_grad_norm: float
    critic_grad_norm: float
    explained_variance: float
    actor_updated: bool
    rollout_log_prob_correction_max: float = 0.0
    actor_optimizer_steps: int = 0


@dataclass(frozen=True)
class ReinFlowPPOProgress:
    """Progress from one likelihood or optimizer minibatch."""

    stage: Literal["rebase", "optimize"]
    completed: int
    total: int
    value_loss: float = 0.0
    policy_loss: float | None = None
    approx_kl: float | None = None
    log_prob_correction: float = 0.0
    actor_steps: int = 0


ReinFlowPPOProgressCallback = Callable[[ReinFlowPPOProgress], None]


class ReinFlowPPOUpdater:
    """Jointly optimize ReinFlow velocity/noise and a separate value critic."""

    def __init__(
        self,
        *,
        policy: ReinFlowActorCritic,
        reference: ReinFlowActorCritic | None,
        cfg: RootConfig,
    ) -> None:
        """Create actor/critic optimizers and freeze the optional reference."""

        self.policy = policy
        self.reference = reference
        self.cfg = cfg
        self.actor_optimizer = torch.optim.AdamW(
            policy.actor.parameters(),
            lr=float(cfg.rl.actor_lr),
            weight_decay=float(cfg.rl.actor_weight_decay),
        )
        self.critic_optimizer = torch.optim.AdamW(
            policy.critic.parameters(),
            lr=float(cfg.rl.critic_lr),
            weight_decay=float(cfg.rl.critic_weight_decay),
        )
        self.actor_scheduler = _make_scheduler(
            self.actor_optimizer,
            total_updates=int(cfg.rl.total_updates),
            warmup_updates=int(cfg.rl.actor_lr_warmup_updates),
            name=str(cfg.rl.lr_scheduler),
        )
        self.critic_scheduler = _make_scheduler(
            self.critic_optimizer,
            total_updates=int(cfg.rl.total_updates),
            warmup_updates=0,
            name=str(cfg.rl.lr_scheduler),
        )
        if reference is not None:
            reference.eval()
            for parameter in reference.parameters():
                parameter.requires_grad = False

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        """Expose the actor optimizer for legacy checkpoint callers."""

        return self.actor_optimizer

    def update(
        self,
        buffer: ReinFlowRolloutBuffer,
        *,
        device: torch.device,
        update_index: int,
        bounds: ActionBounds,
        progress: ReinFlowPPOProgressCallback | None = None,
    ) -> ReinFlowPPOUpdateStats:
        """Run critic warm-up or a complete ReinFlow PPO update."""

        warmup = int(update_index) < int(self.cfg.rl.critic_warmup_updates)
        totals = _empty_totals()
        critic_batches = 0
        actor_batches = 0
        actor_steps = 0
        actor_enabled = not warmup
        log_prob_correction = 0.0
        if actor_enabled:
            log_prob_correction = self._canonicalize_old_log_probs(
                buffer,
                device=device,
                bounds=bounds,
                progress=progress,
            )
        epochs = int(self.cfg.rl.critic_warmup_epochs if warmup else self.cfg.rl.epochs_per_update)
        total_batches = epochs * buffer.minibatch_count(int(self.cfg.rl.minibatch_size))
        for _epoch in range(epochs):
            for batch in buffer.minibatches(int(self.cfg.rl.minibatch_size), device):
                with self._autocast(device):
                    values = self.policy.value(batch.images, batch.proprio)
                value_loss = 0.5 * (batch.returns - values).square().mean()
                critic_grad = self._step_critic(float(self.cfg.rl.value_coef) * value_loss)
                row = {
                    "value_loss": value_loss.detach(),
                    "critic_grad_norm": critic_grad,
                }
                if actor_enabled:
                    actor_row, step_applied = self._step_actor(
                        batch,
                        bounds,
                        require_old_policy_match=actor_batches == 0,
                    )
                    row.update(actor_row)
                    actor_batches += 1
                    actor_steps += int(step_applied)
                    target_kl = self.cfg.rl.target_kl
                    if target_kl is not None and actor_row["approx_kl"].item() > float(target_kl):
                        actor_enabled = False
                _accumulate(totals, row)
                critic_batches += 1
                if progress is not None:
                    progress(
                        ReinFlowPPOProgress(
                            stage="optimize",
                            completed=critic_batches,
                            total=total_batches,
                            value_loss=float(value_loss.detach().cpu()),
                            policy_loss=_optional_metric(row, "policy_loss"),
                            approx_kl=_optional_metric(row, "approx_kl"),
                            actor_steps=actor_steps,
                        )
                    )
        if actor_steps > 0:
            self.actor_scheduler.step()
        self.critic_scheduler.step()
        explained = _explained_variance(buffer)
        return _stats_from_totals(
            totals,
            actor_batches,
            critic_batches,
            actor_steps,
            explained,
            log_prob_correction,
        )

    def _canonicalize_old_log_probs(
        self,
        buffer: ReinFlowRolloutBuffer,
        *,
        device: torch.device,
        bounds: ActionBounds,
        progress: ReinFlowPPOProgressCallback | None,
    ) -> float:
        """Recompute the frozen rollout policy at the PPO minibatch shape.

        CUDA BF16 matrix kernels can produce different rounding for rollout and
        optimization batch sizes. A full flow path sums hundreds of likelihood
        terms, so the old policy is evaluated once at the optimization shape
        before any actor parameter changes.
        """

        maximum = 0.0
        self.policy.actor.likelihood_mode()
        total = buffer.minibatch_count(int(self.cfg.rl.minibatch_size))
        for completed, batch in enumerate(
            buffer.likelihood_batches(int(self.cfg.rl.minibatch_size), device),
            start=1,
        ):
            with torch.no_grad(), self._autocast(device):
                evaluation = self.policy.evaluate_path(
                    batch.images,
                    batch.proprio,
                    batch.paths,
                    bounds=bounds,
                )
            corrected = evaluation.log_prob.float()
            _require_finite("canonical old log_prob", corrected)
            delta = (corrected - batch.old_log_probs.float()).abs().max()
            maximum = max(maximum, float(delta.detach().cpu()))
            buffer.replace_log_probs(batch.start, batch.stop, corrected)
            if progress is not None:
                progress(
                    ReinFlowPPOProgress(
                        stage="rebase",
                        completed=completed,
                        total=total,
                        log_prob_correction=maximum,
                    )
                )
        return maximum

    def _step_actor(
        self,
        batch: ReinFlowRolloutBatch,
        bounds: ActionBounds,
        *,
        require_old_policy_match: bool,
    ) -> tuple[dict[str, torch.Tensor], bool]:
        """Compute and apply one clipped ReinFlow actor update."""

        with self._autocast(batch.images.device):
            evaluation = self.policy.evaluate_path(batch.images, batch.proprio, batch.paths, bounds=bounds)
        log_ratio = evaluation.log_prob.float() - batch.old_log_probs.float()
        _require_finite("log_ratio", log_ratio)
        if require_old_policy_match:
            _require_matching_old_policy(log_ratio)
        ratio = torch.exp(log_ratio.clamp(-20.0, 20.0))
        clip_ratio = float(self.cfg.rl.clip_ratio)
        unclipped = ratio * batch.advantages
        clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * batch.advantages
        policy_loss = -torch.minimum(unclipped, clipped).mean()
        approx_kl = ((ratio - 1.0) - log_ratio).mean().clamp_min(0.0)
        clip_fraction = ((ratio - 1.0).abs() > clip_ratio).float().mean()
        with self._autocast(batch.images.device):
            w2 = self._reference_w2(batch.images, batch.proprio, batch.paths[:, 0], bounds)
            transition_kl = self._transition_kl(batch.images, batch.proprio, batch.paths)
            velocity_anchor = self._velocity_anchor(batch.images, batch.proprio, batch.paths)
        actor_loss = (
            policy_loss
            - float(self.cfg.rl.entropy_coef) * evaluation.entropy.mean()
            + float(self.cfg.rl.reference_w2_coef) * w2
            + float(self.cfg.rl.reference_transition_kl_coef) * transition_kl
            + float(self.cfg.rl.velocity_anchor_coef) * velocity_anchor
        )
        _require_finite("actor_loss", actor_loss)
        row = {
            "policy_loss": policy_loss.detach(),
            "entropy": evaluation.entropy.mean().detach(),
            "approx_kl": approx_kl.detach(),
            "clip_fraction": clip_fraction.detach(),
            "ratio_min": ratio.min().detach(),
            "ratio_max": ratio.max().detach(),
            "reference_w2": w2.detach(),
            "transition_kl": transition_kl.detach(),
            "velocity_anchor": velocity_anchor.detach(),
        }
        target_kl = self.cfg.rl.target_kl
        if target_kl is not None and float(approx_kl.detach()) > float(target_kl):
            row["actor_grad_norm"] = actor_loss.new_zeros(())
            return row, False
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        grad_norm = _clip_grad(self.policy.actor.parameters(), float(self.cfg.rl.max_grad_norm))
        self.actor_optimizer.step()
        row["actor_grad_norm"] = grad_norm
        return row, True

    def _step_critic(self, value_loss: torch.Tensor) -> torch.Tensor:
        """Apply one critic-only optimization step."""

        _require_finite("value_loss", value_loss)
        self.critic_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        grad_norm = _clip_grad(self.policy.critic.parameters(), float(self.cfg.rl.max_grad_norm))
        self.critic_optimizer.step()
        return grad_norm

    def _reference_w2(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor,
        initial_noise: torch.Tensor,
        bounds: ActionBounds,
    ) -> torch.Tensor:
        """Return paired deterministic chunk MSE from shared initial noise."""

        if self.reference is None or float(self.cfg.rl.reference_w2_coef) == 0.0:
            return initial_noise.new_zeros(())
        actor_cond = self.policy.actor.encode_conditioning(images, proprio)
        actor_chunk = self.policy.actor.deterministic_from_conditioning(
            actor_cond,
            bounds=bounds,
            initial_noise=initial_noise,
        )
        with torch.no_grad():
            ref_cond = self.reference.actor.encode_conditioning(images, proprio)
            ref_chunk = self.reference.actor.deterministic_from_conditioning(
                ref_cond,
                bounds=bounds,
                initial_noise=initial_noise,
            )
        return (actor_chunk - ref_chunk).square().mean()

    def _transition_kl(self, images: torch.Tensor, proprio: torch.Tensor, paths: torch.Tensor) -> torch.Tensor:
        """Return mean pre-clipping Gaussian transition KL to the reference."""

        if self.reference is None or float(self.cfg.rl.reference_transition_kl_coef) == 0.0:
            return paths.new_zeros(())
        actor_cond = self.policy.actor.encode_conditioning(images, proprio)
        actor_params = self.policy.actor.transition_parameters(actor_cond, paths)
        with torch.no_grad():
            ref_cond = self.reference.actor.encode_conditioning(images, proprio)
            ref_params = self.reference.actor.transition_parameters(ref_cond, paths)
        return diagonal_gaussian_kl(actor_params, ref_params).mean()

    def _velocity_anchor(self, images: torch.Tensor, proprio: torch.Tensor, paths: torch.Tensor) -> torch.Tensor:
        """Return optional velocity-field MSE against the reference actor."""

        if self.reference is None or float(self.cfg.rl.velocity_anchor_coef) == 0.0:
            return paths.new_zeros(())
        actor_cond = self.policy.actor.encode_conditioning(images, proprio)
        actor_velocity = self.policy.actor.velocity_path(actor_cond, paths)
        with torch.no_grad():
            ref_cond = self.reference.actor.encode_conditioning(images, proprio)
            ref_velocity = self.reference.actor.velocity_path(ref_cond, paths)
        return (actor_velocity - ref_velocity).square().mean()

    def _autocast(self, device: torch.device):
        """Use bf16 only for neural forwards on CUDA."""

        enabled = str(self.cfg.rl.dtype).strip().lower() == "bf16" and device.type == "cuda"
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


FlowPPOUpdater = ReinFlowPPOUpdater
FlowPPOUpdateStats = ReinFlowPPOUpdateStats


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_updates: int,
    warmup_updates: int,
    name: str,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a small update-level warm-up and decay schedule."""

    normalized_name = name.strip().lower()
    if normalized_name not in {"constant", "cosine"}:
        raise ValueError("rl.lr_scheduler must be 'constant' or 'cosine'.")

    def multiplier(step: int) -> float:
        if warmup_updates > 0 and step < warmup_updates:
            return float(step + 1) / float(warmup_updates)
        if normalized_name == "constant":
            return 1.0
        decay_steps = max(1, total_updates - warmup_updates)
        progress = min(1.0, max(0.0, (step - warmup_updates) / decay_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _clip_grad(parameters: Iterable[torch.nn.Parameter], maximum: float) -> torch.Tensor:
    """Clip and return a detached gradient norm."""

    parameter_list = list(parameters)
    if maximum > 0.0:
        norm = torch.nn.utils.clip_grad_norm_(parameter_list, maximum)
    else:
        norms = [parameter.grad.norm() for parameter in parameter_list if parameter.grad is not None]
        norm = torch.stack(norms).norm() if norms else torch.tensor(0.0)
    return torch.as_tensor(norm).detach()


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    """Fail fast when an optimization tensor contains NaN or Inf."""

    if not torch.isfinite(tensor).all():
        raise ReinFlowNumericalError(f"Non-finite tensor encountered in {name}.")


def _require_matching_old_policy(log_ratio: torch.Tensor) -> None:
    """Fail before the first actor step when old-policy likelihoods disagree."""

    maximum = float(log_ratio.detach().abs().max().cpu())
    if maximum > _OLD_POLICY_LOG_RATIO_TOLERANCE:
        raise ReinFlowNumericalError(
            "Old-policy path likelihood mismatch before the first PPO actor step: "
            f"max |log_ratio|={maximum:.3e}, expected <= {_OLD_POLICY_LOG_RATIO_TOLERANCE:.1e}. "
            "No actor update was applied."
        )


def _empty_totals() -> dict[str, float]:
    """Create scalar accumulators for all update diagnostics."""

    return {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "ratio_min": math.inf,
        "ratio_max": -math.inf,
        "reference_w2": 0.0,
        "transition_kl": 0.0,
        "velocity_anchor": 0.0,
        "actor_grad_norm": 0.0,
        "critic_grad_norm": 0.0,
    }


def _accumulate(totals: dict[str, float], row: dict[str, torch.Tensor]) -> None:
    """Accumulate one minibatch row while preserving ratio extrema."""

    for name, value in row.items():
        scalar = float(value.detach().cpu())
        if name == "ratio_min":
            totals[name] = min(totals[name], scalar)
        elif name == "ratio_max":
            totals[name] = max(totals[name], scalar)
        else:
            totals[name] += scalar


def _optional_metric(row: dict[str, torch.Tensor], name: str) -> float | None:
    """Return one optional minibatch metric as a Python float."""

    value = row.get(name)
    return None if value is None else float(value.detach().cpu())


def _stats_from_totals(
    totals: dict[str, float],
    actor_batches: int,
    critic_batches: int,
    actor_steps: int,
    explained_variance: float,
    rollout_log_prob_correction_max: float,
) -> ReinFlowPPOUpdateStats:
    """Convert accumulators into a typed update summary."""

    actor_denominator = max(1, int(actor_batches))
    critic_denominator = max(1, int(critic_batches))
    ratio_min = totals["ratio_min"] if math.isfinite(totals["ratio_min"]) else 1.0
    ratio_max = totals["ratio_max"] if math.isfinite(totals["ratio_max"]) else 1.0
    return ReinFlowPPOUpdateStats(
        policy_loss=totals["policy_loss"] / actor_denominator,
        value_loss=totals["value_loss"] / critic_denominator,
        entropy=totals["entropy"] / actor_denominator,
        approx_kl=totals["approx_kl"] / actor_denominator,
        clip_fraction=totals["clip_fraction"] / actor_denominator,
        ratio_min=ratio_min,
        ratio_max=ratio_max,
        reference_w2=totals["reference_w2"] / actor_denominator,
        transition_kl=totals["transition_kl"] / actor_denominator,
        velocity_anchor=totals["velocity_anchor"] / actor_denominator,
        actor_grad_norm=totals["actor_grad_norm"] / actor_denominator,
        critic_grad_norm=totals["critic_grad_norm"] / critic_denominator,
        explained_variance=explained_variance,
        actor_updated=actor_steps > 0,
        rollout_log_prob_correction_max=float(rollout_log_prob_correction_max),
        actor_optimizer_steps=int(actor_steps),
    )


def _explained_variance(buffer: ReinFlowRolloutBuffer) -> float:
    """Return critic explained variance over the collected rollout."""

    if buffer.returns is None:
        return 0.0
    returns = buffer.returns.flatten().float()
    values = buffer.old_values.flatten().float()
    variance = returns.var(unbiased=False)
    if float(variance) < 1e-12:
        return 0.0
    return float((1.0 - (returns - values).var(unbiased=False) / variance).cpu())
