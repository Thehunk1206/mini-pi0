# ReinFlow PPO for Visuomotor Flow Policies

`mini-pi0` uses `rl.algorithm: reinflow_ppo` for path-space reinforcement
learning of `mini_pi0_fm`. The implementation follows the discrete stochastic
flow construction in the [ReinFlow paper](https://arxiv.org/abs/2505.22094)
and its [reference implementation](https://github.com/ReinFlow/ReinFlow),
adapted from VLA policies to this repository's image-and-proprioception policy.

`gaussian_ppo_baseline` remains available as a surrogate ablation. It places a
Gaussian around the final FM action and is not PPO over the flow policy.

## Stochastic Flow Policy

The pretrained actor predicts the FM velocity
`v_theta(x_k, t_k, o)`. ReinFlow adds a learned diagonal transition standard
deviation:

```text
x_0 ~ Normal(0, I)
mu_k = x_k + dt * v_theta(x_k, t_k, o)
x_(k+1) ~ Normal(mu_k, sigma_psi(x_k, t_k, o))
```

The stored path has shape:

```text
[batch, flow_steps + 1, chunk_size, action_dim]
```

The joint path log-likelihood is:

```text
log pi(path | o) = sum_k sum_h sum_a log p(x_(k+1,h,a) | x_k, o)
```

`p(x_0)` is omitted because the standard-normal base distribution has no
trainable policy parameters. PPO therefore optimizes a likelihood that is
native to the stochastic flow process instead of inventing a final-action
Gaussian.

The noise network uses smooth bounded standard deviations:

```text
sigma = sigma_min + (sigma_max - sigma_min) * sigmoid(z + b)
```

The output bias is initialized so `sigma` starts exactly at
`noise_std_init`. The upper bound can decay to `noise_std_final_max` after the
configured hold fraction.

## Bounded Transitions

When intermediate denoising states are clipped, the likelihood is a censored
diagonal Normal:

- Interior values use the Gaussian density.
- Values at the lower boundary use the lower-tail probability mass.
- Values at the upper boundary use the upper-tail probability mass.

This keeps PPO ratios valid at clipped boundaries. Reported entropy is the
unclipped Gaussian entropy, normalized per flow-step, chunk-token, and action
symbol by default.

Policy-space limits come from simulator bounds. Checkpoint fine-tuning maps
environment bounds through the saved dataset mean and standard deviation.
Scratch mode uses a bounded pre-tanh latent and maps it through the simulator's
action range. Environment actions are clipped once more before stepping.

## Actor, Critic, and Rollout Mode

`ReinFlowActor` owns the FM policy and transition kernel. `FlowCritic` consumes
pooled, detached observation features, so critic warm-up and critic updates
cannot modify the actor. `ReinFlowActorCritic` is only their composition root.

The actor stays in evaluation mode during rollout and PPO likelihood
evaluation. This disables untracked dropout and mutable normalization state
while retaining gradients during actor updates. Frozen vision backbones also
remain in evaluation mode. Re-evaluating a stored path before an optimizer step
must reproduce its old log-probability.

## Macro Actions and GAE

One policy decision samples a complete action chunk and executes at most
`execution_horizon` primitive actions without reconditioning. Execution stops
for an environment on termination, truncation, or configured success.

For a macro transition lasting `m` primitive steps:

```text
R_t = sum_(j=0)^(m-1) gamma^j r_(t,j)
d_t = 0                         on true termination
d_t = gamma^m                   otherwise
c_t = 0                         on termination or truncation
c_t = 1                         otherwise
delta_t = R_t + d_t V(o_(t+1)) - V(o_t)
A_t = delta_t + d_t lambda c_t A_(t+1)
```

Truncation bootstraps from the final observation but stops the GAE trace, so an
advantage never crosses an episode reset. The fixed-capacity rollout buffer
stores one row per macro decision: observation tensors, full flow path, old
joint log-probability, value, next value, macro reward, duration, bootstrap
discount, and trace mask.

## PPO Objective

The actor minimizes the clipped path-space PPO loss:

```text
r_t(theta) = exp(log pi_theta(path_t|o_t) - log pi_old(path_t|o_t))
L_policy = -mean(min(r_t A_t, clip(r_t, 1-eps, 1+eps) A_t))
```

The visual-policy default is `eps = 0.001`. Log-ratios are clamped only before
exponentiation for numerical safety. NaN or Inf in optimization tensors raises
an error immediately.

The complete actor loss is:

```text
L_actor = L_policy
          - entropy_coef * H
          + reference_w2_coef * L_W2
          + reference_transition_kl_coef * L_transition_KL
          + velocity_anchor_coef * L_velocity
```

`L_W2` integrates the trainable actor and frozen BC reference deterministically
from the same initial noise and compares their final chunks. Transition KL is
the analytical KL between pre-clipping Gaussian transitions. It is an optional
extension, not a full censored path KL. `L_velocity` is a non-ReinFlow ablation
and is disabled by default. Do not enable W2 and transition KL together in the
initial coefficient sweep.

The critic uses a separate AdamW optimizer and value loss. During
`critic_warmup_updates`, rollout uses the frozen actor and only the critic is
updated. CUDA neural forwards may use bf16; distributions, ratios, GAE, and
losses stay fp32.

## Initialization Modes

Scratch is an experimental no-dataset baseline:

```yaml
rl:
  algorithm: reinflow_ppo
  init_mode: scratch
  action_normalization: env_bounds
  use_reference_policy: false
  reference_w2_coef: 0.0
  reference_transition_kl_coef: 0.0
```

Production fine-tuning starts from matching FM and action-stat artifacts:

```yaml
rl:
  algorithm: reinflow_ppo
  init_mode: checkpoint
  action_normalization: dataset_stats
  checkpoint: runs/<bc-run>/checkpoints/best.pt
  action_stats_path: runs/<bc-run>/artifacts/action_stats.json
  use_reference_policy: true
```

Resume uses `rl.resume_from` instead of `rl.checkpoint`. ReinFlow checkpoints
atomically store actor, noise network, critic, frozen reference, optimizers,
schedulers, counters, resolved config, action statistics, RNG states, Git
commit, versions, environment manifest, and metrics.

## Commands

ManiSkill smoke and scratch update:

```bash
mini-pi0 rl-smoke --config examples/configs/maniskill3_pickcube_reinflow_scratch.yaml
mini-pi0 rl-train --config examples/configs/maniskill3_pickcube_reinflow_scratch.yaml
```

PegInsertion checkpoint fine-tuning:

```bash
mini-pi0 rl-smoke --config examples/configs/maniskill3_peginsertion_reinflow_finetune.yaml
mini-pi0 rl-train --config examples/configs/maniskill3_peginsertion_reinflow_finetune.yaml
```

Resume and deterministic evaluation:

```bash
mini-pi0 rl-train --config <matching-config.yaml> --resume_from runs/<run>/checkpoints/latest_rl.pt
mini-pi0 rl-eval --config <matching-config.yaml> --resume_from runs/<run>/checkpoints/latest_rl.pt
```

Isaac Lab commands are documented in
[ISAACLAB_DOCKER.md](ISAACLAB_DOCKER.md).

## PegInsertion Protocol

The main PegInsertion config keeps ManiSkill's native dense reward unchanged.
The separate `maniskill3_peginsertion_reinflow_potential.yaml` adds the declared
potential ablation:

```text
Phi = I_grasp + 2(1 - tanh(20 e_align))
      + 4 clip(d_insert / d_hole, 0, 1)
r' = r_native + gamma^m Phi(s') - Phi(s)
```

Per-environment diagnostics include grasp, alignment error, axis error,
insertion depth, peg-box force, and jam duration. Jamming is logged and is not
directly penalized by the first ablation.

Use training seeds `0, 1, 2`. Evaluate BC and RL deterministically on episode
seeds `10000-10099`. Report success with Wilson intervals, return, episode
length, insertion depth, jam rate, clipping, and paired bootstrap intervals for
RL minus BC. Claim improvement only when mean success rises and the paired 95%
interval excludes zero.

## Current Validation Status

- Host suite: 122 passed.
- Native ManiSkill: two-environment PickCube and PegInsertion reset, step, and
  selective reset passed on the installed GPU runtime.
- PickCube scratch: two PPO updates with four environments completed and saved
  metrics/checkpoint artifacts.
- PegInsertion checkpoint: smoke plus one tiny actor/critic update with two
  environments completed.
- These short runs validate engineering paths only. They do not establish a
  success-rate improvement.

## Failure Modes

- Mismatched action statistics invalidate checkpoint fine-tuning.
- Large path ratios can saturate the narrow visual-policy clipping range.
- Too much transition noise destabilizes contact behavior; too little prevents
  useful exploration.
- A critic trained from too little rollout data can dominate early updates.
- Native GPU vector simulators still advance masked rows with zero actions until
  the current macro chunk ends; terminal observations are frozen for learning
  and rows are selectively reset afterward.
- Higher return without fixed-seed success improvement is not evidence of a
  better manipulation policy.
