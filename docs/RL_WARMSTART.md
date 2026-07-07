# PPO Warm-Start From Flow Matching

`mini-pi0 rl-train` fine-tunes a trained `mini_pi0_fm` checkpoint with PPO in
simulation. The first runtime target is Dockerized Isaac Lab in headless mode.

## Design

The runner creates two actor-critic policies:

- a trainable policy initialized from the supervised FM checkpoint
- a frozen reference policy initialized from the same checkpoint

The actor reuses the FM observation encoder and action denoiser. For PPO, it
forms a differentiable Gaussian action distribution by evaluating the denoiser
at the clean end of the flow field and using a learned diagonal log standard
deviation. This keeps the policy initialized from the visuomotor FM model while
making policy-gradient optimization practical.

The PPO loss includes:

- clipped policy objective
- value loss
- entropy bonus
- KL/drift penalty against the frozen reference policy

The KL term is important for robot learning: it discourages large jumps away
from the demonstrated behavior while the simulator reward is still being
validated.

## Checkpoint Requirements

`rl.checkpoint` must point to a mini-pi0 checkpoint containing:

- `model`
- `model_name`
- `model_config`

`rl.action_stats_path` must match the checkpoint's action normalization stats.
The runner uses these stats to denormalize PPO actions before stepping the
simulator.

Example:

```bash
mini-pi0 rl-train \
  --config examples/configs/isaaclab_franka_lift_ppo.yaml \
  --checkpoint runs/<bc-run>/checkpoints/best.pt \
  --action_stats runs/<bc-run>/artifacts/action_stats.json
```

Run that command through Docker for Isaac Lab:

```bash
docker compose -f compose.isaaclab.yaml run --rm isaaclab \
  mini-pi0 rl-train --config examples/configs/isaaclab_franka_lift_ppo.yaml \
  --checkpoint runs/<bc-run>/checkpoints/best.pt \
  --action_stats runs/<bc-run>/artifacts/action_stats.json
```

## Metrics

Each PPO update logs:

- `reward_mean`
- `success_rate`
- `completed_episodes`
- `policy_loss`
- `value_loss`
- `entropy`
- `approx_kl`
- `reference_kl`
- `total_loss`

Treat early single-seed results as smoke tests, not scientific evidence.
Use multiple seeds and fixed eval episodes before claiming improvement.

## Failure Modes

- Missing or mismatched action stats causes invalid action scaling.
- Checkpoint/runtime state keys can mismatch the Isaac adapter state vector.
- Reward hacking can improve return while reducing true task success.
- Too small `kl_coef` can destroy the BC prior; too large can prevent learning.
- Camera observations may be blank if Isaac cameras are not enabled.
- Isaac task ids vary across Isaac Lab releases; update
  `mini_pi0/sim/isaaclab_tasks.py` when needed.

## Safety Boundary

This is simulation-only training. Do not deploy an RL-fine-tuned policy on a
real robot without action bounds, workspace limits, supervised slow rollouts,
latency checks, and emergency stop coverage.
