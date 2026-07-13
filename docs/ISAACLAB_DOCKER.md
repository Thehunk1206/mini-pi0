# Isaac Lab Docker Runtime

Isaac Sim and Isaac Lab run only inside the NVIDIA container. The repository
does not install or modify host Isaac packages, CUDA, NVIDIA drivers, or kernel
modules. Docker uses the host driver through NVIDIA Container Toolkit.

## Requirements

- Docker Compose with NVIDIA GPU support.
- Access to `nvcr.io/nvidia/isaac-lab`.
- A host driver compatible with the selected image.
- Explicit EULA and privacy consent in `.env`.

The default base image is pinned in `.env.example`:

```text
ISAAC_LAB_IMAGE=nvcr.io/nvidia/isaac-lab:2.3.0
```

Create `.env` and set the consent values before running:

```bash
cp .env.example .env
# Edit .env: ACCEPT_EULA=Y and PRIVACY_CONSENT=Y
```

If the driver and image are incompatible, the container command fails. The
project does not attempt a host-side driver or CUDA fix.

## Build and Caches

```bash
docker compose -f compose.isaaclab.yaml build isaaclab
```

The image extends NVIDIA's official Isaac Lab image. The repository is mounted
at `/workspace/mini-pi0` and installed editable by the entrypoint. Set
`MINI_PI0_INSTALL_EDITABLE=0` only when the derived image already contains the
required package version.

Persistent caches live under `.cache/isaaclab/` for pip, Hugging Face,
Omniverse, Kit, compute kernels, and logs. These directories are ignored by
Git and avoid repeated shader and package work.

## Validation Order

Run the commands in this order:

```bash
docker compose -f compose.isaaclab.yaml run --rm isaaclab mini-pi0 backends

docker compose -f compose.isaaclab.yaml run --rm isaaclab \
  mini-pi0 isaac-smoke --config examples/configs/isaaclab_franka_lift.yaml

docker compose -f compose.isaaclab.yaml run --rm isaaclab \
  mini-pi0 rl-smoke \
  --config examples/configs/isaaclab_franka_lift_reinflow_scratch.yaml

docker compose -f compose.isaaclab.yaml run --rm isaaclab \
  mini-pi0 rl-train \
  --config examples/configs/isaaclab_franka_lift_reinflow_scratch.yaml
```

The final config performs two small ReinFlow updates with four environments.
For `num_envs > 1`, one Isaac vector environment and one SimulationApp are
created. The RL algorithm and buffer contain no Isaac imports.

## Checkpoint Fine-Tuning

The optional fine-tuning config requires an Isaac-domain FM checkpoint and the
action statistics produced by that same run:

```bash
docker compose -f compose.isaaclab.yaml run --rm isaaclab \
  mini-pi0 rl-train \
  --config examples/configs/isaaclab_franka_lift_reinflow_finetune.yaml \
  --checkpoint runs/<isaac-bc-run>/checkpoints/best.pt \
  --action_stats runs/<isaac-bc-run>/artifacts/action_stats.json
```

Do not use a ManiSkill checkpoint unless its observation schema, controller,
action dimension, camera setup, and action normalization are intentionally
matched to the Isaac task.

## Deterministic Evaluation

```bash
docker compose -f compose.isaaclab.yaml run --rm isaaclab \
  mini-pi0 rl-eval \
  --config examples/configs/isaaclab_franka_lift_reinflow_scratch.yaml \
  --resume_from runs/<reinflow-run>/checkpoints/latest_rl.pt \
  --eval_episodes 20
```

Evaluation discards learned transition noise and integrates the FM velocity
field with deterministic Euler steps. Episode seeds are fixed by
`rl.eval_seed_start`.

## Outputs

Training writes:

- `config_resolved.yaml`
- `metrics/rl_metrics.jsonl`
- `metrics/rl_summary.json`
- `checkpoints/latest_rl.pt`
- `checkpoints/best_rl.pt` only after a periodic deterministic evaluation
  improves success rate

The checkpoint embeds action statistics, model/reference/critic state,
optimizer and scheduler state, RNG state, Git commit, dependency versions, and
the simulator manifest.

## Supported Task Mapping

| mini-pi0 key | Isaac Lab task |
|---|---|
| `franka_lift_cube` | `Isaac-Lift-Cube-Franka-v0` |
| `franka_stack_cube` | `Isaac-Stack-Cube-Franka-v0` |
| `franka_pick_place` | `Isaac-PickPlace-Franka-v0` when present in the image |
| `franka_peg_insertion` | `Isaac-Peg-Insertion-Franka-v0` when present in the image |

Lift is the required first gate. Other aliases depend on task availability in
the pinned Isaac Lab image.
