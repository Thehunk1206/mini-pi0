# Isaac Lab Docker Runtime

This repo runs Isaac Sim/Lab through Docker instead of installing Isaac on the
host. The host still provides the NVIDIA driver and NVIDIA container runtime,
but `mini-pi0` does not install or modify drivers, CUDA, Isaac Sim, or Isaac
Lab outside the container.

## Requirements

- Docker with the NVIDIA runtime.
- Access to `nvcr.io/nvidia/isaac-lab`.
- EULA and privacy consent passed as runtime environment variables.
- A host NVIDIA driver compatible with the selected Isaac Lab container.

The default image is pinned in `.env.example`:

```text
ISAAC_LAB_IMAGE=nvcr.io/nvidia/isaac-lab:2.3.0
```

If the host driver is incompatible, Docker/Isaac will fail at container runtime;
the repo does not attempt any host-side fix.

## Build

```bash
cp .env.example .env
docker compose -f compose.isaaclab.yaml build isaaclab
```

The Dockerfile is intentionally thin. It starts from the official Isaac Lab
image and installs this repository editable at container startup:

```text
python -m pip install -e ".[vision]"
```

Set `MINI_PI0_INSTALL_EDITABLE=0` only if the image already contains the desired
repo install.

## Cache Volumes

`compose.isaaclab.yaml` bind-mounts cache paths under `.cache/isaaclab/`:

- `pip`
- `hf`
- `ov`
- `kit`
- `computecache`
- `logs`

These paths are ignored by git. Keeping them persistent avoids repeated
downloads and shader/cache rebuilds.

## Smoke Test

```bash
docker compose -f compose.isaaclab.yaml run --rm isaaclab mini-pi0 backends

docker compose -f compose.isaaclab.yaml run --rm isaaclab \
  mini-pi0 isaac-smoke --config examples/configs/isaaclab_franka_lift.yaml
```

`isaac-smoke` launches Isaac Lab headlessly, resolves the configured task, runs
one reset, applies one zero action, and reports observation keys, action
dimension, reward, done, and success.

## Headless PPO Fine-Tuning

```bash
docker compose -f compose.isaaclab.yaml run --rm isaaclab \
  mini-pi0 rl-train --config examples/configs/isaaclab_franka_lift_ppo.yaml \
  --checkpoint runs/<bc-run>/checkpoints/best.pt \
  --action_stats runs/<bc-run>/artifacts/action_stats.json
```

The command writes:

- `runs/<experiment>-rl/runN/config_resolved.yaml`
- `runs/<experiment>-rl/runN/metrics/rl_metrics.jsonl`
- `runs/<experiment>-rl/runN/metrics/rl_summary.json`
- `runs/<experiment>-rl/runN/checkpoints/latest_rl.pt`
- `runs/<experiment>-rl/runN/checkpoints/best_rl.pt`

## Notes

- The first supported task key is `franka_lift_cube`.
- Broader task keys are registered in `mini_pi0/sim/isaaclab_tasks.py`.
- GUI/WebRTC is intentionally out of scope for the first Docker path.
- Isaac-dependent tests should run inside Docker; host tests mock or skip Isaac.
