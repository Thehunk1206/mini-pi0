# mini-pi0 Isaac Lab Container

This Docker setup runs Isaac Sim/Lab inside NVIDIA's Isaac Lab container and
bind-mounts this repository into `/workspace/mini-pi0`.

It does not install Isaac Sim, Isaac Lab, CUDA, or NVIDIA drivers on the host.
The container uses the host NVIDIA driver through the NVIDIA container runtime.

## Build

```bash
cp .env.example .env
docker compose -f compose.isaaclab.yaml build isaaclab
```

## Smoke Test

```bash
docker compose -f compose.isaaclab.yaml run --rm isaaclab \
  mini-pi0 backends

docker compose -f compose.isaaclab.yaml run --rm isaaclab \
  mini-pi0 isaac-smoke --config examples/configs/isaaclab_franka_lift.yaml
```

## ReinFlow Scratch Smoke

```bash
docker compose -f compose.isaaclab.yaml run --rm isaaclab \
  mini-pi0 rl-smoke --config examples/configs/isaaclab_franka_lift_reinflow_scratch.yaml

docker compose -f compose.isaaclab.yaml run --rm isaaclab \
  mini-pi0 rl-train --config examples/configs/isaaclab_franka_lift_reinflow_scratch.yaml
```

The first run downloads/builds container layers and fills cache volumes. Keep
`.cache/isaaclab/` out of git.
