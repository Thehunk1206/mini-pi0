# Simulation Environment

This repo supports three simulator backends through one adapter interface:

- `robosuite`: full runtime support
- `maniskill3`: implemented custom task + dataset collection
- `isaaclab`: scaffold only

Adapter contract is defined in `mini_pi0/sim/base.py` and registry wiring is in `mini_pi0/sim/registry.py`.

## ManiSkill3 Custom Task

Implemented task id:

- `MiniPi0MultiObjectTray-v1`

Task behavior:

- Sample object count uniformly from `1..10`
- Object types cycle through `cube`, `sphere`, `cone` (cone is a stable primitive proxy)
- Robot picks each object from table and places in tray
- Per-step progress signal: `success_fraction = placed_count / total_objects` in `[0, 1]`
- Boolean success compatibility: `success = (success_fraction == 1.0)`

Core files:

- `mini_pi0/sim/maniskill3_custom_env.py`
- `mini_pi0/sim/maniskill3_adapter.py`
- `mini_pi0/dataset/maniskill_collect.py`
- `examples/configs/maniskill3_multiobject_tray.yaml`

## Reward Design

Per-step reward:

`r_t = r_progress + r_place + r_terminal + r_shaping - r_penalties - r_step`

Logged terms in `info` every step:

- `reward_total`
- `reward_progress`
- `reward_place`
- `reward_terminal`
- `reward_shaping`
- `reward_penalties`
- `reward_step`
- `success_fraction`
- `placed_count`
- `total_objects`

Defaults are configured in `RewardWeights` in `mini_pi0/sim/maniskill3_custom_env.py`.

## Install + Backend Check

```bash
uv sync --extra dev --extra lerobot --extra vision --extra hardware
.venv/bin/python -m pip install mani_skill
.venv/bin/python -m mini_pi0 backends
```

## Smoke Run (ManiSkill backend)

```bash
.venv/bin/python -m mini_pi0 eval \
  --config examples/configs/maniskill3_multiobject_tray.yaml \
  --set simulator.backend=maniskill3 \
  --set simulator.task=MiniPi0MultiObjectTray-v1 \
  --set eval.n_episodes=1 \
  --set eval.max_steps=50
```

## Dataset Collection (robomimic-style HDF5)

```bash
.venv/bin/python -m mini_pi0 collect-maniskill-demos \
  --config examples/configs/maniskill3_multiobject_tray.yaml \
  --task MiniPi0MultiObjectTray-v1 \
  --num_episodes 50 \
  --max_steps 300 \
  --out_hdf5 data/robomimic/custom/maniskill3_multiobject/ph/low_dim_v15.hdf5 \
  --overwrite
```

HDF5 layout is robomimic-style under `/data/demo_k` with:

- `actions`, `rewards`, `dones`
- `obs/*` canonical keys
- `info/*` reward/progress traces
- attrs: `success_bool`, `placed_count`, `total_objects`, `final_success_fraction`

