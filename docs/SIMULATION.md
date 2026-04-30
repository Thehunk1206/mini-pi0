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
  --collector_name mini_pi0_multiobject_tray \
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

## Collector Plugins (Task-Specific)

The data-collection entrypoint now resolves a task-specific collector plugin.

Selection precedence:

1. `--collector_name` if provided
2. auto-resolve by `simulator.task` via plugin `supports(cfg)`

Current built-in collector:

- `mini_pi0_multiobject_tray` (for `MiniPi0MultiObjectTray-v1`)

### Adding a new collector

Starter template is provided at:

- `mini_pi0/dataset/maniskill_collectors/plugins/new_task_template.py`

Quick scaffold flow:

```bash
cp mini_pi0/dataset/maniskill_collectors/plugins/new_task_template.py \
   mini_pi0/dataset/maniskill_collectors/plugins/my_new_task.py
```

1. Edit `my_new_task.py`: rename class, set `name`, set `supported_tasks`, implement collect methods.
2. Implement the collector interface:
   - `name`
   - `supports(cfg)`
   - `collect_episode(req)`
   - `collect_vectorized(req, episodes_target)`
   - `finalize_episode(final_info)`
3. Register it with `register_collector(...)`.
4. Import the plugin in `mini_pi0/dataset/maniskill_collectors/__init__.py` for side-effect registration.

Required output fields in `finalize_episode(...)`:

- `success` (bool)
- `success_fraction` (float)
- `placed_count` (int)
- `total_objects` (int)

Minimal registration pattern inside your new plugin:

```python
from mini_pi0.dataset.maniskill_collectors.registry import register_collector

register_collector(MyNewTaskCollector())
```

Import pattern in `mini_pi0/dataset/maniskill_collectors/__init__.py`:

```python
from .plugins import my_new_task as _my_new_task  # noqa: F401
```

Validation command with explicit collector selection:

```bash
.venv/bin/python -m mini_pi0 collect-maniskill-demos \
  --config examples/configs/maniskill3_multiobject_tray.yaml \
  --task ReplaceTask-v1 \
  --collector_name replace_with_collector_name \
  --collector_backend scripted \
  --num_envs 1 \
  --num_episodes 2 \
  --max_steps 80 \
  --no-only_success \
  --overwrite \
  --out_hdf5 artifacts/smoke_new_task_collector.hdf5
```

## Collector Architecture (Detailed)

This section documents how the new collector system is structured internally.

### 1) Orchestrator (`maniskill_collect.py`)

`collect_maniskill_demos(...)` is now orchestration-only:

- resolves collector plugin (`resolve_collector(...)`)
- creates HDF5 file and `data` group
- handles seeding/retry loop and stop conditions
- delegates rollout work to plugin methods
- writes episodes with shared writer (`write_episode(...)`)
- computes summary stats (`summarize_collection_stats(...)`)

It does **not** contain task-specific policy logic anymore.

### 2) Plugin Interface (`interfaces.py`)

Task behavior is defined through:

- `CollectorRequest`
  - normalized runtime inputs passed to plugins:
    - `cfg`, `image_keys`, `state_keys`, `num_envs`, `max_steps`, `only_success`, `backend`
- `TaskCollector` protocol
  - `name`: stable collector id used by CLI
  - `supports(cfg)`: auto-resolution predicate
  - `collect_episode(req)`: single-env episode collection
  - `collect_vectorized(req, episodes_target)`: batched env collection
  - `finalize_episode(final_info)`: normalize task metrics before write

### 3) Registry (`registry.py`)

Registry is explicit and fail-fast:

- `register_collector(collector)`: validates non-empty unique `name`
- `get_collector(name)`: explicit lookup by `--collector_name`
- `resolve_collector(cfg, collector_name)`:
  - explicit name wins
  - otherwise, find plugin where `supports(cfg)` is true
  - error on zero matches (no collector)
  - error on multiple matches (ambiguous collector)
- `list_collectors()`: introspection helper

### 4) Plugin Implementation (`plugins/multi_object_tray.py`)

Current plugin:

- `name = "mini_pi0_multiobject_tray"`
- `supports(...)` matches task aliases for `MiniPi0MultiObjectTray-v1`
- `collect_episode(...)`:
  - uses `req.backend`
  - `mplib` path if runtime check passes
  - otherwise scripted backend
- `collect_vectorized(...)`:
  - scripted-first vectorized backend for stability
- `finalize_episode(...)`:
  - guarantees required metrics exist with defaults

Registration is done at module import:

- `register_collector(MiniPi0MultiObjectTrayCollector())`

### 5) Backend Helpers (`backends.py`)

Backends are reusable low-level rollout executors:

- `collect_single_scripted_episode(...)`
- `collect_vectorized_scripted_episodes(...)`
- `collect_single_mplib_episode(...)`
- `mplib_runtime_check()`

Policy/backends remain task-specific for now, but they are now isolated behind plugin calls.

### 6) CLI Contract (`mini_pi0/cli/main.py`)

`collect-maniskill-demos` now exposes:

- `--collector_name`: explicit task collector selection
- `--collector_backend`: plugin backend hint (`scripted` / `mplib`)

Backward compatibility is preserved:

- if `--collector_name` is omitted, collector is inferred from `simulator.task`

### 7) HDF5 Schema Ownership

The schema is still centralized in shared writer path:

- per-demo datasets:
  - `actions`, `rewards`, `dones`, `obs/*`, `info/*`
- per-demo attrs:
  - `num_samples`, `collector_type`
  - `success_bool`, `final_success_fraction`, `placed_count`, `total_objects`

This means different plugins can produce data uniformly for the same downstream loader.

### 8) Call Flow (End-to-End)

1. CLI parses args and loads config.
2. Orchestrator resolves plugin.
3. Orchestrator builds `CollectorRequest`.
4. Plugin collects episode(s) using selected backend.
5. Plugin normalizes final info in `finalize_episode(...)`.
6. Orchestrator filters by `only_success` (if enabled).
7. Orchestrator writes episode and accumulates stats.
8. Orchestrator returns JSON summary.

### 9) Minimal New-Collector Checklist

When adding a new task collector:

1. Create `plugins/<task_name>.py`.
2. Implement `TaskCollector` methods.
3. Ensure `finalize_episode(...)` always outputs required fields.
4. Register with `register_collector(...)`.
5. Import plugin module in `maniskill_collectors/__init__.py`.
6. Smoke-test both:
   - explicit `--collector_name`
   - inferred collector (no `--collector_name`)
