from __future__ import annotations

import ast
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_origin, get_type_hints

import yaml

from mini_pi0.config.schema import RootConfig, to_dict

T = TypeVar("T")


def _deep_update(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries.

    Args:
        base: Base dictionary that provides default values.
        patch: Override dictionary whose values take precedence.

    Returns:
        A merged dictionary where nested dictionaries are merged recursively.
    """

    out = dict(base)
    for k, v in patch.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _parse_value(text: str) -> Any:
    """Parse CLI override text into a Python value.

    Args:
        text: Raw string from ``--set key=value``.

    Returns:
        Parsed Python value (bool / None / literal / original string).
    """

    low = text.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def _apply_override(d: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Apply one dotted-key override into a nested dictionary in-place.

    Args:
        d: Target dictionary to mutate.
        dotted_key: Dotted key path such as ``train.epochs``.
        value: Parsed value assigned at the final key.
    """

    parts = dotted_key.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _normalize_dataclass_type(field_type: Any) -> Any:
    """Normalize optional dataclass typing hints for recursive construction.

    Args:
        field_type: Raw type hint from dataclass field annotations.

    Returns:
        A dataclass type when discoverable inside Optional/Union, otherwise the
        original hint.
    """

    origin = get_origin(field_type)
    if origin is None:
        return field_type
    # Handle Optional[T] / Union[T, None]
    if origin is list:
        return field_type
    if origin is dict:
        return field_type
    args = getattr(field_type, "__args__", ())
    for a in args:
        if a is type(None):
            continue
        if is_dataclass(a):
            return a
    return field_type


def _dataclass_from_dict(cls: type[T], data: dict[str, Any]) -> T:
    """Recursively instantiate a dataclass from nested dictionaries.

    Args:
        cls: Dataclass type to instantiate.
        data: Mapping of field names to values.

    Returns:
        Instance of ``cls`` with nested dataclasses created recursively.
    """

    if not is_dataclass(cls):
        return data  # type: ignore[return-value]

    hints = get_type_hints(cls)
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        val = data[f.name]
        t = _normalize_dataclass_type(hints.get(f.name, f.type))
        if is_dataclass(t) and isinstance(val, dict):
            kwargs[f.name] = _dataclass_from_dict(t, val)
        else:
            kwargs[f.name] = val
    return cls(**kwargs)  # type: ignore[arg-type]


def apply_overrides(base: dict[str, Any], overrides: list[str] | None = None) -> dict[str, Any]:
    """Apply ``key=value`` override strings onto a base configuration dict.

    Args:
        base: Base configuration dictionary.
        overrides: Optional list of dotted-key overrides from CLI.

    Returns:
        Updated configuration dictionary.

    Raises:
        ValueError: If an override does not follow ``key=value`` format.
    """

    out = dict(base)
    if overrides:
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"Invalid override '{item}'. Expected key=value format.")
            k, v = item.split("=", 1)
            _apply_override(out, k, _parse_value(v))
    return out


def load_config(config_path: str | None = None, overrides: list[str] | None = None) -> RootConfig:
    """Load, merge, and validate repository configuration.

    Args:
        config_path: Optional YAML path. If ``None``, only defaults + overrides are used.
        overrides: Optional CLI overrides in dotted ``key=value`` format.

    Returns:
        Fully-resolved typed ``RootConfig`` object.

    Raises:
        ValueError: If YAML content is not a mapping.
    """

    cfg_dict: dict[str, Any] = {}
    if config_path:
        with open(config_path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"Config at {config_path} must be a YAML mapping.")
            cfg_dict = loaded

    cfg_dict = apply_overrides(cfg_dict, overrides)
    default = RootConfig()
    merged = _deep_update(to_dict(default), cfg_dict)
    return _dataclass_from_dict(RootConfig, merged)


def dump_config(path: str | Path, cfg: RootConfig) -> None:
    """Persist resolved configuration to YAML.

    Args:
        path: Destination YAML path.
        cfg: Root config object to serialize.
    """

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(to_dict(cfg), f, sort_keys=False)
