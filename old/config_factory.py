from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import tomllib

from old.models import MigrationConfig

GIB = 1024 * 1024 * 1024
DEFAULT_OVERRIDE_ENV = "SYNC_MIGRATION_OVERRIDES"


@dataclass
class DerivationDetails:
    applied_overrides: Dict[str, Any]
    override_errors: list[str]
    override_sources: list[str]
    notes: list[str]
    derived_values: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "applied_overrides": self.applied_overrides,
            "override_errors": self.override_errors,
            "override_sources": self.override_sources,
            "notes": self.notes,
            "derived": self.derived_values,
        }


def derive_migration_config(
    *,
    source_root: Path,
    sync_root: Path,
    max_local_gb: float,
    migration_subdir: str = "external_migration",
    overrides: Optional[Dict[str, Any]] = None,
    override_env: str = DEFAULT_OVERRIDE_ENV,
) -> Tuple[MigrationConfig, DerivationDetails]:
    """
    Build a minimal MigrationConfig for the current engine.

    Only a subset of the original knobs are still relevant (max_local_bytes,
    migration_subdir, event queue sizing). Everything else falls back to the
    dataclass defaults on MigrationConfig.
    """
    override_sources: list[str] = []
    env_overrides, env_errors, env_source = load_overrides_from_env(override_env)
    override_errors = list(env_errors)
    if env_overrides or env_errors:
        override_sources.append(env_source or f"env:{override_env}")

    combined_overrides: Dict[str, Any] = {}
    combined_overrides.update(env_overrides)
    if overrides is not None:
        if not isinstance(overrides, dict):
            override_errors.append("Overrides argument must be a mapping/dict.")
        else:
            combined_overrides.update(overrides)
            override_sources.append("function overrides")

    applied_overrides: Dict[str, Any] = {}
    max_local_gb = _consume_float_override(
        combined_overrides, "max_local_gb", default=max_local_gb, applied=applied_overrides, errors=override_errors
    )
    migration_subdir = _consume_str_override(
        combined_overrides, "migration_subdir", default=migration_subdir, applied=applied_overrides
    )
    max_local_bytes_override = _consume_int_override(
        combined_overrides, "max_local_bytes", default=None, applied=applied_overrides, errors=override_errors, min_value=1
    )
    event_queue_size = _consume_int_override(
        combined_overrides,
        "event_queue_size",
        default=MigrationConfig.event_queue_size,
        applied=applied_overrides,
        errors=override_errors,
        min_value=0,
    )

    if max_local_bytes_override is not None:
        max_local_bytes = max_local_bytes_override
    else:
        max_local_bytes = max(int(max_local_gb * GIB), 1)

    config = MigrationConfig(
        source_root=source_root,
        sync_root=sync_root,
        migration_subdir=migration_subdir,
        max_local_bytes=max_local_bytes,
        event_queue_size=event_queue_size,
    )

    derived_values = {
        "max_local_gb": max_local_gb,
        "max_local_bytes": max_local_bytes,
        "migration_subdir": migration_subdir,
        "event_queue_size": event_queue_size,
    }
    details = DerivationDetails(
        applied_overrides=applied_overrides,
        override_errors=override_errors,
        override_sources=override_sources,
        notes=[],
        derived_values=derived_values,
    )
    return config, details


def load_overrides_from_env(env_var: str = DEFAULT_OVERRIDE_ENV) -> Tuple[Dict[str, Any], list[str], Optional[str]]:
    """
    Load override mapping from an env var pointing to a file path or inline JSON.

    Returns (overrides, errors, source).
    """
    raw = os.getenv(env_var)
    if raw is None or not raw.strip():
        return {}, [], None
    raw = raw.strip()
    path = Path(raw)
    errors: list[str] = []
    content = raw
    source = f"env:{env_var}"
    hint_ext: Optional[str] = None

    if path.exists() and path.is_file():
        try:
            content = path.read_text()
            hint_ext = path.suffix.lower()
            source = str(path)
        except OSError as exc:  # noqa: BLE001
            return {}, [f"Failed to read overrides from {path}: {exc}"], str(path)

    overrides, parse_errors = _parse_overrides(content, hint_ext)
    errors.extend(parse_errors)
    return overrides, errors, source


def _parse_overrides(content: str, hint_ext: Optional[str]) -> Tuple[Dict[str, Any], list[str]]:
    errors: list[str] = []
    data: Dict[str, Any] = {}

    ext = hint_ext or ""
    cleaned = content.strip()
    if not cleaned:
        return data, errors

    if not ext or ext == ".json":
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed, errors
            errors.append("Overrides JSON must decode to an object.")
        except json.JSONDecodeError as exc:
            errors.append(f"Failed to parse JSON overrides: {exc}")
        if not ext:
            return data, errors

    if ext in (".toml", ".tml"):
        try:
            parsed = tomllib.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed, errors
            errors.append("Overrides TOML must decode to a table/object.")
        except tomllib.TOMLDecodeError as exc:
            errors.append(f"Failed to parse TOML overrides: {exc}")
        return data, errors

    if ext in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Cannot parse YAML overrides (PyYAML missing?): {exc}")
            return data, errors
        try:
            parsed = yaml.safe_load(cleaned)
            if isinstance(parsed, dict):
                return parsed, errors
            errors.append("Overrides YAML must decode to a mapping/object.")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Failed to parse YAML overrides: {exc}")
        return data, errors

    errors.append(f"Unsupported override format: {ext or 'unknown'}")
    return data, errors


def _consume_float_override(
    overrides: Dict[str, Any],
    key: str,
    default: float,
    applied: Dict[str, Any],
    errors: list[str],
) -> float:
    if key not in overrides:
        return default
    try:
        value = float(overrides[key])
        applied[key] = value
        return value
    except (TypeError, ValueError):
        errors.append(f"{key} override must be numeric.")
        return default


def _consume_str_override(
    overrides: Dict[str, Any],
    key: str,
    default: str,
    applied: Dict[str, Any],
) -> str:
    if key not in overrides:
        return default
    value = str(overrides[key])
    applied[key] = value
    return value


def _consume_int_override(
    overrides: Dict[str, Any],
    key: str,
    default: Optional[int],
    applied: Dict[str, Any],
    errors: list[str],
    *,
    min_value: int = 1,
) -> Optional[int]:
    if key not in overrides:
        return default
    value = overrides[key]
    try:
        int_value = int(value)
    except (TypeError, ValueError):
        errors.append(f"{key} override must be an integer.")
        return default
    if int_value < min_value:
        errors.append(f"{key} override must be >= {min_value}.")
        return default
    applied[key] = int_value
    return int_value
