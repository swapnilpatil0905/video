"""Load, merge, interpolate and validate project configuration.

Order of precedence (lowest → highest):
    configs/default.yaml  →  projects/<id>/config.yaml  →  --email  →  --set key=value
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from ..core.errors import ConfigError
from ..core.secrets import find_literal_secrets, resolve_secret_uri
from .schema import ProjectConfig

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings. Lists and scalars in ``override`` replace those in ``base``."""
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def apply_overrides(tree: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Apply ``dotted.key=value`` overrides; values are parsed as YAML scalars/collections."""
    result = deepcopy(tree)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"--set expects key=value, got '{item}'")
        dotted, raw_value = item.split("=", 1)
        keys = [k for k in dotted.strip().split(".") if k]
        if not keys:
            raise ConfigError(f"--set has an empty key in '{item}'")
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError as exc:
            raise ConfigError(f"--set value for '{dotted}' is not valid YAML: {exc}") from exc
        node = result
        for key in keys[:-1]:
            existing = node.get(key)
            if existing is None:
                node[key] = {}
            elif not isinstance(existing, dict):
                raise ConfigError(f"--set '{dotted}': '{key}' is not a mapping")
            node = node[key]
        node[keys[-1]] = value
    return result


def interpolate(tree: Any, env: Mapping[str, str], missing: list[str] | None = None, path: str = "") -> Any:
    """Resolve ``${VAR}``, ``${VAR:-default}`` and ``secret://env/VAR`` references."""
    if missing is None:
        missing = []
    if isinstance(tree, dict):
        return {k: interpolate(v, env, missing, f"{path}.{k}" if path else str(k)) for k, v in tree.items()}
    if isinstance(tree, list):
        return [interpolate(v, env, missing, f"{path}[{i}]") for i, v in enumerate(tree)]
    if not isinstance(tree, str):
        return tree
    if tree.startswith("secret://"):
        try:
            return resolve_secret_uri(tree, env)
        except ConfigError:
            missing.append(f"{path} ({tree})")
            return None

    def _replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        if name in env:
            return env[name]
        if default is not None:
            return default
        missing.append(f"{path} (${{{name}}})")
        return ""

    return _ENV_REF.sub(_replace, tree)


def _format_validation_error(error: ValidationError) -> str:
    lines = []
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"]) or "<root>"
        lines.append(f"  - {location}: {item['msg']}")
    return "\n".join(lines)


def find_platform_root(start: Path | None = None) -> Path:
    """Directory containing ``configs/pipeline.yaml``.

    Search order: $VP_PLATFORM_ROOT, the parents of ``start``, then the source checkout.
    """
    env_root = os.environ.get("VP_PLATFORM_ROOT")
    if env_root:
        return Path(env_root).resolve()
    candidates = []
    if start is not None:
        candidates.extend([start.resolve(), *start.resolve().parents])
    candidates.extend(Path(__file__).resolve().parents)
    for candidate in candidates:
        if (candidate / "configs" / "pipeline.yaml").exists():
            return candidate
    raise ConfigError("Could not locate the platform root (a directory containing configs/pipeline.yaml).")


@dataclass(frozen=True)
class LoadedConfig:
    config: ProjectConfig
    config_path: Path
    project_dir: Path
    platform_root: Path

    @property
    def project_id(self) -> str:
        return self.config.project.id

    @property
    def source_video_path(self) -> Path:
        path = Path(self.config.project.source_video)
        return path if path.is_absolute() else (self.project_dir / path).resolve()

    @property
    def artifact_root(self) -> Path:
        root = Path(self.config.storage.root)
        return root if root.is_absolute() else self.project_dir / root

    @property
    def state_dir(self) -> Path:
        return self.project_dir / "state"


def load_project_config(
    config_path: str | Path,
    *,
    overrides: Sequence[str] = (),
    email: str | None = None,
    env: Mapping[str, str] | None = None,
    platform_root: Path | None = None,
) -> LoadedConfig:
    config_path = Path(config_path).resolve()
    env = dict(os.environ if env is None else env)
    root = platform_root or find_platform_root(config_path.parent)

    default_path = root / "configs" / "default.yaml"
    base = load_yaml(default_path) if default_path.exists() else {}
    project_raw = load_yaml(config_path)
    merged = deep_merge(base, project_raw)

    if email is not None:
        notifications = merged.setdefault("notifications", {})
        if not isinstance(notifications, dict):
            raise ConfigError("notifications must be a mapping")
        notifications["approval_email"] = email

    merged = apply_overrides(merged, overrides)

    literal = find_literal_secrets(merged)
    if literal:
        raise ConfigError(
            "Secrets must not be stored literally in configuration. Use ${ENV_VAR} or secret://env/NAME for: "
            + ", ".join(literal)
        )

    missing: list[str] = []
    resolved = interpolate(merged, env, missing)
    if missing:
        raise ConfigError("Missing environment variables for configuration values:\n  - " + "\n  - ".join(missing))

    try:
        config = ProjectConfig.model_validate(resolved)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration in {config_path}:\n{_format_validation_error(exc)}") from exc

    return LoadedConfig(config=config, config_path=config_path, project_dir=config_path.parent, platform_root=root)
