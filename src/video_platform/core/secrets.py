"""Secret handling rules.

Secrets must never appear as literal values in configuration files. A config key that
looks like a secret must hold either an environment reference (``${VAR}``) or a secret
reference (``secret://env/VAR``). Values are resolved at load time and wrapped in
pydantic ``SecretStr`` by the config schema so they never print.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .errors import ConfigError

_EXACT_SECRET_KEYS = {"password", "api_key", "admin_api_token", "secret", "token", "secret_access_key"}
_SECRET_SUFFIXES = ("_secret", "_password", "_token", "_api_key", "_secret_key")


def is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return lowered in _EXACT_SECRET_KEYS or lowered.endswith(_SECRET_SUFFIXES)


def is_secret_reference(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    return (stripped.startswith("${") and stripped.endswith("}")) or stripped.startswith("secret://")


def find_literal_secrets(tree: Any, path: str = "") -> list[str]:
    """Return dotted paths of secret-like keys holding literal (non-reference) values."""
    offenders: list[str] = []
    if isinstance(tree, Mapping):
        for key, value in tree.items():
            child = f"{path}.{key}" if path else str(key)
            if is_secret_key(str(key)) and value not in (None, "") and not isinstance(value, (Mapping, list)):
                if not is_secret_reference(value):
                    offenders.append(child)
            offenders.extend(find_literal_secrets(value, child))
    elif isinstance(tree, list):
        for index, item in enumerate(tree):
            offenders.extend(find_literal_secrets(item, f"{path}[{index}]"))
    return offenders


def resolve_secret_uri(uri: str, env: Mapping[str, str]) -> str:
    """Resolve ``secret://<backend>/<name>``. Only the ``env`` backend exists in M1."""
    remainder = uri[len("secret://"):]
    backend, _, name = remainder.partition("/")
    if backend != "env" or not name:
        raise ConfigError(
            f"Unsupported secret reference '{uri.split('/')[0]}//{backend}/…'. "
            "Supported: secret://env/<VAR> (other backends are planned for later milestones)."
        )
    if name not in env:
        raise ConfigError(f"Secret reference secret://env/{name} is not set in the environment.")
    return env[name]
