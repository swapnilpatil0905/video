"""Network policy: keep the project on public sources only.

Packages, images and weights come from public sources (pypi.org, docker.io, ghcr.io, nvcr.io,
quay.io, huggingface.co, github.com). Hosts you must avoid (e.g. an employer's internal
mirrors) are listed in ``packages.forbid_hosts_matching``.

This module never edits machine-wide configuration. It *detects* forbidden settings and
fails fast; ``run.sh`` overrides package indexes for this project only.
"""

from __future__ import annotations

import fnmatch
import os
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from urllib.parse import urlparse

from .errors import PolicyViolation

DEFAULT_FORBIDDEN_HOST_PATTERNS: tuple[str, ...] = ()

_INDEX_ENV_VARS = (
    "PIP_INDEX_URL",
    "PIP_EXTRA_INDEX_URL",
    "UV_INDEX_URL",
    "UV_DEFAULT_INDEX",
    "UV_EXTRA_INDEX_URL",
    "UV_INDEX",
    "HF_ENDPOINT",
)


def host_of(url_or_host: str) -> str:
    """Extract a lowercase hostname from a URL, a ``host:port`` or a bare registry name."""
    candidate = url_or_host.strip()
    if "://" not in candidate:
        candidate = "//" + candidate
    return (urlparse(candidate).hostname or "").lower()


def is_forbidden_host(host: str, patterns: Sequence[str] = DEFAULT_FORBIDDEN_HOST_PATTERNS) -> bool:
    host = host.lower().rstrip(".")
    return bool(host) and any(fnmatch.fnmatch(host, pattern.lower()) for pattern in patterns)


def assert_allowed(url_or_host: str, patterns: Sequence[str] = DEFAULT_FORBIDDEN_HOST_PATTERNS, *, what: str = "URL") -> None:
    host = host_of(url_or_host)
    if is_forbidden_host(host, patterns):
        raise PolicyViolation(f"{what} '{url_or_host}' uses forbidden host '{host}' (project policy: no {', '.join(patterns)}).")


@dataclass(frozen=True)
class Finding:
    check: str
    status: str  # "ok" | "fail" | "warn"
    detail: str


def _effective_pip_index(env: Mapping[str, str]) -> tuple[str | None, str]:
    if env.get("PIP_INDEX_URL"):
        return env["PIP_INDEX_URL"], "PIP_INDEX_URL"
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "config", "get", "global.index-url"],
            capture_output=True,
            text=True,
            timeout=30,
            env=dict(env),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, "pip unavailable"
    value = result.stdout.strip()
    if result.returncode != 0 or not value:
        return "https://pypi.org/simple", "pip default"
    return value, "pip config"


def preflight(
    *,
    patterns: Sequence[str] = DEFAULT_FORBIDDEN_HOST_PATTERNS,
    configured_urls: Iterable[tuple[str, str]] = (),
    env: Mapping[str, str] | None = None,
    check_pip: bool = True,
) -> list[Finding]:
    """Inspect the environment and configured endpoints for forbidden hosts."""
    env = dict(os.environ if env is None else env)
    findings: list[Finding] = []

    for var in _INDEX_ENV_VARS:
        value = env.get(var)
        if not value:
            continue
        for part in value.split():
            if is_forbidden_host(host_of(part), patterns):
                findings.append(Finding(f"env:{var}", "fail", f"{var} points at forbidden host {host_of(part)}"))
                break
        else:
            findings.append(Finding(f"env:{var}", "ok", f"{var}={value}"))

    if check_pip:
        index, source = _effective_pip_index(env)
        if index is None:
            findings.append(Finding("pip:index", "warn", "pip not available in this interpreter"))
        elif is_forbidden_host(host_of(index), patterns):
            findings.append(
                Finding(
                    "pip:index",
                    "fail",
                    f"effective pip index ({source}) is {index}. Run via ./run.sh, which sets PIP_CONFIG_FILE=./pip.conf "
                    "and PIP_INDEX_URL=https://pypi.org/simple for this project only.",
                )
            )
        else:
            findings.append(Finding("pip:index", "ok", f"effective pip index ({source}) is {index}"))

    for label, url in configured_urls:
        if not url:
            continue
        status = "fail" if is_forbidden_host(host_of(url), patterns) else "ok"
        findings.append(Finding(f"config:{label}", status, url))

    return findings
