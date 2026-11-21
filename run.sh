#!/usr/bin/env bash
# Wrapper for run.py.
#  - Forces public package indexes for this project only (the machine-wide pip/uv
#    configuration may point at an internal mirror; it is overridden here, not edited).
#  - Uses the project virtualenv (.venv-platform) if present.
#
# Usage:
#   ./run.sh --bootstrap                       # create .venv-platform from public PyPI
#   ./run.sh --project projects/project_001/config.yaml --validate-only
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export PIP_CONFIG_FILE="$ROOT/pip.conf"
export PIP_INDEX_URL="https://pypi.org/simple"
unset PIP_EXTRA_INDEX_URL || true
export UV_INDEX_URL="https://pypi.org/simple"
export UV_DEFAULT_INDEX="https://pypi.org/simple"
unset UV_EXTRA_INDEX_URL || true
export UV_CONFIG_FILE="$ROOT/uv.toml"

VENV="$ROOT/.venv-platform"
BASE_PYTHON="${VP_BASE_PYTHON:-python3.11}"

if [[ "${1:-}" == "--bootstrap" ]]; then
    "$BASE_PYTHON" -m venv "$VENV"
    "$VENV/bin/python" -m pip install --upgrade pip
    "$VENV/bin/python" -m pip install -e ".[dev]"
    echo "Bootstrapped $VENV from https://pypi.org/simple"
    exit 0
fi

if [[ "${1:-}" == "--test" ]]; then
    shift
    if "$VENV/bin/python" -c "import pytest" 2>/dev/null; then
        exec "$VENV/bin/python" -m pytest "$@"
    fi
    PYTHONPATH="$PWD/src" exec "$VENV/bin/python" scripts/minitest.py "$@"   # no pytest available: built-in runner
fi

PYTHON="${VP_PYTHON:-$VENV/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo "Project environment not found. Run: ./run.sh --bootstrap" >&2
    exit 2
fi

exec "$PYTHON" "$ROOT/run.py" "$@"
