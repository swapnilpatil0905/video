#!/usr/bin/env bash
# Milestone 1 verification on a Vast.ai instance (or any Linux box with internet).
#
# Recommended image: docker.io/library/python:3.11-bookworm (any cheap instance; no GPU needed for M1).
#
# Usage (on the instance, from the copied project directory):
#   bash scripts/vast_verify.sh                 # tests + preflight + registry licence check
#   bash scripts/vast_verify.sh --with-legacy   # also migrate the legacy workspace (copy the data dirs too)
#
# Produces verification_report/ — copy it back to the Mac and share the summary.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
chmod +x run.sh run.py scripts/*.sh 2>/dev/null || true
REPORT="$ROOT/verification_report"
mkdir -p "$REPORT"
WITH_LEGACY=0
[[ "${1:-}" == "--with-legacy" ]] && WITH_LEGACY=1

log() { echo "== $*" | tee -a "$REPORT/summary.txt"; }
: > "$REPORT/summary.txt"

# ---- interpreter (>= 3.11) -------------------------------------------------
PY=""
for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
        PY="$(command -v "$candidate")"; break
    fi
done
if [[ -z "$PY" ]]; then log "FAIL: need Python >= 3.11 (use image python:3.11-bookworm)"; exit 2; fi
log "python: $PY ($("$PY" --version 2>&1))"

# ---- ffprobe (only needed for --with-legacy) ---------------------------------
if [[ $WITH_LEGACY -eq 1 ]] && ! command -v ffprobe >/dev/null 2>&1; then
    if [[ $(id -u) -eq 0 ]] && command -v apt-get >/dev/null; then
        apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null
    fi
    command -v ffprobe >/dev/null || { log "FAIL: ffprobe missing (apt-get install ffmpeg)"; exit 2; }
fi

# ---- environment from public PyPI --------------------------------------------
export VP_BASE_PYTHON="$PY"
rm -rf .venv-platform
./run.sh --bootstrap > "$REPORT/bootstrap.log" 2>&1 && log "bootstrap: ok" || { log "bootstrap: FAILED (see bootstrap.log)"; exit 2; }
"$ROOT/.venv-platform/bin/python" -m pip freeze > "$REPORT/pip_freeze.txt"

# ---- checks -------------------------------------------------------------------
run_step() {  # name, command...
    local name="$1"; shift
    if "$@" > "$REPORT/$name.log" 2>&1; then log "$name: ok"; else log "$name: FAILED (exit $?, see $name.log)"; fi
}

run_step preflight ./run.sh --preflight
run_step tests ./run.sh --test -q --junitxml="$REPORT/junit.xml"
tail -n 3 "$REPORT/tests.log" | sed 's/^/   /' | tee -a "$REPORT/summary.txt"
run_step config_schema ./run.sh --config-schema
run_step init_template ./run.sh --init /tmp/vp_check/projects/project_check
run_step verify_registry ./run.sh --verify-registry
cp -f model_registry/_verification.json "$REPORT/" 2>/dev/null || true

if [[ $WITH_LEGACY -eq 1 ]]; then
    P=projects/project_001/config.yaml
    run_step validate_project_001 ./run.sh --project $P --validate-only
    run_step migrate_legacy ./run.sh --project $P --migrate-legacy
    run_step mode_analyze ./run.sh --project $P --stage speaker_mapping_prior
    run_step verify_manifest ./run.sh --project $P --verify-manifest
    run_step status_json ./run.sh --project $P --status --json
    grep -q "Nothing to do" "$REPORT/mode_analyze.log" \
        && log "ACCEPTANCE: re-running analysis triggers zero work" \
        || log "ACCEPTANCE: analysis would re-run stages (see mode_analyze.log)"
fi

log "report: $REPORT"
