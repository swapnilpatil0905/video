#!/usr/bin/env bash
# ONE command on the GPU machine. Upload + unzip the bundle, then:
#
#   bash start.sh               # setup (first time) → analysis → trailers (high + balanced) → stops for review
#   bash start.sh approve balanced   # approve one trailer → full-length production with that profile
#   bash start.sh resume        # continue whatever was interrupted (used by the on-start script)
#   bash start.sh reject        # stop: no full production
#   bash start.sh status        # where things stand
#   bash start.sh stop          # stop the pipeline (safe to re-run start later; work is cached)
#   bash start.sh smoke         # quick end-to-end test on the first 3 minutes (separate project)
#   bash start.sh regen-shot SH_00042 | regen-line SEG_000123   # redo one item, then: bash start.sh
#
# Optional: put HF_TOKEN=... in .env (needed only for gated models, e.g. pyannote on new projects).
# Optional: PROJECT=projects/<id>/config.yaml to use another project.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
chmod +x run.sh run.py scripts/*.sh 2>/dev/null || true
load_env() {  # KEY=VALUE lines only (quotes optional); a bad line is reported and skipped, never executed
    local line key value n=0
    while IFS= read -r line || [[ -n "$line" ]]; do
        n=$((n + 1))
        line="${line%$'\r'}"
        [[ -z "${line// }" || "$line" =~ ^[[:space:]]*# ]] && continue
        if [[ "$line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
            key="${BASH_REMATCH[2]}"; value="${BASH_REMATCH[3]}"
            if [[ "$value" =~ ^\"(.*)\"$ || "$value" =~ ^\'(.*)\'$ ]]; then value="${BASH_REMATCH[1]}"; fi
            if [[ "$value" == *"<"*">"* ]]; then
                echo "== .env line $n: $key still has a <placeholder>; ignored" >&2; continue
            fi
            export "$key=$value"
        else
            echo "== .env line $n is not KEY=VALUE; ignored" >&2
        fi
    done < .env
}
[[ -f .env ]] && load_env
export VP_RUNTIME_ROOT="${VP_RUNTIME_ROOT:-/workspace/vp}"
# one model cache for setup, prefetch and every stage. Cloud templates often preset HF_HOME
# elsewhere (e.g. /workspace/.hf_home), which would split downloads across two caches.
export HF_HOME="${VP_HF_HOME:-$VP_RUNTIME_ROOT/hf}"

PROJECT="${PROJECT:-projects/project_001/config.yaml}"
CMD="${1:-run}"

single_instance() {  # never let two pipelines write to the same project at once
    mkdir -p "$VP_RUNTIME_ROOT"
    exec 9>"$VP_RUNTIME_ROOT/pipeline.lock"
    if ! flock -n 9; then
        echo "Another pipeline is already running. Check: bash start.sh status   Stop it: bash start.sh stop"
        exit 1
    fi
}
case "$CMD" in run|"" |approve|resume|smoke) single_instance ;; esac
run() { ./run.sh --project "$PROJECT" "$@"; }
RUN_LOG="${RUN_LOG:-/workspace/run.log}"

notify_failure() {  # best effort; stdlib-only, works even before setup
    python3 scripts/send_mail.py --subject "AI Video — run FAILED — $1" \
        --body "The pipeline stopped: $1. Fix it (or send me the log), then run: bash start.sh (finished work is reused)." \
        --log "$RUN_LOG" || true
}

unpack_cast() {  # encrypted personal cast photos (data/cast_photos.zip.enc) → input/Male Lead, input/Female Lead
    local enc=data/cast_photos.zip.enc
    [[ -f "$enc" ]] || return 0
    if [[ -d "input/Male Lead" || -d "input/Female Lead" ]] && [[ ! "$enc" -nt input/.cast_unpacked ]]; then
        return 0
    fi
    if [[ -z "${CAST_PASSPHRASE:-}" ]]; then
        echo "== cast photos are encrypted: add CAST_PASSPHRASE=... to .env to use them (skipping)"
        return 0
    fi
    mkdir -p input
    if openssl enc -d -aes-256-cbc -pbkdf2 -iter 200000 -in "$enc" -out input/.cast_photos.zip -pass env:CAST_PASSPHRASE \
            2>/dev/null && python3 -m zipfile -e input/.cast_photos.zip input/; then
        touch input/.cast_unpacked
        echo "== cast photos unpacked into input/ ($(ls "input/Male Lead" 2>/dev/null | wc -l | tr -d ' ') male, $(ls "input/Female Lead" 2>/dev/null | wc -l | tr -d ' ') female)"
    else
        echo "== could not decrypt $enc (wrong CAST_PASSPHRASE?) — continuing without cast photos"
    fi
    rm -f input/.cast_photos.zip
}

setup() {
    unpack_cast
    bash scripts/setup_vast.sh || { notify_failure "setup failed (see /workspace/vp/logs/setup_*.log)";
                                    echo "Setup failed — fix the reported runtime and re-run: bash start.sh"; exit 1; }
}

dry_run() {  # every pending stage with simulated models: code / data-shape errors in seconds, not after hours
    [[ "${VP_SKIP_DRYRUN:-0}" == "1" ]] && return 0
    echo "== unit tests"
    ./run.sh --test -q >/tmp/vp_tests.log 2>&1 || { tail -n 30 /tmp/vp_tests.log; notify_failure "unit tests failed"; exit 1; }
    tail -n 1 /tmp/vp_tests.log
    echo "== dry run (simulated models, real cache untouched)"
    run --dry-run || { notify_failure "dry run failed — a code error was caught before any model loaded"; exit 1; }
}

sample_and_notify() {
    if run --mode analyze-and-sample; then
        run --notify sample || echo "(trailer is ready; the notification email could not be sent)"
    else
        notify_failure "trailer run failed"
        exit 1
    fi
}

migrate_if_legacy() {
    if [[ "$PROJECT" == "projects/project_001/config.yaml" && -f input/source.mp4 ]]; then
        if [[ ! -f audio/source.wav ]]; then   # not shipped in the data zip; identical settings to the legacy script
            mkdir -p audio
            ffmpeg -hide_banner -loglevel error -y -i input/source.mp4 -map 0:a:0 -vn -ac 1 -ar 16000 -c:a pcm_s16le audio/source.wav
        fi
        run --migrate-legacy . || true   # idempotent; skips stages whose settings differ
    fi
}

case "$CMD" in
    run|"")
        setup
        migrate_if_legacy
        run --validate-only || { notify_failure "configuration invalid"; exit 1; }
        dry_run
        sample_and_notify
        echo
        echo "Compare: $(dirname "$PROJECT")/sample/trailer_high.mp4  vs  trailer_balanced.mp4"
        echo "Measured time/cost per profile: $(dirname "$PROJECT")/sample/comparison.json"
        echo "Then:    bash start.sh approve high|balanced    or    bash start.sh reject"
        ;;
    approve)
        if [[ -n "${2:-}" ]]; then run --approve-sample --profile "$2"; else run --approve-sample; fi
        run --mode full
        echo "Final video: $(dirname "$PROJECT")/output/final.mp4"
        ;;
    reject)  run --reject-sample ;;
    stop)
        # stops the pipeline and its model workers (leaves manual downloads alone)
        pkill -f "run.py --project" || true
        pkill -f "video_platform.workers" || true
        pkill -f "bash start.sh (run|resume|approve|smoke)" || true
        pkill -f "bash scripts/setup_vast.sh" || true
        sleep 3
        # model engines run as separate processes (vLLM "VLLM::EngineCore", multiprocessing children)
        pkill -9 -f "VLLM::|vllm.entrypoints|/workspace/vp/venvs/" 2>/dev/null || true
        sleep 2
        ps aux | grep -E "run.py --project|video_platform.workers|VLLM::|/workspace/vp/venvs/" | grep -v grep || echo "pipeline stopped"
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/GPU memory in use: /'
        ;;
    resume)
        setup
        migrate_if_legacy
        dry_run
        if run --approval-status --json | grep -q '"approved": true'; then
            run --resume || { notify_failure "full production failed"; exit 1; }
        else
            sample_and_notify
        fi
        ;;
    notify-test)
        python3 scripts/send_mail.py --subject "AI Video — test email" \
            --body "Email works. You will get a message like this when the trailer is ready (or if a run fails)." ;;
    status)  run --status; run --approval-status ;;
    watch)
        # live, filtered view of the run: stage start/finish, heartbeats, warnings, errors, pruning
        LOG="${2:-/workspace/run.log}"
        tail -n 200 -F "$LOG" | grep --line-buffered -E "▶|✔|⏳|↳|stage|WARNING|ERROR|Traceback|fallback|prun|prefetch|email|done \[" ;;
    progress)
        # one-shot snapshot: running stage, latest worker progress line, GPU, disk
        LOG="${2:-/workspace/run.log}"
        echo "== last stage events";  grep -E "▶|✔ stage" "$LOG" 2>/dev/null | tail -n 5
        echo "== last heartbeat";     grep "⏳" "$LOG" 2>/dev/null | tail -n 1
        W=$(ls -t "$(dirname "$PROJECT")"/state/logs/worker_*.log 2>/dev/null | head -n 1)
        if [[ -n "$W" ]]; then
            echo "== active worker log: $W"
            tail -c 4000 "$W" | tr '\r' '\n' | grep -v '^\s*$' | tail -n 3
        fi
        echo "== warnings (last 5)";  grep -E "WARNING" "$LOG" 2>/dev/null | tail -n 5
        echo "== GPU"; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo n/a
        echo "== disk"; df -h /workspace 2>/dev/null | tail -n 1 ;;
    regen-shot) run --regenerate-shot "${2:?shot id, e.g. SH_00042}" ;;
    regen-line) run --regenerate-dialogue "${2:?segment id, e.g. SEG_000123}" ;;
    dryrun)  run --dry-run ;;
    rerun-stage) run --rerun-stage "${2:?stage name, e.g. active_speaker}" ;;
    setup)   setup ;;
    test)    setup; ./run.sh --test ;;
    smoke)
        setup
        SMOKE=projects/smoke
        if [[ ! -f $SMOKE/config.yaml ]]; then
            ./run.sh --init "$SMOKE"
            ffmpeg -hide_banner -loglevel error -y -i input/source.mp4 -t 180 -c copy "$SMOKE/input/source.mp4"
        fi
        PROJECT="$SMOKE/config.yaml"
        run --set languages.source=zh --set analysis.asr.language=zh --set sample.duration_seconds=30 \
            --set rights.reuse_source_footage=true --set rights.reuse_source_music_and_effects=true \
            --mode analyze-and-sample
        echo "Smoke trailers: $SMOKE/sample/trailer_high.mp4 and trailer_balanced.mp4 (+ comparison.json)"
        ;;
    *) echo "unknown command '$CMD' (run|approve <profile>|reject|resume|stop|status|smoke|regen-shot|regen-line|rerun-stage|setup|test|notify-test|watch|progress|dryrun)"; exit 2 ;;
esac
