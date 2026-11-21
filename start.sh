#!/usr/bin/env bash
# ONE command on the GPU machine. Upload + unzip the bundle, then:
#
#   bash start.sh               # setup (first time) → analysis → trailers (high + balanced) → stops for review
#   bash start.sh approve balanced   # approve one trailer → full-length production with that profile
#   bash start.sh resume        # continue whatever was interrupted (used by the on-start script)
#   bash start.sh reject        # stop: no full production
#   bash start.sh status        # where things stand
#   bash start.sh smoke         # quick end-to-end test on the first 3 minutes (separate project)
#   bash start.sh regen-shot SH_00042 | regen-line SEG_000123   # redo one item, then: bash start.sh
#
# Optional: put HF_TOKEN=... in .env (needed only for gated models, e.g. pyannote on new projects).
# Optional: PROJECT=projects/<id>/config.yaml to use another project.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
chmod +x run.sh run.py scripts/*.sh 2>/dev/null || true
[[ -f .env ]] && { set -a; source .env; set +a; }
export VP_RUNTIME_ROOT="${VP_RUNTIME_ROOT:-/workspace/vp}"
export HF_HOME="${HF_HOME:-$VP_RUNTIME_ROOT/hf}"

PROJECT="${PROJECT:-projects/project_001/config.yaml}"
CMD="${1:-run}"
run() { ./run.sh --project "$PROJECT" "$@"; }
RUN_LOG="${RUN_LOG:-/workspace/run.log}"

notify_failure() {  # best effort; stdlib-only, works even before setup
    python3 scripts/send_mail.py --subject "AI Video — run FAILED — $1" \
        --body "The pipeline stopped: $1. Fix it (or send me the log), then run: bash start.sh (finished work is reused)." \
        --log "$RUN_LOG" || true
}

setup() {
    bash scripts/setup_vast.sh || { notify_failure "setup failed (see /workspace/vp/logs/setup_*.log)";
                                    echo "Setup failed — fix the reported runtime and re-run: bash start.sh"; exit 1; }
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
    resume)
        setup
        migrate_if_legacy
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
    regen-shot) run --regenerate-shot "${2:?shot id, e.g. SH_00042}" ;;
    regen-line) run --regenerate-dialogue "${2:?segment id, e.g. SEG_000123}" ;;
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
    *) echo "unknown command '$CMD' (run|approve <profile>|reject|resume|status|smoke|regen-shot|regen-line|setup|test|notify-test)"; exit 2 ;;
esac
