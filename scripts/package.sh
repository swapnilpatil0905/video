#!/usr/bin/env bash
# Build dist/video-platform-<version>.zip (code + configs + registry + legacy analysis data).
#   bash scripts/package.sh            # full bundle incl. legacy data (~2 GB)
#   bash scripts/package.sh --code     # code only (small; verification without legacy import)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)"
NAME="video-platform-$VERSION"
[[ "${1:-}" == "--code" ]] && NAME="$NAME-code"
mkdir -p dist
OUT="$ROOT/dist/$NAME.zip"
rm -f "$OUT"

INCLUDE=(start.sh run.py run.sh pyproject.toml pip.conf uv.toml README.md .gitignore .env.example
         configs model_registry requirements src tests scripts docs pipeline projects/project_001/config.yaml
         projects/project_001/README.md projects/project_template)
if [[ "${1:-}" != "--code" ]]; then
    INCLUDE+=(input audio transcripts diarization scenes faces characters timeline verification models)
fi

# -n: store already-compressed media without recompressing
zip -r -q -n .mp4:.wav:.npz:.jpg:.jpeg:.png:.onnx "$OUT" "${INCLUDE[@]}" \
    -x '*/__pycache__/*' '*.pyc' '*.DS_Store' 'model_registry/_verification.json'
# Bundle a .env (never left on disk): HF token from huggingface_token.text + notification defaults.
if [[ -f huggingface_token.text && "${NO_ENV:-0}" != "1" ]]; then
    STAGE="$(mktemp -d)"
    {
        echo "HF_TOKEN=$(tr -d '[:space:]' < huggingface_token.text)"
        echo "NOTIFY_EMAIL=${NOTIFY_EMAIL:-$(cat .notify_email 2>/dev/null | tr -d '[:space:]')}"
        echo "NTFY_TOPIC=vp-$(python3 -c 'import secrets; print(secrets.token_hex(12))')"
        echo "VP_GPU_PRICE_PER_HOUR=${VP_GPU_PRICE_PER_HOUR:-1.015}"
        echo "# Optional (for e-mails with the trailer preview attached): Gmail app password"
        echo "# SMTP_HOST=smtp.gmail.com"
        echo "# SMTP_PORT=587"
        echo "# SMTP_USERNAME=you@gmail.com"
        echo "# SMTP_PASSWORD=your16charapppassword"
    } > "$STAGE/.env"
    chmod 600 "$STAGE/.env"
    (cd "$STAGE" && zip -q "$OUT" .env)
    rm -rf "$STAGE"
    echo "bundled .env (HF token + notification settings)"
fi
echo "$OUT ($(du -h "$OUT" | cut -f1))"
