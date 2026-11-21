#!/usr/bin/env bash
# Build dist/video-data.zip: the files the GPU machine needs that are NOT in git
# (source video, existing analysis, .env with secrets). Code comes from: git clone.
#
# On the machine:  git clone https://github.com/<you>/video.git /workspace/video
#                  python3 -m zipfile -e video-data.zip /workspace/video
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p dist
OUT="$ROOT/dist/video-data.zip"
rm -f "$OUT"
DATA=(input/source.mp4 transcripts/chinese_transcript.json transcripts/chinese_transcript_clean.json
      diarization/speaker_turns.json diarization/speaker_turns.csv diarization/speaker_summary.json
      scenes/scene_timeline.csv scenes/frame_manifest.csv scenes/frames
      faces/face_manifest.csv faces/crops faces/face_embeddings.npz faces/embedding_manifest.csv
      faces/clustered_faces.csv faces/cluster_summary.csv faces/cluster_audit.csv faces/cluster_representatives.csv
      faces/clustering_evaluation.csv characters timeline models)
zip -r -q -n .mp4:.wav:.npz:.jpg:.jpeg:.png:.onnx "$OUT" "${DATA[@]}" -x '*.DS_Store'
if [[ -f huggingface_token.text ]]; then
    STAGE="$(mktemp -d)"
    {
        echo "HF_TOKEN=$(tr -d '[:space:]' < huggingface_token.text)"
        echo "NOTIFY_EMAIL=${NOTIFY_EMAIL:-$(cat .notify_email 2>/dev/null | tr -d '[:space:]')}"
        echo "NTFY_TOPIC=vp-$(python3 -c 'import secrets; print(secrets.token_hex(12))')"
        echo "VP_GPU_PRICE_PER_HOUR=${VP_GPU_PRICE_PER_HOUR:-1.015}"
    } > "$STAGE/.env"
    chmod 600 "$STAGE/.env"
    (cd "$STAGE" && zip -q "$OUT" .env)
    rm -rf "$STAGE"
fi
echo "$OUT ($(du -h "$OUT" | cut -f1))"
