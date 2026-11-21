#!/usr/bin/env bash
# One-time (idempotent) setup of every model runtime on a Vast.ai / Linux GPU machine.
# Public sources only: pypi.org, github.com, huggingface.co, python-build-standalone.
#
#   bash scripts/setup_vast.sh            # everything
#   bash scripts/setup_vast.sh core wan   # selected runtimes
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RT="${VP_RUNTIME_ROOT:-/workspace/vp}"
LOGS="$RT/logs"
mkdir -p "$RT"/{venvs,repos,weights,hf} "$LOGS"
export HF_HOME="${VP_HF_HOME:-$RT/hf}"
export UV_INDEX_URL="https://pypi.org/simple" UV_DEFAULT_INDEX="https://pypi.org/simple"
export PIP_INDEX_URL="https://pypi.org/simple" UV_PYTHON_PREFERENCE="managed" UV_LINK_MODE="copy"
unset PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL || true
[[ -f "$ROOT/.env" ]] && { set -a; source "$ROOT/.env"; set +a; }
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "WARNING: HF_TOKEN not set (create .env first): Hugging Face downloads will be rate-limited and slow"
fi

say()  { echo -e "\033[1;34m==\033[0m $*"; }
fail() { echo -e "\033[1;31m!!\033[0m $*"; FAILED+=("$1"); }
FAILED=()
TARGETS=("$@"); [[ ${#TARGETS[@]} -eq 0 ]] && TARGETS=(system core diar audio llm parler tts image wan latentsync asd sep indicf5 diarizen)
want() { [[ " ${TARGETS[*]} " == *" $1 "* ]]; }

# ---- system packages ----------------------------------------------------------
if want system; then
    say "system packages"
    if command -v apt-get >/dev/null && [[ $(id -u) -eq 0 ]]; then
        apt-get update -qq >"$LOGS/setup_system.log" 2>&1
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg git git-lfs build-essential libgl1 libglib2.0-0 python3 python3-pip curl \
            >>"$LOGS/setup_system.log" 2>&1 || fail system
    fi
    command -v ffmpeg >/dev/null || fail "system (ffmpeg missing)"
    if command -v nvidia-smi >/dev/null; then
        VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        say "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) (${VRAM} MiB)"
        (( VRAM < 70000 )) && say "WARNING: <80 GB VRAM — Wan2.2-Animate-14B and Qwen-Image will use CPU offload (slow)"
    else
        say "WARNING: no NVIDIA GPU detected"
    fi
    FREE_GB=$(df -BG --output=avail "$RT" | tail -1 | tr -dc '0-9')
    (( FREE_GB < 300 )) && say "WARNING: ${FREE_GB} GB free under $RT; ~300 GB recommended (model weights + renders)"
fi

# ---- uv (fast, isolated Python environments) -----------------------------------
if ! command -v uv >/dev/null; then
    python3 -m pip install -q --user uv >/dev/null 2>&1 || python3 -m pip install -q --break-system-packages uv >/dev/null 2>&1
    export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null || { echo "could not install uv"; exit 2; }

mkvenv() {  # name dir requirements-file [extra uv pip args...]
    local name="$1" dir="$2" req="$3"; shift 3
    if [[ -f "$dir/.vp_ok" && "$dir/.vp_ok" -nt "$req" ]]; then say "$name: ready"; return 0; fi
    say "$name: creating environment (log: $LOGS/setup_$name.log)"
    {
        uv venv --allow-existing --python 3.11 "$dir" && \
        VIRTUAL_ENV="$dir" uv pip install --python "$dir/bin/python" -r "$req" "$@"
    } >"$LOGS/setup_$name.log" 2>&1 && touch "$dir/.vp_ok" || fail "$name"
}

if want core; then
    mkvenv core "$ROOT/.venv-platform" "$ROOT/requirements/core.txt" -e "$ROOT[dev]"
fi
want diar   && mkvenv diar   "$RT/venvs/diar"   "$ROOT/requirements/diar.txt"
want audio  && mkvenv audio  "$RT/venvs/audio"  "$ROOT/requirements/audio.txt"
want llm    && mkvenv llm    "$RT/venvs/llm"    "$ROOT/requirements/llm.txt"
want parler && mkvenv parler "$RT/venvs/parler" "$ROOT/requirements/parler.txt"
want tts    && mkvenv tts    "$RT/venvs/tts"    "$ROOT/requirements/tts.txt"
want image  && mkvenv image  "$RT/venvs/image"  "$ROOT/requirements/image.txt"
want sep     && mkvenv sep     "$RT/venvs/sep"     "$ROOT/requirements/sep.txt"
want indicf5 && mkvenv indicf5 "$RT/venvs/indicf5" "$ROOT/requirements/indicf5.txt"
check_import() {  # name python-code — verify an optional runtime actually works; un-mark it if not
    local name="$1" code="$2" py="$RT/venvs/$1/bin/python"
    [[ -f "$RT/venvs/$name/.vp_ok" ]] || return 0
    if ! "$py" -c "$code" >>"$LOGS/setup_$name.log" 2>&1; then
        rm -f "$RT/venvs/$name/.vp_ok"; fail "$name"
    fi
}
want sep     && check_import sep "import audio_separator"
want indicf5 && check_import indicf5 "import transformers, soundfile"

fetch() {  # repo_id local_dir [allow_patterns...]
    "$ROOT/.venv-platform/bin/python" - "$@" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, local, *patterns = sys.argv[1:]
snapshot_download(repo, local_dir=local, allow_patterns=patterns or None, max_workers=16)
PY
}

# ---- Wan2.2 (character replacement) ------------------------------------------------
if want wan; then
    REPO="$RT/repos/Wan2.2"
    if [[ ! -f "$RT/venvs/wan/.vp_ok" ]]; then
        say "wan: repository + environment + weights (large download; log: $LOGS/setup_wan.log)"
        {
            [[ -d "$REPO/.git" ]] || git clone --depth 1 https://github.com/Wan-Video/Wan2.2.git "$REPO"
            uv venv --allow-existing --python 3.11 "$RT/venvs/wan"
            PY="$RT/venvs/wan/bin/python"
            uv pip install --python "$PY" torch torchvision packaging ninja psutil wheel setuptools
            grep -v -i '^flash_attn' "$REPO/requirements.txt" > "$RT/venvs/wan/req.txt"
            uv pip install --python "$PY" -r "$RT/venvs/wan/req.txt"
            for extra in "$REPO"/requirements_animate.txt "$REPO"/requirements_s2v.txt; do
                [[ -f "$extra" ]] && uv pip install --python "$PY" -r "$extra"
            done
            uv pip install --python "$PY" --no-build-isolation flash_attn || echo "flash_attn unavailable (slower attention)"
            uv pip install --python "$PY" sageattention safetensors || echo "sageattention unavailable (balanced profile uses default attention)"
            fetch Wan-AI/Wan2.2-Animate-14B "$RT/weights/Wan2.2-Animate-14B"
        } >"$LOGS/setup_wan.log" 2>&1 && touch "$RT/venvs/wan/.vp_ok" || fail wan
    else say "wan: ready"; fi
fi

# ---- LatentSync (lip-sync) -----------------------------------------------------------
if want latentsync; then
    REPO="$RT/repos/LatentSync"
    if [[ ! -f "$RT/venvs/latentsync/.vp_ok" ]]; then
        say "latentsync: repository + environment + checkpoints (log: $LOGS/setup_latentsync.log)"
        {
            [[ -d "$REPO/.git" ]] || git clone --depth 1 https://github.com/bytedance/LatentSync.git "$REPO"
            uv venv --allow-existing --python 3.11 "$RT/venvs/latentsync"
            uv pip install --python "$RT/venvs/latentsync/bin/python" -r "$REPO/requirements.txt"
            fetch ByteDance/LatentSync-1.6 "$REPO/checkpoints" "latentsync_unet.pt" "whisper/tiny.pt"
        } >"$LOGS/setup_latentsync.log" 2>&1 && touch "$RT/venvs/latentsync/.vp_ok" || fail latentsync
    else say "latentsync: ready"; fi
fi

# ---- DiariZen (speaker diarization) -------------------------------------------------------
if want diarizen; then
    REPO="$RT/repos/DiariZen"
    if [[ ! -f "$RT/venvs/diarizen/.vp_ok" || "$0" -nt "$RT/venvs/diarizen/.vp_ok" ]]; then
        say "diarizen: repository + environment (log: $LOGS/setup_diarizen.log)"
        {
            [[ -d "$REPO/.git" ]] || git clone --recursive --depth 1 https://github.com/BUTSpeechFIT/DiariZen.git "$REPO"
            uv venv --allow-existing --python 3.11 "$RT/venvs/diarizen"
            PY="$RT/venvs/diarizen/bin/python"
            uv pip install --python "$PY" torch torchaudio
            [[ -f "$REPO/requirements.txt" ]] && uv pip install --python "$PY" -r "$REPO/requirements.txt"
            uv pip install --python "$PY" -e "$REPO"
            [[ -d "$REPO/pyannote-audio" ]] && uv pip install --python "$PY" -e "$REPO/pyannote-audio"
            "$PY" -c "from diarizen.pipelines.inference import DiariZenPipeline; print('diarizen import ok')"
        } >"$LOGS/setup_diarizen.log" 2>&1 && touch "$RT/venvs/diarizen/.vp_ok" || fail diarizen
    else say "diarizen: ready"; fi
fi

# ---- Light-ASD (active speaker detection) ----------------------------------------------
if want asd; then
    REPO="$RT/repos/Light-ASD"
    if [[ ! -f "$RT/venvs/asd/.vp_ok" || "$ROOT/requirements/asd.txt" -nt "$RT/venvs/asd/.vp_ok" ]]; then
        say "asd: Light-ASD repository + environment (log: $LOGS/setup_asd.log)"
        {
            [[ -d "$REPO/.git" ]] || git clone --depth 1 https://github.com/Junhua-Liao/Light-ASD.git "$REPO"
            uv venv --allow-existing --python 3.11 "$RT/venvs/asd"
            uv pip install --python "$RT/venvs/asd/bin/python" -r "$ROOT/requirements/asd.txt"
            ls "$REPO"/weight/*.model
            (cd "$REPO" && "$RT/venvs/asd/bin/python" -c "from ASD import ASD; print('Light-ASD import ok')")
        } >"$LOGS/setup_asd.log" 2>&1 && touch "$RT/venvs/asd/.vp_ok" || fail asd
    else say "asd: ready"; fi
fi

# ---- Real-ESRGAN weights (balanced profile upscaling) ------------------------------
if want image; then
    ESR="$RT/weights/RealESRGAN_x2plus.pth"
    [[ -s "$ESR" ]] || curl -fsSL -o "$ESR" https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth \
        || say "WARNING: Real-ESRGAN weights not downloaded (upscaling falls back to Lanczos)"
fi

OPTIONAL=" asd sep indicf5 diarizen "   # upgrades with automatic fallbacks; never block the run
REQUIRED_FAILED=(); OPTIONAL_FAILED=()
for f in "${FAILED[@]}"; do
    if [[ "$OPTIONAL" == *" $f "* ]]; then OPTIONAL_FAILED+=("$f"); else REQUIRED_FAILED+=("$f"); fi
done
if [[ ${#OPTIONAL_FAILED[@]} -gt 0 ]]; then
    say "optional upgrades unavailable (fallbacks will be used): ${OPTIONAL_FAILED[*]} — see $LOGS/setup_<name>.log"
fi
if [[ ${#REQUIRED_FAILED[@]} -gt 0 ]]; then
    echo; echo "!! setup incomplete: ${REQUIRED_FAILED[*]} — see $LOGS/setup_<name>.log"
    exit 1
fi
say "setup complete (runtimes under $RT)"
