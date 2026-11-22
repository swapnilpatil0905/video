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
source "$ROOT/scripts/load_env.sh"; [[ -f "$ROOT/.env" ]] && load_env "$ROOT/.env"
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
    # a second Hugging Face cache (cloud templates often preset one) silently doubles model storage
    for other in /workspace/.hf_home "$HOME/.cache/huggingface"; do
        if [[ -d "$other/hub" && ! -L "$other" && "$(readlink -f "$other")" != "$(readlink -f "$HF_HOME")" ]]; then
            OTHER_GB=$(du -s "$other/hub" 2>/dev/null | awk '{print int($1/1048576)}')
            if (( OTHER_GB >= 1 )); then
                say "WARNING: second model cache $other holds ${OTHER_GB} GB (this pipeline uses $HF_HOME). Free it: rm -rf $other && ln -s $HF_HOME $other"
            fi
        fi
    done
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
want parler && { "$RT/venvs/parler/bin/python" -c "from sentencepiece import sentencepiece_model_pb2; import parler_tts" \
    >>"$LOGS/setup_parler.log" 2>&1 || { rm -f "$RT/venvs/parler/.vp_ok"; fail "parler (import check)"; }; }
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
want indicf5 && check_import indicf5 "import transformers, soundfile; from f5_tts.infer.utils_infer import load_vocoder"

fetch() {  # repo_id local_dir [allow_patterns...]
    "$ROOT/.venv-platform/bin/python" - "$@" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, local, *patterns = sys.argv[1:]
snapshot_download(repo, local_dir=local, allow_patterns=patterns or None, max_workers=16)
PY
}

# ---- Wan2.2 (character replacement) ------------------------------------------------
wan_extras() {  # python repo — animate/s2v requirements; uv rejects "-e git+…" (SAM2), so install those non-editable
    local py="$1" repo="$2" extra
    for extra in "$repo"/requirements_animate.txt "$repo"/requirements_s2v.txt; do
        [[ -f "$extra" ]] || continue
        sed -E 's/^[[:space:]]*-e[[:space:]]+//' "$extra" > "$RT/venvs/wan/$(basename "$extra")"
        uv pip install --python "$py" --no-build-isolation -r "$RT/venvs/wan/$(basename "$extra")" \
            || echo "!! $(basename "$extra") did not install completely"
    done
}

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
            wan_extras "$PY" "$REPO"
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
            # DiariZen's bundled pyannote-audio imports torchaudio.AudioMetaData, removed in torchaudio 2.9
            PIN="$RT/venvs/diarizen/constraints.txt"; printf 'torch<2.9\ntorchaudio<2.9\n' >"$PIN"
            uv pip install --python "$PY" -c "$PIN" torch torchaudio
            [[ -f "$REPO/requirements.txt" ]] && uv pip install --python "$PY" -c "$PIN" -r "$REPO/requirements.txt"
            uv pip install --python "$PY" -c "$PIN" -e "$REPO"
            [[ -d "$REPO/pyannote-audio" ]] && uv pip install --python "$PY" -c "$PIN" -e "$REPO/pyannote-audio"
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

# ---- GPU check: every model environment must run CUDA kernels on THIS GPU -----------------
# (packages that pin an old torch — chatterbox: 2.6, LatentSync: 2.5.1+cu121 — have no Blackwell
#  sm_120 kernels: "no kernel image is available for execution on the device")
GPU_TEST='import torch; assert torch.cuda.is_available(); d="cuda"; torch.nn.LSTM(4, 4).to(d)(torch.randn(1, 1, 4, device=d)); torch.nn.functional.conv2d(torch.randn(1, 1, 4, 4, device=d), torch.randn(1, 1, 3, 3, device=d)); torch.cuda.synchronize()'
gpu_fix() {  # name — upgrade only the torch family of a venv whose torch cannot run on this GPU
    local name="$1" py="$RT/venvs/$1/bin/python" pkgs=()
    [[ -x "$py" && -f "$RT/venvs/$name/.vp_ok" ]] || return 0
    command -v nvidia-smi >/dev/null || return 0
    "$py" -c "$GPU_TEST" >>"$LOGS/setup_$name.log" 2>&1 && return 0
    say "$name: torch cannot run on this GPU; installing torch 2.8 (CUDA 12.8) (log: $LOGS/setup_$name.log)"
    pkgs=("torch==2.8.0")
    "$py" -c "import torchvision" 2>/dev/null && pkgs+=("torchvision==0.23.0")
    "$py" -c "import torchaudio" 2>/dev/null && pkgs+=("torchaudio==2.8.0")
    uv pip install --python "$py" "${pkgs[@]}" >>"$LOGS/setup_$name.log" 2>&1
    "$py" -c "$GPU_TEST" >>"$LOGS/setup_$name.log" 2>&1 || { rm -f "$RT/venvs/$name/.vp_ok"; say "$name: GPU check still fails"; fail "$name"; }
}
for env in tts indicf5 parler latentsync wan image audio diar asd sep llm diarizen; do
    want "$env" && gpu_fix "$env"
done
if want wan && [[ -f "$RT/venvs/wan/.vp_ok" ]]; then  # animation preprocessing has its own requirements
    WAN_PRE='import sys; sys.path.insert(0, "wan/modules/animate/preprocess"); import process_pipepline'
    if ! (cd "$RT/repos/Wan2.2" && "$RT/venvs/wan/bin/python" -c "$WAN_PRE") >>"$LOGS/setup_wan.log" 2>&1; then
        say "wan: installing animation preprocessing requirements (log: $LOGS/setup_wan.log)"
        wan_extras "$RT/venvs/wan/bin/python" "$RT/repos/Wan2.2" >>"$LOGS/setup_wan.log" 2>&1
        (cd "$RT/repos/Wan2.2" && "$RT/venvs/wan/bin/python" -c "$WAN_PRE") >>"$LOGS/setup_wan.log" 2>&1 \
            || { rm -f "$RT/venvs/wan/.vp_ok"; say "wan: animation preprocessing still cannot be imported"; fail wan; }
    fi
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
