# shellcheck shell=bash
# Read .env safely: KEY=VALUE lines only; bad or <placeholder> lines are reported and skipped, never executed.
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
    done < "${1:-.env}"
}
