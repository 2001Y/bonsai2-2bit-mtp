#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:=/models/Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP.gguf}"
: "${PORT:=8080}"
: "${PORT_HEALTH:=$PORT}"
: "${LLAMA_CTX_SIZE:=32768}"
: "${LLAMA_SPEC_DRAFT_N_MAX:=2}"
: "${LLAMA_SERVER_BIN:=/opt/llama/bin/llama-server}"

if [[ "$PORT_HEALTH" != "$PORT" ]]; then
    printf 'PORT_HEALTH (%s) must match PORT (%s) for the single-port llama-server.\n' \
        "$PORT_HEALTH" "$PORT" >&2
    exit 64
fi

exec "$LLAMA_SERVER_BIN" \
    -m "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --alias ternary-bonsai-2-27b-abliterated-mtp \
    -ngl 99 \
    -fa on \
    -c "$LLAMA_CTX_SIZE" \
    --jinja \
    --reasoning off \
    --parallel 1 \
    --spec-type draft-mtp \
    --spec-draft-n-max "$LLAMA_SPEC_DRAFT_N_MAX"
