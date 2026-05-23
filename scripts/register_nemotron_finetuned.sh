#!/usr/bin/env bash
# Register a merged nemotron-3-nano fine-tuned checkpoint with Ollama.
#
# Imports directly from Safetensors weights (no GGUF conversion needed).
# Preserves the template, renderer, parser, and sampling parameters from
# the original nemotron-3-nano:30b modelfile.
#
# Usage:
#   bash scripts/register_nemotron_finetuned.sh \
#       --model-dir models/nemotron-musique-lora_merged \
#       --model-name nemotron-3-nano-musique:30b \
#       [--ollama-host http://localhost:11434]
#
# --ollama-host  Override OLLAMA_HOST (useful on remote machines where Ollama
#                listens on a non-default address/port). Defaults to whatever
#                OLLAMA_HOST is already set to in the environment, or Ollama's
#                own default (http://localhost:11434).
#
# The model-dir is typically <output-dir>_merged produced by train_sft.py
# with --merge-at-end.

set -euo pipefail

MODEL_DIR=""
MODEL_NAME=""
OLLAMA_HOST_OVERRIDE=""
QUANTIZE="q4_K_M"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-dir)    MODEL_DIR="$2"; shift 2 ;;
        --model-name)   MODEL_NAME="$2"; shift 2 ;;
        --ollama-host)  OLLAMA_HOST_OVERRIDE="$2"; shift 2 ;;
        --quantize)     QUANTIZE="$2"; shift 2 ;;
        --no-quantize)  QUANTIZE=""; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -n "$OLLAMA_HOST_OVERRIDE" ]]; then
    export OLLAMA_HOST="$OLLAMA_HOST_OVERRIDE"
fi

if [[ -z "$MODEL_DIR" || -z "$MODEL_NAME" ]]; then
    echo "Usage: $0 --model-dir <merged-checkpoint-dir> --model-name <ollama-name> [--ollama-host <url>] [--quantize <level>|--no-quantize]" >&2
    echo "  e.g. $0 --model-dir models/nemotron-musique-lora_merged --model-name nemotron-3-nano-musique:30b" >&2
    echo "  default quantization: q4_K_M (matches nemotron-3-nano:30b base). Use --no-quantize to import at native precision." >&2
    exit 1
fi

MODEL_DIR_ABS="$(realpath "$MODEL_DIR")"

if [[ ! -d "$MODEL_DIR_ABS" ]]; then
    echo "Error: directory not found: $MODEL_DIR_ABS" >&2
    exit 1
fi

if ! compgen -G "$MODEL_DIR_ABS/*.safetensors" > /dev/null 2>&1; then
    echo "Warning: no *.safetensors files found in $MODEL_DIR_ABS" >&2
    echo "  Make sure train_sft.py was run with --merge-at-end" >&2
fi

MODELFILE="$MODEL_DIR_ABS/Modelfile"

cat > "$MODELFILE" <<EOF
FROM $MODEL_DIR_ABS

TEMPLATE {{ .Prompt }}
RENDERER nemotron-3-nano
PARSER nemotron-3-nano
PARAMETER temperature 1
PARAMETER top_p 1
EOF

echo "Modelfile written to $MODELFILE"
echo "---"
cat "$MODELFILE"
echo "---"
echo ""

if ! ollama list &>/dev/null; then
    echo "Error: Ollama is not running or not reachable. Start it first." >&2
    exit 1
fi

if [[ -n "$QUANTIZE" ]]; then
    echo "Registering '$MODEL_NAME' with Ollama, quantizing to $QUANTIZE (this may take a while)..."
    ollama create "$MODEL_NAME" --quantize "$QUANTIZE" -f "$MODELFILE"
else
    echo "Registering '$MODEL_NAME' with Ollama at native precision (no quantization)..."
    ollama create "$MODEL_NAME" -f "$MODELFILE"
fi

echo ""
echo "Registration complete. Verify with:"
echo "  ollama run $MODEL_NAME"
