#!/usr/bin/env bash
# Build a token-type-CORRECTED copy of a gemma-4 SFT checkpoint as a SEPARATE ollama tag.
#
# Nothing existing is touched: the production GGUF, its blob, and the
# `gemma4-frames-robust-q4km:latest` registration are all left exactly as they are, so every result
# already collected stays valid and comparable. This only adds a new tag to compare against.
#
# Six control tokens (<|tool_call>, <tool_call|>, <|tool_response>, <tool_response|>, <|channel>,
# <channel|>) were converted as USER_DEFINED instead of CONTROL. Measured effect: responses leak a
# raw `<channel|>` and truncate on the path where the model answers WITHOUT calling the search tool
# -- 89.2% of zero-search rows vs 0.1% of rows with a search. See
# scripts/patch_gguf_token_types.py and docs/frames_cue_robustness_sft.md.
#
# The registration copies the original's `PARAMETER stop "<turn|>"` verbatim, so the ONLY difference
# between the two tags is the six token types.
#
# Disk: a full copy is made, imported by `ollama create` (which re-stores it as a blob), then the
# copy is deleted -- net cost is one extra 18.7 GB blob.
#
# Usage (inside the container, with ollama already serving):
#   bash scripts/setup_gemma_fixed_model.sh <src.gguf> <new-tag>
set -euo pipefail
SRC="${1:-/rg/reichart_prj/dvirla/gemma_gguf/gemma-4-31b-frames-robust-Q4_K_M.gguf}"
TAG="${2:-gemma4-frames-robust-q4km-fixed}"
WORK="$(dirname "$SRC")/$(basename "$SRC" .gguf)-tokfix.gguf"

if ollama show "$TAG" >/dev/null 2>&1; then
  echo "[setup] $TAG already registered; nothing to do."; exit 0
fi
echo "[setup] copying $SRC -> $WORK"
cp "$SRC" "$WORK"
echo "[setup] patching token types on the COPY (source untouched)"
python3 scripts/patch_gguf_token_types.py "$WORK" --apply
echo "[setup] registering $TAG"
MF="$(mktemp)"
printf 'FROM %s\nPARAMETER stop "<turn|>"\n' "$WORK" > "$MF"
ollama create "$TAG" -f "$MF"
rm -f "$MF" "$WORK"
echo "[setup] done: $TAG registered, working copy removed"
ollama show "$TAG" 2>&1 | sed -n '1,20p'
