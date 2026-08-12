#!/usr/bin/env bash
# Serve a local open-weight VLM on vLLM's OpenAI-compatible endpoint, with the
# flags the harness needs: tool calling on, and enough image slots for a long
# episode (the agent accumulates one image per view_image call).
#
# vLLM lives in its own uv venv, not this project's: it hard-pins torch and
# transformers, so sharing an environment with chartsandbox only causes grief.
# Create it once with
#
#     uv venv --python 3.12 ~/.venvs/vllm
#     uv pip install --python ~/.venvs/vllm/bin/python vllm==0.27.1
#
# Usage:  scripts/serve_vllm.sh [MODEL_PATH]
#   GPUS=0,1 PORT=8001 MAX_LEN=131072 scripts/serve_vllm.sh /data2/Qwen/...
set -euo pipefail

MODEL="${1:-/data2/Qwen/Qwen3-VL-8B-Instruct}"
# Served name is what the harness sends as `model`; keep it short, not the path.
SERVED="${SERVED:-$(basename "$MODEL")}"
VLLM="${VLLM:-$HOME/.venvs/vllm/bin/vllm}"
GPUS="${GPUS:-0}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-65536}"
MAX_IMAGES="${MAX_IMAGES:-16}"
# Qwen3-VL emits <tool_call>{json}</tool_call>, which is the hermes format.
# (qwen3_xml is for the <function=...><parameter=...> style — wrong model family.)
PARSER="${PARSER:-hermes}"

# One shard per visible GPU: 8B fits on one A100, bigger weights need more.
TP="$(awk -F, '{print NF}' <<<"$GPUS")"

# torch.compile's inductor backend shells out to `ninja` by bare name, so the venv's
# bin has to be on PATH — calling $VLLM by absolute path alone is not enough, and the
# failure only surfaces minutes in, as a FileNotFoundError during engine startup.
export PATH="$(dirname "$VLLM"):$PATH"

exec env CUDA_VISIBLE_DEVICES="$GPUS" "$VLLM" serve "$MODEL" \
    --served-model-name "$SERVED" \
    --host 0.0.0.0 --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
    --limit-mm-per-prompt "{\"image\": $MAX_IMAGES}" \
    --enable-auto-tool-choice --tool-call-parser "$PARSER"
