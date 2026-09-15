#!/usr/bin/env bash
# Run an OpenRouter ChartReviewBench experiment with durable logging.
#
# Usage:
#   run_openrouter_experiment.sh [TASK_ROOT] [OUTPUT_ROOT] [LIMIT]
#
# LIMIT is applied independently to A, B and C. Omit it for the full tracks.
set -Eeuo pipefail

SANDBOX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$SANDBOX_ROOT/../.." && pwd)"
TASK_ROOT="${1:-$PROJECT_ROOT/data/chartrepairbench-tasks}"
OUTPUT_ROOT="${2:-$PROJECT_ROOT/data/runs/openrouter-experiment}"
LIMIT="${3:-0}"
MODEL="${OPENROUTER_MODEL:-openrouter:deepseek/deepseek-v4.1-flash}"
JUDGE_MODEL="${OPENROUTER_JUDGE_MODEL:-$MODEL}"
WORKERS="${OPENROUTER_WORKERS:-1}"
MAX_TOKENS="${OPENROUTER_MAX_TOKENS:-16000}"
MAX_TURNS="${OPENROUTER_MAX_TURNS:-6}"
EPISODE_TIMEOUT="${OPENROUTER_EPISODE_TIMEOUT:-1800}"

if [[ ! -f "$HOME/.bashrc" ]]; then
  echo "ERROR: $HOME/.bashrc does not exist" >&2
  exit 2
fi

# Required by the experiment contract. A normal non-interactive bash may return
# early from .bashrc, so also evaluate only the key's export assignment when the
# regular source did not expose it. The key value is never printed.
source "$HOME/.bashrc" || true
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  key_assignment="$(awk '/^[[:space:]]*export[[:space:]]+OPENROUTER_API_KEY=/{print; exit}' "$HOME/.bashrc")"
  if [[ -n "$key_assignment" ]]; then
    eval "$key_assignment"
  fi
fi
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: OPENROUTER_API_KEY is not set after source ~/.bashrc" >&2
  exit 2
fi

if [[ ! -d "$SANDBOX_ROOT" || ! -d "$TASK_ROOT" ]]; then
  echo "ERROR: missing sandbox or task root" >&2
  echo "  sandbox: $SANDBOX_ROOT" >&2
  echo "  tasks:   $TASK_ROOT" >&2
  exit 2
fi
if [[ "$OUTPUT_ROOT" == "$TASK_ROOT" || "$OUTPUT_ROOT" == "$TASK_ROOT"/* ]]; then
  echo "ERROR: output must be outside the input task folder" >&2
  exit 2
fi
if ! [[ "$LIMIT" =~ ^[0-9]+$ ]]; then
  echo "ERROR: LIMIT must be a nonnegative integer" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
LOG_FILE="$OUTPUT_ROOT/experiment_$(date +%Y%m%d_%H%M%S).txt"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "ChartReviewBench OpenRouter experiment"
echo "started: $(date -Is)"
echo "task_root: $TASK_ROOT"
echo "output_root: $OUTPUT_ROOT"
echo "model: $MODEL"
echo "judge: $JUDGE_MODEL"
echo "limit_per_track: $LIMIT"
echo "workers: $WORKERS"
echo "max_turns: $MAX_TURNS"
echo "max_tokens: $MAX_TOKENS"
echo "episode_timeout: $EPISODE_TIMEOUT"
echo "log_file: $LOG_FILE"

cd "$SANDBOX_ROOT"
for track in A B C; do
  track_tasks="$TASK_ROOT/track$track"
  track_output="$OUTPUT_ROOT/track$track"
  if [[ ! -d "$track_tasks" ]]; then
    echo "ERROR: missing $track_tasks"
    exit 2
  fi
  echo
  echo "===== Track $track: $(date -Is) ====="
  cmd=(
    .venv/bin/python scripts/run_suite.py "$track_tasks"
    --model "$MODEL"
    --judge "$JUDGE_MODEL"
    --out "$track_output"
    --workers "$WORKERS"
    --max-turns "$MAX_TURNS"
    --max-tokens "$MAX_TOKENS"
    --episode-timeout "$EPISODE_TIMEOUT"
  )
  if [[ "$LIMIT" != 0 ]]; then
    cmd+=(--limit "$LIMIT")
  fi
  "${cmd[@]}"
done

echo
echo "finished: $(date -Is)"
echo "results: $OUTPUT_ROOT"
