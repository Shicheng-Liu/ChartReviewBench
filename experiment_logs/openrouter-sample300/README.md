# OpenRouter 300-per-track experiment logs

This directory contains the core audit logs for the stratified 300-task-per-track
experiments with:

- `anthropic/claude-opus-5.5`
- `openai/gpt-6.1-sol`

Both models have 300 final results for each of Tracks A, B, and C. Rule-based
rescoring is complete for all 1,800 final episodes.

## Contents

- `results/<model>/track{A,B,C}/<task_id>/result.json`: final metrics, complete
  agent transcript, token usage, response metadata, and rule-based scores.
- `results/<model>/track{A,B,C}/<task_id>/trajectory.jsonl`: every tool call,
  arguments, stdout/stderr, warnings, file changes, and render references.
- `results/<model>/track{A,B,C}/<task_id>/process.log`: per-episode runner log.
- `.attempts/`: archived failed/interrupted attempts retained by resume.
- `analysis/openrouter_opus55_gpt61sol_results.md`: concise comparison.
- `analysis/{score_runs.py,final_scores.py}`: scoring and aggregation scripts used.
- `sample/manifest.json`: sample seed, source revision, composition, and task IDs.

Rendered images and copied workspaces are intentionally excluded from Git because
they add about 1.6 GiB. They remain on the experiment host. No API keys are stored
in this directory.
