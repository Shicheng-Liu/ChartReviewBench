# OpenRouter Subsample Results: Claude Opus 5.5 and GPT-6.1 Sol

**Status (2026-11-13 13:48 EST).** Both model runs and rule-based rescoring are complete (900/900 valid episodes and 900/900 rule scores per model, with no scoring errors).

## Setup

- Stratified sample: 300 tasks per track (900 total), seed `20260909`.
- Tracks A/B contain the same 300 flawed charts; Track C contains 259 flawed and 41 clean charts.
- Agent model: `anthropic/claude-opus-5.5` or `openai/gpt-6.1-sol` through OpenRouter.
- XML protocol, 6 turns, 16,000 output-token limit, 1,800 s episode timeout.
- Provider pinned with no fallback: Google for Opus, OpenAI for GPT.

The sample preserves the full benchmark's flaw-count distribution:

| Set | 1 flaw | 2 flaws | 3 flaws | Clean |
|---|---:|---:|---:|---:|
| Full A/B | 400 (21.1%) | 900 (47.4%) | 600 (31.6%) | 0 |
| Sample A/B | 63 (21.0%) | 142 (47.3%) | 95 (31.7%) | 0 |
| Sample C | 54 | 123 | 82 | 41 |

## Run Integrity

| Model | A | B | C | Failures | Parse/evaluation errors |
|---|---:|---:|---:|---:|---:|
| Claude Opus 5.5 | 300/300 | 300/300 | 300/300 | 0 | 0 |
| GPT-6.1 Sol | 300/300 | 300/300 | 300/300 | 0 | 0 |

GPT had two transient evaluator responses with no completion choice; both episodes were rerun from the beginning and completed successfully. The failed attempts are retained under `.attempts/`.

## Rule-Based Results

The requested joint metric is computed per episode:

```text
Joint = 1[executability == 1 AND Rec-B == 1 AND Pre-B == 1]
```

| Model | Track | Exec. | Rec-B | Rec-S | Pre-B | Pre-S | Joint |
|---|---|---:|---:|---:|---:|---:|---:|
| Opus 5.5 | A | 100.00 | 75.00 | 92.30 | 97.00 | 99.27 | **75.00** |
| Opus 5.5 | B | 100.00 | 63.33 | 85.22 | 83.00 | 97.14 | **57.33** |
| Opus 5.5 | C (flawed) | 100.00 | 67.57 | 86.72 | 81.85 | 97.09 | **59.46** |
| GPT-6.1 Sol | A | 100.00 | 71.67 | 90.27 | 91.33 | 98.24 | **70.33** |
| GPT-6.1 Sol | B | 100.00 | 54.33 | 81.73 | 59.33 | 92.07 | **40.67** |
| GPT-6.1 Sol | C (flawed) | 100.00 | 51.74 | 81.62 | 52.90 | 90.20 | **35.52** |

Scores are percentages. Track C recovery is undefined on clean charts, so the main C rows use only the 259 flawed tasks. On the 41 clean tasks, Pre-B / Pre-S are 92.68 / 97.54 for Opus and 60.98 / 92.36 for GPT. Track-wise reporting is preferred because Tracks A and B reuse the same charts.

## Results by Number of Flaws

Three-flaw tasks are well represented (95 in A/B and 82 in C) and are generally harder. Entries are Rec-B / Joint.

| Model | Track | 1 flaw | 2 flaws | 3 flaws |
|---|---|---:|---:|---:|
| Opus 5.5 | A | 82.54 / 82.54 | 76.76 / 76.76 | **67.37 / 67.37** |
| Opus 5.5 | B | 76.19 / 63.49 | 62.68 / 58.45 | **55.79 / 51.58** |
| Opus 5.5 | C | 75.93 / 64.81 | 67.48 / 63.41 | **62.20 / 50.00** |
| GPT-6.1 Sol | A | 80.95 / 80.95 | 72.54 / 71.13 | **64.21 / 62.11** |
| GPT-6.1 Sol | B | 66.67 / 39.68 | 50.70 / 39.44 | **51.58 / 43.16** |
| GPT-6.1 Sol | C | 57.41 / 33.33 | 52.03 / 36.59 | **47.56 / 35.37** |

Thus, strong aggregate performance is not caused by under-sampling three-flaw tasks. A triple-flaw-only extension would mainly reduce subgroup uncertainty or support a targeted stress-test claim, rather than correct a sampling imbalance.

## Tokens and Recorded API Cost

Output tokens already include reasoning tokens. Costs are summed from OpenRouter `usage.cost`; local evaluator calls are included as actual experiment overhead but are not reported as benchmark metrics.

| Model | Agent tokens | Evaluator tokens | Total tokens | Agent cost | Evaluator cost | Retry cost | Total recorded cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Claude Opus 5.5 | 15,056,680 | 6,394,470 | 21,451,150 | $82.7678 | $32.6540 | $0 | **$115.4218** |
| GPT-6.1 Sol | 15,080,236 | 7,884,027 | 22,964,263 | $29.6424 | $15.9467 | $0.0464 | **$45.6356** |

## Artifacts and Reproducibility

- Sample manifest: `data/chartrepairbench-v1.3-sample300/manifest.json`
- Opus results: `data/runs/openrouter-claude-opus-5.5-sample300/`
- GPT results: `data/runs/openrouter-gpt-6.1-sol-sample300/`
- Every final episode retains `result.json`, `trajectory.jsonl`, `process.log`, the full agent transcript, tool inputs/outputs, and rendered images.
- Rule scoring: `scripts/score_runs.py`; final aggregation and Joint metric: `scripts/final_scores.py`.

This is a stratified subsample rather than the full 6,000-task benchmark. Track C clean-chart conclusions are especially uncertain because that subgroup contains only 41 tasks.
