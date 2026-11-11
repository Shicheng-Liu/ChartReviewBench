# VisAudit rebuttal runs

This folder is the code supplement submitted with the paper (`VisAudit-code`, ZIP SHA-256
`410584b8c85ec4127df7a013480d80b9260882018dc3c140568b3597b426492b`), unchanged except for one
opt-in switch in `src/eval/episode_runner.py`. With `VISAUDIT_CAPTURE_STEP_IMAGES=1`, every
image version in the agent workspace is copied to `runs/<digest>/step_images/` after each step,
because later renders overwrite `output.png`. Without the switch the harness behaves exactly as
in the paper.

- `src/eval/episode_runner.py`: submitted SHA-256 `05f31515fb63e9a2c889f9d12477f18e778fa8b55e231e4776ec28e940c03d47`,
  now `09dea37431c6e4777e0dd4433db49c46f59550a4993eb80ca9ad13c1217cfeea`. This is the exact file
  the human-evaluation run used. `PACKAGE_FILES_SHA256.json` is kept as submitted, so
  `verify_code_package.py` stops at this file with exit code 1; that is expected.
- New: this file, `rebuttal/B_humaneval200.json`, `rebuttal/sample_humaneval200.py`, `.gitignore`.
  Everything else is byte-identical to the submitted supplement.

## Runs

All three are Track B with Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) under the settings of
the paper's Haiku Track B run, `history/selected_outcomes/condition-11.jsonl`: temperature 0,
no step or wall-time limit, stagnation guard v2, no deferral scheduling. That run is the
full-input baseline for both ablations.

| Run | Tasks | Inputs | What it measures |
|---|---|---|---|
| Human evaluation | 200 | chart, code, table, summary | repaired charts for human judgement against Rec/Pre |
| No initial chart | all 1,900 | code, table, summary | the given chart image is withheld; the agent can still render `chart.py` and view its own output |
| No images | all 1,900 | code, table, summary, text-only | no image block reaches the model at any step; `view_image` is withheld from the API tool list and answered with a fixed note |

In the text-only run the system prompt is the same as in the baseline. The agent can still read
image files as bytes or inspect the figure in Python; what is removed is visual perception.

## Setup

The benchmark inputs are not in this repository. Get `visaudit-frozen-data.zip` (608 MB,
shared separately), then from this folder:

```sh
unzip -q /path/to/visaudit-frozen-data.zip 'visaudit-code/data/*'
mv visaudit-code/data data && rmdir visaudit-code
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-validated.txt
.venv/bin/python -m src.eval.track_suite --materialize data/frozen-v1.4 --out work/runtime
export ANTHROPIC_API_KEY=...
```

Python 3.12 on macOS is required for the isolated agent path.

## Commands

Written out in full so they work the same in sh, bash and zsh. Each run needs its own `--out`:
`--no-images` is not part of the run identity, so two ablations sharing a folder would silently
reuse each other's episodes.

No initial chart (1,900 tasks):

```sh
.venv/bin/python -m src.eval.run_benchmark --model claude-haiku-4-5-20251001 \
  --samples work/runtime/samples --tasks work/tasks --track B \
  --no-step-limit --no-wall-time-limit --stagnation-guard --workers 8 --retry-errors \
  --dataset work/runtime/B.json --modalities code,table,summary \
  --out work/ablation-no-initial-chart
```

No images at all (1,900 tasks):

```sh
.venv/bin/python -m src.eval.run_benchmark --model claude-haiku-4-5-20251001 \
  --samples work/runtime/samples --tasks work/tasks --track B \
  --no-step-limit --no-wall-time-limit --stagnation-guard --workers 8 --retry-errors \
  --dataset work/runtime/B.json --modalities code,table,summary --no-images \
  --out work/ablation-no-images
```

Human-evaluation sample (200 tasks, with step images):

```sh
VISAUDIT_CAPTURE_STEP_IMAGES=1 .venv/bin/python -m src.eval.run_benchmark \
  --model claude-haiku-4-5-20251001 \
  --samples work/runtime/samples --tasks work/tasks --track B \
  --no-step-limit --no-wall-time-limit --stagnation-guard --workers 8 --retry-errors \
  --dataset rebuttal/B_humaneval200.json --out work/humaneval200
```

The capture switch takes effect on this uncapped path (and whenever `--stagnation-guard` is
set); with both a step and a wall-time limit and no guard the upstream runner is used and the
switch is ignored. `step_000__*` also contains the input files present before the first step.

`rebuttal/B_humaneval200.json` (SHA-256 `a5093175ab76586be98ca659fa6c79ed9ed40c80a9cc81819af802e8fa2644ad`)
is `B.json` restricted to 42 single-, 95 double- and 63 triple-flaw tasks, spread evenly over all
106 flaw scenarios; `rebuttal/sample_humaneval200.py` regenerates it. The completed run used a
byte-identical copy at `work/runtime/B_humaneval200.json`.

Add `--dry-run` to any command to see the episode count without calling the model, and
`--budget-usd N` for a hard cost cap. Rerunning the same command resumes; `--retry-errors`
repeats technical failures (timeouts, connection errors), which are never scored as results.

## Checks after a run

- `report.md`: all metrics, completed and errored counts, cost.
- Text-only run: every `runs/<digest>/result.json` must show `agent_stats.text_only: true`;
  in the no-initial-chart run it must be `false`.
- Image-tool use (reviewer request): count rows in `runs/<digest>/trajectory.jsonl` with
  `tool == "view_image"` and `observation.ok == true`, grouped by `args.path`.
  `tool_counts` in `result.json` also counts failed calls.

## Output

Per run: `episodes/<task_id>.json` with scores, and `runs/<digest>/` with `trajectory.jsonl`
(every tool call with its arguments, including written code), `result.json` (including the
model's text for every turn and `tool_counts`), the final `workspace/`, and `step_images/` when
capture is on. `runs_index.json` maps digests to task IDs.

Cost reference: the paper's Haiku Track B run cost $68.17 for 1,900 episodes ($0.036 per
episode). The 200-task human-evaluation run cost $6.35.
