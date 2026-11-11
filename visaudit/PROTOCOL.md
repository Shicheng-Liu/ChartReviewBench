# Protocol scope and defaults

The evaluated agent's system prompt is in
`runtime/src/chartsandbox/agent.py`; tool schemas are in `tools.py` beside it.
Task wording is built by `src/eval/modalities.py` and `src/eval/tracks.py`.
These instructions describe a persistent Python workspace, one tool call per
turn, viewing the rendered result, and explicitly finishing. Track A appends
the diagnosis; B asks the agent to find defects; C first asks whether the chart
is already faithful and specifies run-without-edit plus finish for clean inputs.

The generic command defaults remain historical: 20 tool actions and 900 seconds.
The main frozen A/B/C suite explicitly used no tool-step and no episode wall-time
cap. Specify `--no-step-limit --no-wall-time-limit` for that condition. Per-call
timeouts, loop guards, provider response limits, and technical retries are
separate settings; absence of an episode cap does not remove those mechanisms.
The initial observation discloses a finite step budget in budget ablations.

The frozen paper population is 1,900 flawed tasks (400 single, 900 double,
600 triple) and 300 additional clean sources for C. Do not silently use legacy
clean-twin expansion as the paper's C population. Use the explicit Track-C
manifest. The synthetic demo has one flawed task and is only a plumbing check.

Provider model IDs must be specified exactly. Token counts sum recorded model
responses in the retained episode; cache input needs to be included where the
provider reports it separately. Failed/discarded attempts are not the same as
retained-episode cost. Provider tokenizers differ.

The source snapshot contains the current core runner, including historical
branches and generic CLI defaults. The paper experiments span multiple guard,
runtime, scheduling and transport versions. Saved campaign settings, per-task
retained-attempt selection and provider configurations are in `history/`.
Benchmark inputs and release data artifacts are not included. See HANDOFF.md
for the included evidence and the historical metadata discrepancy. Raw private
trajectories and discarded workspaces are not included. The bundle does not
collapse versions into a claim of one identical experimental protocol.

Source names related to the private deployment transport are genericized only
in this code supplement. Model IDs, task wording, score definitions, clean handling and
tool semantics are preserved. Frozen result files must not be relabelled or
rewritten to pretend they came from this modified package.
