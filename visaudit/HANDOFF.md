# Code and retained-outcome supplement

This archive contains the five-model evaluation code and runtime, retained
measured outcomes and saved configuration evidence. The frozen benchmark inputs
are not included. The checksum manifest covers the files in this code supplement.

From the extracted `visaudit-code/` directory, the following commands use only
the Python standard library and make no model requests:

```sh
python3 verify_code_package.py
python3 summarize_retained.py > retained_means.csv
```

The first command verifies the code supplement and retained-outcome bindings.
The second reproduces aggregate means from the saved outcomes. It does not
execute charts, rescore model outputs or rerun the benchmark.

## Install and check

Use Python 3.12 on macOS for the isolated agent path. The standalone scoring
algorithm can run elsewhere on trusted code, but identical rendering across
platforms has not been established. No Linux isolation backend is provided.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-validated.txt
.venv/bin/python -m pip check
.venv/bin/python -B verify_code_package.py
.venv/bin/python -B smoke_check.py
```

## Reproduce the measured means

```sh
.venv/bin/python -B summarize_retained.py > retained_means.csv
```

The CSV contains proportions, not percentages, and an individual defined
denominator for each score. Correct-chart recovery stays undefined. Input-token
means include cache tokens where supplied in the source accounting. The default
scope is 15 main conditions and 12 interactive ablations (52,800 outcomes).
The two historical single-response conditions remain archived explicitly; add
`--include-historical-single-response` to include their 3,800 outcomes.

`history/conditions.json` maps every condition to its retained episode records,
historical protocol/scorer IDs and run-configuration files. Every original
retained source-episode hash was checked during assembly. The JSONL extracts
retain the measured metrics, denominators' null values, tool counts, accounting,
termination state and source hashes. Raw model conversations and workspace
attempt histories are not redistributed here.

The numeric model results from collaborators (DeepSeek and Gemini) are not in
this local evidence collection. Their raw outputs, exact served IDs/settings and
token accounting must come from those collaborators for equivalent reproduction.

## Dataset-dependent execution

The bundled code includes task materialization, chart scoring and model-backed
evaluation interfaces. They require benchmark input images, plotting programs,
tables, summaries and manifests that are not included in this supplement.
Consequently, this archive alone does not reproduce full benchmark runs.
`verify_handoff.py` also requires those inputs; use `verify_code_package.py`
to verify this code-only supplement.

`smoke_check.py` uses the included synthetic example and can check the local
execution/scoring interface after installing the documented dependencies. It is
not a benchmark run. `VALIDATION.json` preserves the historical validation report
for the original full bundle, including checks performed against data that is
not part of this supplement. It is not evidence that this ZIP contains the data.

A new provider-backed evaluation requires credentials and constitutes a new
experiment. Hosted providers use their standard environment variables. The
generic deployment adapter uses `ENDPOINT_BASE`, `ENDPOINT_API_KEY` and
`ENDPOINT_KEY_HEADER`; no account or endpoint is supplied.

## Current reruns versus historical results

The runnable source uses generic deployment configuration and the current bundled
runtime. It does not claim to be one historical executable shared by all runs.
The measured campaigns used multiple guard, scheduling and transport versions,
including fresh P90 tail attempts and technical recovery passes.

`history/configurations/` preserves saved run/deployment settings with personal
paths and deployment addresses generalized. Their original hashes remain the
identities; hashes of generalized contents are recorded separately. Missing
fields stay missing rather than being filled from current defaults.
`history/source_versions/` contains recovered recorded source versions as archival
text, with a mapping of original and packaged hashes. These are evidence for the
historical code, not drop-in executable replacements. The fixed scoring and
extraction core is bound to the frozen suite; surrounding runtime versions differ.

One historical Mistral Large Track-A technical-retry `run.json` no longer
matches its recorded analysis-time hash. The exact older metadata file was not
recovered. This discrepancy is listed in `history/conditions.json` and
`VALIDATION.json`; its currently available settings are not presented as an
exact original snapshot. All 56,600 selected episode-file hashes do match.

A new model run measures a new stochastic/API execution. Replaying the historical
aggregate means uses the saved retained outcomes above. Reconstructing every
original discarded workspace or provider exchange would additionally require raw
private run archives, which are not included. Fresh installation and bounded local
smokes are documented in `VALIDATION.json`; no model-backed run was performed.
