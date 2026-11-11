# VisAudit: code and retained-outcome supplement

This code-only archive contains evaluation, runtime, construction and scoring
code, saved configuration evidence, and retained measured outcomes for five
models. It does **not** contain the frozen benchmark images, plotting programs,
tables or summaries. The included scripts reproduce aggregate results from
retained outcomes; rerunning the benchmark requires inputs not included here.

## Verify and reproduce measured means

From this extracted `visaudit-code/` directory, use Python 3 with its standard
library only; no installation, credentials or model requests are needed:

```sh
python3 verify_code_package.py
python3 summarize_retained.py > retained_means.csv
```

The verifier checks every file listed in this code archive's checksum manifest,
the retained-outcome hashes, and the condition/episode counts. The summary uses
saved measured outcomes rather than executing charts or rescoring model outputs.
Scores are proportions with individual defined denominators; input tokens include
cached tokens where recorded. The default scope is 15 main conditions plus 12
interactive ablations: 52,800 outcomes. Two historical single-response conditions
remain archived outside that scope; `--include-historical-single-response` adds
their 3,800 outcomes. Collaborator DeepSeek/Gemini raw runs are not included.

## Contents and execution scope

- `history/`: outcomes, saved run/deployment settings, selection receipts and
  all 43 recovered recorded source versions.
- `src/`, `runtime/`, `scoring/`: evaluation, construction/QC and scoring code.
- `requirements-validated.txt`: dependency versions for the tested environment.
- `VALIDATION.json`: original full-bundle validation and its stated limitations.
- `PACKAGE_FILES_SHA256.json`: checksums for this code-only archive, excluding
  the checksum manifest itself.

See **[HANDOFF.md](HANDOFF.md)** for the supported checks and execution scope.
Frozen-task materialization, the full data verifier and real benchmark runs
require benchmark inputs that are not included in this supplement. The isolated
agent path was tested on Python 3.12/macOS; no Linux isolation backend or
identical cross-platform rendering is claimed. A new provider-backed evaluation
also requires credentials and constitutes a new experiment.

One historical run-metadata file changed after the analysis. Its recorded and
available hashes and the missing original snapshot are disclosed in
`history/conditions.json`. The retained measured outcomes are preserved.
