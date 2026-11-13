# Full-run core logs

These archives contain the complete text-based audit logs and rule scores for the
full 6,000-task OpenRouter experiments. Rendered images and copied workspaces are
excluded.

| Archive | Model | Tasks | SHA-256 |
|---|---|---:|---|
| `deepseek-full-6000-core-logs.tar.zst` | `deepseek/deepseek-v4.1-flash` | 6,000 | `b81d2356f5ee7773e71afbf769d15b2b9bc57363e3a921cd236f8f5b24355b63` |
| `gemini-full-6000-merged-core-logs.tar.zst` | `google/gemini-3.8-flash` | 6,000 | `86e7cdc8e1dc2f19a145a06d9150510c7603f0e10d1f9beddd323987d4cfffcc` |

Each final task includes `result.json`, `trajectory.jsonl`, `process.log`, task and
agent configuration, token usage, and rule-based metrics. Both archives contain
1,900 Track A, 1,900 Track B, and 2,200 Track C results, with no missing rule scores
or rule-scoring errors.

The Gemini archive merges the initial run and `retry-v3` by task ID, preferring the
successful retry on overlap. Its `run_logs/` directory retains launcher and retry
logs. Two historical trajectories contained an old OpenRouter credential; the
credential is replaced with `[REDACTED_OPENROUTER_API_KEY]` in this public archive.

Verify and extract:

```bash
sha256sum -c SHA256SUMS
tar --zstd -xf deepseek-full-6000-core-logs.tar.zst
tar --zstd -xf gemini-full-6000-merged-core-logs.tar.zst
```
