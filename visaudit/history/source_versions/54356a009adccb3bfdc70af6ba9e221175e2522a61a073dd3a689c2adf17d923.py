"""Identify scorer sources and expose version coverage of recorded episodes."""
import hashlib
import json
from collections import Counter
from pathlib import Path


def scorer_provenance() -> dict:
    root = Path(__file__).resolve().parent
    files = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
             for name in ("_extract.py", "_runner.py", "capture.py", "compare.py",
                          "metrics.py", "axis_rule.py", "sandbox_run.py", "to_sandbox.py",
                          "tracks.py", "rescore.py", "run_benchmark.py", "episode_runner.py", "cancellation.py", "provenance.py",
                          "kernel_isolation.py", "modalities.py", "dataset.py", "providers.py")}
    files["../harness.py"] = hashlib.sha256((root.parent / "harness.py").read_bytes()).hexdigest()
    files["../../env-lock.txt"] = hashlib.sha256(
        (root.parent.parent / "env-lock.txt").read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "files": files}


def scorer_coverage(episodes: list) -> dict:
    versions = Counter()
    for episode in episodes:
        scorer = episode.get("scorer")
        digest = scorer.get("sha256") if isinstance(scorer, dict) else None
        if (isinstance(digest, str) and len(digest) == 64
                and all(c in "0123456789abcdef" for c in digest)):
            versions[digest] += 1
    versioned = sum(versions.values())
    unversioned = len(episodes) - versioned
    consistent = len(versions) == 1 and unversioned == 0
    return {
        "status": "consistent" if consistent else "inconsistent" if versions else "unversioned",
        "consistent": consistent,
        "requires_rescore": not consistent,
        "n_versioned": versioned,
        "n_unversioned": unversioned,
        "versions": dict(sorted(versions.items())),
    }
