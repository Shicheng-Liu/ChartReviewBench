"""Verify this code-only archive and retained-outcome counts; standard library only."""
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent


def require(ok, message):
    if not ok:
        raise SystemExit(message)


def package_path(name):
    p = PurePosixPath(name)
    require(not p.is_absolute() and ".." not in p.parts, "Unsafe package path")
    return ROOT / p


def main():
    manifest = json.loads((ROOT / "PACKAGE_FILES_SHA256.json").read_text())
    require("PACKAGE_FILES_SHA256.json" not in manifest, "Manifest must exclude itself")
    require(bool(manifest), "Empty manifest")
    for name, expected in manifest.items():
        path = package_path(name)
        require(path.is_file() and not path.is_symlink(), "Missing package file: " + name)
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                "Package file mismatch: " + name)

    history = json.loads((ROOT / "history/conditions.json").read_text())
    conditions = history["conditions"]
    ids, total, current, current_n = set(), 0, 0, 0
    for condition in conditions:
        require(condition["id"] not in ids, "Duplicate condition")
        ids.add(condition["id"])
        name = condition["outcomes_path"]
        require(name in manifest, "Unindexed outcome file: " + name)
        raw = package_path(name).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == condition["outcomes_sha256"],
                "Outcome hash mismatch: " + name)
        rows = [json.loads(line) for line in raw.splitlines()]
        require(len(rows) == condition["selected_episodes"], "Episode count mismatch: " + name)
        require(len({r["canonical_id"] for r in rows}) == len(rows),
                "Duplicate task within condition: " + name)
        require(all(r["protocol_id"] in condition["protocols"] for r in rows),
                "Unknown protocol: " + name)
        total += len(rows)
        if condition["current_paper_scope"]:
            current += len(rows)
            current_n += 1
    require((len(conditions), total, current_n, current) == (29, 56600, 27, 52800),
            "Unexpected historical/current condition or outcome totals")
    print(json.dumps({"verified": True, "scope": "code and retained outcomes; frozen data separate",
                      "files_verified": len(manifest), "historical_conditions": len(conditions),
                      "historical_outcomes": total, "current_conditions": current_n,
                      "current_outcomes": current, "model_calls": 0}, indent=2))


if __name__ == "__main__":
    main()
