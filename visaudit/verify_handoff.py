"""Verify file hashes and the frozen 1,900+300 task/prompt contract offline."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    index = ROOT / "PACKAGE_FILES_SHA256.json"
    if index.exists():
        for name, expected in json.loads(index.read_text()).items():
            path = ROOT / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise SystemExit("Package file mismatch: " + name)
    from src.eval.track_suite import verify_suite
    suite = verify_suite(ROOT / "data/frozen-v1.4")
    history = json.loads((ROOT / "history/conditions.json").read_text())
    for c in history["conditions"]:
        path = ROOT / c["outcomes_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == c["outcomes_sha256"]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == c["selected_episodes"]
        assert len({r["canonical_id"] for r in rows}) == len(rows)
        assert all(r["protocol_id"] in c["protocols"] for r in rows)
    print(json.dumps({"verified": True, "instances": suite["unique_instances"],
                      "history": history["counts"]}, indent=2))


if __name__ == "__main__":
    main()
