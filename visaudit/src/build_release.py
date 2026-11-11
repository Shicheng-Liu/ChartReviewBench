"""Combine the three v1.3 draws into the one evaluation set and package it.

The 04.09. sync fixed the shape: 40 instances per single-flaw category, 25 per
pair and 10 per triple — 400 + 900 + 600 = 1,900 over all 106 scenarios. reviewer
asked for the mix to be shuffled and split into two groups, and for one shippable
artifact so the collaborators can evaluate on exactly the same data. Only numbers
measured on this set go into the paper.

The package is self-contained: for every item the ground-truth script, table,
summary and render, plus the flawed script and its render, plus a manifest that
names the injected flaws. A collaborator needs nothing from this repository.

    .venv/bin/python src/build_release.py --name chartrepairbench-v1.3-eval
    .venv/bin/python src/build_release.py --name … --package   # + tar.gz
"""
import argparse
import hashlib
import json
import random
import shutil
import sys
import tarfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V0 = ROOT / "data/v0"
PARTS = (("eval-single-400", "single"), ("eval-multi2-900", "pair"),
         ("eval-multi3-600", "triple"))
SEED = 42


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _flaws(row: dict) -> list:
    """The injected edits of an instance, single or chained."""
    if row.get("edits"):
        return [{"category": e["category"], "description": e.get("description", "")}
                for e in row["edits"]]
    return [{"category": row["category"], "description": row.get("description", "")}]


def collect():
    """Every item of the three draws, with its manifest row and arity."""
    manifest = {}
    for name in ("manifest_llm.jsonl", "manifest_multiflaw.jsonl"):
        path = V0 / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                manifest[(row["sample_folder"], row["category"])] = row

    items = []
    for dataset_name, arity in PARTS:
        data = json.loads((V0 / f"datasets/{dataset_name}.json").read_text())
        for item in data["items"]:
            physical = item.get("flaw_dir") or item["category"]
            row = manifest[(item["instance"], physical)]
            items.append({
                "id": row["instance_id"],
                "arity": arity,
                "n_flaws": len(_flaws(row)),
                "combination": (physical.split("__", 1)[1] if "__" in physical
                                else physical),
                "flaws": _flaws(row),
                "library": item.get("library") or row.get("library"),
                "chart_type": item.get("chart_type") or row.get("chart_type"),
                "source_dataset": dataset_name,
                "_folder": item["instance"],
                "_physical": physical,
            })
    return items


def split(items, seed=SEED):
    """Shuffle the mix and cut it into two groups of equal size (reviewer, 04.09.).

    Stratified by scenario, so both groups carry every category and every
    combination: a random cut of 1,900 items over 106 scenarios would otherwise
    give one group six of a rare triple and the other four.
    """
    rng = random.Random(seed)
    by_scenario = {}
    for item in items:
        by_scenario.setdefault((item["arity"], item["combination"]), []).append(item)
    groups = {"A": [], "B": []}
    # An odd bucket has to give its extra item to one side; alternating which
    # side that is keeps the two groups the same size (968/932 otherwise).
    odd_turn = 0
    for key in sorted(by_scenario):
        bucket = sorted(by_scenario[key], key=lambda i: i["id"])
        rng.shuffle(bucket)
        first = "A" if odd_turn % 2 == 0 else "B"
        second = "B" if first == "A" else "A"
        if len(bucket) % 2:
            odd_turn += 1
        for index, item in enumerate(bucket):
            groups[first if index % 2 == 0 else second].append(item)
    for name in groups:
        groups[name].sort(key=lambda i: i["id"])
    return groups


def split_by_source(items, seed=SEED):
    """Split so that a source chart never appears in both groups.

    The per-instance split keeps the scenario counts exactly even but lets the
    same source chart, and with it the same table, summary and ground truth,
    land in both groups: measured on the 04.09. build, 1,140 of 1,900 instances.
    For two independent evaluation halves the source has to be the unit. Sources
    are placed largest-first into whichever group is currently smaller, which
    keeps the halves within a few instances of each other and, measured on the
    same build, still puts every one of the 106 scenarios in both groups.
    """
    rng = random.Random(seed)
    by_source = {}
    for item in items:
        by_source.setdefault(item["id"].split("__")[0], []).append(item)
    order = sorted(by_source, key=lambda s: (-len(by_source[s]), s))
    rng.shuffle(order[:0])                    # order is deterministic by design
    groups = {"A": [], "B": []}
    for source in order:
        target = "A" if len(groups["A"]) <= len(groups["B"]) else "B"
        groups[target].extend(by_source[source])
    for name in groups:
        groups[name].sort(key=lambda i: i["id"])
    return groups


def write(name: str, items, groups, out_dir: Path, with_files=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    group_of = {i["id"]: g for g, rows in groups.items() for i in rows}
    records = []
    for item in items:
        folder, physical = item.pop("_folder"), item.pop("_physical")
        src = V0 / "samples" / folder
        flawed = src / "perturbed_llm" / physical
        rel = Path("instances") / item["id"]
        record = {**item, "group": group_of[item["id"]],
                  "files": {"ground_truth_code": str(rel / "ground_truth.py"),
                            "table": str(rel / "table.csv"),
                            "summary": str(rel / "summary.txt"),
                            "ground_truth_render": str(rel / "ground_truth.png"),
                            "flawed_code": str(rel / "flawed.py"),
                            "flawed_render": str(rel / "flawed.png")}}
        if with_files:
            dest = out_dir / rel
            dest.mkdir(parents=True, exist_ok=True)
            for source, target in ((src / "gt/code.py", "ground_truth.py"),
                                   (src / "gt/table.csv", "table.csv"),
                                   (src / "gt/summary.txt", "summary.txt"),
                                   (src / "gt/render.png", "ground_truth.png"),
                                   (flawed / "code.py", "flawed.py"),
                                   (flawed / "render.png", "flawed.png")):
                if source.exists():
                    shutil.copy2(source, dest / target)
        records.append(record)

    counts = Counter(r["arity"] for r in records)
    meta = {
        "name": name,
        "version": "1.3",
        "task": ("Each instance is a chart whose plotting script carries one or more "
                 "deliberately injected flaws. table.csv and summary.txt are untouched "
                 "and describe the correct chart. The agent must find the contradiction, "
                 "repair flawed.py and produce a faithful chart."),
        "size": len(records),
        "composition": {"single": counts["single"], "pair": counts["pair"],
                        "triple": counts["triple"]},
        "per_scenario": {"single": 40, "pair": 25, "triple": 10},
        "scenarios": {"single_categories": 10, "pair_combinations": 36,
                      "triple_combinations": 60},
        "groups": {g: len(rows) for g, rows in groups.items()},
        "group_note": ("A and B are a stratified random split of the same set: every "
                       "category and every combination appears in both."),
        "validity": ("v1.3: the flawed script runs, the figure-object footprint against "
                     "the ground truth is non-empty, the bound category postconditions "
                     "hold (missing-legend, axis-truncation), and a strict majority of at "
                     "least three model families that are not the generator's family "
                     "calls the flaw detectable from table and summary. Multi-flaw "
                     "instances satisfy this per component."),
        "generator": ("Mistral Medium 3.5 (ENDPOINT) and Claude Sonnet 5 for the historical "
                      "part; axis-truncation is a deterministic clipping transform."),
        "source": "ChartNet core_permissive (CDLA-Permissive-2.0)",
        "seed": SEED,
    }
    (out_dir / "dataset.json").write_text(
        json.dumps({"meta": meta, "items": records}, indent=1, ensure_ascii=False),
        encoding="utf-8")
    (out_dir / "README.md").write_text(_readme(meta), encoding="utf-8")
    return meta


def _readme(meta) -> str:
    c = meta["composition"]
    return f"""# {meta['name']}

{meta['task']}

## What is in here

| part | instances | per scenario | scenarios |
|---|---:|---:|---:|
| single flaw | {c['single']} | 40 | 10 categories |
| two flaws | {c['pair']} | 25 | 36 combinations |
| three flaws | {c['triple']} | 10 | 60 combinations |
| **total** | **{meta['size']}** | | **106** |

`dataset.json` lists every instance: its injected flaws, the library and chart
type, the group, and the six files that belong to it under `instances/<id>/`.

> **`dataset.json` is the answer key.** It names and describes the injected
> flaws. Hand the agent the four files listed below, never the manifest.

- `ground_truth.py`, `ground_truth.png` — the correct script and its render
- `flawed.py`, `flawed.png` — the script the agent receives, and its render
- `table.csv`, `summary.txt` — untouched; they describe the correct chart

Groups **A** ({meta['groups']['A']}) and **B** ({meta['groups']['B']}) are a
stratified random split of the same set; every category and combination appears
in both.

## How an instance counts as valid

{meta['validity']}

## Running and scoring

`PROMPT.md` holds the exact task text the agent is given, per track and per
modality set, and how the four files are laid out in its working directory.

`score_submission.py` scores one repaired script against its instance with the
benchmark's own metric:

    python score_submission.py --instance instances/<id> \
        --submission repaired.py --category "<combination from dataset.json>"

It prints the metric vector: executability, recovery, preservation and data
fidelity, each continuous and as an all-or-nothing binary. Recovery counts only
the figure-object paths the injected flaws touched, preservation only the paths
outside that footprint. There is no weighted overall score, on purpose. Feeding
`ground_truth.py` back in scores 1.0 everywhere; feeding `flawed.py` back in
scores recovery 0.0 and preservation 1.0. Use those two as your smoke test.

Use the pinned environment in `env-lock.txt`: matplotlib decides autoscale
floats, and a different version moves preservation. `CRB_PYTHON` selects the
interpreter that runs the plot scripts.

## Provenance

Charts come from {meta['source']}. Flawed scripts were written by
{meta['generator']} Seed {meta['seed']}.
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default="chartrepairbench-v1.3-eval")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-files", action="store_true", help="manifest only, no copies")
    ap.add_argument("--split-by-source", action="store_true",
                    help="keep every source chart entirely in one group. Without it "
                         "the split is per instance and 60 %% of instances share their "
                         "source chart with the other group, so A and B are not "
                         "independent samples.")
    ap.add_argument("--package", action="store_true", help="also write a tar.gz")
    args = ap.parse_args(argv)

    out_dir = args.out or (ROOT / "release" / args.name)
    items = collect()
    groups = split_by_source(items) if args.split_by_source else split(items)
    meta = write(args.name, items, groups, out_dir, with_files=not args.no_files)
    size = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
    print(f"{meta['size']} instances -> {out_dir}  ({size / 1e6:.1f} MB)")
    print(f"  {meta['composition']} · groups {meta['groups']}")
    if args.package:
        archive = out_dir.parent / f"{args.name}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(out_dir, arcname=args.name)
        print(f"  archive {archive} ({archive.stat().st_size / 1e6:.1f} MB) "
              f"sha256 {_sha256(archive)[:16]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
