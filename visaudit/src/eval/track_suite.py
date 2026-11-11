"""Build and materialize the frozen A/B/C evaluation suite without model calls."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import shutil

from . import modalities, tracks

VERSION = "1.4"
PROTOCOL = "abc-uncapped-v1"
BASE_FIELDS = {
    "ground_truth_code": "gt/code.py", "ground_truth_render": "gt/render.png",
    "table": "gt/table.csv", "summary": "gt/summary.txt",
}
PUBLIC_NAMES = {"ground_truth_code": "ground_truth.py", "ground_truth_render": "ground_truth.png",
                "table": "table.csv", "summary": "summary.txt"}
MODELS = ("google-gemma-4-31b-it-a100", "mistral-large-3-675b-2512-fp4",
          "qwen3-vl-235b-a22b-instruct", "ministral-3-8b-2512", "granite-vision-4-1-4b")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def inside(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"path outside package: {name}")
    return path


def category(item):
    parts = item["combination"].split("+")
    return parts[0] if len(parts) == 1 else f"multi{len(parts)}__{item['combination']}"


def validate_id(ident):
    if not isinstance(ident, str) or Path(ident).name != ident or ident in {".", ".."}:
        raise ValueError("invalid instance identity")


def validate_clean(items, base_sources):
    source_ids = [item["source_id"] for item in items]
    if len(items) != 300 or len(set(source_ids)) != 300 or base_sources.intersection(source_ids):
        raise ValueError("need 300 unique additional clean sources disjoint from the flawed set")
    for item in items:
        source = item["source_id"]
        if len(source) != 8 or any(c not in "0123456789abcdef" for c in source):
            raise ValueError("invalid source identity")


def validate_base(items):
    if len(items) != 1900 or len({i["id"] for i in items}) != 1900:
        raise ValueError("expected 1,900 unique flawed tasks")
    if Counter(i["arity"] for i in items) != {"single": 400, "pair": 900, "triple": 600}:
        raise ValueError("base composition changed")
    if Counter(i["group"] for i in items) != {"A": 950, "B": 950}:
        raise ValueError("base group composition changed")
    scenarios = Counter((i["arity"], i["combination"]) for i in items)
    if len(scenarios) != 106 or any(n != {"single": 40, "pair": 25, "triple": 10}[k[0]]
                                    for k, n in scenarios.items()):
        raise ValueError("base scenario quotas changed")
    for item in items:
        validate_id(item["id"])
        components = tracks.flaw_components(category(item))
        if (len(components) != item["n_flaws"] or len(set(components)) != len(components)
                or item.get("clean", False) is not False
                or item["n_flaws"] != {"single": 1, "pair": 2, "triple": 3}[item["arity"]]
                or sorted(components) != sorted(f["category"] for f in item["flaws"])
                or not set(components) <= tracks.FLAW_HINTS.keys()
                or set(item["files"]) != set(BASE_FIELDS) | {"flawed_code", "flawed_render"}):
            raise ValueError("invalid flaw metadata: " + item["id"])


def build_suite(base, clean_selection, out, *, project, seed=42):
    base, out, project = Path(base).resolve(), Path(out).resolve(), Path(project).resolve()
    if out.exists():
        raise FileExistsError("use a new output directory; frozen suites are never overwritten")
    manifest, selection = read(base / "dataset.json"), read(clean_selection)
    validate_base(manifest["items"])
    if selection.get("validation", {}).get("release_sha256") != sha(base / "dataset.json"):
        raise ValueError("clean selection is not bound to this base dataset")
    selected = selection["items"]
    sources = {item["id"].split("__", 1)[0] for item in manifest["items"]}
    validate_clean(selected, sources)
    original_hashes = read(base / "artifact_hashes.json")
    base_bindings = {}
    for item in manifest["items"]:
        for relative in item["files"].values():
            source = inside(base, relative)
            if not source.is_file() or sha(source) != original_hashes.get(relative):
                raise ValueError("base input hash mismatch: " + relative)
            base_bindings[relative] = {"source": relative, "sha256": original_hashes[relative]}
    base_bindings["dataset.json"] = {"source": "dataset.json", "sha256": sha(base / "dataset.json")}
    if (base / "scoring").is_dir():
        scorer_sources = [p for p in (base / "scoring").rglob("*")
                          if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
        scorer_root = base / "scoring"
    else:
        scorer_sources = [base / "score_submission.py", base / "env-lock.txt"]
        scorer_sources += sorted((base / "chartrepair_eval").glob("*.py"))
        scorer_root = base
        if not (base / "chartrepair_eval" / "__init__.py").is_file():
            raise ValueError("base standalone scorer is incomplete")
    if not scorer_sources:
        raise ValueError("base standalone scorer is missing")
    for source in scorer_sources:
        source = inside(base, str(source.relative_to(base)))
        target = "scoring/" + str(source.relative_to(scorer_root))
        base_bindings[target] = {"source": str(source.relative_to(base)), "sha256": sha(source)}
    # Validate all external inputs before creating a partial package.
    for item in selected:
        for field in BASE_FIELDS:
            source = inside(project, item["files"][field])
            if not source.is_file() or sha(source) != item["files_sha256"][field]:
                raise ValueError("clean input hash mismatch: " + item["source_id"])
    out.mkdir(parents=True)
    for relative, binding in base_bindings.items():
        dest = inside(out, relative)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(inside(base, binding["source"]), dest)
        if sha(dest) != binding["sha256"]:
            raise ValueError("base changed during copy: " + relative)
    write(out / "base_bindings.json", {"schema": 1,
          "base_dataset_sha256": sha(base / "dataset.json"), "files": base_bindings})
    clean_items = []
    for item in selected:
        ident = item["source_id"] + "__reference"
        files = {}
        for field in BASE_FIELDS:
            relative = f"instances/{ident}/{PUBLIC_NAMES[field]}"
            dest = inside(out, relative)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(inside(project, item["files"][field]), dest)
            if sha(dest) != item["files_sha256"][field]:
                raise ValueError("clean input changed during copy: " + item["source_id"])
            files[field] = relative
        clean_items.append({"id": ident, "source_id": item["source_id"], "clean": True,
                            "arity": "clean", "n_flaws": 0, "combination": "", "flaws": [],
                            "group": None, "library": item["library"], "chart_type": item["chart_type"],
                            "source_dataset": "ChartNet core_permissive", "files": files})
    write(out / "clean_dataset.json", {"meta": {"name": "chartrepairbench-clean-300", "size": 300,
          "version": VERSION, "source": "ChartNet core_permissive (CDLA-Permissive-2.0)",
          "selection_sha256": sha(clean_selection)}, "items": clean_items})

    all_items = manifest["items"] + clean_items
    track_files = {}
    base_prompt = modalities.instruction(modalities.ALL)
    for track in tracks.TRACKS:
        items = list(all_items if track == "C" else manifest["items"])
        if track == "C":
            random.Random(seed).shuffle(items)
        rows = []
        for item in items:
            clean = item.get("clean", False)
            cat = "clean" if clean else category(item)
            prompt = tracks.instruction_for(base_prompt, track, cat, clean)
            digest = hashlib.sha256(prompt.encode()).hexdigest()
            relative = f"prompts/{digest}.txt"
            dest = out / relative
            dest.parent.mkdir(exist_ok=True)
            dest.write_text(prompt)
            rows.append({"id": item["id"], "clean": clean, "category": cat,
                         "prompt": relative, "prompt_sha256": digest})
        payload = {"schema": 1, "track": track, "protocol_version": PROTOCOL,
                   "step_budget": None, "wall_time_s": None, "items": rows,
                   "total": len(rows), "seed": seed}
        if track == "C":
            payload["track_c_mode"] = "explicit"
        relative = f"tracks/{track}.json"
        write(out / relative, payload)
        track_files[track] = {"path": relative, "sha256": sha(out / relative), "tasks": len(rows)}
    write(out / "suite.json", {"schema": 1, "version": VERSION, "protocol_version": PROTOCOL,
        "base_dataset_sha256": sha(out / "dataset.json"), "clean_dataset_sha256": sha(out / "clean_dataset.json"),
        "base_bindings_sha256": sha(out / "base_bindings.json"),
        "unique_instances": 2200, "tracks": track_files, "models": list(MODELS),
        "main_episodes": 30000, "step_budget": None, "wall_time_s": None,
        "modalities": list(modalities.ALL), "kernel_timeout_s": 30, "http_timeout_s": 180,
        "capture_timeout_s": 120, "max_output_tokens": 8000, "temperature": 0,
        "stopping": "finish, no-valid-tool guard, provider failure or explicit interruption; no episode step/time cap"})
    write(out / "files.sha256.json", {str(p.relative_to(out)): sha(p)
          for p in sorted(out.rglob("*")) if p.is_file()})
    return verify_suite(out)


def verify_suite(suite):
    suite = Path(suite).resolve()
    expected = read(suite / "files.sha256.json")
    for name, digest in expected.items():
        if sha(inside(suite, name)) != digest:
            raise ValueError("suite file changed: " + name)
    base, clean, meta = read(suite / "dataset.json"), read(suite / "clean_dataset.json"), read(suite / "suite.json")
    for name, field in (("dataset.json", "base_dataset_sha256"),
                        ("clean_dataset.json", "clean_dataset_sha256"),
                        ("base_bindings.json", "base_bindings_sha256")):
        if sha(suite / name) != meta[field]:
            raise ValueError("suite manifest binding mismatch: " + name)
    bindings = read(suite / "base_bindings.json")
    if bindings["base_dataset_sha256"] != meta["base_dataset_sha256"]:
        raise ValueError("base dataset identity mismatch")
    for name, binding in bindings["files"].items():
        inside(suite, binding["source"])
        if sha(inside(suite, name)) != binding["sha256"]:
            raise ValueError("base content changed: " + name)
    validate_base(base["items"])
    ids = {i["id"] for i in base["items"]}
    clean_ids = {i["id"] for i in clean["items"]}
    validate_clean(clean["items"], {ident.split("__", 1)[0] for ident in ids})
    if len(clean_ids) != 300 or ids & clean_ids:
        raise ValueError("invalid clean population")
    for item in clean["items"]:
        validate_id(item["id"])
        if (item["id"] != item["source_id"] + "__reference" or item.get("clean") is not True
                or item["n_flaws"] != 0 or item["flaws"] or item["combination"]
                or item["arity"] != "clean" or set(item["files"]) != set(BASE_FIELDS)):
            raise ValueError("invalid clean metadata")
    if (meta["step_budget"] is not None or meta["wall_time_s"] is not None
            or meta["unique_instances"] != 2200 or meta["main_episodes"] != 30000):
        raise ValueError("suite metadata differs from the declared protocol")
    by_id = {item["id"]: item for item in base["items"] + clean["items"]}
    base_prompt = modalities.instruction(modalities.ALL)
    for track, wanted in (("A", ids), ("B", ids), ("C", ids | clean_ids)):
        manifest = read(suite / f"tracks/{track}.json")
        binding = meta["tracks"][track]
        if (binding["path"] != f"tracks/{track}.json" or binding["tasks"] != len(wanted)
                or sha(inside(suite, binding["path"])) != binding["sha256"]):
            raise ValueError("track manifest binding mismatch: " + track)
        if manifest["track"] != track or manifest["total"] != len(wanted):
            raise ValueError("track metadata mismatch")
        if track == "C" and manifest.get("track_c_mode") != "explicit":
            raise ValueError("Track C requires the explicit population mode")
        if len(manifest["items"]) != len(wanted) or {i["id"] for i in manifest["items"]} != wanted:
            raise ValueError("track selection mismatch: " + track)
        if manifest["step_budget"] is not None or manifest["wall_time_s"] is not None:
            raise ValueError("suite protocol is not uncapped")
        for item in manifest["items"]:
            if item["clean"] != (item["id"] in clean_ids):
                raise ValueError("clean flag mismatch")
            cat = "clean" if item["clean"] else category(by_id[item["id"]])
            if item["category"] != cat:
                raise ValueError("track category mismatch")
            prompt = inside(suite, item["prompt"])
            if sha(prompt) != item["prompt_sha256"]:
                raise ValueError("prompt binding mismatch")
            if prompt.read_text() != tracks.instruction_for(base_prompt, track, cat, item["clean"]):
                raise ValueError("track instruction differs from its declared diagnosis")
    return meta


def materialize_runtime(suite, out):
    """Create pool-independent runner inputs for all tracks from a verified suite."""
    suite, out = Path(suite).resolve(), Path(out).resolve()
    verify_suite(suite)
    if out.exists():
        raise FileExistsError("runtime output already exists")
    items = read(suite / "dataset.json")["items"] + read(suite / "clean_dataset.json")["items"]
    records = {}
    for item in items:
        clean = item.get("clean", False)
        cat = "clean" if clean else category(item)
        files = dict(BASE_FIELDS)
        if not clean:
            files.update(flawed_code=f"perturbed_llm/{cat}/code.py", flawed_render=f"perturbed_llm/{cat}/render.png")
        for field, dest in files.items():
            target = inside(out, str(Path("samples") / item["id"] / dest))
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(inside(suite, item["files"][field]), target)
        if not clean:
            write(out / "samples" / item["id"] / "perturbed_llm" / cat / "perturbation.json",
                  {"sample_folder": item["id"], "category": cat, "flaws": item["flaws"],
                   "description": " ".join(f.get("description", "") for f in item["flaws"])})
        records[item["id"]] = {"instance": item["id"], "public_id": item["id"], "category": cat,
            "clean": clean, "source_id": item["id"].split("__", 1)[0], "arity": item["arity"],
            "group": item["group"], "library": item["library"], "chart_type": item["chart_type"]}
    paths = {}
    for track in tracks.TRACKS:
        selection = read(suite / f"tracks/{track}.json")
        manifest = {"name": f"v{VERSION}-{track}-{PROTOCOL}", "total": selection["total"],
                    "protocol_version": PROTOCOL, "track": track, "step_budget": None, "wall_time_s": None,
                    "source_manifest_sha256": sha(suite / f"tracks/{track}.json"),
                    "items": [records[i["id"]] for i in selection["items"]]}
        if track == "C":
            manifest["track_c_mode"] = "explicit"
        paths[track] = out / f"{track}.json"
        write(paths[track], manifest)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path)
    ap.add_argument("--clean-selection", type=Path)
    ap.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--materialize", type=Path, help="materialize this existing suite instead of building")
    args = ap.parse_args()
    if args.materialize:
        print(json.dumps({k: str(v) for k, v in materialize_runtime(args.materialize, args.out).items()}))
    else:
        if not args.base or not args.clean_selection:
            ap.error("--base and --clean-selection are required")
        print(json.dumps(build_suite(args.base, args.clean_selection, args.out, project=args.project), indent=2))


if __name__ == "__main__":
    main()
