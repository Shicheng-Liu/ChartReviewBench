"""Run the QC judge on ENDPOINT (a configured inference service) — cross-family + prompt sensitivity.

Two checks in one runner:

  1. Cross-family. The quality check has to use a different language model than
     the generator. Opus 4.8 is a different model but the same family, so the
     *identical* prompt is re-run across five open-weight families on ENDPOINT to
     measure whether the verdict is model-dependent.
  2. Prompt sensitivity. The runner also takes prompt variants and measures how
     far the verdict moves when the wording moves.

The system prompt and the per-instance prompt are IMPORTED from llm_judge.py —
never re-typed — so a ENDPOINT verdict and an Anthropic verdict answer the exact same
question. The only unavoidable difference: Anthropic gets the schema through its
structured-output API, ENDPOINT models get the same schema as an appended JSON
instruction (identical for every ENDPOINT model, so cross-model comparison is fair).

Results are keyed by (instance_id, judge_model, prompt_variant) and merged into
data/v0/judge_results_endpoint.json — nothing is ever overwritten (storage mandate).

Usage:
    ENDPOINT_API_KEY=... .venv/bin/python src/endpoint_judge.py --model llama-3-3-70b-instruct
    ENDPOINT_API_KEY=... .venv/bin/python src/endpoint_judge.py --model qwen2-5-72b-instruct \
        --variant strict --limit 100
    .venv/bin/python src/endpoint_judge.py --list
"""
import argparse
import fcntl
import json
import os
import re
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_judge import SYSTEM, build_prompt, CSV_HEAD  # identical prompt, by import

ROOT = Path(__file__).resolve().parent.parent
V0 = ROOT / "data/v0"
SAMPLES = V0 / "samples"
OUT = V0 / "judge_results_endpoint.json"
BASE = os.environ.get("ENDPOINT_BASE", "")

# endpoint slug -> served model name is resolved live from /v1/models, so a
# renamed checkpoint cannot silently 404 a whole run.
MODELS = [
    "llama-3-3-70b-instruct",
    "qwen2-5-72b-instruct",
    "deepseek-v3-2",
    "gpt-oss-120b-a100",
    "granite-4-1-30b",
]

JSON_TAIL = """

Respond with a single JSON object and nothing else:
{"detectable": true or false,
 "basis": "summary" | "table" | "both" | "none",
 "evidence_quote": "the exact fragment that exposes the flaw, empty if not detectable",
 "match_type": "explicit" | "semantic",
 "reasoning": "one sentence"}"""

# Prompt variants for the sensitivity study. "current" is the production
# wording — the other three are deliberate perturbations of the framing, not
# improvements.
VARIANTS = {
    "current": SYSTEM,
    # scaffolding removed: no EXPLICIT/SEMANTIC teaching, no strictness clause
    "minimal": """You are the quality gate of ChartRepairBench, a benchmark of deliberately flawed charts.

Each benchmark instance was built by taking a correct chart and injecting exactly one flaw into
its plotting code. The chart's underlying table (table.csv) and its text summary (summary.txt)
were NOT touched, so they still describe the CORRECT chart.

Your single question: given only this summary and this table, could an agent notice this flaw?""",
    # same content, thumb on the reject side
    "strict": SYSTEM + """

Additional instruction: apply a high bar. Only answer detectable when the contradiction is
beyond reasonable doubt. When you are uncertain, answer not detectable.""",
    # same content, thumb on the accept side
    "lenient": SYSTEM + """

Additional instruction: apply a generous bar. Answer detectable whenever a careful agent could
plausibly notice the contradiction. When you are uncertain, answer detectable.""",
}

_lock = threading.Lock()
_served_cache = {}


def served_name(client, slug):
    """Resolve the served model name for an endpoint (cached per process)."""
    with _lock:
        if slug in _served_cache:
            return _served_cache[slug]
    r = client.get(f"{BASE}/{slug}/v1/models", timeout=60)
    r.raise_for_status()
    name = r.json()["data"][0]["id"]
    with _lock:
        _served_cache[slug] = name
    return name


def parse_verdict(text):
    """Lenient JSON extraction — open models like to wrap or annotate their JSON."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if "detectable" not in d:
        return None
    det = d["detectable"]
    if isinstance(det, str):
        det = det.strip().lower() in ("true", "yes", "1")
    return {"llm_detectable": bool(det),
            "basis": str(d.get("basis", ""))[:20],
            "match_type": str(d.get("match_type", ""))[:20],
            "quote": str(d.get("evidence_quote", ""))[:160],
            "reasoning": str(d.get("reasoning", ""))[:220]}


def wait_for_endpoint(client, slug, max_wait=6 * 3600):
    """Block while the gateway does not answer; True once it does again.

    ENDPOINT drops out for minutes at a time.  Without this, every verdict during
    an outage became an error, the dataset builder saw missing coverage, and
    the top-up planner generated instances that only lacked a judge.
    """
    t0 = time.time()
    while time.time() - t0 < max_wait:
        try:
            if client.get(f"{BASE}/{slug}/v1/models", timeout=30).status_code == 200:
                return True
        except Exception:  # noqa: BLE001 - that is the outage we are waiting out
            pass
        time.sleep(60)
    return False


def judge_one(client, inst, slug, variant, retries=3):
    d = SAMPLES / inst["sample_folder"]
    summary = (d / "gt/summary.txt").read_text(encoding="utf-8")
    csv_text = (d / "gt/table.csv").read_text(encoding="utf-8")
    rec = {"instance_id": inst["instance_id"], "category": inst["category"],
           "judge_model": f"endpoint/{slug}", "prompt_variant": variant,
           "rule_verdict": inst["gates"]["solvable"],
           "description": inst.get("description", "")[:150]}
    body = {
        "model": served_name(client, slug),
        "messages": [{"role": "system", "content": VARIANTS[variant] + JSON_TAIL},
                     {"role": "user", "content": build_prompt(inst, summary, csv_text)}],
        "max_tokens": 700,
        "temperature": 0.0,
    }
    last = ""
    for attempt in range(retries):
        try:
            r = client.post(f"{BASE}/{slug}/v1/chat/completions", json=body,
                            timeout=httpx.Timeout(240.0, connect=30.0))
            if r.status_code != 200:
                last = f"http_{r.status_code}"
                if r.status_code >= 500 or r.status_code == 429:
                    # gateway or pod trouble, not this instance: wait it out
                    wait_for_endpoint(client, slug)
                else:
                    time.sleep(2 * (attempt + 1))
                continue
            msg = r.json()["choices"][0]["message"]
            # reasoning models (gpt-oss) may put the answer in a reasoning field
            text = msg.get("content") or msg.get("reasoning") or ""
            v = parse_verdict(text)
            if v is None:
                last = "unparsed"
                time.sleep(1)
                continue
            rec.update(v)
            u = r.json().get("usage") or {}
            rec["usage"] = {"in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0)}
            return rec
        except Exception as exc:                     # noqa: BLE001 - report, keep going
            last = type(exc).__name__
            if isinstance(exc, httpx.TransportError):
                wait_for_endpoint(client, slug)          # an outage, not this instance's fault
            else:
                time.sleep(3 * (attempt + 1))
    rec["error"] = last or "failed"
    return rec


def load_prev():
    return json.loads(OUT.read_text()) if OUT.exists() else []


def save(results):
    """Merge by (instance_id, judge_model, prompt_variant) — never lose a verdict.

    Four judge families run as four processes on the same file, so the merge
    happens under an exclusive file lock and the write is a rename. Without
    that, two processes checkpointing at the same moment drop each other's
    verdicts, and a reader can see a half-written file.
    """
    lock_path = OUT.with_suffix(OUT.suffix + ".lock")
    with _lock, open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            prev = load_prev()
            seen = {(r["instance_id"], r["judge_model"], r.get("prompt_variant", "current"))
                    for r in results}
            keep = [p for p in prev
                    if (p["instance_id"], p["judge_model"], p.get("prompt_variant", "current")) not in seen]
            fd, tmp = tempfile.mkstemp(dir=OUT.parent, prefix=OUT.name + ".", suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(results + keep, stream, indent=2, ensure_ascii=False)
            os.replace(tmp, OUT)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def multiflaw_components():
    """One judging item per (multi-flaw instance, component flaw).

    The judge question is single-flaw by construction ("exactly one flaw was
    injected"), and a multi-flaw instance is a chain of exactly such edits,
    each with its own description. So every component is judged on its own
    description against the untouched summary and table, and the dataset
    builder demands a cross-family majority for every component. The verdict
    key is instance_id::category, so the votes never mix with the single-flaw
    instance of the same source chart.
    """
    path = V0 / "manifest_multiflaw.jsonl"
    if not path.exists():
        return []
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("gates", {}).get("exec_ok"):
            continue
        for edit in row.get("edits") or []:
            items.append({
                "instance_id": "%s::%s" % (row["instance_id"], edit["category"]),
                "category": edit["category"],
                "description": edit.get("description", ""),
                "sample_folder": row["sample_folder"],
                "gates": {"solvable": row.get("gates", {}).get("solvable")},
            })
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-3-3-70b-instruct",
                    help=f"ENDPOINT endpoint slug (known: {', '.join(MODELS)})")
    ap.add_argument("--variant", default="current", choices=sorted(VARIANTS))
    ap.add_argument("--limit", type=int, default=None, help="first N runnable instances")
    ap.add_argument("--sample", type=int, default=None,
                    help="seeded random sample of N runnable instances (same seed as annotation)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--redo", action="store_true", help="re-judge instances already done")
    ap.add_argument("--list", action="store_true", help="show what is already in the result file")
    ap.add_argument("--multiflaw", action="store_true",
                    help="judge the components of multi-flaw instances "
                         "(manifest_multiflaw.jsonl) instead of single-flaw instances; "
                         "one verdict per (instance, component), keyed instance_id::category")
    ap.add_argument("--probe", action="store_true",
                    help="one short chat completion against --model; exit 0 if it answers, "
                         "1 otherwise (a pod can answer /v1/models and still drop every "
                         "chat request, as qwen2-5-72b did on 02.09.)")
    args = ap.parse_args()

    if args.probe:
        key = os.environ.get("ENDPOINT_API_KEY")
        if not key:
            sys.exit("ENDPOINT_API_KEY not set")
        # A real judge request, not a four-token ping: on 02.09. qwen2-5-72b
        # answered pings in a second and dropped every judge-sized request.
        manifest = [json.loads(l) for l in (V0 / "manifest_llm.jsonl").read_text().splitlines()
                    if l.strip()]
        inst = next(m for m in manifest if m["gates"]["exec_ok"])
        try:
            with httpx.Client(headers={os.environ.get("ENDPOINT_KEY_HEADER", "X-API-Key"): key, "Content-Type": "application/json"}) as client:
                served_name(client, args.model)
                rec = judge_one(client, inst, args.model, "current", retries=1)
        except Exception as exc:  # noqa: BLE001 - the probe's whole point
            print(f"{args.model}: {type(exc).__name__}")
            sys.exit(1)
        ok = "llm_detectable" in rec
        print(f"{args.model}: {'answers' if ok else rec.get('error', 'failed')}")
        sys.exit(0 if ok else 1)

    if args.list:
        cnt = Counter((r["judge_model"], r.get("prompt_variant", "current"))
                      for r in load_prev() if "error" not in r)
        for (m, v), n in sorted(cnt.items()):
            print(f"{n:5d}  {m:40s} {v}")
        return

    key = os.environ.get("ENDPOINT_API_KEY")
    if not key:
        sys.exit("ENDPOINT_API_KEY not set")

    if args.multiflaw:
        runnable = multiflaw_components()
    else:
        manifest = [json.loads(l) for l in (V0 / "manifest_llm.jsonl").read_text().splitlines()]
        runnable = [m for m in manifest if m["gates"]["exec_ok"]]
    if args.sample:
        import random
        pool = sorted(runnable, key=lambda m: m["instance_id"])
        random.Random(7).shuffle(pool)          # same seed as make_annotation.py
        runnable = pool[:args.sample]

    todo = runnable
    if not args.redo:
        done = {r["instance_id"] for r in load_prev()
                if r["judge_model"] == f"endpoint/{args.model}"
                and r.get("prompt_variant", "current") == args.variant and "error" not in r}
        todo = [m for m in runnable if m["instance_id"] not in done]
    if args.limit:
        todo = todo[:args.limit]

    print(f"{args.model} · variant={args.variant} · judging {len(todo)} of {len(runnable)}",
          flush=True)
    if not todo:
        return

    t0 = time.time()
    headers = {os.environ.get("ENDPOINT_KEY_HEADER", "X-API-Key"): key, "Content-Type": "application/json"}
    limits = httpx.Limits(max_connections=args.workers + 4, max_keepalive_connections=args.workers)
    with httpx.Client(headers=headers, limits=limits) as client:
        # A dead gateway at start is an outage to wait out, not a bad slug:
        # on 02.09. a four-hour outage made all three gate judges exit within
        # two seconds, and the pipeline went on as if they had run.
        if not wait_for_endpoint(client, args.model):
            sys.exit("ENDPOINT did not come back within the wait limit")
        served_name(client, args.model)          # fail fast on a bad slug
        results = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for i, rec in enumerate(ex.map(lambda m: judge_one(client, m, args.model, args.variant),
                                           todo), 1):
                results.append(rec)
                if i % 50 == 0:
                    rate = i / (time.time() - t0)
                    print(f"  {i}/{len(todo)} · {rate:.1f}/s · "
                          f"{sum(1 for r in results if 'error' in r)} errors", flush=True)
                if i % 100 == 0:
                    save(results)                # checkpoint: an interrupted run keeps its work

    save(results)
    ok = [r for r in results if "error" in r or "llm_detectable" in r]
    good = [r for r in results if "llm_detectable" in r]
    errs = Counter(r["error"] for r in results if "error" in r)
    det = sum(1 for r in good if r["llm_detectable"])
    print(f"\n{len(good)} verdicts, {len(results) - len(good)} errors {dict(errs) if errs else ''}")
    print(f"detectable {det} · not detectable {len(good) - det} "
          f"({det / max(1, len(good)):.1%} valid) · {time.time() - t0:.0f}s")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
