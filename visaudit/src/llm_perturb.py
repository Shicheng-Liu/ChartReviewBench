"""LLM-generated perturbations: the model rewrites the whole script, we verify it.

Flow per job (one sample x one category):
  1. cheap static preconditions (job_preconditions: is this pair worth a call?)
  2. cache lookup — keyed by (sample_id, category, hash of the GT code)
  3. one API call: system prompt (cached) + category instruction + GT code + summary + CSV head
  4. structured output: full perturbed script + edit description + evidence + applicable flag
  5. gates.check_llm_edit verifies the model changed ONLY what the category allows;
     on failure, retry once with the validator's complaint fed back in
  6. write to cache; run_pipeline picks it up from there

Costs nothing and calls nothing without --allow-api: the default is a dry run that
reports how many calls would be made.

Usage:
    .venv/bin/python src/llm_perturb.py                      # dry run + cost estimate
    .venv/bin/python src/llm_perturb.py --report-jobs        # job counts per category, no cost
    .venv/bin/python src/llm_perturb.py --allow-api --limit 20
    .venv/bin/python src/llm_perturb.py --allow-api          # everything missing
"""
import argparse
import fcntl
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import httpx
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gates
import job_preconditions
import perturb_prompts as P
import transforms

ROOT = Path(__file__).resolve().parent.parent
# The ENDPOINT backend lives in src/v2 and imports its siblings as `src.…`, so the
# repo root has to be importable too when this file runs as a script.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
V0 = ROOT / "data/v0"
SAMPLES = V0 / "samples"
CACHE = V0 / "llm_perturbations.json"
JOBS = V0 / "llm_jobs.json"

# override with CRB_MODEL if the id needs to change
MODEL = os.environ.get("CRB_MODEL", "claude-opus-4-8")
MAX_RETRIES = 0   # no retry, no rejection: measure first
# Decision of 01.09.: the generator is ENDPOINT-only (Mistral Medium 3.5). The
# Anthropic path stays for the historical Sonnet caches and for comparisons.
ENDPOINT_MODEL = os.environ.get("CRB_ENDPOINT_MODEL", "mistral-medium-3-5-128b-a100")


class ENDPOINTClient:
    """A generator on a ENDPOINT endpoint with the same prompts as the Anthropic path.

    Uses the provider-neutral V2 backend (shared credential router over every
    configured key, bounded per-key concurrency, status-aware retries) and the
    V2 JSON contract parser, and maps the answer onto the v1.2 cache record.
    """

    def __init__(self, slug: str = ENDPOINT_MODEL, *, concurrency_per_key: int = 8,
                 timeout_seconds: float = 300.0):
        from v2.generator import (  # noqa: WPS433 - optional provider import
            DecodingConfig, ENDPOINTBackend, ENDPOINTCredentialRouter, endpoint_api_keys,
        )
        router = ENDPOINTCredentialRouter(endpoint_api_keys(),
                                      max_in_flight_per_key=concurrency_per_key)
        self.backend = ENDPOINTBackend(slug, credential_router=router,
                                   timeout_seconds=timeout_seconds)
        self.decoding = DecodingConfig(temperature=0.0, max_tokens=16_000, seed=42)
        self.slug = slug

    @property
    def model(self) -> str:
        try:
            return self.backend.resolve_served_model()
        except Exception:  # noqa: BLE001 - served id is provenance, not a gate
            return self.slug


ENDPOINT_BASE = os.environ.get("ENDPOINT_BASE", "")
_outage_lock = threading.Lock()


def _wait_for_endpoint(slug: str, max_wait: float = 6 * 3600.0) -> bool:
    """Block while the gateway does not answer; True once it does again.

    ENDPOINT drops out for minutes at a time.  A wave that keeps going through an
    outage turns every job of that stretch into an uncached refusal, and the
    top-up rounds then re-buy them.  One thread probes, the others queue on
    the lock, so the whole wave pauses and resumes together.
    """
    from v2.generator import endpoint_api_keys

    with _outage_lock:
        t0 = time.time()
        while True:
            try:
                r = httpx.get(f"{ENDPOINT_BASE}/{slug}/v1/models",
                              headers={os.environ.get("ENDPOINT_KEY_HEADER", "X-API-Key"): endpoint_api_keys()[0]}, timeout=30)
                if r.status_code == 200:
                    return True
            except Exception:  # noqa: BLE001 - that is the outage we are waiting out
                pass
            if time.time() - t0 > max_wait:
                return False
            print("  ENDPOINT unreachable, waiting 60 s", flush=True)
            time.sleep(60)


def _generate_endpoint(client, job, code, summary, csv_text, previous_flaws=()):
    from v2.generator import GenerationRequest, build_prompt, parse_response

    req = GenerationRequest(
        source_id=str(job["sample_id"]), category=job["category"],
        library=str(job["library"]), chart_type=str(job["chart_type"]),
        code=code, summary=summary, csv_text=csv_text,
        previous_flaws=tuple(previous_flaws),
    )
    system, user = build_prompt(req)
    reply = None
    for attempt in range(2):
        try:
            reply = client.backend.complete(system, user, client.decoding)
            break
        except Exception as exc:  # noqa: BLE001 - one dead call is one refusal
            # First failure: is the gateway down?  Then wait it out and retry
            # once; a healthy gateway means the error was this call's own.
            if attempt == 0 and _wait_for_endpoint(client.slug):
                continue
            return None, "endpoint_error_%s" % type(exc).__name__[:24]
    if reply is None:
        return None, "endpoint_error_exhausted"
    try:
        # v1.3 binds no target for LLM categories, so an edit with an incomplete
        # V2 target is still an edit (see parse_response).
        out, _ = parse_response(reply.raw_text, job["category"], strict_target=False)
    except Exception as exc:  # noqa: BLE001
        return None, "unparsed_%s" % type(exc).__name__[:24]
    served = reply.served_model or client.model
    if not out.get("applicable"):
        return {"category": job["category"], "applicable": False,
                "decline_reason": str(out.get("decline_reason", ""))[:300],
                "model": served, "provider": "endpoint"}, \
            f"declined: {str(out.get('decline_reason', ''))[:80]}"
    perturbed = out.get("perturbed_code", "")
    ok, note, metrics = gates.check_llm_edit(code, perturbed, job["category"])
    return {
        "category": job["category"],
        "perturbed_code": perturbed,
        "description": out.get("edit_description", ""),
        "evidence": out.get("evidence") or {},
        "target": out.get("target") or {},
        "model": served,
        "provider": "endpoint",
        "prompt": "v2-system+shared-category-instructions",
        "parse_mode": "lenient-target",
        "edit_ok": ok,
        "edit_note": note,
        "edit_metrics": metrics,
    }, "ok"
# adaptive thinking is not available on every tier (Haiku rejects it with a 400)
NO_THINKING = ("haiku",)


def THINKING(model):
    return {} if any(t in model for t in NO_THINKING) else {"thinking": {"type": "adaptive"}}


CSV_HEAD_LINES = 12
# What the mentors asked for: 100 flawed charts per category, 1000 in total.
TARGET_PER_CATEGORY = 100


class Evidence(BaseModel):
    """What the flaw contradicts. A bare `dict` here would serialise to a schema
    with no properties, so the model can literally return nothing — every field
    must be spelled out."""
    summary: list[str] = Field(
        default_factory=list,
        description="Strings that must appear VERBATIM in summary.txt — quote the ORIGINAL "
                    "(true) value the flaw contradicts, e.g. the original title")
    min_summary: int = Field(
        default=0,
        description="If >0, only this many of the 'summary' strings need to match")
    table: list[str] = Field(
        default_factory=list, description="Strings that must appear verbatim in table.csv")
    table_nums: list[float] = Field(
        default_factory=list,
        description="Numbers that must appear in table.csv — the ORIGINAL value, not the injected one")
    either: list[str] = Field(
        default_factory=list, description="Strings that must appear in summary.txt OR table.csv")
    summary_any_color: bool = Field(
        default=False, description="Weakest form: only assert the summary names some colour")

    def to_dict(self) -> dict:
        d = {}
        if self.summary:
            d["summary"] = self.summary
            if self.min_summary:
                d["min_summary"] = self.min_summary
        if self.table:
            d["table"] = self.table
        if self.table_nums:
            d["table_nums"] = self.table_nums
        if self.either:
            d["either"] = self.either
        if self.summary_any_color:
            d["summary_any_color"] = True
        return d


class Perturbation(BaseModel):
    applicable: bool = Field(description="False if this category does not fit this chart")
    decline_reason: str = Field(default="", description="One sentence, only when applicable is False")
    perturbed_code: str = Field(default="", description="The COMPLETE modified script")
    edit_description: str = Field(default="", description="One sentence describing the injected flaw")
    evidence: Evidence = Field(default_factory=Evidence,
                               description="Which untouched evidence the flaw contradicts")


def cache_key(sample_id, category, gt_code):
    h = hashlib.sha256(gt_code.encode("utf-8")).hexdigest()[:12]
    return f"{sample_id}|{category}|{h}"


def load_cache(path: Path = CACHE) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_cache(new_records: dict, path: Path = CACHE) -> dict:
    """Merge `new_records` into the cache file under a lock, atomically.

    Two producers write this file (the ENDPOINT generator and the deterministic
    axis injector), and a wave of one can run while the other is still going.
    A plain read-at-start / write-at-checkpoint would let the later writer
    drop everything the earlier one added in between, and a reader (the
    instance builder) could see a half-written file. So: take an exclusive
    lock, re-read, merge, write to a temp file and rename. Returns the merged
    cache so the caller's in-memory view stays current.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            merged = load_cache(path)
            merged.update(new_records)
            fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(merged, stream, indent=2, ensure_ascii=False)
            os.replace(tmp, path)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    return merged


def build_jobs():
    """One job per (viable sample, applicable category).

    The set is the UNION of two answers to "is this pair worth an API call":
    the deterministic transform finding a rewritable AST node, and the much
    looser static precondition in job_preconditions. The transform asks the wrong
    question for this path — it has to splice source text, while the model
    rewrites the whole script — and asking it starved swapped-legend-labels,
    dropped-series and axis-truncation far below the 100 instances the benchmark
    needs. Every job records which side proposed it, so the widening stays
    measurable instead of asserted (see --report-jobs).
    """
    index = json.loads((V0 / "index.json").read_text())
    # donor titles first — t_wrong_title needs them to propose a candidate at all
    donors = []
    for e in index:
        t = transforms.extract_title((SAMPLES / e["folder"] / "gt/code.py").read_text(encoding="utf-8"))
        if t:
            donors.append(t)
    jobs = []
    for e in index:
        d = SAMPLES / e["folder"]
        exec_json = d / "gt/exec.json"
        if not exec_json.exists():
            continue
        st = json.loads(exec_json.read_text())
        if st.get("status") != "ok":
            continue
        code = (d / "gt/code.py").read_text(encoding="utf-8")
        meta = json.loads((d / "meta.json").read_text())
        summary = (d / "gt/summary.txt").read_text(encoding="utf-8")
        csv_text = (d / "gt/table.csv").read_text(encoding="utf-8")
        deterministic = {c["category"]
                         for c in transforms.perturb_all(code, meta, summary, csv_text, donors)
                         if c.get("status") == "candidate"}
        loose = {cat for cat in job_preconditions.CATEGORIES
                 if job_preconditions.applies(cat, code, meta, summary, csv_text,
                                              donor_titles=donors)}
        for category in sorted(deterministic | loose):
            jobs.append({"folder": e["folder"], "sample_id": meta["id"],
                         "category": category, "chart_type": meta["chart_type"],
                         "library": meta["library"],
                         "source": ("both" if category in deterministic and category in loose
                                    else "deterministic" if category in deterministic
                                    else "precondition")})
    return jobs


def report_jobs(jobs):
    """Per-category job counts before and after the widening. The 'vorher' column
    is what the deterministic gate alone would have proposed."""
    before = Counter(j["category"] for j in jobs if j["source"] in ("deterministic", "both"))
    after = Counter(j["category"] for j in jobs)
    print(f"\nJobs je Kategorie · Ziel {TARGET_PER_CATEGORY} Instanzen")
    print(f"  {'Kategorie':28s} {'vorher':>7s} {'nachher':>8s} {'delta':>7s}   Ziel erreichbar")
    for cat in job_preconditions.CATEGORIES:
        b, a = before[cat], after[cat]
        print(f"  {cat:28s} {b:7d} {a:8d} {a - b:+7d}   "
              f"{'ja' if a >= TARGET_PER_CATEGORY else 'NEIN'}")
    print(f"  {'SUMME':28s} {sum(before.values()):7d} {sum(after.values()):8d} "
          f"{sum(after.values()) - sum(before.values()):+7d}")
    short = [c for c in job_preconditions.CATEGORIES if after[c] < TARGET_PER_CATEGORY]
    if short:
        print(f"  unter Ziel: {', '.join(short)}")


def build_user_turn(job, code, summary, csv_text, preamble=""):
    csv_head = "\n".join(csv_text.splitlines()[:CSV_HEAD_LINES])
    return (
        f"{preamble}{P.CATEGORY_INSTRUCTIONS[job['category']]}\n\n"
        f"--- sample_id: {job['sample_id']} | category: {job['category']} | "
        f"library: {job['library']} | chart_type: {job['chart_type']} ---\n\n"
        f"## ground-truth script (code.py)\n```python\n{code}\n```\n\n"
        f"## summary.txt (TRUE — never edit)\n{summary}\n\n"
        f"## table.csv, first rows (TRUE — never edit)\n```\n{csv_head}\n```"
    )


def generate_one(client, job, code, summary, csv_text, preamble="", previous_flaws=()):
    """Returns (record | None, status). Retries once on gate failure.

    `preamble` is prepended to the category instruction. Multi-flaw chaining uses
    it to say that the input script already carries a deliberate flaw: without
    that, the second call sees a script contradicting the summary it is told to
    trust, and helpfully repairs the first flaw while injecting the second.
    `previous_flaws` carries the same information for the ENDPOINT path, whose
    prompt builder phrases it itself.
    """
    if isinstance(client, ENDPOINTClient):
        return _generate_endpoint(client, job, code, summary, csv_text, previous_flaws)
    user_turn = build_user_turn(job, code, summary, csv_text, preamble)
    feedback = ""
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.messages.parse(
                model=MODEL,
                max_tokens=16000,
                **THINKING(MODEL),
                system=[{"type": "text", "text": P.SYSTEM_PROMPT,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user_turn + feedback}],
                output_format=Perturbation,
            )
        except anthropic.APIStatusError as exc:
            return None, f"api_error_{exc.status_code}"
        except anthropic.APIConnectionError:
            return None, "connection_error"
        except Exception as exc:            # noqa: BLE001 - one bad answer, not a dead run
            # A model answer that does not parse (measured: JSON with a trailing
            # comma) raised straight out of here and killed a 2600-job sweep
            # after 1053 jobs. One malformed response is a refusal for this
            # instance, never a reason to stop the run.
            return None, "unparsed_%s" % type(exc).__name__[:24]

        if resp.stop_reason == "refusal" or resp.parsed_output is None:
            return None, "refusal_or_unparsed"
        out = resp.parsed_output

        if not out.applicable:
            # Cached like any other answer. A decline is a real, paid-for verdict
            # about this (sample, category); not recording it means every later
            # run buys the same "no" again.
            return {"category": job["category"], "applicable": False,
                    "decline_reason": out.decline_reason[:300],
                    "model": MODEL}, f"declined: {out.decline_reason[:80]}"

        # NOT a gate: the verdict is recorded so we can measure what a gate
        # WOULD have rejected, but nothing is thrown away here.
        ok, note, metrics = gates.check_llm_edit(code, out.perturbed_code, job["category"])
        return {
            "category": job["category"],
            "perturbed_code": out.perturbed_code,
            "description": out.edit_description,
            "evidence": out.evidence.to_dict(),
            "model": MODEL,
            "edit_ok": ok,
            "edit_note": note,
            "edit_metrics": metrics,
        }, "ok"
    return None, "exhausted"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-api", action="store_true", help="actually spend money")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--category", default=None, help="restrict to one category")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--per-category", type=int, default=None,
                    help="cap jobs per category — the main cost lever")
    ap.add_argument("--report-jobs", action="store_true",
                    help="print the per-category job count before and after the "
                         "precondition widening, then exit")
    ap.add_argument("--provider", choices=("endpoint", "anthropic"), default="endpoint",
                    help="endpoint (default since 01.09.: Mistral Medium 3.5 over the "
                         "shared credential router) or anthropic (historical path)")
    ap.add_argument("--model", default=None,
                    help="ENDPOINT endpoint slug or Anthropic model id (defaults: "
                         f"{ENDPOINT_MODEL} / {MODEL})")
    ap.add_argument("--endpoint-concurrency", type=int, default=8,
                    help="in-flight requests per ENDPOINT key")
    ap.add_argument("--llm-axis", action="store_true",
                    help="also send axis-truncation jobs to the LLM (historical path). "
                         "Since 01.09. the axis flaw is the deterministic clip written by "
                         "axis_inject.py, so these jobs are skipped by default")
    ap.add_argument("--retry-failures", action="store_true",
                    help="only the (sample, category) jobs listed in llm_perturb_failures.jsonl "
                         "that are still not in the cache")
    args = ap.parse_args()
    if args.provider == "anthropic" and args.model:
        globals()["MODEL"] = args.model

    jobs = build_jobs()
    JOBS.write_text(json.dumps(jobs, indent=2))
    if args.report_jobs:
        report_jobs(jobs)
        return
    cache = load_cache()
    retry = None
    if args.retry_failures:
        failures = V0 / "llm_perturb_failures.jsonl"
        retry = set()
        if failures.exists():
            for line in failures.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    retry.add((rec["folder"], rec["category"]))

    todo, per_cat_taken, in_chars = [], Counter(), 0
    for job in jobs:
        if args.category and job["category"] != args.category:
            continue
        if job["category"] == "axis-truncation" and not args.llm_axis:
            continue
        if retry is not None and (job["folder"], job["category"]) not in retry:
            continue
        if args.per_category and per_cat_taken[job["category"]] >= args.per_category:
            continue
        d = SAMPLES / job["folder"]
        code = (d / "gt/code.py").read_text(encoding="utf-8")
        if cache_key(job["sample_id"], job["category"], code) in cache:
            continue
        per_cat_taken[job["category"]] += 1
        in_chars += len(code) + len((d / "gt/summary.txt").read_text(encoding="utf-8"))
        todo.append(job)
    if args.limit:
        todo = todo[:args.limit]

    by_cat = Counter(j["category"] for j in todo)
    n_declined = sum(1 for r in cache.values() if r.get("applicable") is False)
    print(f"{len(jobs)} mögliche Jobs · {len(cache) - n_declined} generiert + "
          f"{n_declined} abgelehnt im Cache · {len(todo)} offen")
    for cat, n in sorted(by_cat.items()):
        print(f"   {cat:28s} {n:5d}")

    if not args.allow_api:
        # estimate from the real files: ~4 chars per token, plus the shared system
        # prompt (cached after the first call) and the model echoing the script back
        code_tok = in_chars / 4 if todo else 0
        sys_tok = len(P.SYSTEM_PROMPT) / 4 * len(todo)
        out_tok = code_tok * 1.15          # full script back + description + evidence
        price = {"opus": (5.0, 25.0), "sonnet": (3.0, 15.0), "haiku": (1.0, 5.0)}
        report_jobs(jobs)
        print(f"\nDRY RUN — kein API-Call. Geschätzt für {len(todo)} Jobs:")
        print(f"  ~{code_tok / 1e3:.0f}k Input-Tokens (Code+Summary) + "
              f"~{sys_tok / 1e3:.0f}k System-Prompt · ~{out_tok / 1e3:.0f}k Output")
        for name, (pin, pout) in price.items():
            cost = (code_tok + sys_tok * 0.1) / 1e6 * pin + out_tok / 1e6 * pout
            print(f"  {name:7s} ≈ ${cost:6.2f}")
        print("\nKostenhebel: --per-category N (z. B. 60) · --limit N · --category X")
        print("Mit --allow-api ausführen, um wirklich zu generieren.")
        return

    if args.provider == "endpoint":
        client = ENDPOINTClient(args.model or ENDPOINT_MODEL,
                            concurrency_per_key=args.endpoint_concurrency)
        print(f"generator: ENDPOINT {client.slug} (served as {client.model})")
    else:
        client = anthropic.Anthropic()
        print(f"generator: Anthropic {MODEL}")
    stats = Counter()
    done = [0]
    fresh = {}

    def work(job):
        d = SAMPLES / job["folder"]
        code = (d / "gt/code.py").read_text(encoding="utf-8")
        summary = (d / "gt/summary.txt").read_text(encoding="utf-8")
        csv_text = (d / "gt/table.csv").read_text(encoding="utf-8")
        rec, status = generate_one(client, job, code, summary, csv_text)
        return job, code, rec, status

    failures = V0 / "llm_perturb_failures.jsonl"
    with ThreadPoolExecutor(max_workers=args.workers) as ex, \
            failures.open("a", encoding="utf-8") as failed:
        for job, code, rec, status in ex.map(work, todo):
            stats[status.split(":")[0]] += 1
            if rec:
                fresh[cache_key(job["sample_id"], job["category"], code)] = rec
            elif status != "ok":
                # An uncached refusal (transport error, unparsable answer) is
                # retried by the next wave; log it so a systematic failure is
                # visible before it eats a whole wave.
                failed.write(json.dumps({"folder": job["folder"], "category": job["category"],
                                         "status": status, "model": getattr(client, "slug", MODEL)})
                             + "\n")
                failed.flush()
            done[0] += 1
            if done[0] % 25 == 0 or done[0] == len(todo):
                cache = save_cache(fresh)
                fresh = {}
                print(f"  {done[0]}/{len(todo)} · {dict(stats)}", flush=True)

    cache = save_cache(fresh)
    n_declined = sum(1 for r in cache.values() if r.get("applicable") is False)
    print(f"\nfertig: {dict(stats)} — Cache hält {len(cache) - n_declined} "
          f"Perturbationen und {n_declined} Ablehnungen")


if __name__ == "__main__":
    main()
