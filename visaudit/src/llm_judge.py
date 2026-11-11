"""Stage-2 quality check: an LLM decides whether an injected flaw is detectable.

The rule-based string match has a false-positive problem. When the summary states the title explicitly, an exact match is right; but
when the summary only describes the chart, a paraphrased title is still perfectly
detectable and the string match wrongly rejects it. That case needs a semantic judge.

The judge sees only what a benchmark agent would see (the untouched summary and
table) plus a description of what was changed, and answers one question: could an
agent notice this from these inputs?

Usage:
    ANTHROPIC_API_KEY=... .venv/bin/python src/llm_judge.py --set rejected
    ANTHROPIC_API_KEY=... .venv/bin/python src/llm_judge.py --set accepted --limit 40
    CRB_JUDGE_MODEL=claude-opus-4-8 .venv/bin/python src/llm_judge.py --set unjudged

Results are keyed by (instance_id, judge_model): re-judging with a different
model adds verdicts side by side instead of overwriting earlier ones.
"""
import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
V0 = ROOT / "data/v0"
SAMPLES = V0 / "samples"
OUT = V0 / "judge_results.json"
MODEL = os.environ.get("CRB_JUDGE_MODEL", "claude-sonnet-5")
CSV_HEAD = 15
# USD per MTok (input, output), for the cost line printed after a run
PRICE = {"claude-sonnet-5": (3.0, 15.0), "claude-opus-4-8": (5.0, 25.0)}


class Verdict(BaseModel):
    detectable: bool = Field(
        description="True if an agent holding only the chart, the summary and the table "
                    "could notice that something is wrong")
    basis: str = Field(
        description="Which evidence exposes it: 'summary', 'table', 'both', or 'none'")
    evidence_quote: str = Field(
        default="", description="The exact fragment of the summary or table that exposes "
                                "the flaw. Empty if not detectable.")
    match_type: str = Field(
        default="", description="'explicit' if the evidence names the changed element "
                                "literally, 'semantic' if it only follows from the meaning")
    reasoning: str = Field(description="One sentence.")


SYSTEM = """You are the quality gate of ChartRepairBench, a benchmark of deliberately flawed charts.

Each benchmark instance was built by taking a correct chart and injecting exactly one flaw into
its plotting code. The chart's underlying table (table.csv) and its text summary (summary.txt)
were NOT touched, so they still describe the CORRECT chart. A benchmark agent is later shown the
flawed chart plus that truthful table and summary, and must notice that something is wrong.

Your single question: given only this summary and this table, could an agent notice this flaw?

An instance is only fair if the answer is yes. Judge the two cases separately:

  * EXPLICIT: the evidence names the changed element literally. If the summary says
    'the title reads "Sales by Month"' and the title was changed, that is explicit and
    the contradiction is direct.
  * SEMANTIC: the evidence does not name the element, but the meaning still contradicts
    the flaw. If the summary describes a chart of expenditure percentages over time and
    the y-axis label was changed to "Total Enrollment", the summary never quotes the axis
    label, yet the contradiction is unmistakable. This still counts as detectable.

Be strict about the opposite case: if the summary is generic enough that the flawed version
would read as perfectly consistent with it, the flaw is NOT detectable and the instance must
be rejected. Example: a legend is removed, but the summary never names the individual series,
so nobody could reconstruct which series is which.

Ignore whether the flaw is visually obvious. Judge only whether the given text and numbers
contradict it. Quote the deciding fragment verbatim."""


def build_prompt(inst, summary, csv_text):
    csv_head = "\n".join(csv_text.splitlines()[:CSV_HEAD])
    return (
        f"## The injected flaw\ncategory: {inst['category']}\n"
        f"what was changed: {inst.get('description', '')}\n\n"
        f"## summary.txt (untouched, describes the CORRECT chart)\n{summary}\n\n"
        f"## table.csv, first rows (untouched)\n```\n{csv_head}\n```\n\n"
        "Could an agent notice this flaw from the summary and table above?"
    )


def judge_one(client, inst):
    d = SAMPLES / inst["sample_folder"]
    summary = (d / "gt/summary.txt").read_text(encoding="utf-8")
    csv_text = (d / "gt/table.csv").read_text(encoding="utf-8")
    rec = {"instance_id": inst["instance_id"], "category": inst["category"],
           "judge_model": MODEL,
           "rule_verdict": inst["gates"]["solvable"],
           "rule_note": inst["gates"]["solvability_note"],
           "description": inst.get("description", "")[:150]}
    try:
        resp = client.messages.parse(
            model=MODEL, max_tokens=4000,
            thinking={"type": "adaptive"},
            system=[{"type": "text", "text": SYSTEM,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": build_prompt(inst, summary, csv_text)}],
            output_format=Verdict,
        )
    except anthropic.APIStatusError as exc:
        rec["error"] = f"api_{exc.status_code}"
        return rec
    if resp.stop_reason == "refusal" or resp.parsed_output is None:
        rec["error"] = "unparsed"
        return rec
    v = resp.parsed_output
    rec.update({"llm_detectable": v.detectable, "basis": v.basis,
                "match_type": v.match_type, "quote": v.evidence_quote[:160],
                "reasoning": v.reasoning[:220],
                "usage": {"in": resp.usage.input_tokens,
                          "out": resp.usage.output_tokens}})
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["rejected", "accepted", "both", "unjudged"],
                    default="rejected")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    manifest = [json.loads(l) for l in (V0 / "manifest_llm.jsonl").read_text().splitlines()]
    runnable = [m for m in manifest if m["gates"]["exec_ok"]]
    rejected = [m for m in runnable if not m["gates"]["solvable"]]
    accepted = [m for m in runnable if m["gates"]["solvable"]]

    prev = json.loads(OUT.read_text()) if OUT.exists() else []
    for p in prev:
        p.setdefault("judge_model", "claude-sonnet-5")  # runs before 2026-08-03

    todo = []
    if args.set == "unjudged":
        done = {p["instance_id"] for p in prev
                if p["judge_model"] == MODEL and "error" not in p}
        todo = [m for m in runnable if m["instance_id"] not in done]
    if args.set in ("rejected", "both"):
        todo += rejected
    if args.set in ("accepted", "both"):
        # stratified sample across categories
        by_cat = {}
        for m in accepted:
            by_cat.setdefault(m["category"], []).append(m)
        per = max(1, (args.limit or 40) // max(1, len(by_cat)))
        for ms in by_cat.values():
            todo += ms[:per]
    if args.limit:
        todo = todo[:args.limit] if args.set != "both" else todo

    print(f"{len(rejected)} rule-rejected · {len(accepted)} rule-accepted · "
          f"judging {len(todo)} with {MODEL}")
    client = anthropic.Anthropic()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(lambda m: judge_one(client, m), todo))

    seen = {(r["instance_id"], r["judge_model"]) for r in results}
    OUT.write_text(json.dumps(
        results + [p for p in prev if (p["instance_id"], p["judge_model"]) not in seen],
        indent=2, ensure_ascii=False))

    ok = [r for r in results if "llm_detectable" in r]
    print(f"\n{len(ok)} beurteilt, {len(results) - len(ok)} Fehler\n")
    conf = Counter((r["rule_verdict"], r["llm_detectable"]) for r in ok)
    print(f"{'Regel':>8s} {'LLM':>8s}  {'n':>4s}   Bedeutung")
    labels = {
        (False, True): "FALSE POSITIVE der Regel: zu Unrecht aussortiert",
        (False, False): "korrekt aussortiert",
        (True, True): "beide einig: valide",
        (True, False): "Regel zu lax: Instanz ist nicht lösbar",
    }
    for (rule, llm), n in sorted(conf.items(), key=lambda x: -x[1]):
        print(f"{str(rule):>8s} {str(llm):>8s}  {n:4d}   {labels[(rule, llm)]}")

    tok_in = sum(r["usage"]["in"] for r in ok if "usage" in r)
    tok_out = sum(r["usage"]["out"] for r in ok if "usage" in r)
    if tok_in and MODEL in PRICE:
        pi, po = PRICE[MODEL]
        print(f"\nTokens: {tok_in:,} in / {tok_out:,} out"
              f" ≈ ${tok_in / 1e6 * pi + tok_out / 1e6 * po:.2f}")

    fp = [r for r in ok if not r["rule_verdict"] and r["llm_detectable"]]
    if fp:
        rate = len(fp) / max(1, len([r for r in ok if not r["rule_verdict"]])) * 100
        print(f"\nFalse-Positive-Rate der Regelprüfung: {len(fp)}/"
              f"{len([r for r in ok if not r['rule_verdict']])} = {rate:.0f}%")
        print(f"Davon per Semantik erkannt: "
              f"{sum(1 for r in fp if r.get('match_type') == 'semantic')}\n")
        for r in fp[:6]:
            print(f"  {r['category']:24s} {r['reasoning'][:95]}")
            print(f"  {'':24s} Beleg: {r['quote'][:90]!r}")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
