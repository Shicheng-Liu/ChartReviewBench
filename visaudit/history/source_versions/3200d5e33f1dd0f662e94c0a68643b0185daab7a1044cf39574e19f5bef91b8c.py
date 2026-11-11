r"""Run a model over a whole dataset and report the benchmark numbers.

This is the top of the evaluation stack: `dataset` says which instances are in,
`to_sandbox` turns each into a task directory, the sandbox drives the episode,
`metrics` scores the repair, and this module aggregates the result into one
evaluation report.

    .venv/bin/python -m src.eval.run_benchmark --model claude-haiku-4-5 \
        --dataset v0-1000 --modalities chart,code,table,summary \
        --workers 6 --budget-usd 10 --out data/v0/runs/haiku-icts

Four properties this file exists for, in the order they matter.

**The budget is a stop, not a warning.** A run is launched unattended against a
paid API, so before every episode the cap is re-checked against what has already
been spent plus what the episodes currently in flight are expected to cost. When
it would be exceeded the run stops cleanly, writes the report over what finished,
and says how many items it never started. Resumed episodes count as spent: the
cap bounds the total price of the run, not of one attempt.

**Every episode is its own file.** `<out>/episodes/<task_id>.json` is written the
moment an episode ends, so a run that dies at item 700 keeps 699 results and the
next start resumes at 700. `--aggregate-only` rebuilds the report from those
files alone.

**Undefined is not zero.** Recovery is undefined when the flaw left an empty
footprint, preservation when there was nothing outside it to keep. That is a
property of the instance, not a failure of the agent, so a None is excluded from
the mean and counted separately. Averaging it as a zero would quietly punish the
agent for the benchmark's own edge cases.

**One completed submission is one vote — and the categories are not the same
size.** The headline figure is the mean over completed submissions; the macro
figure is the mean over the category means. The two differ and both are reported,
labelled.

Errors are counted, never hidden, and never silently turned into model failures:
an episode that raises is recorded with its traceback and contributes to
`n_errors`, but is excluded from capability means because no completed submission
exists to score. Reports therefore expose both questions separately:
`n_completed` is the denominator for model capability, while `n` and `n_errors`
measure endpoint and harness reliability.

**Executability is our re-execution, not the sandbox's file check.** The two used
to be one column and are not one question. `executability` re-runs the agent's
final `chart.py` and asks whether it draws a chart; it is the gate recovery,
preservation and fidelity already sit behind, so the row is internally
consistent. `produced_png` is the sandbox's `renders_ok` subgoal, an
artifact-existence check on `output.png` that never runs the submission. They
disagree in both directions — a correct script the agent never ran, a stale png
from before the agent broke its script — so the disagreement rate is reported as
a number, per category and overall. It is a measurement of the sandbox, and its
author should see it.

**The agent's working directory must not name the flaw.** The sandbox does not
confine its Python kernel, so the run directory's own name is readable from
inside the episode; named after the task id it spelled out the category, and one
`os.listdir("../..")` gave up every sibling instance's category too. Run
directories are therefore `runs/<digest of the task id>`, the map back lives in
`runs_index.json` beside that directory rather than in it, and the task id is
kept where the agent cannot reach it — in `episodes/<task_id>.json`, which is
also this file's resume key. `sandbox_run` holds the mechanism and the residual
risk; read its docstring before trusting a run.
"""
import argparse
import concurrent.futures
import copy
import datetime
import fcntl
import functools
import hashlib
import json
import math
import os
import shutil
import statistics
import threading
import time
import traceback
from collections import Counter
from pathlib import Path

from . import dataset as dataset_mod
from . import modalities
from . import cancellation, tracks, sandbox_run, to_sandbox
from .provenance import scorer_coverage, scorer_provenance

SCHEMA = 3                       # 3: infrastructure errors are separate from capability means
MODES = ("strict", "lenient")
METRICS = ("data_fidelity", "data_fidelity_binary",
           "recovery", "recovery_binary",
           "preservation", "preservation_binary")

# Cost estimates scale a single-episode calibration by relative model prices;
# the estimate is not a bound on any individual episode's cost.
ANCHOR_MODEL = "claude-sonnet-5"
ANCHOR_USD = 0.038
ANCHOR_STEPS = 6

# The no-op agent from `sandbox_run --offline`: runs the flawed script once and
# stops. It repairs nothing and exists so the whole path can be exercised for
# free, in tests and in a smoke run.
OFFLINE_SCRIPT = [
    {"tool": "execute_python",
     "args": {"code": "exec(compile(open('chart.py').read(), 'chart.py', 'exec'))"}},
    {"tool": "finish", "args": {"message": "offline no-op agent"}},
]


# Error text of a gateway that is down, as opposed to a model that misbehaved.
_OUTAGE_MARKERS = ("ReadTimeout", "ConnectTimeout", "ConnectError", "RemoteProtocolError",
                   "APIConnectionError", "APITimeoutError", "Connection error",
                   "no healthy upstream", "Bad gateway", "502", "503", "504")


def _looks_like_outage(exc) -> bool:
    text = str(exc)
    return any(marker in text for marker in _OUTAGE_MARKERS)


class EpisodeFailed(Exception):
    """An episode that did not finish, carrying what is still known about it.

    The cost matters even in failure: tokens spent before the crash are on the
    bill and must reach the budget, or a run that fails half its episodes would
    keep starting new ones on money it has already spent.

    `produced_png` is the sandbox's verdict and survives a scoring crash, because
    the episode itself ran. Executability does not: it is our own re-execution,
    and if scoring is what broke, we never asked the question.

    `scores` is the metric block for an episode with no scorable submission, taken
    from the instance where that could still be computed (see
    `sandbox_run.score_no_submission`). None means even that failed, and the
    caller falls back to `failed_scores`.
    """

    def __init__(self, message, cost_usd=None, produced_png=None, scores=None,
                 stop_reason=None):
        super().__init__(message)
        self.cost_usd = cost_usd
        self.produced_png = produced_png
        self.scores = scores
        self.stop_reason = stop_reason


# --------------------------------------------------------------------------- cost

def _price(model: str):
    """(input, output) USD per Mtok for a model, or None when it has no entry."""
    try:
        from .providers import load_providers
        load_providers()                              # puts the sibling clone on the path
        from chartsandbox.providers.base import PRICE_PER_MTOK
    except Exception:                                 # no clone, no price table, no estimate
        return None
    ident = (model or "").partition(":")[2] or (model or "")
    ident = ident.strip().lower().partition("@")[0]
    if ident in PRICE_PER_MTOK:
        return PRICE_PER_MTOK[ident]
    for known, price in PRICE_PER_MTOK.items():       # dated snapshot of a known id
        if ident.startswith(known + "-"):
            return price
    return None


def cost_per_episode(model: str):
    """Expected USD for one full-modality episode, or None when unpriced.

    Extrapolated from the one measured episode by the price ratio to the anchor
    model, averaging the input and output ratios because the token split of that
    episode was not recorded. It is an anchor, not a forecast.
    """
    price, anchor = _price(model), _price(ANCHOR_MODEL)
    if price is None or anchor is None:
        return None
    ratio = (price[0] / anchor[0] + price[1] / anchor[1]) / 2
    return round(ANCHOR_USD * ratio, 6)


class Budget:
    """A hard spending cap, safe to consult from several worker threads.

    `start` is the gate: it returns False once the cap would be crossed and from
    then on stays closed, so the remaining workers drain instead of racing each
    other past the limit. Episodes in flight are counted at the running mean, so
    N workers cannot each pass a check the last one would have failed.
    """

    def __init__(self, cap, prior=None, spent=0.0, n_costed=0, enforced=True):
        self.cap = cap
        self.prior = prior
        self.spent = float(spent)
        self.n_costed = int(n_costed)
        self.enforced = bool(enforced) and cap is not None
        self.inflight = 0
        self.skipped = 0
        self.tripped = False
        self._lock = threading.Lock()

    def estimate(self) -> float:
        if self.n_costed:
            return self.spent / self.n_costed
        return self.prior or 0.0

    def start(self) -> bool:
        with self._lock:
            if not self.enforced:
                self.inflight += 1
                return True
            if self.tripped:
                self.skipped += 1
                return False
            projected = self.spent + (self.inflight + 1) * self.estimate()
            if projected > self.cap:
                self.tripped = True
                self.skipped += 1
                return False
            self.inflight += 1
            return True

    def finish(self, cost) -> None:
        with self._lock:
            self.inflight = max(0, self.inflight - 1)
            if cost is not None:
                self.spent += float(cost)
                self.n_costed += 1


# --------------------------------------------------------------------------- jobs

def select(ds: dict, categories=None, limit=None) -> list:
    """The dataset's items after --categories and --limit.

    The dataset's own order is the round-robin draw order, so a truncated run
    stays spread over libraries and chart types instead of collapsing onto
    whichever stratum sorts first.
    """
    items = dataset_mod.items(ds)
    if categories:
        wanted = set(categories)
        unknown = sorted(wanted - {i["category"] for i in items})
        if unknown:
            raise ValueError("dataset %s holds no instance of %s"
                             % (ds.get("name", "?"), ", ".join(unknown)))
        items = [i for i in items if i["category"] in wanted]
    if limit is not None and limit >= 0:
        items = items[:limit]
    return items


def job_variants(track=None, item=None, track_c_mode="twins") -> tuple:
    """The (clean, label) variants one dataset item expands to under a track.

    Explicit Track C populations run each declared clean/flawed item once.
    Legacy populations retain one ground-truth twin per flawed instance.
    """
    if tracks.validate_track(track) != "C":
        if item and item.get("clean"):
            raise ValueError("clean instances are only allowed in Track C")
        return ((False, ""),)
    if track_c_mode == "explicit":
        if item is None or not isinstance(item.get("clean"), bool):
            raise ValueError("explicit Track C requires a boolean clean flag on every item")
        if (item.get("category") == "clean") != item["clean"]:
            raise ValueError("explicit Track C clean flag and category disagree")
        return ((item["clean"], ""),)
    if track_c_mode != "twins":
        raise ValueError("unknown Track C mode %r" % track_c_mode)
    if item and item.get("clean"):
        raise ValueError("clean inputs require track_c_mode='explicit'")
    return ((False, ""), (True, "#clean"))


def track_mode(ds, track):
    return ds.get("track_c_mode", "twins") if track == "C" else "flawed_only"


def selected_protocol_jobs(items, track, track_c_mode):
    """One enumeration for task building, dry-run counts and resumable identity."""
    jobs = [(item, clean, label) for item in items
            for clean, label in job_variants(track, item, track_c_mode)]
    identities = [(item["instance"], dataset_mod.flaw_dir(item), clean) for item, clean, _ in jobs]
    if len(set(identities)) != len(identities):
        raise ValueError("selection contains duplicate task identities")
    return jobs


def build_jobs(items: list, tasks_root: Path, mods, log=print, track=None,
               step_budget=tracks.DEFAULT_STEP_BUDGET, *,
               wall_time_s=tracks.DEFAULT_WALL_TIME_S, track_c_mode="twins") -> tuple:
    """Task directories for the items, built where they do not exist yet.

    Runs in the calling thread on purpose: `build_task` writes into a shared
    tasks root, and doing it up front keeps the workers off each other's files.
    An item whose sources cannot define a task is returned separately rather than
    quietly dropped.
    """
    track = tracks.validate_track(track)
    step_budget = tracks.validate_step_budget(step_budget)
    wall_time_s = tracks.validate_wall_time(wall_time_s)
    jobs, unbuildable = [], []
    for item, clean, label in selected_protocol_jobs(items, track, track_c_mode):
        cancellation.check()
        # The physical flaw folder, which for a split category is not its name.
        fdir = dataset_mod.flaw_dir(item)
        tid = to_sandbox.task_id(item["instance"], fdir, mods, track, step_budget, clean,
                                wall_time_s=wall_time_s)
        task_dir = Path(tasks_root) / tid
        if not (task_dir / "task.yaml").exists():
            try:
                to_sandbox.build_task(to_sandbox.SAMPLES / item["instance"],
                                      fdir, Path(tasks_root), mods,
                                      track=track, step_budget=step_budget,
                                      clean=clean, wall_time_s=wall_time_s)
            except (FileNotFoundError, KeyError, ValueError) as exc:
                log("skipped %s: %s" % (tid, exc))
                unbuildable.append({**item, "task_id": tid, "reason": str(exc)})
                continue
        jobs.append({"task_id": tid, "task_dir": task_dir,
                     "instance": item["instance"],
                     "category": item["category"] + label,
                     "track": track, "step_budget": step_budget, "clean": clean,
                     "wall_time_s": wall_time_s})
    return jobs, unbuildable


# --------------------------------------------------------------------------- the agent

def agent_factory(model: str, *, offline=False, no_images=False, max_tokens=8000, log=print):
    """A zero-argument callable that builds a FRESH agent for every episode.

    Fresh matters twice. The sandbox's provider accumulates `calls` and `usage`
    on the instance and `LLMAgent.reset` does not clear them, so one shared
    provider would report episode 7 as costing everything episodes 1 to 7 cost —
    which is exactly the number the budget is built on. A provider per episode
    also removes the only unsynchronised counter in the parallel path.

    The HTTP client is shared, because it is the expensive object and it is the
    one piece that is safe to share: the SDK clients are thread-safe and pool
    their connections.
    """
    if offline:
        from chartsandbox.agent import ScriptedAgent
        log("agent: offline no-op (no model, nothing is repaired)")
        return (lambda: ScriptedAgent(list(OFFLINE_SCRIPT))), "offline"

    from chartsandbox.agent import LLMAgent
    if sandbox_run.is_hosted(model):
        from .providers import make_provider
        first = make_provider(model, text_only=no_images, max_tokens=max_tokens)
        client = getattr(first, "client", None)
        log("agent: %s (hosted%s)" % (model, ", text-only" if no_images else ""))

        def make():
            provider = (make_provider(model, text_only=no_images, max_tokens=max_tokens,
                                      client=client) if client is not None else
                        make_provider(model, text_only=no_images, max_tokens=max_tokens))
            return LLMAgent(provider)
        return make, model

    # ENDPOINT: the endpoint is asked for its served id once, not once per episode.
    from chartsandbox.providers.vllm_provider import VLLMProvider
    client = sandbox_run.endpoint_client(model)
    served = sandbox_run.endpoint_served_model(client)
    strict = any(k in model for k in sandbox_run.STRICT_ALTERNATION)
    cls = sandbox_run.endpoint_provider(VLLMProvider, no_images, strict)
    log("agent: %s  (served as %s%s)"
        % (model, served, ", strict role alternation" if strict else ""))

    def make():
        return LLMAgent(cls(served, client=client, max_tokens=max_tokens,
                            base_url=str(client.base_url)))
    return make, model


def agent_factory_when_ready(model, *, offline=False, no_images=False, log=print):
    """Wait through ENDPOINT startup outages without consuming episode retries."""
    while True:
        cancellation.check()
        try:
            return agent_factory(model, offline=offline, no_images=no_images, log=log)
        except Exception as exc:
            if offline or sandbox_run.is_hosted(model) or not sandbox_run.transient_provider_error(exc):
                raise
            log("agent: ENDPOINT endpoint unavailable; waiting before starting episodes")
            while not sandbox_run.wait_for_endpoint_endpoint(model):
                cancellation.check()
                log("agent: ENDPOINT endpoint still unavailable; continuing to wait")


# --------------------------------------------------------------------------- one episode

def interrupted_episode(run_dir, result=None, agent=None):
    """Keep partial evidence without treating a cancelled attempt as a submission."""
    if result is None:
        result = {}
        for name in ("checkpoint.json", "result.json"):
            try:
                result = json.loads((Path(run_dir) / name).read_text())
                break
            except (FileNotFoundError, json.JSONDecodeError):
                pass
    stats = result.get("agent_stats") or (getattr(agent, "stats", None) if agent else None) or {}
    return {
        "status": "interrupted", "stop_reason": "interrupted", "error": cancellation.reason(),
        "scored_against_instance": False, "executability": None, "produced_png": None,
        "metrics": {name: {mode: None for mode in MODES} for name in METRICS},
        "footprint": [], "steps": result.get("steps_taken"), "wall_s": result.get("wall_time_s"),
        "cost_usd": stats.get("cost_usd"), "finish_origin": result.get("finish_origin"),
        "edited": None, "sandbox": {"stop_reason": "interrupted"},
        "agent_stats": {key: value for key, value in stats.items() if key != "transcript"},
        "resume_mode": "fresh_attempt_after_archive",
    }


def deferred_episode(run_dir, result=None, agent=None):
    ep = interrupted_episode(run_dir, result, agent)
    return {**ep, "status": "deferred", "stop_reason": "deferred", "error": None,
            "sandbox": {"stop_reason": "deferred"},
            "resume_mode": "fresh_uncapped_attempt_after_first_pass"}


def sandbox_episode(job: dict, run_dir: Path, make_agent, run_episode, timeout=120,
                    stagnation_policy=None) -> dict:
    """Drive one episode and score it, or raise EpisodeFailed with the cost.

    Executability and the three metrics all come from OUR re-execution of the
    agent's final `chart.py` against the full ground truth, whatever the agent was
    allowed to see, so the whole row answers one question. The sandbox's
    `renders_ok` travels alongside as `produced_png`, which is what it measures.
    """
    agent = make_agent()
    try:
        options = {"stagnation_policy": stagnation_policy} if stagnation_policy else {}
        res = run_episode(job["task_dir"], agent, out_dir=run_dir, **options)
    except cancellation.Cancelled:
        return interrupted_episode(run_dir, agent=agent)
    except cancellation.Deferred:
        return deferred_episode(run_dir, agent=agent)
    except Exception as exc:
        if cancellation.requested():
            return interrupted_episode(run_dir, agent=agent)
        raise EpisodeFailed("%s: %s\n%s" % (type(exc).__name__, exc, traceback.format_exc()),
                            cost_usd=_agent_cost(agent),
                            scores=_scores_for_no_submission(job, timeout),
                            stop_reason="transport_error" if _looks_like_outage(exc)
                            else "episode_exception") from exc
    cost = (res.get("agent_stats") or {}).get("cost_usd")
    stop_reason = sandbox_run.episode_stop_reason(res, run_dir)
    if stop_reason == "interrupted" or cancellation.requested():
        return interrupted_episode(run_dir, res, agent)
    if stop_reason == "deferred" or res.get("status") == "deferred":
        return deferred_episode(run_dir, res, agent)
    interrupted = stop_reason in sandbox_run.INTERRUPTED_STOP_REASONS
    try:
        ours = sandbox_run.score_repair(job["task_dir"], run_dir, timeout)
    except Exception as exc:
        if cancellation.requested():
            return interrupted_episode(run_dir, res, agent)
        # The episode itself ran, so the sandbox's verdict and the price are real
        # and are kept; our own re-execution is what never happened.
        raise EpisodeFailed("scoring failed: %s: %s\n%s"
                            % (type(exc).__name__, exc, traceback.format_exc()),
                            cost_usd=cost,
                            produced_png=sandbox_run.produced_png(res),
                            scores=_scores_for_no_submission(job, timeout),
                            stop_reason=stop_reason if interrupted else "scoring_error") from exc
    if cancellation.requested():
        return interrupted_episode(run_dir, res, agent)
    cfg = to_sandbox.read_task_config(job["task_dir"])
    return {
        "status": "error" if interrupted else "ok",
        "error": "sandbox interrupted: %s" % stop_reason if interrupted else None,
        "stop_reason": stop_reason,
        "scored_against_instance": True,
        "executability": ours["executability"],
        "produced_png": sandbox_run.produced_png(res),
        "metrics": {m: ours[m] for m in METRICS},
        "footprint": ours["diagnostics"].get("footprint", []),
        "steps": res["steps_taken"],
        "wall_s": res["wall_time_s"],
        "cost_usd": cost,
        "modalities": cfg["modalities"],
        "modality_code": cfg["modality_code"],
        "track": cfg.get("track", tracks.DEFAULT_TRACK),
        "step_budget": cfg.get("step_budget", tracks.DEFAULT_STEP_BUDGET),
        "episode_wall_time_limit_s": cfg.get("wall_time_s", tracks.DEFAULT_WALL_TIME_S),
        "finish_origin": res.get("finish_origin"),
        "stagnation_policy": res.get("stagnation_policy"),
        "stagnation_evidence": res.get("stagnation_evidence"),
        "clean": bool(cfg.get("clean", False)),
        # Did the agent change the script it was handed?  On a clean twin that
        # is a false repair; on a flawed chart, leaving it alone is a miss.
        "edited": _edited(job["task_dir"], run_dir),
        "sandbox": {**{k: res[k] for k in ("final_score", "subgoals_passed", "subgoals_total",
                                           "finished", "steps_taken", "step_budget")},
                    "stop_reason": stop_reason},
        "agent_stats": {k: v for k, v in (res.get("agent_stats") or {}).items()
                        if k != "transcript"},
        **{k: res[k] for k in ("interaction_protocol", "model_turn_budget", "model_turns",
                               "single_shot_protocol", "submission_sha256", "empty_submission",
                               "finish_reason", "resume_mode") if k in res},
    }


def _edited(task_dir, run_dir):
    """Whether the submitted chart.py differs from the workspace's input script."""
    given = Path(task_dir) / "workspace" / "chart.py"
    try:
        final = sandbox_run.harness.read_workspace_file(Path(run_dir) / "workspace", "chart.py")
    except (ValueError, OSError):
        return True  # An unsafe artifact also replaced the original submission.
    if final is None:
        return None if not given.exists() else True
    if not given.exists():
        return True
    return given.read_bytes() != final


def _agent_cost(agent):
    stats = getattr(agent, "stats", None)
    return stats.get("cost_usd") if isinstance(stats, dict) else None


def _pick_scores(scored: dict) -> dict:
    """The executability + metric block of a `metrics.score` result."""
    return {"executability": scored["executability"],
            "metrics": {m: {mode: scored[m][mode] for mode in MODES} for m in METRICS}}


def _scores_for_no_submission(job: dict, timeout: int):
    """Score a crashed episode against its own instance, or None if even that fails.

    An episode that raised produced no submission, which is the same outcome as a
    submission that does not render — so it gets the same verdict, from the same
    function. What matters is that `metrics.score` derives the footprint from gt
    and flawed, never from the submission: whether recovery or preservation even
    APPLIES is a property of the instance and is still answerable when the agent's
    side is gone. Writing 0.0 here instead would record "the agent failed to
    repair" for a question the benchmark never asked.

    It costs two captures per failed episode. Failures are rare, both captures are
    bounded by the same timeout the normal path uses, and a wrong number in the
    report is more expensive.
    """
    try:
        return _pick_scores(sandbox_run.score_no_submission(
            job["task_dir"], timeout, "the episode raised before it could be scored"))
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def _failed_scores() -> dict:
    """Last resort: a failure we could not tie back to an instance at all.

    Reached when the crash happened outside an episode (a bug in our own glue, a
    fake episode function in a test) or when scoring the instance failed too. The
    instance is what says whether a metric applies, so without it the honest
    answer is unknown — and between the two ways of being wrong, 0.0 can only
    count against the agent and None would quietly excuse the episode. Zeros it
    is, and `n_errors` in the same report says how many rows are this.
    """
    return {"executability": 0,
            "metrics": {m: {mode: 0 if m.endswith("_binary") else 0.0 for mode in MODES}
                        for m in METRICS}}


def failed_scores() -> dict:
    """A fresh copy of `_failed_scores`, so an episode record never aliases the cache."""
    return copy.deepcopy(_failed_scores())


# --------------------------------------------------------------------------- the run

def episodes_dir(out) -> Path:
    return Path(out) / "episodes"


def load_episodes(out) -> dict:
    """Every finished episode already on disk, keyed by task id."""
    found = {}
    for path in sorted(episodes_dir(out).glob("*.json")):
        try:
            ep = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue                       # a half-written file from a hard kill
        found[ep.get("task_id", path.stem)] = ep
    return found


def resumable_completed(out, *, retry_errors=False):
    return {key: ep for key, ep in load_episodes(out).items()
            if ep.get("status") not in {"interrupted", "deferred"} and
            (ep.get("stop_reason") or (ep.get("sandbox") or {}).get("stop_reason")) not in {"interrupted", "deferred"}
            and (not retry_errors or ep.get("status") == "ok")}


def _write_episode(out, ep: dict) -> None:
    d = episodes_dir(out)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / ("." + ep["task_id"] + ".tmp")
    tmp.write_text(json.dumps(ep, indent=1), encoding="utf-8")
    tmp.replace(d / (ep["task_id"] + ".json"))   # atomic: a kill leaves no torn file


def scheduling_policy(seconds=None, reference=None, *, step_budget=None):
    if seconds is None:
        if reference is not None:
            raise ValueError("--deferral-reference requires --defer-after-seconds")
        return None
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("--defer-after-seconds must be finite and positive")
    step_budget = tracks.validate_step_budget(step_budget)
    # The full calibration can contain thousands of rows. Bind its bytes here;
    # run.json retains it once, rather than duplicating it in every episode.
    reference = {"sha256": reference["sha256"]} if reference is not None else None
    policy = {"name": "p90-defer-once-v1", "defer_after_seconds": float(seconds),
            "first_attempt": "defer_and_discard_partial_workspace",
            "tail_attempt": "fresh_after_all_first_pass_attempts_drain",
            "max_deferrals": 1, "tail_step_budget": None, "tail_wall_time_s": None,
            "completed_submission_scoring": "outside_deferral_deadline",
            "reference": reference}
    if step_budget is not None:
        policy.update(name="p90-defer-once-finite-steps-v1", tail_step_budget=step_budget,
                      step_budget_scope="per_attempt", discarded_first_attempt_step_budget=step_budget)
    return policy


def _deferred_resume_mode(policy):
    return ("fresh_attempt_with_original_step_budget_after_first_pass"
            if policy and policy.get("tail_step_budget") is not None
            else "fresh_uncapped_attempt_after_first_pass")


def _durable_json(path, value):
    """Commit the minimal recovery record before discarding any partial files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=1)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def load_deferred(out):
    found = {}
    for path in sorted((Path(out) / "deferred").glob("*.json")):
        entry = json.loads(path.read_text(encoding="utf-8"))
        if entry["task_id"] != path.stem or entry["phase"] not in {"pending", "tail_running", "finished"}:
            raise ValueError("invalid deferral ledger: %s" % path)
        found[entry["task_id"]] = entry
    return found


def _save_deferred(out, entry):
    _durable_json(Path(out) / "deferred" / (entry["task_id"] + ".json"), entry)


def _discard_deferred_workspace(out, task_id):
    """Only the scheduler-owned partial directory; caller has closed the kernel."""
    run_dir = Path(out) / "runs" / sandbox_run.run_slug(task_id)
    if run_dir.is_symlink():
        run_dir.unlink()
    elif run_dir.exists():
        shutil.rmtree(run_dir)


def _deferral_evidence(ep):
    stats = ep.get("agent_stats") or {}
    return {"status": "deferred", "stop_reason": "deferred",
            "steps": ep.get("steps"), "wall_s": ep.get("wall_s"),
            "cost_usd": ep.get("cost_usd"), "finished_at": ep["finished_at"],
            "agent_stats": {key: stats[key] for key in ("calls", "usage", "cost_usd") if key in stats}}


def record_deferred(out, ep, policy, *, reason="deadline", origin=None):
    """Persist a deferred first attempt; also usable for documented migration.

    No workspace is touched here. The caller must close its kernel before
    discarding that workspace with `_discard_deferred_workspace`.
    """
    if (Path(out) / "deferred" / (ep["task_id"] + ".json")).exists():
        raise ValueError("first deferral is already recorded: %s" % ep["task_id"])
    prior = episodes_dir(out) / (ep["task_id"] + ".json")
    if prior.exists() and json.loads(prior.read_text()).get("status") == "ok":
        raise ValueError("cannot replace a completed episode with a deferral")
    ep = {**ep, **deferred_episode(Path(out) / "unused", {}),
          **{key: ep.get(key) for key in ("steps", "wall_s", "cost_usd", "agent_stats")},
          "finished_at": ep.get("finished_at") or datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "scheduling_policy": policy, "scheduling_pass": "first",
          "resume_mode": _deferred_resume_mode(policy)}
    # Only accounting survives a discarded attempt, never messages/tool outputs.
    ep["agent_stats"] = _deferral_evidence(ep)["agent_stats"]
    entry = {"schema": 1, "task_id": ep["task_id"], "instance": ep["instance"],
             "category": ep["category"], "scheduling_policy": policy,
             "phase": "pending", "reason": reason, "origin": origin,
             "first_attempt": _deferral_evidence(ep)}
    _save_deferred(out, entry)
    _durable_json(episodes_dir(out) / (ep["task_id"] + ".json"), ep)
    return entry


def run_config(out, meta: dict, force=False) -> dict:
    """Pin what this output directory belongs to, and refuse to mix runs.

    Resuming with a different model or dataset into the same directory would
    produce a report averaging two agents, which is worse than an error.
    """
    path = Path(out) / "run.json"
    keys = ("model", "dataset", "modality_code", "track", "step_budget",
            "wall_time_s", "track_c_mode", "scheduling_policy", "stagnation_policy",
            "interaction_protocol", "model_turn_budget", "single_shot_protocol")
    if path.exists() and not force:
        old = json.loads(path.read_text(encoding="utf-8"))
        defaults = {"wall_time_s": tracks.DEFAULT_WALL_TIME_S,
                    "track_c_mode": "twins" if old.get("track") == "C" else "flawed_only"}
        clash = [k for k in keys if old.get(k, defaults.get(k)) != meta.get(k, defaults.get(k))]
        if (old.get("dataset_sha256") and meta.get("dataset_sha256")
                and old["dataset_sha256"] != meta["dataset_sha256"]):
            clash.append("dataset_sha256")
        old_scorer = (old.get("scorer") or {}).get("sha256")
        new_scorer = (meta.get("scorer") or {}).get("sha256")
        if old_scorer and new_scorer and old_scorer != new_scorer:
            clash.append("scorer")
        if clash:
            raise SystemExit(
                "%s already holds a run with %s. Use a different --out, or --force to "
                "overwrite it." % (path, ", ".join("%s=%s" % (k, old.get(k)) for k in clash)))
        meta = {**old, **meta, "resumed_at": meta.get("started_at")}
    Path(out).mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


class RunLock:
    """A process-held lock; recorded PIDs are diagnostic and cannot block resume."""

    def __init__(self, out: Path):
        self.path = Path(out) / "run.lock"
        self.held = False
        self._stream = None

    def acquire(self, log=print) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.held:
            return
        stream = self.path.open("a+")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.close()
            raise SystemExit("another run is already writing to %s; use a different --out or wait for it" % self.path.parent)
        try:
            stream.seek(0)
            stream.truncate()
            json.dump({"pid": os.getpid()}, stream)
            stream.flush()
        except BaseException:
            stream.close()
            raise
        self._stream = stream
        self.held = True

    def release(self) -> None:
        if self.held:
            # Keep the inode: unlinking permits contenders to lock different files.
            self._stream.close()
            self._stream = None
            self.held = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False


def run_episodes(jobs: list, out, episode_fn, *, meta=None, workers=4, budget=None,
                 force=False, retry_errors=False, log=print, model=None) -> tuple:
    """Run every job that is not already finished, in parallel, and return them all.

    THREADS, not processes, and the reason is that nothing heavy actually happens
    in this interpreter. Each episode gets its own run directory, and the sandbox
    gives that directory its own `Sandbox` and therefore its own `PersistentKernel`
    subprocess (`sandbox/src/chartsandbox/runtime/kernel.py`); the only `chdir` in
    the whole path is inside that child (`runtime/worker_main.py`), never here.
    Scoring goes through `capture` -> `harness.run_in_sandbox`, which writes into
    its own `TemporaryDirectory` and runs its own subprocess with its own env
    (`src/harness.py`), so two scorings cannot see each other's files either. What
    is left in-process is JSON, HTTP waits and subprocess waits, all of which
    release the GIL. The one shared mutable object that would have been unsafe is
    the provider's token accounting, and `agent_factory` gives every episode its
    own provider.

    Two things must therefore stay in the calling thread, and do: importing the
    sandbox (its verifier registry fills at import time) and warming the
    matplotlib font cache, both done in `main` before the pool starts.
    """
    meta = copy.deepcopy(meta or {})
    policy = meta.get("scheduling_policy")
    deferred = load_deferred(out)
    if deferred and (force or not policy):
        raise ValueError("a deferred queue requires its original policy and cannot be forced")
    loaded = load_episodes(out)
    for task_id, entry in deferred.items():
        if entry["scheduling_policy"] != policy:
            raise ValueError("deferral policy changed for %s" % task_id)
        prior = loaded.get(task_id, {})
        if entry["phase"] == "pending":
            if prior.get("status") == "ok" or prior.get("scheduling_pass") == "tail":
                raise ValueError("pending deferral conflicts with a completed/tail record: %s" % task_id)
            # A crash between the durable ledger and deletion must not turn the
            # discarded P90 workspace into an archived retry on restart.
            _discard_deferred_workspace(out, task_id)
            if prior and prior.get("status") != "deferred":
                (episodes_dir(out) / (task_id + ".json")).unlink()
                loaded.pop(task_id, None)
        elif prior.get("status") in {"ok", "error"}:
            if prior.get("scheduling_pass") != "tail" or prior.get("scheduling_policy") != policy:
                raise ValueError("deferral ledger has no matching tail result: %s" % task_id)
            retry = retry_errors and prior["status"] == "error"
            entry.update(phase="tail_running" if retry else "finished", final_status=prior["status"])
            _save_deferred(out, entry)
        elif entry["phase"] == "finished":
            raise ValueError("finished deferred attempt has no final episode: %s" % task_id)
    done = {} if force else resumable_completed(out, retry_errors=retry_errors)
    todo = [j for j in jobs if j["task_id"] not in done]
    first_pass = [j for j in todo if j["task_id"] not in deferred]
    episodes = [done[j["task_id"]] for j in jobs if j["task_id"] in done]
    if episodes:
        log("resuming: %d of %d episodes are already on disk" % (len(episodes), len(jobs)))

    if budget is not None:
        for ep in episodes:                # already paid for, so already spent
            budget.finish(ep.get("cost_usd"))
        for entry in deferred.values():
            budget.finish(entry["first_attempt"].get("cost_usd"))

    runs_root = Path(out) / "runs"
    # The digest -> task id map, written once and merged, so a resume can still
    # name the directories an earlier attempt produced. It lands beside the runs
    # root, never inside it: see `sandbox_run.index_path`.
    if todo:
        sandbox_run.record_run_dirs(runs_root, [j["task_id"] for j in todo])
    counter, lock = [0], threading.Lock()
    total = len(todo)

    def _work(job, tail=False):
        if cancellation.requested():
            return None
        if budget is not None and not budget.start():
            return None
        started = time.time()
        base = {
            "task_id": job["task_id"], "instance": job["instance"],
            "category": job["category"], **(meta or {}),
            "clean": bool(job.get("clean", False)),
            # The task id is not in the directory name any more, so it is recorded
            # here — this file is the only place the two are tied together.
            "run_slug": sandbox_run.run_slug(job["task_id"]),
            "finished_at": None,
        }
        if policy:
            base["scheduling_pass"] = "tail" if tail else "first"
        if tail:
            entry = deferred[job["task_id"]]
            ep_path = episodes_dir(out) / (job["task_id"] + ".json")
            if ep_path.exists() and json.loads(ep_path.read_text()).get("status") == "deferred":
                ep_path.unlink()
            entry.update(phase="tail_running")
            _save_deferred(out, entry)
            base["deferred_first_attempt"] = entry["first_attempt"]

        def attempt():
            if meta.get("interaction_protocol") == "single-shot-v1":
                from .single_shot import prepare_run_dir
                run_dir = prepare_run_dir(runs_root, job["task_id"], force=force)
            else:
                run_dir = sandbox_run.prepare_run_dir(runs_root, job["task_id"])
            seconds = policy["defer_after_seconds"] if policy and not tail else None
            with cancellation.episode_deadline(seconds):
                return episode_fn(job, run_dir)

        try:
            result = attempt()
            if cancellation.requested():
                result = interrupted_episode(runs_root / base["run_slug"])
            ep = {**base, "status": "ok", "error": None, **result}
        except cancellation.Deferred:
            cancellation.check()
            ep = {**base, **deferred_episode(runs_root / base["run_slug"])}
        except EpisodeFailed as exc:
            cancellation.check()
            retry_succeeded = False
            # Wait for a transient gateway outage to clear before giving the
            # episode one additional attempt. A deferred tail stays uncapped;
            # the first pass must never retry outside its scheduling deadline.
            if (meta.get("interaction_protocol") != "single-shot-v1"
                    and (not policy or tail) and _looks_like_outage(exc)
                    and sandbox_run.wait_for_endpoint_endpoint(model)):
                cancellation.check()
                started = time.time()
                try:
                    result = attempt()
                    ep = {**base, "status": "ok", "error": None, **result}
                    retry_succeeded = True
                except Exception as retry_exc:      # noqa: BLE001 - fall through
                    cancellation.check()
                    exc = retry_exc if isinstance(retry_exc, EpisodeFailed) else exc
            if not retry_succeeded:
                ep = {**base, "status": "error", "error": str(exc)[-4000:],
                      "stop_reason": exc.stop_reason,
                      **(exc.scores or failed_scores()),
                      "scored_against_instance": exc.scores is not None,
                      "produced_png": exc.produced_png, "footprint": [],
                      "steps": None, "wall_s": round(time.time() - started, 2),
                      "cost_usd": exc.cost_usd}
        except Exception as exc:           # a bug in our own glue, not in the episode
            cancellation.check()
            why = "%s: %s\n%s" % (type(exc).__name__, exc, traceback.format_exc())
            ep = {**base, "status": "error", "error": why[-4000:],
                  "stop_reason": "bridge_exception",
                  **failed_scores(), "scored_against_instance": False,
                  "produced_png": None, "footprint": [],
                  "steps": None, "wall_s": round(time.time() - started, 2), "cost_usd": None}
        if meta.get("scorer"):
            ep["scorer"] = meta["scorer"]
        ep["finished_at"] = datetime.datetime.now(datetime.timezone.utc).replace(
            microsecond=0).isoformat()
        if ep.get("status") == "deferred" and (not policy or tail):
            ep.update(status="error", stop_reason="unexpected_deferral",
                      error="the uncapped attempt unexpectedly requested deferral")
        if ep.get("status") == "deferred":
            ep["wall_s"] = ep.get("wall_s") or round(time.time() - started, 3)
            ep["resume_mode"] = _deferred_resume_mode(policy)
            entry = record_deferred(out, ep, policy)
            deferred[job["task_id"]] = entry
            ep["agent_stats"] = entry["first_attempt"]["agent_stats"]
            _discard_deferred_workspace(out, job["task_id"])
        if budget is not None:
            budget.finish(ep.get("cost_usd"))
        if ep.get("status") != "deferred":
            if tail:
                _durable_json(episodes_dir(out) / (job["task_id"] + ".json"), ep)
            else:
                _write_episode(out, ep)
        if tail and ep.get("status") in {"ok", "error"}:
            entry.update(phase="finished", final_status=ep["status"])
            _save_deferred(out, entry)
        with lock:
            counter[0] += 1
            m = ep.get("metrics") or {}
            log("[%d/%d] %-52s %-5s exec=%s png=%s rec=%s pres=%s%s"
                % (counter[0], total, job["task_id"][:52], ep["status"],
                   ep.get("executability"), ep.get("produced_png"),
                   _fmt((m.get("recovery") or {}).get("strict")),
                   _fmt((m.get("preservation") or {}).get("strict")),
                   "" if ep.get("cost_usd") is None else "  $%.4f" % ep["cost_usd"]))
        return ep

    def work(job, tail=False):
        if cancellation.requested():
            return None
        try:
            return _work(job, tail)
        except (cancellation.Cancelled, KeyboardInterrupt) as exc:
            if isinstance(exc, KeyboardInterrupt):
                cancellation.request("SIGINT")
            ep = {**meta, "task_id": job["task_id"], "instance": job["instance"],
                  "category": job["category"], "clean": bool(job.get("clean", False)),
                  "run_slug": sandbox_run.run_slug(job["task_id"]),
                  **interrupted_episode(runs_root / sandbox_run.run_slug(job["task_id"])),
                  "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
            if policy:
                ep["scheduling_pass"] = "tail" if tail else "first"
            if tail:
                ep["deferred_first_attempt"] = deferred[job["task_id"]]["first_attempt"]
            _write_episode(out, ep)
            if budget is not None:
                budget.finish(ep.get("cost_usd"))
            return ep

    queue_last_write = [0.0]

    def queue_snapshot(phase, active=(), pending=0):
        queue_last_write[0] = time.monotonic()
        _durable_json(Path(out) / "queue_state.json", {
            "schema": 1, "pid": os.getpid(),
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "phase": phase, "active_task_ids": list(active),
            "first_pass_drained": phase in {"tail", "completed"},
            "tail_pending": pending if phase == "tail" else 0,
            "first_pending": pending if phase == "first" else 0,
        })

    def run_pass(pass_jobs, tail=False):
        produced = []
        queue_phase = "tail" if tail else "first"
        queue_snapshot(queue_phase, pending=len(pass_jobs))
        if pass_jobs and workers <= 1 and not os.environ.get("CHARTREPAIR_WORKER_LIMIT_FILE"):
            try:
                for position, job in enumerate(pass_jobs):
                    if cancellation.requested():
                        break
                    queue_snapshot(queue_phase, [job["task_id"]], len(pass_jobs) - position - 1)
                    produced.append(work(job, tail))
                    queue_snapshot(queue_phase, pending=len(pass_jobs) - position - 1)
            except KeyboardInterrupt:
                cancellation.request("SIGINT")
                log("interrupted: stopping after %d episodes" % len(produced))
        elif pass_jobs:
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
            remaining, pending, exhausted = iter(pass_jobs), set(), False
            admitted, pending_ids = 0, {}
            admission_limit, invalid_limit = workers, None

            def desired_workers():
                nonlocal admission_limit, invalid_limit
                path = os.environ.get("CHARTREPAIR_WORKER_LIMIT_FILE")
                if not path:
                    return workers
                try:
                    limit = int(Path(path).read_text().strip())
                    if not 0 <= limit <= workers:
                        raise ValueError(f"expected an integer between 0 and {workers}")
                except (OSError, ValueError) as exc:
                    if str(exc) != invalid_limit:
                        log(f"worker admission unchanged at {admission_limit}: {exc}")
                        invalid_limit = str(exc)
                else:
                    if limit != admission_limit:
                        log(f"worker admission: {admission_limit} -> {limit} (active episodes continue)")
                    admission_limit, invalid_limit = limit, None
                return admission_limit

            def submit_available():
                nonlocal exhausted, admitted
                limit = desired_workers()
                while len(pending) < limit and not cancellation.requested():
                    job = next(remaining, None)
                    if job is None:
                        exhausted = True
                        break
                    admitted += 1
                    queue_snapshot(queue_phase, [*pending_ids.values(), job["task_id"]],
                                   len(pass_jobs) - admitted)
                    future = pool.submit(work, job, tail)
                    pending.add(future)
                    pending_ids[future] = job["task_id"]

            def collect(future):
                if future.cancelled():
                    return
                try:
                    produced.append(future.result())
                except Exception as exc:
                    log("episode result lost: %s: %s" % (type(exc).__name__, exc))

            try:
                submit_available()
                while pending or not exhausted:
                    if not pending:
                        if cancellation.requested():
                            break
                        try:
                            cancellation.wait(0.2)
                        except cancellation.Cancelled:
                            break
                        submit_available()
                        continue
                    ready, _ = concurrent.futures.wait(pending, timeout=0.2,
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in ready:
                        pending.remove(future)
                        collect(future)
                        pending_ids.pop(future, None)
                        queue_snapshot(queue_phase, pending_ids.values(), len(pass_jobs) - admitted)
                    if time.monotonic() - queue_last_write[0] >= 5:
                        queue_snapshot(queue_phase, pending_ids.values(), len(pass_jobs) - admitted)
                    if cancellation.requested():
                        for future in pending:
                            future.cancel()
                    else:
                        submit_available()
            except KeyboardInterrupt:
                cancellation.request("SIGINT")
                for future in pending:
                    future.cancel()
            finally:
                pool.shutdown(wait=True)
                for future in pending:
                    collect(future)
                queue_snapshot("interrupted" if cancellation.requested() else queue_phase,
                               pending=len(pass_jobs) - admitted)
        return [ep for ep in produced if ep is not None]

    produced = run_pass(first_pass)
    first_ran = {ep["task_id"] for ep in produced}
    if (not cancellation.requested() and not (budget and budget.tripped)
            and all(job["task_id"] in first_ran for job in first_pass)):
        tail_jobs = [job for job in todo if job["task_id"] in deferred
                     and deferred[job["task_id"]]["phase"] != "finished"]
        if tail_jobs:
            if policy.get("tail_step_budget") is None:
                log("first pass drained; starting %d fresh uncapped deferred attempts" % len(tail_jobs))
            else:
                log("first pass drained; starting %d fresh deferred attempts, %d tool calls per attempt" % (
                    len(tail_jobs), policy["tail_step_budget"]))
            produced += run_pass(tail_jobs, tail=True)
    latest = {ep["task_id"]: ep for ep in episodes + produced}
    # Keep a deferred row visible after a restart that stops before reaching the
    # tail. It is pending work, never a completed or erroneous submission.
    for job in todo:
        task_id = job["task_id"]
        if task_id not in latest and task_id in deferred:
            entry = deferred[task_id]
            latest[task_id] = {**meta, "task_id": task_id, "instance": job["instance"],
                              "category": job["category"], "clean": bool(job.get("clean")),
                              **deferred_episode(runs_root / sandbox_run.run_slug(task_id), {}),
                              **entry["first_attempt"], "resume_mode": _deferred_resume_mode(policy)}
    ran = set(latest)
    queue_snapshot("interrupted" if cancellation.requested() or (budget and budget.tripped) else "completed")
    return ([latest[job["task_id"]] for job in jobs if job["task_id"] in latest],
            [job for job in todo if job["task_id"] not in ran])


# --------------------------------------------------------------------------- aggregation

def _stats(values: list) -> dict:
    """Mean over the values that are defined. None is excluded and counted.

    This is the whole undefined rule in three lines: a metric that does not apply
    to an instance leaves the numerator AND the denominator, and the number of
    times that happened travels with the mean so nobody reads a mean over 12
    items as a mean over 100.
    """
    got = [v for v in values if v is not None]
    return {"mean": round(sum(got) / len(got), 4) if got else None,
            "n": len(got), "undefined": len(values) - len(got)}


def _exec_vs_png(eps: list) -> dict:
    """How often our re-execution and the sandbox's file check disagree, and which way.

    Only episodes where both verdicts exist are compared; an episode whose task
    carried no `renders_ok` subgoal, or that crashed before the sandbox could
    answer, has nothing to disagree about and is not counted either way.

    `exec_only` is the agent that wrote a working script and never ran it, so no
    output.png was ever produced. `png_only` is the opposite: an output.png left
    over from an execution while the submitted script no longer works. Therefore
    executability is determined by re-execution independently of `renders_ok`.
    """
    both = [(int(e["executability"]), int(e["produced_png"])) for e in eps
            if e.get("executability") is not None and e.get("produced_png") is not None]
    exec_only = sum(1 for x, p in both if x and not p)
    png_only = sum(1 for x, p in both if p and not x)
    n = len(both)
    return {
        "n_compared": n,
        "n_disagree": exec_only + png_only,
        "rate": round((exec_only + png_only) / n, 4) if n else None,
        "exec_only": exec_only,
        "png_only": png_only,
        "n_uncomparable": len(eps) - n,
    }


def _block(eps: list) -> dict:
    """The full metric block for one group of attempted episodes.

    Provider, transport, and harness failures are not model submissions.  They
    stay visible as attempts/errors, but capability metrics are computed only
    over completed episodes.  Counting a gateway outage as a failed repair made
    the reported score depend on endpoint availability; worse, a later retry
    changed the score without changing a single model answer.
    """
    completed = [e for e in eps if e.get("status") == "ok"]
    costs = [e.get("cost_usd") for e in eps if e.get("cost_usd") is not None]
    first_attempts = [e["deferred_first_attempt"] for e in eps if e.get("deferred_first_attempt")]
    costs += [e["cost_usd"] for e in first_attempts if e.get("cost_usd") is not None]
    reasons = [e.get("stop_reason") or (e.get("sandbox") or {}).get("stop_reason")
               or "unknown" for e in eps]
    return {
        "n": len(eps),
        "n_completed": len(completed),
        "n_stagnation": reasons.count("stagnation"),
        "n_errors": sum(e.get("status") not in {"ok", "interrupted", "deferred"} for e in eps),
        "n_interrupted": sum(e.get("status") == "interrupted" for e in eps),
        "n_deferred": sum(e.get("status") == "deferred" for e in eps),
        "n_deferred_first_attempts": len(first_attempts) + sum(e.get("status") == "deferred" for e in eps),
        "stop_reasons": dict(sorted(Counter(reasons).items())),
        "n_harness_aborted": sum(e.get("status") != "ok"
                                 and reason in sandbox_run.INTERRUPTED_STOP_REASONS
                                 for e, reason in zip(eps, reasons)),
        # error rows we could not tie back to an instance, so their zeros are the
        # last resort rather than a real "this metric applies and was not met"
        "n_errors_unscored": sum(1 for e in eps if e.get("status") not in {"ok", "interrupted", "deferred"}
                                 and not e.get("scored_against_instance")),
        # ours: the submitted chart.py was re-run and drew a chart
        "executability": _stats([e.get("executability") for e in completed]),
        # the sandbox's: an output.png exists in the workspace and opens
        "produced_png": _stats([e.get("produced_png") for e in completed]),
        "exec_vs_png": _exec_vs_png(completed),
        "metrics": {m: {mode: _stats([(e.get("metrics") or {}).get(m, {}).get(mode)
                                      for e in completed])
                        for mode in MODES} for m in METRICS},
        "steps": _stats([e.get("steps") for e in completed]),
        # Detection: the agent moved at least one flawed property towards the
        # truth (strict recovery > 0). Undefined recovery stays excluded.
        "detection": _stats([
            (1.0 if value > 0 else 0.0) if value is not None else None
            for value in (
                (e.get("metrics") or {}).get("recovery", {}).get("strict") for e in completed
            )
        ]),
        "wall_time_s": round(sum(e.get("wall_s") or 0.0 for e in eps + first_attempts), 1),
        "cost_usd": round(sum(costs), 4) if costs else None,
        "n_priced": len(costs),
    }


def _macro(by_cat: dict) -> dict:
    """Mean over the category means, for categories where the mean is defined.

    The categories are not the same size, so this answers a different question
    than the item mean: it weights every flaw type equally instead of weighting
    it by how many instances the pool happened to yield.
    """
    cats = sorted(by_cat)
    def over(pick):
        vals = [pick(by_cat[c]) for c in cats]
        got = [v for v in vals if v is not None]
        return {"mean": round(sum(got) / len(got), 4) if got else None,
                "n_categories": len(got), "undefined_categories": len(vals) - len(got)}
    return {
        "n_categories": len(cats),
        "executability": over(lambda b: b["executability"]["mean"]),
        "produced_png": over(lambda b: b["produced_png"]["mean"]),
        "metrics": {m: {mode: over(lambda b, m=m, mode=mode: b["metrics"][m][mode]["mean"])
                        for mode in MODES} for m in METRICS},
    }


def aggregate(episodes: list, meta=None, skipped=None) -> dict:
    """One report over the episodes: overall, per category, and the macro mean."""
    coverage = scorer_coverage(episodes)
    by_cat: dict = {}
    for ep in episodes:
        by_cat.setdefault(ep.get("category", "unknown"), []).append(ep)
    cats = {c: _block(by_cat[c]) for c in sorted(by_cat)}
    ns = [b["n_completed"] for b in cats.values()]
    typical = statistics.median(ns) if ns else 0
    for c, b in cats.items():
        b["small_n"] = bool(typical and b["n_completed"] < 0.5 * typical)

    return {
        "schema": SCHEMA,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).replace(
            microsecond=0).isoformat(),
        "run": dict(meta or {}),
        "scorer": episodes[0]["scorer"] if coverage["consistent"] else None,
        "scorer_coverage": coverage,
        "requires_rescore": coverage["requires_rescore"],
        "aggregation": {
            "headline": "item mean: every completed submission is one vote",
            "macro": "mean of the category means: every flaw type is one vote",
            "undefined": ("a metric that does not apply to an instance (an empty footprint "
                          "for recovery, nothing outside it for preservation) is excluded "
                          "from the mean and counted under 'undefined', never averaged as 0"),
            "errors": ("an episode interrupted by the harness or an error is excluded from "
                       "capability metrics and reported separately; partial scores remain "
                       "diagnostic. Step-budget exhaustion remains a completed attempt. "
                       "Stagnation guard stops are scored completed attempts with a distinct stop reason. "
                       "n_completed is the denominator for model scores, while n records "
                       "endpoint and harness reliability; stop_reasons counts current episode records. "
                       "Deferred first attempts have unknown scores, are counted separately and "
                       "their time/cost remains included after the fresh tail attempt. "
                       "n_harness_aborted counts wall-time and interrupted errors"),
            "binary": "1 only when the score is exactly 1.0, so a binary mean is the fraction "
                      "of instances that were fully repaired",
            "executability": ("our re-execution of the agent's final chart.py: does the "
                              "submitted script run and draw a chart. It is the gate the "
                              "other three metrics already sit behind"),
            "produced_png": ("the sandbox's renders_ok subgoal: an output.png exists in the "
                             "workspace and opens as an image. It never executes the "
                             "submission, so it is a diagnostic, not the headline"),
            "exec_vs_png": ("how often the two disagree. exec_only: a working script the "
                            "agent never ran. png_only: an output.png left from an earlier "
                            "run while the submitted script no longer works"),
            "modes": list(MODES),
        },
        "overall": _block(episodes),
        "macro": _macro(cats),
        "categories": cats,
        "track": {
            "track": (meta or {}).get("track", tracks.DEFAULT_TRACK),
            "step_budget": (meta or {}).get("step_budget", tracks.DEFAULT_STEP_BUDGET),
            "wall_time_s": (meta or {}).get("wall_time_s", tracks.DEFAULT_WALL_TIME_S),
            "population_mode": (meta or {}).get("track_c_mode"),
            "detection": tracks.track_summary(episodes),
        },
        "skipped": dict(skipped or {}),
    }


# --------------------------------------------------------------------------- the report

def _fmt(x, digits=3):
    if x is None:
        return "n/a"
    return ("%%.%df" % digits) % x


def _row(name, n, block, flag=""):
    m, mode = block["metrics"], "strict"
    return "| %s%s | %d | %s | %s | %s | %s | %s | %s |" % (
        name, flag, n,
        _fmt(block["executability"]["mean"]),
        _fmt(block["produced_png"]["mean"]),
        _fmt(m["recovery"][mode]["mean"]),
        _fmt(m["recovery_binary"][mode]["mean"]),
        _fmt(m["preservation"][mode]["mean"]),
        _fmt(m["preservation_binary"][mode]["mean"]))


def _disagreement_section(overall: dict, cats: dict) -> list:
    """The exec / produced_png disagreement, spelled out. It is a number about the
    sandbox, not about the agent, and its author should be able to read it here."""
    d = overall["exec_vs_png"]
    out = ["", "## Executability vs produced_png", "",
           "`executability` re-runs the agent's final `chart.py` and asks whether it draws a "
           "chart. `produced_png` is the sandbox's `renders_ok` subgoal, which asks whether an "
           "`output.png` exists in the workspace and opens as an image "
           "(`sandbox/src/chartsandbox/verifiers/execution.py`) — it never executes the "
           "submission. The three repair metrics sit behind the first, so that is the "
           "headline; this is what the second says instead.", ""]
    if not d["n_compared"]:
        out.append("No episode carried both verdicts, so there is nothing to compare.")
        return out
    out += ["| | |", "|---|---|",
            "| compared | %d of %d episodes carried both verdicts |"
            % (d["n_compared"], overall["n"]),
            "| disagree | **%d (%s)** |" % (d["n_disagree"], _fmt(d["rate"])),
            "| exec 1, png 0 | %d — the script runs, but no `output.png` was ever written "
            "(the agent never ran it) |" % d["exec_only"],
            "| exec 0, png 1 | %d — an `output.png` is present, but the submitted script "
            "does not run (a stale render) |" % d["png_only"]]
    worst = sorted(((c, cats[c]["exec_vs_png"]) for c in cats
                    if cats[c]["exec_vs_png"]["n_disagree"]),
                   key=lambda kv: -kv[1]["rate"])
    if worst:
        out += ["", "| category | compared | disagree | rate | exec 1, png 0 | exec 0, png 1 |",
                "|---|---:|---:|---:|---:|---:|"]
        out += ["| %s | %d | %d | %s | %d | %d |"
                % (c, b["n_compared"], b["n_disagree"], _fmt(b["rate"]),
                   b["exec_only"], b["png_only"]) for c, b in worst]
    return out


def format_markdown(report: dict) -> str:
    run, overall, cats = report["run"], report["overall"], report["categories"]
    skipped = report.get("skipped") or {}
    n_skipped = sum(v for k, v in skipped.items() if isinstance(v, int))
    out = []
    out.append("# ChartRepairBench — %s on %s" % (run.get("model", "?"), run.get("dataset", "?")))
    out.append("")
    out.append("| | |")
    out.append("|---|---|")
    rows = [
        ("model", run.get("model", "?")),
        ("dataset", "%s (%s items selected%s)" % (
            run.get("dataset", "?"), run.get("n_items", overall["n"]),
            ", %d skipped" % n_skipped if n_skipped else "")),
        ("modalities", "%s (%s)" % (", ".join(run.get("modalities") or []),
                                    run.get("modality_code", "?"))),
        ("episodes", "%d run, %d completed, %d errors, %d interrupted, %d deferred pending" % (
            overall["n"], overall["n_completed"], overall["n_errors"], overall.get("n_interrupted", 0),
            overall.get("n_deferred", 0))),
        ("stop reasons", ", ".join("%s: %d" % (reason, n) for reason, n in
                                  sorted(overall.get("stop_reasons", {}).items()))
                         or "not recorded"),
        ("harness aborts", "%d excluded from capability means; partial scores are diagnostic" %
                          overall["n_harness_aborted"]
                          if "n_harness_aborted" in overall else "not recorded"),
        # Episode time is the sum over episodes; the clock is one attempt of the run,
        # and a resume is a new attempt, so the two are labelled apart.
        ("wall time", "%.1f min of episode time over %s workers%s" % (
            overall["wall_time_s"] / 60.0, run.get("workers", "?"),
            "" if run.get("run_wall_time_s") is None
            else " (last attempt: %.1f min on the clock)" % (run["run_wall_time_s"] / 60.0))),
        ("cost", ("$%.4f over %d priced attempts" % (overall["cost_usd"], overall["n_priced"]))
                 if overall["cost_usd"] is not None else "not priced (no cost reported)"),
        ("steps", "%s per episode on average" % _fmt(overall["steps"]["mean"], 1)),
        ("track", "%s, step budget %s" % (run.get("track", tracks.DEFAULT_TRACK),
                                          run.get("step_budget", tracks.DEFAULT_STEP_BUDGET))),
        ("generated", report.get("generated_at", "")),
    ]
    if run.get("interaction_protocol") == "single-shot-v1":
        rows.append(("interaction", "single-shot: one model response with all inputs preloaded; no tool calls or repair feedback"))
    if run.get("scheduling_policy"):
        policy = run["scheduling_policy"]
        if policy.get("tail_step_budget") is None:
            rows.append(("scheduling", "%s; first-attempt threshold %ss; one fresh uncapped tail attempt; %d first attempts deferred" % (
                policy["name"], policy["defer_after_seconds"], overall.get("n_deferred_first_attempts", 0))))
        else:
            rows.append(("scheduling", "%s; first-attempt threshold %ss; fresh tail without P90 deadline, retaining %d tool calls per attempt; %d first attempts deferred" % (
                policy["name"], policy["defer_after_seconds"], policy["tail_step_budget"],
                overall.get("n_deferred_first_attempts", 0))))
            rows.append(("step-budget accounting", "The cap applies per attempt, not cumulatively. A discarded first attempt can additionally consume up to %d tool calls; infrastructure retries remain separate attempts." % policy["tail_step_budget"]))
    if run.get("inherited_scheduler_epoch"):
        rows.append(("inherited scheduler epoch", json.dumps(run["inherited_scheduler_epoch"], sort_keys=True)))
    detection = (report.get("track") or {}).get("detection")
    coverage = report.get("scorer_coverage")
    if coverage:
        rows.append(("scorer versions", "%s; %d versioned, %d unversioned; %d distinct hashes" % (
            coverage["status"], coverage["n_versioned"], coverage["n_unversioned"],
            len(coverage["versions"]))))
        rows.append(("requires_rescore", "true — scores are provisional" if
                     coverage["requires_rescore"] else "false"))
    if report.get("scorer"):
        rows.append(("scorer SHA-256", report["scorer"]["sha256"]))
    if report.get("rescore"):
        rows.append(("rescore", "complete" if report["rescore"].get("complete") else "incomplete"))
    if detection and detection.get("clean_n"):
        rows.append(("track C", "false repairs %d/%d clean charts (%s), missed repairs %d/%d "
                                "flawed (%s)" % (
                         detection["false_repair_n"], detection["clean_n"],
                         _fmt(detection["false_repair_rate"]),
                         detection["missed_repair_n"], detection["flawed_n"],
                         _fmt(detection["missed_repair_rate"]))))
    out += ["| %s | %s |" % (k, v) for k, v in rows]
    if report.get("requires_rescore"):
        out += ["", "> **Scorer versions are missing or differ.** Re-score every episode "
                    "with one scorer before treating these numbers as a comparable result."]
    if n_skipped:
        detail = ", ".join("%s %d" % (k, v) for k, v in sorted(skipped.items())
                           if isinstance(v, int) and v)
        out += ["", "> **%d items were never run** (%s). Every mean below is over what "
                    "finished, not over the dataset." % (n_skipped, detail)]

    out += ["", "## Per category", "",
            "Scores are the **strict** tolerance and the **item mean**. "
            "`rec-bin` / `pres-bin` are all-or-nothing: the fraction of instances scored "
            "exactly 1.0.", "",
            "`exec` is **our** re-execution of the submitted `chart.py` and is the gate the "
            "other columns already sit behind. `png` is the sandbox's `renders_ok`, which "
            "only checks that an `output.png` exists — a different question, kept as a "
            "diagnostic. Where they differ, see *Executability vs produced_png* below.", "",
            "| category | scored n | exec | png | rec-score | rec-binary | pres-score | pres-binary |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    small = []
    for c in sorted(cats):
        b = cats[c]
        flag = " **(small n)**" if b.get("small_n") else ""
        if b.get("small_n"):
            small.append("%s (n=%d)" % (c, b["n_completed"]))
        out.append(_row(c, b["n_completed"], b, flag))
    out.append("| | | | | | | | |")
    out.append(_row("**overall (item mean)**", overall["n_completed"], overall))
    macro = report["macro"]
    out.append("| **macro (mean of %d category means)** | %d | %s | %s | %s | %s | %s | %s |" % (
        macro["n_categories"], overall["n_completed"],
        _fmt(macro["executability"]["mean"]),
        _fmt(macro["produced_png"]["mean"]),
        _fmt(macro["metrics"]["recovery"]["strict"]["mean"]),
        _fmt(macro["metrics"]["recovery_binary"]["strict"]["mean"]),
        _fmt(macro["metrics"]["preservation"]["strict"]["mean"]),
        _fmt(macro["metrics"]["preservation_binary"]["strict"]["mean"])))
    if small:
        out += ["", "**(small n)**: %s — fewer than half the instances of a typical category. "
                    "A mean over these is not comparable to a mean over a full one."
                % "; ".join(small)]
    out += ["", "The headline row is the **item mean**: every completed submission is one "
                "vote. The macro "
                "row is the mean of the ten category means: every flaw type is one vote. They "
                "differ because the categories are not the same size."]

    out += ["", "## Overall, both tolerances", "",
            "| metric | strict | lenient | undefined (strict) |",
            "|---|---:|---:|---:|"]
    for name in ("executability", "produced_png"):
        st = overall[name]
        out.append("| %s | %s | %s | %d |" % (name.replace("_", " "), _fmt(st["mean"]),
                                              _fmt(st["mean"]), st["undefined"]))
    for m in METRICS:
        s, l = overall["metrics"][m]["strict"], overall["metrics"][m]["lenient"]
        out.append("| %s | %s | %s | %d |" % (m.replace("_", " "), _fmt(s["mean"]),
                                              _fmt(l["mean"]), s["undefined"]))
    out += _disagreement_section(overall, cats)

    und = [(c, cats[c]) for c in sorted(cats)
           if any(cats[c]["metrics"][m]["strict"]["undefined"] for m in METRICS)]
    out += ["", "## Undefined values, excluded from every mean above", "",
            "Recovery is undefined when the flaw left an empty footprint, preservation when "
            "there was nothing outside it to keep. That is a property of the instance, not a "
            "failure of the agent, so it is excluded from the mean and never counted as 0.", ""]
    if und:
        out += ["| category | n | recovery | preservation | data fidelity |",
                "|---|---:|---:|---:|---:|"]
        for c, b in und:
            out.append("| %s | %d | %d | %d | %d |" % (
                c, b["n"], b["metrics"]["recovery"]["strict"]["undefined"],
                b["metrics"]["preservation"]["strict"]["undefined"],
                b["metrics"]["data_fidelity"]["strict"]["undefined"]))
    else:
        out.append("No metric was undefined in this run.")
    if overall["n_errors"]:
        out += ["", "## Errors", "",
                "%d of %d attempted episodes were interrupted or raised an error. Each is "
                "recorded with its reason or traceback in "
                "`episodes/<task_id>.json` and excluded from capability metrics: an endpoint "
                "or harness failure is not a model submission. The score denominator is the "
                "%d completed episodes; the error count separately reports operational "
                "reliability. Partial scores remain diagnostic. Exhausting the step budget "
                "still counts as a completed attempt."
                % (overall["n_errors"], overall["n"], overall["n_completed"])]
        if overall["n_errors_unscored"]:
            out.append("%d of them could not be tied back to an instance (the crash happened "
                       "outside the episode, or scoring the instance failed too). They remain "
                       "in the error count and never enter a capability mean."
                       % overall["n_errors_unscored"])
    out += ["", "## Where the episodes ran", "",
            "Run directories are `runs/<digest>`, not `runs/<task_id>`: the sandbox does not "
            "confine its Python kernel, so an agent that prints `os.getcwd()` would otherwise "
            "read its own flaw category out of the path, and `os.listdir(\"../..\")` would "
            "give up every sibling instance's category. `runs_index.json` maps the digests "
            "back, and every `episodes/<task_id>.json` carries its own `run_slug`. This hides "
            "the name; it does not confine the kernel — see the module docstring of "
            "`src/eval/sandbox_run.py` for what an agent can still reach."]
    return "\n".join(out) + "\n"


def write_report(out, report: dict) -> tuple:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rj, rm = out / "report.json", out / "report.md"
    rj.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    rm.write_text(format_markdown(report), encoding="utf-8")
    return rj, rm


# --------------------------------------------------------------------------- cli

def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "-" for c in str(name))


def _warm_font_cache(log=print) -> None:
    """Build the shared matplotlib font cache once, before any worker starts.

    It is the one piece of state the scoring subprocesses share (MPLCONFIGDIR),
    and several cold processes building it at once is the only race the parallel
    path would have.
    """
    try:
        import harness                                   # capture.py put src/ on the path
    except ImportError:
        return
    if not list(Path(harness.MPL_CACHE).glob("fontlist-*.json")):
        log("warming the matplotlib font cache once ...")
        harness.warm_cache()


def expected_task_ids(meta: dict, root: Path) -> set[str]:
    """Task IDs belonging to the selection pinned in a run config."""
    source = meta.get("dataset_path") or meta.get("dataset")
    mods = tuple(meta.get("modalities") or ())
    if not source or not mods:
        raise ValueError("run.json lacks dataset_path/dataset or modalities")
    ds = dataset_mod.load(source, Path(root), check=False)
    items = select(ds, meta.get("categories"), meta.get("limit"))
    track = meta.get("track", tracks.DEFAULT_TRACK)
    step_budget = meta.get("step_budget", tracks.DEFAULT_STEP_BUDGET)
    wall_time_s = meta.get("wall_time_s", tracks.DEFAULT_WALL_TIME_S)
    mode = meta.get("track_c_mode", track_mode(ds, track))
    return {
        to_sandbox.task_id(item["instance"], dataset_mod.flaw_dir(item), mods,
                           track, step_budget, clean, wall_time_s=wall_time_s)
        for item, clean, _ in selected_protocol_jobs(items, track, mode)
    }


def main(argv=None) -> int:
    with cancellation.install_signal_handlers():
        try:
            return _main(argv)
        except cancellation.Cancelled:
            print("INTERRUPTED: stop requested; saved attempts will be retried on resume.")
            return 130


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="claude-haiku-4-5",
                    help="hosted id (claude-*, gpt-*) or a ENDPOINT endpoint slug")
    ap.add_argument("--dataset", help="dataset name under data/v0/datasets, or a path to one")
    ap.add_argument("--modalities", help="what the agent may see, comma-separated out of %s "
                                         "(default: all four)" % ", ".join(modalities.ALL))
    ap.add_argument("--out", help="run directory (default data/v0/runs/<dataset>__<model>__<code>)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--budget-usd", type=float, default=None,
                    help="hard cap: no further episode is started once the run would "
                         "cross it")
    ap.add_argument("--allow-unpriced", action="store_true",
                    help="run despite --budget-usd when the model reports no cost "
                         "(free endpoints only; a billed model without a price entry "
                         "would otherwise run unbounded)")
    ap.add_argument("--categories", help="comma separated, a subset of the dataset's")
    ap.add_argument("--limit", type=int, help="only the first N items after --categories")
    ap.add_argument("--force", action="store_true", help="re-run finished episodes")
    ap.add_argument("--retry-errors", action="store_true",
                    help="re-run only the episodes that failed")
    ap.add_argument("--defer-after-seconds", type=float,
                    help="defer the first attempt after this duration; fresh tail has no P90 deadline and retains the original tool-step cap per attempt")
    ap.add_argument("--deferral-reference", type=Path,
                    help="JSON evidence for the fixed per-model deferral threshold; content and SHA enter run identity")
    ap.add_argument("--stagnation-guard", action="store_true",
                    help="stop repeated tool/output cycles without workspace changes; part of the episode protocol")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="rebuild the report from the episode files, run nothing")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would run and what it would cost, call nothing")
    ap.add_argument("--offline", action="store_true",
                    help="no-op scripted agent instead of a model: free, repairs nothing")
    ap.add_argument("--no-images", action="store_true", help="run the agent text-only")
    ap.add_argument("--single-shot", action="store_true",
                    help="Track B: preload all four inputs and request one repaired program, without tool interaction")
    ap.add_argument("--track", default=tracks.DEFAULT_TRACK,
                    help="evaluation protocol: A diagnosed repair (the flaw is named), "
                         "B autonomous repair (default), C open-world verification "
                         "(an explicit clean/flawed population or legacy clean twins)")
    step_limits = ap.add_mutually_exclusive_group()
    step_limits.add_argument("--max-steps", type=int, default=tracks.DEFAULT_STEP_BUDGET,
                    help="agent step budget per episode (default %d); part of the "
                         "task identity" % tracks.DEFAULT_STEP_BUDGET)
    step_limits.add_argument("--no-step-limit", dest="max_steps", action="store_const", const=None,
                             help="no tool-step cap; finish and no-tool guards still apply")
    wall_limits = ap.add_mutually_exclusive_group()
    wall_limits.add_argument("--wall-time-limit", type=int, default=tracks.DEFAULT_WALL_TIME_S,
                             help="episode wall-time cap in seconds (default 900)")
    wall_limits.add_argument("--no-wall-time-limit", dest="wall_time_limit", action="store_const",
                             const=None, help="no episode wall-time cap; operation timeouts remain")
    ap.add_argument("--samples", default=None,
                    help="instance root (default data/v0/samples); point it at an exported "
                         "V2 instance root to run the same protocols on V2 releases")
    ap.add_argument("--tasks", default="data/v0/sandbox_tasks")
    ap.add_argument("--root", default=str(dataset_mod.DEFAULT_ROOT))
    ap.add_argument("--timeout", type=int, default=120, help="seconds per scoring capture")
    args = ap.parse_args(argv)

    try:
        mods = modalities.validate(args.modalities)
        track = tracks.validate_track(args.track)
        step_budget = tracks.validate_step_budget(args.max_steps)
        wall_time_s = tracks.validate_wall_time(args.wall_time_limit)
        reference = None
        if args.deferral_reference:
            raw = args.deferral_reference.read_bytes()
            reference = {"sha256": hashlib.sha256(raw).hexdigest(), "content": json.loads(raw)}
        from .stagnation import POLICY as STAGNATION_POLICY
        stagnation_policy = copy.deepcopy(STAGNATION_POLICY) if args.stagnation_guard else None
        policy = scheduling_policy(args.defer_after_seconds, reference, step_budget=step_budget)
        if policy and wall_time_s is not None:
            raise ValueError("deferral requires --no-wall-time-limit; the tool-step cap applies per attempt")
        if args.single_shot:
            if track != "B" or tuple(mods) != tuple(modalities.DEFAULT):
                raise ValueError("single-shot requires Track B and all four modalities")
            if step_budget is not None or wall_time_s is not None:
                raise ValueError("single-shot requires --no-step-limit --no-wall-time-limit; its budget is one model response")
            if (args.no_images or args.offline or policy or args.deferral_reference
                    or args.stagnation_guard):
                raise ValueError("single-shot cannot use text-only, offline, deferral or stagnation settings")
            from .single_shot import MODELS
            if args.model not in MODELS:
                raise ValueError("single-shot is prepared for Gemma and Qwen only")
    except ValueError as exc:
        ap.error(str(exc))
    protocol_suffix = tracks.task_suffix(track, step_budget, wall_time_s=wall_time_s)
    if args.single_shot:
        protocol_suffix += "__single-shot-v1"
    if args.samples:
        # The instance root is read at call time by task building and scoring,
        # so pointing it elsewhere switches every protocol to that stock.
        to_sandbox.SAMPLES = Path(args.samples)

    # ---- aggregate-only: the episode files are the whole input
    if args.aggregate_only:
        if not args.out:
            ap.error("--aggregate-only needs --out (the run directory to rebuild)")
        out = Path(args.out)
        cfg_path = out / "run.json"
        meta = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        if args.samples:
            to_sandbox.SAMPLES = Path(args.samples)
        elif meta.get("samples"):
            to_sandbox.SAMPLES = Path(meta["samples"])
        loaded = load_episodes(out)
        expected = expected_task_ids(meta, Path(args.root))
        extras = sorted(set(loaded) - expected)
        eps = [loaded[task_id] for task_id in expected if task_id in loaded]
        if not eps:
            raise SystemExit("no episode files under %s" % episodes_dir(out))
        if extras:
            print("ignored %d stale episode files outside run.json selection" % len(extras))
        report = aggregate(eps, meta, meta.get("skipped"))
        rj, rm = write_report(out, report)
        print(format_markdown(report))
        print("wrote %s and %s" % (rj, rm))
        return 0

    if not args.dataset:
        ap.error("--dataset is required (a name under data/v0/datasets, or a path)")
    ds = dataset_mod.load(args.dataset, Path(args.root), check=False)
    mode = track_mode(ds, track)
    cats = [c.strip() for c in args.categories.split(",")] if args.categories else None
    try:
        items = select(ds, cats, args.limit)
        protocol_jobs = selected_protocol_jobs(items, track, mode)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not items:
        raise SystemExit("the selection is empty")

    model_label = "offline" if args.offline else args.model
    out = Path(args.out) if args.out else Path(args.root) / "runs" / (
        "%s__%s__%s%s" % (_safe(ds.get("name", "ds")), _safe(model_label),
                          modalities.code(mods), protocol_suffix))
    if policy and not args.out:
        digest = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()[:12]
        out = out.with_name(out.name + "__defer-" + digest)
    prior = None if args.offline else cost_per_episode(args.model)

    # ---- dry run: no sandbox, no task building, no API
    if args.dry_run:
        done = resumable_completed(out, retry_errors=args.retry_errors)
        todo = [(i, clean) for i, clean, _ in protocol_jobs
                if to_sandbox.task_id(i["instance"], dataset_mod.flaw_dir(i), mods,
                                      track, step_budget, clean,
                                      wall_time_s=wall_time_s) not in done]
        n = len(protocol_jobs) if args.force else len(todo)
        print("dataset     %s  (%d items selected of %d)"
              % (ds.get("name", args.dataset), len(items), ds.get("total", len(items))))
        print("modalities  %s (%s)" % (", ".join(mods), modalities.code(mods)))
        print("track       %s, tool-step limit %s, wall-time limit %s, population %s" % (
            track, step_budget, wall_time_s, mode))
        print("out         %s" % out)
        if args.single_shot:
            print("interaction single-shot-v1: one model response, inputs preloaded, no tools or feedback")
        if policy:
            if policy.get("tail_step_budget") is None:
                print("scheduling  first attempt %.3fs, fresh uncapped tail retry once" % policy["defer_after_seconds"])
            else:
                print("scheduling  first attempt %.3fs, fresh tail without P90 deadline; %d tool calls per attempt" % (
                    policy["defer_after_seconds"], policy["tail_step_budget"]))
                print("accounting  a discarded first attempt can additionally consume up to %d tool calls; this is not a cumulative task cap" % policy["tail_step_budget"])
        print("episodes    %d to run%s"
              % (n, "" if args.force else " (%d already on disk)" % len(done)))
        if prior is None:
            print("cost        not priced: %s has no entry in the sandbox price table"
                  % model_label)
        else:
            print("cost        ~$%.2f  (%d x $%.4f per episode)" % (n * prior, n, prior))
            print("            extrapolated from ONE measured episode: a full-modality %s "
                  "episode cost $%.3f over %d steps, scaled by the price ratio. A different "
                  "model, a smaller modality set or a longer episode moves this."
                  % (ANCHOR_MODEL, ANCHOR_USD, ANCHOR_STEPS))
        if args.budget_usd is not None and prior is not None and n * prior > args.budget_usd:
            print("budget      $%.2f would stop the run after about %d episodes"
                  % (args.budget_usd, int(args.budget_usd / prior)))
        print("\ndry run: nothing was called.")
        return 0

    # ---- the real thing
    cancellation.check()
    run_scorer = scorer_provenance()
    run_episode, _, _ = sandbox_run.load_sandbox()   # imports here, before the pool starts
    _warm_font_cache()
    cancellation.check()
    single_shot_protocol = None
    if args.single_shot:
        from . import single_shot
        make_agent, model_label = single_shot.build_agent_factory(args.model)
        run_episode = single_shot.run_episode
        single_shot_protocol = single_shot.protocol_manifest()
    else:
        make_agent, model_label = agent_factory_when_ready(
            args.model, offline=args.offline, no_images=args.no_images)

    ds_path = dataset_mod.dataset_path(args.dataset, Path(args.root))
    meta = {
        "model": model_label,
        "dataset": ds.get("name", str(args.dataset)),
        "dataset_path": str(ds_path),
        "dataset_sha256": hashlib.sha256(ds_path.read_bytes()).hexdigest(),
        "modalities": list(mods),
        "modality_code": modalities.code(mods),
        "track": track,
        "step_budget": step_budget,
        "wall_time_s": wall_time_s,
        "track_c_mode": mode,
        "scheduling_policy": policy,
        "stagnation_policy": stagnation_policy,
        "scheduling_reference": reference,
        "samples": str(to_sandbox.SAMPLES),
        "scorer": run_scorer,
        "n_items": len(items),
        "workers": args.workers,
        "budget_usd": args.budget_usd,
        "categories": cats,
        "limit": args.limit,
        "started_at": datetime.datetime.now(datetime.timezone.utc).replace(
            microsecond=0).isoformat(),
    }
    if args.single_shot:
        meta.update(interaction_protocol="single-shot-v1", model_turn_budget=1,
                    single_shot_protocol=single_shot_protocol)
    meta = run_config(out, meta, force=args.force)
    ep_meta = {k: meta[k] for k in ("model", "dataset", "modalities", "modality_code",
                                    "track", "step_budget", "wall_time_s", "track_c_mode", "scorer", "scheduling_policy", "stagnation_policy")}
    if args.single_shot:
        ep_meta.update({k: meta[k] for k in ("interaction_protocol", "model_turn_budget",
                                            "single_shot_protocol")})

    jobs, unbuildable = build_jobs(items, Path(args.tasks), mods,
                                   track=track, step_budget=step_budget,
                                   wall_time_s=wall_time_s, track_c_mode=mode)
    print("modalities: %s (%s)" % (", ".join(mods), modalities.code(mods)))
    print("track:      %s, tool-step limit %s, wall-time limit %s" % (
        track, step_budget, wall_time_s))
    print("dataset:    %s, %d items, %d task dirs ready%s"
          % (meta["dataset"], len(items), len(jobs),
             ", %d unbuildable" % len(unbuildable) if unbuildable else ""))

    enforced = args.budget_usd is not None and prior is not None
    if args.budget_usd is not None and not enforced:
        # A cap that cannot be enforced is not a cap. Refuse instead of warning:
        # an unpriced-but-billed model (a new Anthropic id missing from the price
        # table) would otherwise run unbounded. Free endpoints opt out explicitly.
        if not args.allow_unpriced:
            raise SystemExit(
                "%s reports no cost prior, so --budget-usd $%.2f cannot be enforced. "
                "For a free endpoint pass --allow-unpriced; for a billed model add "
                "its price so the cap works." % (model_label, args.budget_usd))
        print("WARNING: %s reports no cost; running uncapped (--allow-unpriced)."
              % model_label)
    elif args.budget_usd is None and prior is not None:
        print("WARNING: no --budget-usd. This run is uncapped and will spend about "
              "$%.2f." % (len(jobs) * prior))
    budget = Budget(args.budget_usd, prior=prior, enforced=enforced)
    if enforced:
        print("budget:     $%.2f, starting estimate $%.4f per episode"
              % (args.budget_usd, prior))

    def episode_fn(job, run_dir):
        return sandbox_episode(job, run_dir, make_agent, run_episode, args.timeout,
                               stagnation_policy=stagnation_policy)

    t0 = time.time()
    with RunLock(out) as _lock:            # noqa: F841  (held for the whole run)
        episodes, not_run = run_episodes(
            jobs, out, episode_fn, meta=ep_meta, workers=args.workers, budget=budget,
            force=args.force, retry_errors=args.retry_errors, model=args.model)

    # A job that never ran was either refused by the budget or cancelled by a
    # Ctrl-C. The budget counts its own refusals, so the rest is the interruption.
    n_budget = min(budget.skipped, len(not_run))
    skipped = {"budget": n_budget, "interrupted": len(not_run) - n_budget,
               "unbuildable": len(unbuildable),
               "not_run": [j["task_id"] for j in not_run][:50],
               "unbuildable_items": [u["task_id"] for u in unbuildable][:50]}
    meta = {**meta, "skipped": skipped,
            "finished_at": datetime.datetime.now(datetime.timezone.utc).replace(
                microsecond=0).isoformat(),
            "run_wall_time_s": round(time.time() - t0, 1)}
    (out / "run.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    report = aggregate(episodes, meta, skipped)
    rj, rm = write_report(out, report)
    if budget.tripped:
        print("\nBUDGET STOP: $%.4f of $%.2f spent over %d priced episodes. "
              "%d items were NOT run."
              % (budget.spent, budget.cap, budget.n_costed, n_budget))
    if skipped["interrupted"]:
        print("INTERRUPTED: %d items were not run." % skipped["interrupted"])
    if unbuildable:
        print("%d items had no buildable task and were not run." % len(unbuildable))
    print()
    print(format_markdown(report))
    print("wrote %s and %s" % (rj, rm))
    return 130 if cancellation.requested() else 0


if __name__ == "__main__":
    raise SystemExit(main())
