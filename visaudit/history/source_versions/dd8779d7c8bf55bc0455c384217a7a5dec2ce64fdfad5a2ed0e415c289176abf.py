r"""Run a real agent through the sandbox, then score its repair with our metrics.

One command: missing task directories are built from our instances on the way in,
the sandbox drives the episode (tools, step budget, its own verifiers), and the
agent's final `chart.py` is scored with the rule-based metrics.

Two verdicts on execution, and they are not the same question. `executability`
comes from OUR re-execution of the agent's final `chart.py` (`metrics.score`):
does the submitted script run and draw a chart. `produced_png` is the sandbox's
`renders_ok` subgoal, which only checks that an `output.png` exists in the
workspace and opens as an image — it never runs the submission. For a repair
benchmark the first is the question, so it is the headline and the gate the other
metrics already use; the second is kept as a diagnostic under its real name.

Which of the four modalities the agent may see is a configuration, not a
constant. `--modalities` is the difficulty knob; the set travels in the task id
(`to_sandbox.task_id`), so the same instance at two levels never shares a task
directory, and every result row records what the agent was actually given.

    ENDPOINT_API_KEY=... .venv/bin/python -m src.eval.sandbox_run \
        --model granite-4-1-30b --no-images \
        --instances 0035_094d6db2__wrong-title

    ANTHROPIC_API_KEY=... .venv/bin/python -m src.eval.sandbox_run \
        --model claude-haiku-4-5 --modalities code,table,summary --limit 5

RUN DIRECTORY IDENTITIES
------------------------
`run_slug` uses a deterministic task-id digest to keep flaw labels out of paths
visible to the agent and let resume locate the same directory. Task-id mappings
and result records stay outside the accessible workspace. `prepare_run_dir`
clears an existing attempt before a retry so stale artifacts cannot affect it.

KERNEL CONFINEMENT
------------------
`kernel_isolation` replaces the local kernel import binding with a Seatbelt
worker. The OS permits only the workspace and explicit runtime files, denying
oracle/dataset/other episode reads and writes, including through subprocesses.
A real filesystem/network/render preflight must pass before the runner returns;
unsupported platforms fail closed. The sibling sandbox repository is unchanged.
The digest naming remains useful for keeping task labels out of visible paths.
"""
import argparse
import ast
import base64
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path

from . import cancellation, modalities, to_sandbox
from .capture import capture, capture_path, harness
from .compare import TableIndex
from .metrics import score
from .providers import TEXT_ONLY_NOTE, make_provider, strip_view_image

SANDBOX = Path(__file__).resolve().parents[2].parent / "ChartReviewBench" / "sandbox"
ENDPOINT_BASE = "${ENDPOINT_BASE}"
HOSTED = ("gpt", "o1", "o3", "o4", "claude", "fable", "mythos")
# Open-weight models whose names share a hosted prefix but live on ENDPOINT.
ENDPOINT_DESPITE_PREFIX = ("gpt-oss",)


def wait_for_endpoint_endpoint(model: str | None, max_wait: float = 6 * 3600) -> bool:
    """Block while a ENDPOINT endpoint does not answer; True once it does again.

    Hosted models (Anthropic, OpenAI) have their own retries, so they return
    immediately. For a ENDPOINT slug this polls `/v1/models` once a minute, which is
    what turns a gateway outage from "every episode of the next hours fails"
    into "the run pauses and continues".
    """
    if not model or is_hosted(model):
        return False
    key = os.environ.get("ENDPOINT_API_KEY", "").split(",")[0].strip()
    if not key:
        return False
    import httpx

    deadline = time.time() + max_wait
    while time.time() < deadline:
        cancellation.check()
        try:
            if httpx.get("%s/%s/v1/models" % (ENDPOINT_BASE, model),
                         headers={"ENDPOINT_API_KEY": key},
                         timeout=deadline_request_timeout(30)).status_code == 200:
                cancellation.check()
                return True
        except Exception:  # noqa: BLE001 - that is the outage we are waiting out
            pass
        cancellation.wait(60)
    return False


def is_hosted(model: str) -> bool:
    """Whether `model` is a hosted SDK id (OpenAI, Anthropic) rather than a ENDPOINT slug.

    Open-weight `gpt-oss` models are served by ENDPOINT despite sharing a hosted
    model prefix; keep them on the ENDPOINT credential and endpoint path.
    """
    return model.startswith(HOSTED) and not model.startswith(ENDPOINT_DESPITE_PREFIX)


MAX_TOKENS = 8000
SEED = 42

# The suffix `to_sandbox.task_id` appends for a non-default modality set. Matched
# here so a whole task directory name can be pasted into --instances.
MOD_SUFFIX = re.compile(r"__m-([a-z]+)$")

# Length of the run-directory digest. 16 hex characters is 64 bits: far past any
# collision over a 10k-instance pool, and short enough to still paste.
SLUG_LEN = 16
SLUG_RE = re.compile(r"^[0-9a-f]{%d}$" % SLUG_LEN)

# One writer at a time for the index; `run_benchmark` merges into it from several
# worker threads.
_INDEX_LOCK = threading.Lock()


# --------------------------------------------------------------------------- run dirs

def run_slug(task_id: str) -> str:
    """The opaque directory name for one episode. See the module docstring.

    A pure function of the task id and nothing else, which is what makes a resume
    land in the directory its own earlier attempt used.
    """
    return hashlib.sha256(str(task_id).encode("utf-8")).hexdigest()[:SLUG_LEN]


def prepare_run_dir(runs_root, task_id) -> Path:
    """Archive an existing attempt and create a clean, opaque run directory."""
    runs_root = Path(runs_root)
    slug = run_slug(task_id)
    d = runs_root / slug
    episode = runs_root.parent / "episodes" / (str(task_id) + ".json")
    if d.exists() or episode.exists():
        archive = runs_root.parent / "attempts" / slug / (str(time.time_ns()) + "-" + uuid.uuid4().hex[:8])
        archive.mkdir(parents=True)
        (archive / "attempt.json").write_text(json.dumps({
            "task_id": task_id, "run_slug": slug, "resume_mode": "fresh_attempt",
            "note": "Prior workspace, partial checkpoints and episode record are preserved; model history is not resumed.",
        }, indent=2))
        if d.exists():
            d.rename(archive / "run")
        if episode.exists():
            episode.rename(archive / "episode.json")
    d.mkdir(parents=True, exist_ok=True)
    return d


def index_path(runs_root) -> Path:
    """Where the digest -> task id map lives: beside the runs root, never inside.

    Inside it, the map would be adjacent to the agent's own working directory and
    would give away in one read what naming the directories by digest bought.
    """
    runs_root = Path(runs_root)
    return runs_root.parent / (runs_root.name + "_index.json")


def record_run_dirs(runs_root, task_ids) -> Path:
    """Merge `digest -> task id` for these tasks into the index and return its path.

    Merged rather than overwritten because a resumed run only knows about the
    episodes it still has to do, and the index has to keep naming the ones that
    finished in an earlier attempt.

    A collision would put two episodes in one directory and each would overwrite
    the other's workspace, which is the kind of failure that shows up as an
    inexplicable score rather than as an error. 64 bits over a 10k pool make it
    vanishingly unlikely; it is checked anyway, because the check is free and the
    failure is not.
    """
    path = index_path(runs_root)
    with _INDEX_LOCK:
        old = {}
        if path.exists():
            try:
                old = (json.loads(path.read_text(encoding="utf-8")) or {}).get("runs", {})
            except (json.JSONDecodeError, AttributeError):
                old = {}                      # a torn file is not worth losing the run over
        fresh = {run_slug(t): t for t in task_ids}
        if len(fresh) != len(set(task_ids)):
            raise ValueError("run_slug collision inside one run; two task ids share a "
                             "directory. Widen SLUG_LEN.")
        clash = [(s, old[s], fresh[s]) for s in fresh if s in old and old[s] != fresh[s]]
        if clash:
            raise ValueError("run_slug collision with %s: %s" % (path, clash))
        runs = {**old, **fresh}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name("." + path.name + ".tmp")
        tmp.write_text(json.dumps({
            "schema": 1,
            "note": ("The run directories under %s/ are named by a digest of the task id so "
                     "the agent's own working directory does not spell out the flaw "
                     "category. This maps them back. It lives outside that directory on "
                     "purpose: task labels stay out of agent-visible paths; the kernel "
                     "also enforces OS workspace confinement." % runs_root.name),
            "runs": dict(sorted(runs.items())),
        }, indent=1), encoding="utf-8")
        tmp.replace(path)
    return path


def endpoint_client(slug: str):
    """OpenAI client pointed at a ENDPOINT endpoint (key travels in a header)."""
    import openai
    key = os.environ.get("ENDPOINT_API_KEY")
    if not key:
        raise SystemExit("ENDPOINT_API_KEY is not set")
    return openai.OpenAI(
        base_url="%s/%s/v1" % (ENDPOINT_BASE, slug),
        api_key=key,
        default_headers={"ENDPOINT_API_KEY": key},
        # Bound stalled requests; the provider retry layer handles transient
        # timeouts independently of the episode's interaction budget.
        timeout=180,
        max_retries=0,       # retries are handled in ENDPOINT._create, with backoff
    )


def endpoint_served_model(client, attempts=4):
    """Resolve the served model id without letting one gateway timeout kill a run."""
    delay = 4
    for attempt in range(attempts):
        cancellation.check()
        try:
            kwargs = {} if cancellation.remaining() is None else {
                "timeout": deadline_request_timeout(getattr(client, "timeout", 180))}
            models = client.models.list(**kwargs).data
            cancellation.check()
            if not models:
                raise RuntimeError("ENDPOINT /models returned no models")
            return models[0].id
        except Exception as exc:  # noqa: BLE001 - classified below
            cancellation.check()
            if not transient_provider_error(exc) or attempt == attempts - 1:
                raise
            cancellation.wait(delay)
            delay *= 2


def _balanced_json_end(text: str, start: int) -> int | None:
    """Index after one balanced JSON object/array at ``start``, or ``None``.

    Matching delimiter types as well as depth matters here: ``{]`` must not let a
    recovery heuristic skip ahead and reinterpret a later object as the call's
    arguments.  ``json.loads`` remains the authority on whether the balanced
    span is valid JSON.
    """
    if start >= len(text) or text[start] not in "{[":
        return None
    pairs = {"{": "}", "[": "]"}
    stack, in_str, esc = [], False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c in pairs:
            stack.append(pairs[c])
        elif c in "}]":
            if not stack or c != stack.pop():
                return None
            if not stack:
                return i + 1
    return None


def _json_candidates(text: str):
    """Yield every balanced top-level {...} or [...] span in the text, string-aware.

    A regex cannot pair braces, and the arguments of a write_file/execute_python
    call contain nested dicts and braces inside string literals as a matter of
    course. Balance is tracked by hand; whether a span is real JSON is decided by
    json.loads at the call site.
    """
    i, n = 0, len(text)
    while i < n:
        if text[i] not in "{[":
            i += 1
            continue
        end = _balanced_json_end(text, i)
        if end is not None:
            yield text[i:end]
            i = end
        else:
            i += 1


def _named_json_calls(text: str, known: set) -> list:
    """Recover bare ``tool_name{...}`` calls emitted by some chat templates.

    Only an exact known tool name at an identifier boundary is considered.  The
    suffix must be one complete balanced JSON object and must decode to a dict;
    prose after an unclosed object is never searched for a smaller nested match.
    """
    if not known:
        return []
    names = "|".join(re.escape(name) for name in sorted(known, key=len, reverse=True))
    pattern = re.compile(r"(?<![\w.])(?:%s)(?!\w)\s*(?=\{)" % names)
    out = []
    for match in pattern.finditer(text):
        name = match.group(0).strip()
        start = match.end()
        while start < len(text) and text[start].isspace():
            start += 1
        end = _balanced_json_end(text, start)
        if end is None:
            continue
        try:
            args = json.loads(text[start:end])
        except json.JSONDecodeError:
            continue
        if isinstance(args, dict):
            out.append({"name": name, "args": args})
    return out


def _normalize_calls(obj, known: set) -> list:
    """Flatten the call shapes models actually emit into [{'name', 'args'}]."""
    out = []

    def one(d):
        if not isinstance(d, dict):
            return
        if isinstance(d.get("function"), dict):      # OpenAI tool_calls element
            d = d["function"]
        name = d.get("name")
        args = next((d[k] for k in ("arguments", "args", "parameters", "input")
                     if k in d), None)
        if not isinstance(name, str) or name not in known:
            return
        if isinstance(args, str):                     # OpenAI stringifies arguments
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return
        if isinstance(args, dict):
            out.append({"name": name, "args": args})

    if isinstance(obj, list):
        for el in obj:
            one(el)
    elif isinstance(obj, dict):
        for key in ("functions", "function_calls", "tool_calls", "calls"):
            if isinstance(obj.get(key), list):
                for el in obj[key]:
                    one(el)
        if not out:
            one(obj)
    return out


def _python_literal_calls(text: str, known: set) -> list:
    """Decode explicit Python-style tool expressions without executing their arguments."""
    blocks = [text.strip()] + re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, re.S)
    parameters = {"execute_python": ("code",), "read_file": ("path", "max_bytes"),
                  "write_file": ("path", "content"), "list_files": ("path",),
                  "view_image": ("path",), "finish": ("message",)}
    out = []
    for block in blocks:
        try:
            tree = ast.parse(block.strip())
        except (SyntaxError, ValueError):
            continue
        # Ordinary plotting programs are not implicit execute_python requests.
        for statement in tree.body:
            if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
                continue
            call = statement.value
            if not isinstance(call.func, ast.Name) or call.func.id not in known:
                continue
            name = call.func.id
            try:
                values = [ast.literal_eval(value) for value in call.args]
                keywords = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
                if None in keywords or len(keywords) != len(call.keywords):
                    continue
                if len(values) == 1 and isinstance(values[0], dict) and not keywords:
                    args = values[0]
                else:
                    names = parameters.get(name, ())
                    if len(values) > len(names) or set(names[:len(values)]) & keywords.keys():
                        continue
                    args = {**dict(zip(names, values)), **keywords}
                if not all(isinstance(key, str) for key in args):
                    continue
                json.dumps(args)
            except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                continue
            out.append({"name": name, "args": args})
    return out


def text_tool_calls(text: str, known: set) -> list:
    """Recover tool calls a model wrote into its message instead of the API field.

    Endpoints without a native tool-call parser can return calls as text.
    Supported forms include named JSON objects, call arrays, stringified or
    nested JSON arguments, Python calls with literal arguments, and XML.
    Argument payloads are parsed without executing model-generated code.
    """
    calls, seen = [], set()

    def add(call):
        key = (call["name"], json.dumps(call["args"], sort_keys=True))
        if key not in seen:
            seen.add(key)
            calls.append(call)

    for cand in _json_candidates(text):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        for c in _normalize_calls(obj, known):
            add(c)
    for c in _named_json_calls(text, known):
        add(c)
    for c in _python_literal_calls(text, known):
        add(c)
    # <invoke name="read_file"><parameter name="path">chart.py</parameter></invoke>
    for m in re.finditer(r'<invoke\s+name="([^"]+)"\s*>(.*?)</invoke>', text, re.S):
        if m.group(1) not in known:
            continue
        args = {k: v.strip() for k, v in
                re.findall(r'<parameter\s+name="([^"]+)"[^>]*>(.*?)</parameter>', m.group(2), re.S)}
        add({"name": m.group(1), "args": args})
    for i, c in enumerate(calls):
        c["id"] = recovered_id(i)
    return calls


def recovered_id(i: int) -> str:
    """An id for a recovered call: exactly nine alphanumeric characters.

    Mistral chat templates require this shape. The id is returned with the tool
    result in the next request, so an invalid id would fail a subsequent turn.
    """
    return "rec%06d" % i


# These chat templates require strict role alternation. Image tool results
# become separate user messages, so the bridge inserts an assistant turn when
# needed to keep that sequence valid.
STRICT_ALTERNATION = ("mistral", "ministral")


def _bridge_tool_to_user(native: list) -> list:
    """Insert a minimal assistant turn wherever a user message follows a tool one.

    The alternative — folding the text into the preceding tool message — cannot
    carry an image block, which is exactly what these messages exist for. A
    one-line assistant bridge keeps every observation intact and costs the model
    a token it would have produced anyway.
    """
    out = []
    for msg in native:
        if (msg.get("role") == "user" and out and out[-1].get("role") == "tool"):
            out.append({"role": "assistant", "content": "Continuing."})
        out.append(msg)
    return out


def deadline_request_timeout(configured):
    """Bound each HTTP phase by the remaining first-pass scheduling time."""
    cancellation.check()
    left = cancellation.remaining()
    if left is None:
        return configured
    if hasattr(configured, "read"):
        import httpx
        return httpx.Timeout(**{
            phase: min(value, left) if value is not None else left
            for phase in ("connect", "read", "write", "pool")
            for value in [getattr(configured, phase)]})
    return min(configured, left) if isinstance(configured, (float, int)) else left


def endpoint_provider(base_cls, text_only: bool, strict_alternation: bool = False):
    """Subclass the sandbox's vLLM provider for the ENDPOINT endpoints.

    Two quirks live there, both worked around here: the endpoints never return
    native tool calls, and a text-only model rejects content sent as a list of
    typed blocks ("is not a multimodal model"). The second is why the redaction
    stays local instead of using `providers.redact_images`: here every non-text
    block has to go, not only the images.
    """
    class ENDPOINT(base_cls):
        def _create(self, **kwargs):
            """Retry the transient failures of a shared gateway.

            Retry classified transport and server errors with backoff to avoid
            making task coverage depend on temporary endpoint availability.
            Propagate failures that persist through four attempts.
            """
            delay = 4
            for attempt in range(4):
                cancellation.check()
                try:
                    request_kwargs = kwargs
                    if cancellation.remaining() is not None:
                        configured = kwargs.get("timeout", getattr(getattr(self, "client", None), "timeout", 180))
                        request_kwargs = {**kwargs, "timeout": deadline_request_timeout(configured)}
                    response = super()._create(**request_kwargs)
                    # Some ENDPOINT deployments return HTTP 200 with an OpenAI-shaped
                    # error body and no choices.  The SDK accepts that payload as a
                    # ChatCompletion with ``choices=None``; the upstream provider
                    # then crashes at ``resp.choices[0]`` and mislabels a deployment
                    # failure as a local TypeError.  Promote the embedded error into
                    # the normal retry/classification path instead.
                    if hasattr(response, "choices") and not response.choices:
                        extra = getattr(response, "model_extra", None) or {}
                        detail = getattr(response, "error", None) or extra.get("error")
                        raise RuntimeError(
                            "ENDPOINT internal server error returned no choices: %s" % detail)
                    return response
                except Exception as exc:            # noqa: BLE001 - classified below
                    cancellation.check()
                    if not transient_provider_error(exc) or attempt == 3:
                        raise
                    cancellation.wait(delay)
                    delay *= 2

        def _to_native(self, messages):
            out = super()._to_native(messages)
            for m in out:
                c = m.get("content")
                if not isinstance(c, list):
                    continue
                if text_only:
                    # A text-only agent can still pull an image in by reading a
                    # PNG, which the server rejects outright. Swap it for a note.
                    c = [b if b.get("type") == "text" else
                         {"type": "text", "text": TEXT_ONLY_NOTE}
                         for b in c]
                    m["content"] = c
                if c and all(b.get("type") == "text" for b in c):
                    m["content"] = "\n".join(b.get("text", "") for b in c)
            return _bridge_tool_to_user(out) if strict_alternation else out

        def complete(self, system, messages, tools):
            if text_only:
                tools = strip_view_image(tools)
            turn = super().complete(system, messages, tools)
            if turn.tool_calls or not turn.text:
                return turn
            found = text_tool_calls(turn.text, {t["name"] for t in tools})
            if found:
                turn.tool_calls = found
                # Keep only the parsed calls in history: replaying raw control
                # tokens such as `[TOOL_CALLS]` alongside them can invalidate
                # the next request's chat template.
                turn.text = ""
                # The recovered calls must also reach the history, or the next
                # request carries tool results with no call to attach them to.
                # Dropping the native `raw` makes the provider rebuild the
                # assistant message from these blocks instead of replaying it.
                turn.raw = None
                # Drop the text blocks too, for the same reason: the provider
                # rebuilds the assistant message from these blocks, and a text
                # block holding the raw call syntax would put it straight back.
                turn.blocks = [b for b in (turn.blocks or [])
                               if b["type"] not in ("tool_call", "text")]
                turn.blocks += [{"type": "tool_call", **c} for c in found]
            return turn
    return ENDPOINT


def transient_provider_error(exc):
    """Recognize retryable gateway failures even after a provider wraps them.

    The sandbox provider converts OpenAI ``APITimeoutError`` and connection
    exceptions to ``RuntimeError``. Classifying only the outer type therefore
    made the advertised retry loop run exactly once. Walk the exception chain
    and use the narrow transport phrases emitted by that wrapper.
    """
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        name = type(exc).__name__
        message = str(exc).lower()
        if (any(token in name for token in
                ("Redirect", "Timeout", "APIConnection", "InternalServer"))
                or getattr(exc, "status_code", None) in (429, 500, 502, 503, 504)
                or any(token in message for token in (
                    "request timed out", "connection error", "connection reset",
                    "maximum allowed redirects", "internal server error",
                    "bad gateway", "service unavailable", "gateway timeout",
                    "endpoint /models returned no models"))):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def load_sandbox():
    """Import chartsandbox from the sibling clone without installing it here."""
    sandbox_path = Path(os.environ.get("CHARTREPAIR_SANDBOX", SANDBOX))
    src = sandbox_path / "src"
    if not src.exists():
        raise SystemExit("sandbox clone not found at %s" % sandbox_path)
    sys.path.insert(0, str(src))
    for extra in sorted((sandbox_path / ".venv/lib").glob("python*/site-packages")):
        sys.path.append(str(extra))
    from chartsandbox import runner  # noqa: E402
    from chartsandbox.agent import LLMAgent  # noqa: E402
    from chartsandbox.providers.vllm_provider import VLLMProvider  # noqa: E402
    from .kernel_isolation import install_kernel_isolation
    install_kernel_isolation()
    install_safe_tool_dispatch(runner)
    from .episode_runner import run_episode
    return run_episode, LLMAgent, VLLMProvider


def install_safe_tool_dispatch(runner):
    """Return malformed tool arguments to the agent instead of killing its episode.

    The upstream dispatcher calls file tools with ``**tc.args``. A model omitting
    ``path`` therefore raises ``TypeError`` outside the sandbox loop, so it never
    sees the tool error and cannot repair its call. Expose invalid arguments
    as a failed observation so the next turn can correct them.
    """
    if getattr(runner, "_chartrepair_safe_dispatch", False):
        return
    original = runner._dispatch

    def safe_dispatch(sandbox, tool_call):
        try:
            result = original(sandbox, tool_call)
            if tool_call.tool == "view_image" and result.get("ok"):
                import io
                from PIL import Image
                try:
                    raw = base64.b64decode(result.get("base64", ""), validate=True)
                    with Image.open(io.BytesIO(raw)) as im:
                        mime = {"PNG": "image/png", "JPEG": "image/jpeg", "GIF": "image/gif",
                                "WEBP": "image/webp"}.get(im.format)
                        width, height = im.size
                        im.verify()
                    if not mime or width < 1 or height < 1:
                        raise ValueError("unsupported image format")
                except (ValueError, OSError, SyntaxError):
                    return {"ok": False, "error": "view_image requires a valid PNG, JPEG, GIF or WebP image; use read_file for text."}
                result = {**result, "mime": mime, "width": width, "height": height}
            return result
        except TypeError as exc:
            return {
                "ok": False,
                "error": "invalid arguments for %s: %s" % (tool_call.tool, exc),
            }

    runner._dispatch = safe_dispatch
    runner._chartrepair_safe_dispatch = True


def split_spec(spec: str, mods: tuple) -> tuple:
    """(instance, category) from an --instances entry.

    Accepts a bare instance folder, `<instance>__<category>`, and a whole task
    directory name. The modality suffix of a pasted task id is checked rather
    than ignored: it would otherwise be read as part of the category name and
    fail much later with an unrelated message.
    """
    m = MOD_SUFFIX.search(spec)
    if m:
        try:
            given = modalities.from_code(m.group(1))
        except ValueError as exc:
            raise SystemExit("%s: %s" % (spec, exc)) from exc
        if given != mods:
            raise SystemExit(
                "%s names modalities %s, but this run builds %s. Pass "
                "--modalities %s, or drop the suffix."
                % (spec, ",".join(given), ",".join(mods), ",".join(given)))
        spec = spec[:m.start()]
    name, _, category = spec.partition("__")
    return name, category


def ensure_tasks(tasks_root: Path, limit: int, instances, mods=None) -> list:
    """The task dirs to run; whatever is missing is converted from our instances.

    `mods` is the modality set the tasks are built for and is part of the task
    id, so two difficulty levels of one instance sit side by side on disk instead
    of overwriting each other.
    """
    mods = modalities.validate(mods)
    wanted = []
    for spec in instances or []:
        # '<instance>__<category>' is one pair, a bare instance means all of its
        name, category = split_spec(spec, mods)
        instance = to_sandbox.SAMPLES / name
        cats = [category] if category else sorted(
            p.name for p in (instance / "perturbed_llm").iterdir() if p.is_dir())
        wanted += [(instance, c) for c in cats]
    if not wanted:
        wanted = to_sandbox.pick(limit, SEED)

    dirs = []
    for instance, category in wanted:
        task_dir = tasks_root / to_sandbox.task_id(instance.name, category, mods)
        if not (task_dir / "task.yaml").exists():
            try:
                to_sandbox.build_task(instance, category, tasks_root, mods)
            except FileNotFoundError as exc:
                # A source file this set needs is missing and what is left does
                # not define a task. Dropping the instance keeps the rest of the
                # sweep running; a silently different difficulty would not.
                print("skipped %s: %s" % (task_dir.name, exc))
                continue
            print("converted %s" % task_dir.name)
        dirs.append(task_dir)
    return dirs


def produced_png(res: dict):
    """The sandbox's `renders_ok` verdict: 1, 0, or None when there is no such subgoal.

    Named for what it measures. `renders_ok` runs the `execution` verifier with
    `produces: output.png` (`sandbox/src/chartsandbox/verifiers/execution.py`),
    which asks whether that file exists in the workspace and opens as an image —
    it never executes the agent's final `chart.py`. So it answers a different
    question than `metrics.score`'s `executability` does, and disagrees in both
    directions: a correct script the agent never ran leaves no png, and a script
    the agent broke after one good run leaves a stale one. Executability comes
    from our re-execution; this stays a diagnostic.
    """
    for sg in res["subgoal_results"]:
        if sg["id"] == "renders_ok":
            return int(bool(sg["passed"]))
    return None


def no_submission(reason: str) -> dict:
    """A capture record standing in for a submission that does not exist.

    `metrics.score` reads `status` and `spec` to decide there is nothing to score;
    the rest of the shape is here so the record can travel anywhere a real capture
    can.
    """
    return {"status": "no_script", "duration": 0.0, "png": None, "spec": None,
            "flat": {}, "stderr": reason, "capture_error": ""}


INTERRUPTED_STOP_REASONS = frozenset({"wall_time_exceeded", "interrupted"})


def episode_stop_reason(result: dict, run_dir: Path | None = None) -> str:
    """Recover the runner's termination cause, including pre-stop-reason runs.

    The upstream runner returns normally after its wall-clock guard fires. Its
    trajectory marker, not elapsed time or a failed repair, proves that abort.
    A model consuming its full step budget still completed a benchmark attempt.
    """
    aliases = {"finish": "finished", "agent_finish": "finished",
               "step_budget": "step_budget_exhausted", "wallclock_timeout": "wall_time_exceeded",
               "wall_time_limit": "wall_time_exceeded", "aborted": "interrupted"}
    explicit = result.get("stop_reason")
    explicit = aliases.get(explicit, explicit) if isinstance(explicit, str) else None
    if explicit == "unknown":
        explicit = None
    if explicit in INTERRUPTED_STOP_REASONS:
        return explicit
    trajectory = result.get("trajectory") or []
    if run_dir is not None:
        path = Path(run_dir) / "trajectory.jsonl"
        if path.exists():
            trajectory = list(trajectory)
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    trajectory.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # interrupted legacy writes may end in a partial row
    if any(isinstance(row, dict) and row.get("tool") == "<wall_time_exceeded>"
           for row in trajectory):
        return "wall_time_exceeded"
    if explicit:
        return explicit
    if result.get("finished") is True:
        return "finished"
    steps, budget = result.get("steps_taken"), result.get("step_budget")
    if (result.get("finished") is False and isinstance(steps, int)
            and isinstance(budget, int) and budget > 0):
        return "step_budget_exhausted" if steps >= budget else "interrupted"
    return "unknown"


def reference(task_dir: Path, timeout: int = 120) -> tuple:
    """(GT, initial chart, table, scoring category, extra files) for one task.

    Everything scoring needs except the submission. Split out so an episode that
    produced no submission at all can still be scored against its own instance:
    the footprint comes from gt vs flawed, so whether a metric applies is a
    property of the instance and stays answerable when the agent's side is gone.
    """
    cfg = to_sandbox.read_task_config(task_dir)
    instance, category = to_sandbox.SAMPLES / cfg["sample_folder"], cfg["category"]
    table_path = instance / "gt" / "table.csv"
    # The scoring sandbox must offer the same files the agent's workspace held.
    # A submitted script that reads summary.txt is otherwise failed here for a
    # FileNotFoundError rather than for what it actually does, and the penalty
    # lands hardest on an agent that left the original script alone.
    extra = {"table.csv": table_path.read_text(encoding="utf-8", errors="replace")}
    summary_path = instance / "gt" / "summary.txt"
    if summary_path.exists():
        extra["summary.txt"] = summary_path.read_text(encoding="utf-8", errors="replace")
    gt = capture_path(instance / "gt" / "code.py", extra, timeout, isolate=True)
    # A clean twin starts at GT: there is nothing to recover and every original
    # property must be preserved, including the paired flawed task's old target.
    if cfg.get("clean", False):
        flawed, category = gt, ""
    else:
        flawed = capture_path(instance / "perturbed_llm" / category / "code.py", extra, timeout,
                              isolate=True)
    return gt, flawed, TableIndex.from_path(table_path), category, extra


def score_repair(task_dir: Path, run_dir: Path, timeout: int = 120) -> dict:
    """Score the agent's final chart.py against the instance it came from.

    The task config says which modalities the episode shipped; scoring itself
    always uses the full ground truth, whatever the agent got to see.
    """
    gt, flawed, table, category, extra = reference(task_dir, timeout)
    # Without the code modality the agent has to write chart.py itself, so a
    # missing script is a real, zero-scoring outcome and not a broken run.
    try:
        source = harness.read_workspace_file(run_dir / "workspace", "chart.py")
        sub = (capture(source.decode("utf-8"), extra, timeout, isolate=True)
               if source is not None else
               no_submission("the agent left no chart.py in the workspace"))
    except (ValueError, UnicodeDecodeError) as exc:
        sub = no_submission("invalid chart.py submission: " + str(exc))
    return score(gt, flawed, sub, table, category,
                 summary_text=extra.get("summary.txt"))


def score_no_submission(task_dir: Path, timeout: int = 120, reason: str = "") -> dict:
    """Score an episode that produced nothing scorable, against its own instance.

    For an episode that crashed there is no submission to run, but the question
    "does this metric even apply here" still has an answer, and it lives entirely
    on the gt/flawed side. Asking it is the difference between recording a metric
    as a failed repair and recording it as one the benchmark never asked.
    """
    gt, flawed, table, category, extra = reference(task_dir, timeout)
    return score(gt, flawed, no_submission(reason or "the episode produced no submission"),
                 table, category, summary_text=extra.get("summary.txt"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="granite-4-1-30b",
                    help="ENDPOINT endpoint slug, or a hosted id (claude-*, gpt-*)")
    ap.add_argument("--judge", help="ENDPOINT slug or hosted id for the vlm_judge "
                                    "(a different family than the agent)")
    ap.add_argument("--instances", nargs="*",
                    help="instance folders, <instance>__<category> ids or task dir names; "
                         "without this, --limit are picked across the categories")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--modalities",
                    help="what the agent may see, comma-separated out of %s "
                         "(default: all four). A smaller set is the same instance at a "
                         "harder level and gets its own task id."
                         % ", ".join(modalities.ALL))
    ap.add_argument("--no-images", action="store_true",
                    help="run as a text-only agent: view_image is not offered at all")
    ap.add_argument("--offline", action="store_true",
                    help="no-op scripted agent instead of a model: demo and tests without ENDPOINT")
    ap.add_argument("--tasks", default="data/v0/sandbox_tasks")
    ap.add_argument("--out", default="data/v0/sandbox_runs")
    args = ap.parse_args(argv)

    try:
        mods = modalities.validate(args.modalities)
    except ValueError as exc:
        ap.error(str(exc))

    run_episode, LLMAgent, VLLMProvider = load_sandbox()
    model_label = args.model
    if args.offline:
        # Runs the flawed script once and stops. Repairs nothing; it exists so the
        # whole path can be exercised without a model.
        from chartsandbox.agent import ScriptedAgent  # noqa: E402
        script = [{"tool": "execute_python",
                   "args": {"code": "exec(compile(open('chart.py').read(), 'chart.py', 'exec'))"}},
                  {"tool": "finish", "args": {"message": "offline no-op agent"}}]
        make_agent = lambda: ScriptedAgent(script)  # noqa: E731
        model_label = "offline"
        print("agent: offline no-op (no model, nothing is repaired)")
    elif is_hosted(args.model):
        # Hosted models speak native tool calls, but two things still have to be
        # handled: the effort parameter is rejected below the Claude 5 line, and
        # --no-images has to withhold view_image. Both live in .providers.
        provider = make_provider(args.model, text_only=args.no_images, max_tokens=MAX_TOKENS)
        make_agent = lambda: LLMAgent(provider)  # noqa: E731
        print("agent: %s (hosted%s)" % (args.model, ", text-only" if args.no_images else ""))
    else:
        client = endpoint_client(args.model)
        served = endpoint_served_model(client)
        provider = endpoint_provider(
            VLLMProvider, args.no_images,
            any(k in args.model for k in STRICT_ALTERNATION))(
            served, client=client, max_tokens=MAX_TOKENS,
            base_url=str(client.base_url))  # so the run stats name the real endpoint
        make_agent = lambda: LLMAgent(provider)  # noqa: E731
        print("agent: %s  (served as %s)" % (args.model, served))
    print("modalities: %s (%s)" % (", ".join(mods), modalities.code(mods)))

    if args.judge:
        if is_hosted(args.judge):
            # The judge exists to look at the chart, so text_only never applies to
            # it — but it would trip over the same effort parameter, so it goes
            # through make_provider too instead of install_judge's get_provider.
            from chartsandbox.judge import make_judge_backend  # noqa: E402
            from chartsandbox.verifiers import set_judge_backend  # noqa: E402
            set_judge_backend(make_judge_backend(
                make_provider(args.judge, max_tokens=MAX_TOKENS)))
            print("judge: %s (hosted)" % args.judge)
        else:
            from chartsandbox.judge import install_judge  # noqa: E402
            jclient = endpoint_client(args.judge)
            jserved = endpoint_served_model(jclient)
            install_judge("vllm:%s" % jserved, client=jclient)
            print("judge: %s  (served as %s)" % (args.judge, jserved))

    tasks = ensure_tasks(Path(args.tasks), args.limit, args.instances, mods)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    # One level deeper than the old `<out>/<task_id>`: the run directories now sit
    # alone under `runs/`, so listing the agent's own parent shows nothing but
    # digests. summary.json (which names the tasks) stays at `<out>`.
    runs_root = out_root / "runs"
    record_run_dirs(runs_root, [t.name for t in tasks])
    rows = []
    for i, task_dir in enumerate(tasks, 1):
        print("\n[%d/%d] %s" % (i, len(tasks), task_dir.name), flush=True)
        run_dir = prepare_run_dir(runs_root, task_dir.name)
        res = run_episode(task_dir, make_agent(), out_dir=run_dir)
        ours = score_repair(task_dir, run_dir)
        # What the task actually shipped, not what was asked for: build_task drops
        # a modality whose source file is missing.
        cfg = to_sandbox.read_task_config(task_dir)
        rows.append({
            "task": task_dir.name,
            "run_slug": run_dir.name,
            "model": model_label,
            "modalities": cfg["modalities"],
            "modality_code": cfg["modality_code"],
            # our re-execution of the submitted script, the headline
            "executability": ours["executability"],
            # the sandbox's artifact-existence check, a diagnostic
            "produced_png": produced_png(res),
            "ours": {k: ours[k] for k in ("data_fidelity", "data_fidelity_binary",
                                          "recovery", "recovery_binary",
                                          "preservation", "preservation_binary")},
            "footprint": ours["diagnostics"].get("footprint", []),
            "steps": res["steps_taken"],
            "wall_s": res["wall_time_s"],
            # absent for the scripted agent, which talks to no model at all
            "cost_usd": (res.get("agent_stats") or {}).get("cost_usd"),
            "sandbox": {k: res[k] for k in ("final_score", "subgoals_passed",
                                            "subgoals_total", "steps_taken", "wall_time_s")},
        })
        r, s, o = rows[-1], rows[-1]["sandbox"], rows[-1]["ours"]
        print("   executability %s | fidelity %s | recovery %s | preservation %s  (strict)" % (
            r["executability"], o["data_fidelity"]["strict"],
            o["recovery"]["strict"], o["preservation"]["strict"]))
        print("   sandbox: score=%s subgoals=%s/%s produced_png=%s steps=%s wall=%ss%s" % (
            s["final_score"], s["subgoals_passed"], s["subgoals_total"], r["produced_png"],
            s["steps_taken"], s["wall_time_s"],
            "" if r["cost_usd"] is None else "  cost=$%.4f" % r["cost_usd"]))
        if r["produced_png"] is not None and r["produced_png"] != r["executability"]:
            print("   NOTE: executability %s but produced_png %s — the sandbox checks that "
                  "output.png exists, not that chart.py runs."
                  % (r["executability"], r["produced_png"]))
        # written after every episode, so a run that dies later keeps its results
        (out_root / "summary.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")

    print("\ngeschrieben:", out_root / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
