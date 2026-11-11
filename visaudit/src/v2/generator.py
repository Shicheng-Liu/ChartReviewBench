"""Provider-neutral perturbation generation for the V2 production pipeline.

This module deliberately separates the wire protocol from the perturbation
contract.  ENDPOINT, Anthropic, or a test double may serve the request, while the
same prompt, JSON schema, parser, provenance fields, and cache key are used for
every provider.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from queue import Queue
from typing import Any, Iterable, Mapping, Protocol

from src import perturb_prompts


SCHEMA_VERSION = "chartrepairbench.perturbation.v2"

V2_SYSTEM_PROMPT = perturb_prompts.SYSTEM_PROMPT + """

## V2 output-contract override

For this V2 pipeline, the JSON contract at the end of the user request replaces
the legacy structured-output paragraph above. Return that JSON object directly;
there is no `edits` field. `target` is mandatory and identifies the exact
Figure-Object entity the postcondition must prove. For color contradictions,
`evidence.summary_color_mapping` is an additional allowed and mandatory V2 key;
it maps the exact summary-named entity to its exact truth color, and
`evidence.summary_quote` must copy the complete, direct affirmative clause that
maps exactly that one entity to exactly that one named color. Decline when the
clause contains another chart entity or color, negation, contrast, modality, a
condition, or a past/former-state qualifier, and also decline when any other
clause in the full summary gives that target a different color or blocks its
truth-color claim.

Repair truth must be reconstructable from the physical summary, table, and
candidate code; `target` and `evidence` never supply otherwise hidden truth.
For wrong-title, the physical summary must contain one current, direct
affirmative title-to-exact-GT-text clause, that text must be the only distinct
title claim in the summary, and the summary must not claim the injected title.
For wrong-axis-label, require either a clean direct axis/quantity-to-exact-GT-
label summary relation or an exact unique CSV column whose complete values or
locators bind to that GT axis. Every label claim in the active axis/quantity
scope must agree with the GT label, and no competing injected-label CSV
relation may exist; header presence alone, an unused code string or an
arbitrary table cell does not count. V2 changes exactly one
canonical axis label with replacement mechanism (a); do not use the legacy
two-axis swap because the binding target identifies one axis. For wrong-chart-type,
require a clean direct current relation from the whole chart or target series
to the exact GT kind, with no competing kind claim in that scope; a different
non-target series may legitimately have another kind. For missing-legend, the
summary is a conflict check only and never reconstruction evidence. Every one
of at least two GT labels must instead bind through at least one of two structural
routes: the exact full target-axis sequence of definitely-live plotted
`label=`/`name=` literals occurs once in GT legend order and maps to captured GT
series roots, or one unique complete wide named CSV series relation occurs in
exact GT legend order and the captured target-axis series set/count exactly
equals that legend group. A subset with a hidden extra series, comment, unused
assignment, call inside a function/branch/condition/comprehension, or unordered
label group does not count. For dropped-series, the complete missing
label-to-value relation must bind uniquely to the physical CSV. Its GT mark kind
must additionally be recoverable either from a clean direct current target-series-
to-kind summary statement (a whole-chart statement alone is insufficient), or
from uniform same-axis GT kinds plus at least one definitely-live surviving
labeled plot call of that kind, with every such target-axis survivor call
resolving to that same kind. Decline on negation, contrast, modality, conditions,
or former-state wording.
"""

TARGET_EXAMPLES = {
    "wrong-title": {
        "path": "axes[0].title", "truth": "original", "injected": "wrong",
    },
    "wrong-axis-label": {
        "path": "axes[0].xlabel", "axis_index": 0, "axis": "x",
        "truth": "original", "injected": "wrong",
    },
    "missing-legend": {
        "path": "axes[0].legend.labels", "labels": ["series A", "series B"],
    },
    "swapped-legend-labels": {
        "path": "axes[0].legend.labels", "labels": ["series A", "series B"],
    },
    "color-contradicts-summary": {
        "series_label": "series A", "summary_color": "blue",
        "injected_color": "red",
    },
    "wrong-chart-type": {
        "series_labels": ["series A"], "truth_kind": "bar", "injected_kind": "line",
    },
    "dropped-series": {"series_label": "series A"},
    "axis-truncation": {"axis_index": 0, "axis": "y"},
    "value-perturbation": {
        "series_label": "series A", "point_index": 2,
        "truth_value": 10, "injected_value": 20,
    },
    "swapped-series": {"labels": ["series A", "series B"]},
}

EVIDENCE_EXAMPLES = {
    "wrong-title": {"summary": ["The title is Original chart title."]},
    "wrong-axis-label": {"either": ["The y-axis label is Original axis label."]},
    "missing-legend": {
        "summary": ["series A", "series B"], "min_summary": 2,
    },
    "swapped-legend-labels": {"either": ["series A", "series B"]},
    "color-contradicts-summary": {
        "summary_color_mapping": {"series A": "blue"},
        "summary_quote": "Series A is shown in blue.",
    },
    "wrong-chart-type": {"summary": ["Series A is shown as a bar chart."]},
    "dropped-series": {"either": ["series A"]},
    "axis-truncation": {"table_nums": [10]},
    "value-perturbation": {"table_nums": [10]},
    "swapped-series": {"table_nums": [10, 20]},
}


@dataclass(frozen=True)
class DecodingConfig:
    temperature: float = 0.0
    max_tokens: int = 16_000
    seed: int | None = None


@dataclass(frozen=True)
class GenerationRequest:
    source_id: str
    category: str
    library: str
    chart_type: str
    code: str
    summary: str
    csv_text: str
    previous_flaws: tuple[str, ...] = ()
    # Canonical, capture-qualified target seeds for this category.  Empty for
    # historical (production/1-/7) requests, which keeps their prompt hashes
    # byte-identical; production/8 steers the generator onto exactly one seed.
    qualified_candidates: tuple[Mapping[str, Any], ...] = ()


@dataclass
class BackendReply:
    provider: str
    requested_model: str
    served_model: str
    raw_text: str
    usage: dict[str, Any] = field(default_factory=dict)
    request_id: str | None = None
    latency_seconds: float | None = None
    retries: int = 0


class ChatBackend(Protocol):
    def complete(self, system: str, user: str, decoding: DecodingConfig) -> BackendReply:
        ...


def endpoint_error_class(exc: Exception) -> str:
    """Classify provider failures without retaining messages, headers, or keys."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        if status == 429:
            return "http_429"
        if 500 <= status < 600:
            return "http_5xx"
        return f"http_{status}"
    name = type(exc).__name__
    lowered = name.lower()
    if isinstance(exc, TimeoutError) or "timeout" in lowered:
        return "timeout"
    if name in {
        "ConnectError",
        "ConnectionError",
        "NetworkError",
        "RemoteProtocolError",
    }:
        return "transport"
    return name


def _endpoint_retryable(exc: Exception) -> bool:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        return status in {408, 409, 425, 429} or 500 <= status < 600
    name = type(exc).__name__
    return (
        isinstance(exc, (TimeoutError, ConnectionError, json.JSONDecodeError))
        or "timeout" in name.lower()
        or name in {
            "ConnectError",
            "NetworkError",
            "RemoteProtocolError",
        }
        or (isinstance(exc, RuntimeError) and str(exc) == "ENDPOINT returned no choices")
    )


def _endpoint_retry_delay(exc: Exception, attempt: int) -> float:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    retry_after = headers.get("retry-after") if headers is not None else None
    if retry_after is not None:
        try:
            parsed = float(retry_after)
            if math.isfinite(parsed):
                return min(max(parsed, 0.0), 30.0)
        except (TypeError, ValueError):
            pass
    base = min(2 ** attempt, 4)
    return base + random.uniform(0.0, base * 0.25)


class ENDPOINTCredentialValidationError(RuntimeError):
    """Secret-free aggregate failure from an endpoint credential health pass."""

    def __init__(self, key_count: int, errors_by_type: dict[str, int]):
        self.key_count = key_count
        self.errors_by_type = dict(sorted(errors_by_type.items()))
        failures = sum(self.errors_by_type.values())
        details = ", ".join(
            f"{name}={count}" for name, count in self.errors_by_type.items()
        )
        super().__init__(
            f"ENDPOINT credential validation failed for {failures}/{key_count} "
            f"slots ({details})"
        )


def endpoint_api_keys(raw: str | None = None) -> tuple[str, ...]:
    """Return unique ENDPOINT credentials without ever exposing them in metadata."""
    if raw is None:
        raw = os.environ.get("ENDPOINT_API_KEYS") or os.environ.get("ENDPOINT_API_KEY", "")
    keys = []
    for value in raw.strip().strip('"').strip("'").split(","):
        key = value.strip().strip('"').strip("'")
        if key and key not in keys:
            keys.append(key)
    if not keys:
        raise ValueError("ENDPOINT_API_KEY or ENDPOINT_API_KEYS is not set")
    return tuple(keys)


class ENDPOINTCredentialRouter:
    """Fairly lease bounded credential slots across every ENDPOINT endpoint."""

    def __init__(self, api_keys: tuple[str, ...], *, max_in_flight_per_key: int):
        keys = tuple(dict.fromkeys(key.strip() for key in api_keys if key.strip()))
        if not keys:
            raise ValueError("at least one ENDPOINT credential is required")
        if max_in_flight_per_key < 1:
            raise ValueError("ENDPOINT per-key concurrency must be positive")
        self._keys = keys
        self.max_in_flight_per_key = max_in_flight_per_key
        self._available: Queue[int] = Queue()
        for _ in range(max_in_flight_per_key):
            for slot in range(len(keys)):
                self._available.put(slot)

    @property
    def key_count(self) -> int:
        return len(self._keys)

    @property
    def capacity(self) -> int:
        return self.key_count * self.max_in_flight_per_key

    @contextmanager
    def lease(self):
        slot = self._available.get()
        try:
            yield slot, self._keys[slot]
        finally:
            self._available.put(slot)


@contextmanager
def _single_credential_lease(api_key: str | None):
    yield 0, api_key


def _json_schema_tail(category: str) -> str:
    target = json.dumps(TARGET_EXAMPLES[category], ensure_ascii=False)
    evidence = json.dumps(
        EVIDENCE_EXAMPLES[category],
        ensure_ascii=False,
    )
    return f"""

Return one JSON object and no Markdown. It must match this contract:
{{
  "applicable": true,
  "decline_reason": "",
  "perturbed_code": "the complete modified Python script",
  "edit_description": "one sentence",
  "evidence": {evidence},
  "target": {target}
}}

If the category does not apply, return applicable=false, a concise decline_reason,
empty perturbed_code, and empty evidence/target objects. Never guess target metadata:
it is consumed by a fail-closed semantic gate after the script renders.
The values above are examples of the exact keys, not values to copy. Replace every
example with the actual original/injected value and captured series or axis identity.
"""



def qualified_candidate_seeds(
    candidates: Iterable[Mapping[str, Any]] | None,
) -> tuple[dict[str, Any], ...]:
    """Reduce qualification candidates to the JSON-safe seeds the prompt shows."""
    seeds = []
    for candidate in candidates or ():
        if not isinstance(candidate, Mapping):
            continue
        target = candidate.get("target")
        if not isinstance(target, Mapping) or not target:
            continue
        seed = {"target": json.loads(json.dumps(target, sort_keys=True))}
        required = candidate.get("required_generation_fields")
        if isinstance(required, (list, tuple)) and required:
            seed["required_generation_fields"] = [str(item) for item in required]
        seeds.append(seed)
    return tuple(seeds)


def _qualified_candidate_block(req: GenerationRequest) -> str:
    seeds = qualified_candidate_seeds(req.qualified_candidates)
    if not seeds:
        return ""
    rendered = json.dumps(list(seeds), sort_keys=True, ensure_ascii=False, indent=1)
    return (
        "\n\nQUALIFIED TARGETS. A capture-based qualification pass has already "
        "proven that each target below is recoverable by a repair agent from "
        "summary.txt and table.csv alone. You MUST inject the flaw on exactly one "
        "of these targets and report that target's identity verbatim in the "
        "`target` object (add only the fields the contract requires, such as the "
        "injected value). A flaw on any other element is rejected by the gate. "
        "Whatever you inject (a text, a label, a color, a number) must be plausible "
        "but must not occur anywhere in summary.txt or table.csv, because the repair "
        "agent must be able to identify the truth from those files alone.\n"
        f"```json\n{rendered}\n```"
    )


def build_prompt(req: GenerationRequest) -> tuple[str, str]:
    if req.category not in perturb_prompts.CATEGORY_INSTRUCTIONS:
        raise ValueError(f"unknown category {req.category!r}")
    previous = ""
    if req.previous_flaws:
        previous = (
            "The input already contains these intentional flaws, which must remain "
            "unchanged: " + ", ".join(req.previous_flaws) + ".\n\n"
        )
    csv_head = "\n".join(req.csv_text.splitlines()[:12])
    user = (
        previous
        + perturb_prompts.CATEGORY_INSTRUCTIONS[req.category]
        + _qualified_candidate_block(req)
        + _json_schema_tail(req.category)
        + f"\n--- source_id: {req.source_id} | category: {req.category} | "
          f"library: {req.library} | chart_type: {req.chart_type} ---\n\n"
        + f"## code.py\n```python\n{req.code}\n```\n\n"
        + f"## summary.txt (truth)\n{req.summary}\n\n"
        + f"## table.csv, first rows (truth)\n```\n{csv_head}\n```"
    )
    return V2_SYSTEM_PROMPT, user


def prompt_hash(req: GenerationRequest) -> str:
    system, user = build_prompt(req)
    return hashlib.sha256((system + "\0" + user).encode("utf-8")).hexdigest()


def request_fingerprint(
    req: GenerationRequest,
    *,
    provider: str,
    served_model: str,
    decoding: DecodingConfig,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "provider": provider,
        "served_model": served_model,
        "prompt_hash": prompt_hash(req),
        "source_hash": hashlib.sha256(req.code.encode("utf-8")).hexdigest(),
        "source_id": req.source_id,
        "category": req.category,
        "decoding": asdict(decoding),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _json_candidates(text: str):
    stripped = text.strip()
    if stripped:
        yield "direct", stripped
    for match in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S | re.I):
        yield "markdown_fence", match.group(1)
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield "embedded_object", text[index:index + end]


def _valid_target_field(name: str, value: Any) -> bool:
    if name in {"axis_index", "point_index"}:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0
    if name in {"labels", "series_labels"}:
        return isinstance(value, list) and bool(value) and all(
            isinstance(item, str) and item.strip() for item in value
        )
    if name in {"truth_value", "injected_value"}:
        return isinstance(value, (int, float)) and not isinstance(value, bool) \
            and math.isfinite(float(value))
    return isinstance(value, str) and bool(value.strip())


def parse_response(text: str, category: str, *,
                   strict_target: bool = True) -> tuple[dict[str, Any], str]:
    """Parse the generator's JSON answer.

    ``strict_target`` is the V2 contract: the structured ``target`` must be
    complete and well-formed for the category, or the answer is unusable. The
    v1.3 recipe (01.09.) uses this backend only for the edit itself and binds
    no target for LLM categories (the deterministic axis path records its own),
    so it parses with ``strict_target=False``: a complete edit with a missing or
    malformed target is kept, and only an answer without code or description is
    rejected. Measured in wave 1: one in five wrong-chart-type and
    swapped-legend-labels answers carried a usable edit but an incomplete target.
    """
    if category not in TARGET_EXAMPLES:
        raise ValueError(f"unknown category {category!r}")
    last_error = "no JSON object found"
    for path, candidate in _json_candidates(text):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = str(exc)
            continue
        if not isinstance(value, dict) or not isinstance(value.get("applicable"), bool):
            last_error = "applicable must be a boolean"
            continue
        if value["applicable"] is False:
            value.setdefault("decline_reason", "")
            value.setdefault("perturbed_code", "")
            value.setdefault("evidence", {})
            value.setdefault("target", {})
            return value, path
        required = (("perturbed_code", "edit_description", "evidence", "target")
                    if strict_target else ("perturbed_code", "edit_description"))
        missing = [key for key in required if key not in value]
        if missing:
            # A top-level object with the discriminator is the model's answer,
            # not noise around another object.  Fail on its incomplete contract
            # instead of later treating a nested evidence object as a second
            # candidate and hiding the useful error.
            raise ValueError(
                "unparseable perturbation response: missing keys: " + ", ".join(missing)
            )
        if not isinstance(value["perturbed_code"], str) or not value["perturbed_code"].strip():
            last_error = "perturbed_code must be a non-empty string"
            continue
        if not strict_target:
            for key in ("evidence", "target"):
                if not isinstance(value.get(key), dict):
                    value[key] = {}
            return value, path
        if not isinstance(value["evidence"], dict) or not isinstance(value["target"], dict):
            last_error = "evidence and target must be objects"
            continue
        missing_target = [
            key for key in TARGET_EXAMPLES[category]
            if key not in value["target"]
        ]
        if missing_target:
            raise ValueError(
                "unparseable perturbation response: target misses required keys: "
                + ", ".join(missing_target)
            )
        invalid_target = [
            key for key in TARGET_EXAMPLES[category]
            if not _valid_target_field(key, value["target"].get(key))
        ]
        if invalid_target:
            raise ValueError(
                "unparseable perturbation response: target has invalid values: "
                + ", ".join(invalid_target)
            )
        target_labels = value["target"].get("labels")
        if category == "missing-legend" and len(target_labels) < 2:
            raise ValueError(
                "unparseable perturbation response: missing-legend needs "
                "at least two target labels"
            )
        if category in {"swapped-legend-labels", "swapped-series"} and \
                len(target_labels) != 2:
            raise ValueError(
                f"unparseable perturbation response: {category} needs exactly "
                "two target labels"
            )
        if category == "color-contradicts-summary" and not (
            isinstance(value["evidence"].get("summary_color_mapping"), dict)
            and value["evidence"]["summary_color_mapping"]
            and isinstance(value["evidence"].get("summary_quote"), str)
            and 0 < len(value["evidence"]["summary_quote"].strip()) <= 500
        ):
            raise ValueError(
                "unparseable perturbation response: color evidence needs "
                "summary_color_mapping and a verbatim summary_quote"
            )
        return value, path
    raise ValueError(f"unparseable perturbation response: {last_error}")


class PerturbationGenerator:
    def __init__(self, backend: ChatBackend, decoding: DecodingConfig | None = None):
        self.backend = backend
        self.decoding = decoding or DecodingConfig()

    def generate(self, req: GenerationRequest) -> dict[str, Any]:
        system, user = build_prompt(req)
        reply = self.backend.complete(system, user, self.decoding)
        parse_error = None
        try:
            parsed, parse_path = parse_response(reply.raw_text, req.category)
        except ValueError as exc:
            parse_error = str(exc)
            parse_path = "failed"
            parsed = {
                "applicable": False,
                "decline_reason": f"response_contract_error: {parse_error}",
                "perturbed_code": "",
                "evidence": {},
                "target": {},
            }
        record = {
            "schema_version": SCHEMA_VERSION,
            "request_fingerprint": request_fingerprint(
                req,
                provider=reply.provider,
                served_model=reply.served_model,
                decoding=self.decoding,
            ),
            "source_id": req.source_id,
            "source_hash": hashlib.sha256(req.code.encode("utf-8")).hexdigest(),
            "category": req.category,
            "prompt_hash": prompt_hash(req),
            "decoding": asdict(self.decoding),
            "provider": reply.provider,
            "requested_model": reply.requested_model,
            "served_model": reply.served_model,
            "request_id": reply.request_id,
            "usage": reply.usage,
            "latency_seconds": reply.latency_seconds,
            "retries": reply.retries,
            "raw_response": reply.raw_text,
            "parse_path": parse_path,
            "output": parsed,
        }
        if parse_error:
            record["parse_error"] = parse_error
        return record


class ENDPOINTBackend:
    """OpenAI-compatible ENDPOINT chat-completions backend with served-id capture."""

    DEFAULT_BASE = os.environ.get("ENDPOINT_BASE", "")

    def __init__(
        self,
        slug: str,
        *,
        api_key: str | None = None,
        credential_router: ENDPOINTCredentialRouter | None = None,
        base_url: str = DEFAULT_BASE,
        client=None,
        timeout_seconds: float = 300.0,
        retries: int = 2,
    ):
        if client is not None and credential_router is not None:
            raise ValueError("an injected ENDPOINT client cannot use a credential router")
        self.slug = slug
        self.credential_router = credential_router
        if client is None and credential_router is None:
            keys = endpoint_api_keys(api_key)
            if len(keys) > 1:
                self.credential_router = ENDPOINTCredentialRouter(
                    keys, max_in_flight_per_key=1
                )
                self.api_key = None
            else:
                self.api_key = keys[0]
        else:
            self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.client = client
        self._clients = {0: client} if client is not None else {}
        self._owns_client = client is None
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self._served_model: str | None = None
        self._served_lock = threading.Lock()
        self._state = threading.Condition()
        self._active_operations = 0
        self._completion_attempts = 0
        self._closed = False

    def _make_client(self, api_key: str | None = None):
        import httpx

        key = api_key or self.api_key
        if not key:
            raise RuntimeError("ENDPOINT credential routing produced an empty key")
        headers = {os.environ.get("ENDPOINT_KEY_HEADER", "X-API-Key"): key, "Content-Type": "application/json"}
        return httpx.Client(headers=headers, timeout=self.timeout_seconds)

    def _client_for(self, slot: int, api_key: str | None):
        client = self._clients.get(slot)
        if client is None:
            client = self._make_client(api_key)
            self._clients[slot] = client
            if slot == 0:
                self.client = client
        return client

    @contextmanager
    def _operation(self):
        with self._state:
            if self._closed:
                raise RuntimeError("ENDPOINTBackend is closed")
            self._active_operations += 1
        try:
            yield
        finally:
            with self._state:
                self._active_operations -= 1
                if self._active_operations == 0:
                    self._state.notify_all()

    @contextmanager
    def _borrow_client(self):
        lease = (
            self.credential_router.lease()
            if self.credential_router is not None
            else _single_credential_lease(self.api_key)
        )
        with lease as (slot, api_key):
            with self._state:
                client = self._client_for(slot, api_key)
            yield slot, client

    def _fetch_served_model(self, client) -> str:
        response = client.get(f"{self.base_url}/{self.slug}/v1/models")
        response.raise_for_status()
        data = response.json().get("data") or []
        if not data or not data[0].get("id"):
            raise RuntimeError("ENDPOINT /models returned no served model id")
        return str(data[0]["id"])

    def _resolve_served_model(self) -> str:
        if self._served_model:
            return self._served_model
        with self._served_lock:
            if self._served_model:
                return self._served_model
            last_error: Exception | None = None
            attempts = max(
                self.retries + 1,
                self.credential_router.key_count
                if self.credential_router is not None else 1,
            )
            for attempt in range(attempts):
                try:
                    with self._borrow_client() as (_, client):
                        self._served_model = self._fetch_served_model(client)
                    return self._served_model
                except Exception as exc:
                    last_error = exc
                    if attempt + 1 == attempts:
                        break
                    credential_failover = (
                        self.credential_router is not None
                        and attempt < self.credential_router.key_count - 1
                    )
                    if credential_failover:
                        continue
                    if not _endpoint_retryable(exc):
                        break
                    time.sleep(min(2 ** attempt, 4))
            assert last_error is not None
            raise last_error

    def resolve_served_model(self) -> str:
        """Resolve and cache the served ID before a resumable production run."""
        with self._operation():
            return self._resolve_served_model()

    def validate_credentials(self) -> str:
        """Verify every routed credential once against this endpoint."""
        if self.credential_router is None:
            return self.resolve_served_model()
        with self._operation(), self._served_lock:
            seen = set()
            served_models = set()
            errors: dict[str, int] = {}
            for _ in range(self.credential_router.capacity):
                with self._borrow_client() as (slot, client):
                    if slot in seen:
                        continue
                    seen.add(slot)
                    try:
                        served_models.add(self._fetch_served_model(client))
                    except Exception as exc:
                        error = endpoint_error_class(exc)
                        errors[error] = errors.get(error, 0) + 1
                if len(seen) == self.credential_router.key_count:
                    break
            if len(seen) != self.credential_router.key_count:
                errors["router_coverage"] = (
                    self.credential_router.key_count - len(seen)
                )
            if len(served_models) > 1:
                errors["served_model_mismatch"] = len(served_models)
            if errors:
                raise ENDPOINTCredentialValidationError(
                    self.credential_router.key_count, errors
                )
            if not served_models:
                raise ENDPOINTCredentialValidationError(
                    self.credential_router.key_count,
                    {"empty_served_model_set": self.credential_router.key_count},
                )
            self._served_model = next(iter(served_models))
            return self._served_model

    @property
    def completion_attempts(self) -> int:
        """Return the exact number of physical chat-completion HTTP attempts."""
        with self._state:
            return self._completion_attempts

    def close(self) -> None:
        """Close an internally-created pool after all in-flight calls finish."""
        with self._state:
            if self._closed:
                return
            self._closed = True
            while self._active_operations:
                self._state.wait()
            clients = list(
                {id(client): client for client in self._clients.values()}.values()
            )
            if self._owns_client:
                self.client = None
                self._clients.clear()
        if self._owns_client:
            for client in clients:
                client.close()

    def __enter__(self):
        with self._state:
            if self._closed:
                raise RuntimeError("ENDPOINTBackend is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False

    def complete(self, system: str, user: str, decoding: DecodingConfig) -> BackendReply:
        with self._operation():
            started = time.monotonic()
            last_error: Exception | None = None
            served = self._served_model or self._resolve_served_model()
            body: dict[str, Any] = {
                "model": served,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": decoding.temperature,
                "max_tokens": decoding.max_tokens,
            }
            if decoding.seed is not None:
                body["seed"] = decoding.seed
            for attempt in range(self.retries + 1):
                with self._borrow_client() as (_, client):
                    try:
                        with self._state:
                            self._completion_attempts += 1
                        response = client.post(
                            f"{self.base_url}/{self.slug}/v1/chat/completions",
                            json=body,
                        )
                        response.raise_for_status()
                        payload = response.json()
                        choices = payload.get("choices") or []
                        if not choices:
                            raise RuntimeError("ENDPOINT returned no choices")
                        message = choices[0].get("message") or {}
                        text = message.get("content") or message.get("reasoning") or ""
                        usage = payload.get("usage") or {}
                        return BackendReply(
                            provider="endpoint",
                            requested_model=self.slug,
                            served_model=served,
                            raw_text=str(text),
                            usage={
                                "input_tokens": usage.get("prompt_tokens", 0),
                                "output_tokens": usage.get("completion_tokens", 0),
                                "total_tokens": usage.get("total_tokens", 0),
                            },
                            request_id=response.headers.get("x-request-id"),
                            latency_seconds=round(time.monotonic() - started, 3),
                            retries=attempt,
                        )
                    except Exception as exc:
                        last_error = exc
                assert last_error is not None
                if attempt < self.retries and _endpoint_retryable(last_error):
                    time.sleep(_endpoint_retry_delay(last_error, attempt))
                    continue
                break
            assert last_error is not None
            raise last_error


class AnthropicBackend:
    """Anthropic backend using ordinary messages JSON for the same V2 contract."""

    def __init__(self, model: str, *, client=None, api_key: str | None = None):
        self.model = model
        if client is None:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
        self.client = client

    def complete(self, system: str, user: str, decoding: DecodingConfig) -> BackendReply:
        started = time.monotonic()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "max_tokens": decoding.max_tokens,
            "temperature": decoding.temperature,
        }
        response = self.client.messages.create(**kwargs)
        text = "".join(
            block.text for block in response.content
            if getattr(block, "type", None) == "text"
        )
        usage = getattr(response, "usage", None)
        return BackendReply(
            provider="anthropic",
            requested_model=self.model,
            served_model=getattr(response, "model", None) or self.model,
            raw_text=text,
            usage={
                "input_tokens": getattr(usage, "input_tokens", 0),
                "output_tokens": getattr(usage, "output_tokens", 0),
            },
            request_id=getattr(response, "id", None),
            latency_seconds=round(time.monotonic() - started, 3),
        )
