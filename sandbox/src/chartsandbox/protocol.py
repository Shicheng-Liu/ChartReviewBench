"""The tagged-text response protocol: `<reasoning>` / `<code>` / `<decision>`.

An alternative to tool calling. One model turn is one *iteration* of the
edit-execute-inspect-revise loop, and it looks like this::

    <reasoning>
    analysis of the current chart, the issues identified, and the planned fixes
    </reasoning>
    <code>
    # Python / Matplotlib code
    </code>
    <decision>continue</decision>

Why this exists alongside tool calling: a self-hosted open-weight model needs a
working tool-call parser on the serving side to be evaluable at all, and when that
parser is weak the episode fails for reasons that have nothing to do with charts.
Plain tags need nothing but text generation, so the same task can be put to models
that tool calling would exclude — and running both protocols on one model turns
"how much does the response format cost you" into a measurable ablation.

`<decision>` is also where the agent's *stopping* judgement becomes observable: the
environment never decides an episode is done. It executes what it is given, hands
back stdout/stderr and the rendered image, and stops when the agent says stop or
the turn budget runs out — never because a verifier passed. Whether stopping was
the right call is then a property of the trajectory we can score, not something the
harness quietly fixed.

Parsing is deliberately forgiving. The measurement is chart repair, not tag
discipline, so recoverable deviations (a fenced code block instead of `<code>`, a
missing `<decision>`, a truncated closing tag) are accepted and recorded in
`warnings`; a turn is only rejected when it carries no runnable action at all.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

CONTINUE = "continue"
STOP = "stop"

#: Words a model reaches for when it means "I am done" / "keep going".
_STOP_WORDS = {"stop", "stopped", "done", "finish", "finished", "complete",
               "completed", "end", "terminate", "halt"}
_CONTINUE_WORDS = {"continue", "continued", "continuing", "iterate", "iterating",
                   "revise", "revising", "keep going", "again", "next", "retry"}

_FENCE = re.compile(r"\A\s*```[a-zA-Z0-9_+#-]*[ \t]*\r?\n(.*?)\r?\n?[ \t]*```\s*\Z", re.S)
#: A bare fenced block, used only to recover code when `<code>` is missing.
_LOOSE_FENCE = re.compile(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", re.S | re.I)


def _closed(name: str) -> re.Pattern[str]:
    return re.compile(rf"<{name}\s*>(.*?)</\s*{name}\s*>", re.I | re.S)


def _unclosed(name: str) -> re.Pattern[str]:
    """Opening tag with no closing one — what a max_tokens cutoff leaves behind."""
    return re.compile(rf"<{name}\s*>(.*)\Z", re.I | re.S)


_RE = {n: (_closed(n), _unclosed(n)) for n in ("reasoning", "code", "decision")}


@dataclass
class TaggedTurn:
    """One parsed model turn. Never raised from — failure shows up in `warnings`."""

    reasoning: str = ""
    code: str | None = None
    decision: str = CONTINUE
    warnings: list[str] = field(default_factory=list)

    @property
    def actionable(self) -> bool:
        """True when the turn carries something the environment can act on.

        Code to run, or a decision to stop. A turn with neither is a no-op that
        would burn an iteration on nothing, so the caller re-prompts instead.
        """
        return bool(self.code and self.code.strip()) or self.decision == STOP

    @property
    def stopping(self) -> bool:
        return self.decision == STOP


def strip_fence(code: str) -> str:
    """Drop a ```-fence wrapping the whole block; leave anything else alone."""
    m = _FENCE.match(code)
    return m.group(1) if m else code.strip("\r\n")


def _extract(name: str, text: str, warnings: list[str]) -> str | None:
    closed, unclosed = _RE[name]
    found = closed.findall(text)
    if found:
        if len(found) > 1:
            # Keep the first: a model that emits two code blocks usually means the
            # second as commentary, and running both halves would double-execute.
            warnings.append(f"multiple <{name}> blocks; used the first")
        return found[0]
    m = unclosed.search(text)
    if m:
        warnings.append(f"unclosed <{name}> tag; read to end of response")
        return m.group(1)
    return None


def _normalize_decision(raw: str | None, warnings: list[str]) -> str:
    if raw is None:
        # Defaulting to stop would end episodes on a formatting slip and score it
        # as a stopping decision the model never made. Continue is the safe default.
        warnings.append("no <decision> tag; assumed continue")
        return CONTINUE
    word = re.sub(r"[^a-z ]+", "", raw.strip().lower()).strip()
    if word in _STOP_WORDS:
        return STOP
    if word in _CONTINUE_WORDS:
        return CONTINUE
    # Not one of the accepted words: fall back to a substring read before giving up,
    # which catches "decision: continue" and similar.
    for w in _STOP_WORDS:
        if re.search(rf"\b{w}\b", word):
            warnings.append(f"unrecognised <decision> {raw.strip()[:40]!r}; read as stop")
            return STOP
    for w in _CONTINUE_WORDS:
        if re.search(rf"\b{w}\b", word):
            warnings.append(f"unrecognised <decision> {raw.strip()[:40]!r}; read as continue")
            return CONTINUE
    warnings.append(f"unrecognised <decision> {raw.strip()[:40]!r}; assumed continue")
    return CONTINUE


def parse_tagged(text: str) -> TaggedTurn:
    """Parse one model response. Always returns a turn; check `.actionable`."""
    text = text or ""
    warnings: list[str] = []

    reasoning = _extract("reasoning", text, warnings) or ""
    code = _extract("code", text, warnings)
    decision_raw = _extract("decision", text, warnings)

    if code is None:
        # No <code>, but a fenced block: the model had the right idea in the wrong
        # wrapper. Recover it rather than burn an iteration on the tag.
        loose = _LOOSE_FENCE.findall(text)
        if loose:
            warnings.append("code was fenced, not in <code>; recovered")
            code = loose[0]

    return TaggedTurn(
        reasoning=reasoning.strip(),
        code=strip_fence(code) if code is not None else None,
        decision=_normalize_decision(decision_raw, warnings),
        warnings=warnings,
    )
