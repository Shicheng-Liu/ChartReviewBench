"""Persistent execution worker (the "kernel").

Runs as a child process. Reads one JSON message per line from stdin:
    {"code": "<python>", "timeout": <seconds>}
and replies with exactly one JSON line on the *real* fd 1 (sys.__stdout__):
    {"stdout", "stderr", "error", "warnings", "files_changed", "images_created",
     "figures_captured"}

State (the exec namespace) persists across messages — this is what makes
long-horizon, stateful trajectories possible. During exec, the user's stdout/
stderr are redirected to buffers so fd 1 stays clean for the protocol.

Determinism: forces the Agg backend and a fixed hash/PYTHONHASHSEED so renders
are reproducible. Hard network isolation is a Docker-mode concern; here we install
a best-effort socket guard.

Bounded three ways, because one bound does not cover the others. SIGALRM caps
wall-clock per execution; RLIMIT_CPU caps CPU per execution, which is what catches
a C-level loop that never reaches a bytecode boundary where Python could deliver
SIGALRM; RLIMIT_AS caps address space for the life of the worker, turning a runaway
allocation into a MemoryError the agent can read instead of an OOM kill it cannot.
The parent's hard timeout remains the last resort.

Warnings are captured, not just leaked to stderr: "runs without errors or critical
warnings" is a scored property, and a warning is often the only sign of a real
rendering defect (a missing glyph renders as a box, a tight-layout failure crops
labels) that no exception and no figure property would reveal.
"""
import contextlib
import io
import json
import os
import resource
import signal
import sys
import traceback
import warnings as _warnings

WORKDIR = sys.argv[1]
os.chdir(WORKDIR)

# --- deterministic, headless rendering -------------------------------------
os.environ.setdefault("MPLBACKEND", "Agg")
try:
    import matplotlib

    matplotlib.use("Agg")
except Exception:  # pragma: no cover - matplotlib always present in our env
    pass

# --- resource ceilings ------------------------------------------------------
# Address-space capping is OFF by default, and that is a deliberate concession to
# what this benchmark actually renders. RLIMIT_AS is inherited by children, and
# plotly's static export runs through kaleido, which ships a Chromium-derived
# binary that reserves tens of gigabytes of *virtual* address space up front —
# far more than it ever touches. Under any sane cap that reservation fails and the
# export hangs until the step timeout, silently costing every plotly task in the
# suite (a fifth of this dataset) with a timeout that looks like a slow model.
# Set SANDBOX_MEM_MB to opt in where no browser-based renderer is involved; CPU
# time, the per-step alarm and the parent's hard timeout bound a runaway either way.
_MEM_MB = int(os.environ.get("SANDBOX_MEM_MB", "0"))
if _MEM_MB > 0:
    try:
        _soft, _hard = resource.getrlimit(resource.RLIMIT_AS)
        _want = _MEM_MB * 1024 * 1024
        if _hard == resource.RLIM_INFINITY or _want < _hard:
            resource.setrlimit(resource.RLIMIT_AS, (_want, _hard))
    except (ValueError, OSError):
        pass  # a platform that will not take it is not a reason to refuse work


class _CpuExceeded(Exception):
    pass


def _on_xcpu(signum, frame):
    raise _CpuExceeded()


try:
    signal.signal(signal.SIGXCPU, _on_xcpu)
except (AttributeError, ValueError):
    pass


def _cpu_used() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def _cap_cpu(seconds: int) -> None:
    """Give this execution `seconds` of CPU on top of what the worker has used.

    The limit is cumulative per process, so it is re-armed each turn rather than
    set once; the hard limit is left alone so it can always be raised again.
    """
    try:
        _soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
        want = int(_cpu_used()) + max(1, seconds) + 1
        if hard != resource.RLIM_INFINITY:
            want = min(want, hard)
        resource.setrlimit(resource.RLIMIT_CPU, (want, hard))
    except (ValueError, OSError):
        pass


# --- best-effort network guard (real isolation = Docker mode, future) ------
# Block outbound *connections* rather than replacing socket.socket — the latter
# breaks stdlib class definitions like `class SSLSocket(socket)` that many libs
# (e.g. seaborn -> urllib -> ssl) trigger at import time.
if os.environ.get("SANDBOX_ALLOW_NET") != "1":
    for _m in ("ssl", "http.client", "urllib.request"):  # bind classes to the real socket first
        try:
            __import__(_m)
        except Exception:
            pass
    import socket as _socket_mod

    def _no_net(*_a, **_k):
        raise OSError("network access is disabled in the sandbox")

    _socket_mod.socket.connect = _no_net          # type: ignore[assignment]
    _socket_mod.socket.connect_ex = _no_net       # type: ignore[assignment]
    _socket_mod.create_connection = _no_net       # type: ignore[assignment]

# --- persistent namespace ---------------------------------------------------
NS: dict = {"__name__": "__sandbox__", "__builtins__": __builtins__}


class _Timeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _Timeout()


signal.signal(signal.SIGALRM, _on_alarm)

IMG_EXT = {"png", "jpg", "jpeg", "gif", "webp", "svg", "pdf"}

#: Where a figure the agent left open but never saved gets written, so it can still
#: be shown back. Deliberately a dot-directory and never `output.png`: the agent's
#: deliverable has to be something the agent actually produced, or a forgotten
#: savefig would score as a rendered chart.
FIG_DIR = ".sandbox_figures"


def _capture_open_figures(turn: int) -> list[str]:
    """Save any still-open matplotlib figure. Returns workspace-relative paths."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    saved = []
    try:
        nums = plt.get_fignums()
    except Exception:
        return []
    for i, num in enumerate(nums):
        path = os.path.join(FIG_DIR, f"turn{turn:02d}_fig{i}.png")
        try:
            os.makedirs(FIG_DIR, exist_ok=True)
            plt.figure(num).savefig(path, dpi=100, bbox_inches="tight")
            saved.append(path)
        except Exception:
            continue
    return saved


def _snapshot() -> dict[str, float]:
    files: dict[str, float] = {}
    for root, _dirs, names in os.walk("."):
        for n in names:
            p = os.path.normpath(os.path.join(root, n))
            try:
                files[p] = os.path.getmtime(p)
            except OSError:
                pass
    return files


def _reply(obj: dict) -> None:
    sys.__stdout__.write(json.dumps(obj) + "\n")
    sys.__stdout__.flush()


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            _reply({"error": "KernelError: malformed message", "stdout": "", "stderr": "",
                    "warnings": [], "files_changed": [], "images_created": [],
                    "figures_captured": []})
            continue

        code = msg.get("code", "")
        timeout = int(msg.get("timeout", 30))
        turn = int(msg.get("turn", 0))
        before = _snapshot()
        out, err = io.StringIO(), io.StringIO()
        error = None

        _cap_cpu(timeout)
        signal.alarm(timeout)
        caught: list = []
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), \
                    _warnings.catch_warnings(record=True) as caught:
                _warnings.simplefilter("always")
                exec(compile(code, "<agent_code>", "exec"), NS)
        except _Timeout:
            error = f"TimeoutError: execution exceeded {timeout}s of wall-clock"
        except _CpuExceeded:
            error = f"TimeoutError: execution exceeded {timeout}s of CPU time"
        except MemoryError:
            error = ("MemoryError: allocation failed"
                     + (f"; the sandbox caps address space at {_MEM_MB} MB "
                        f"(SANDBOX_MEM_MB)" if _MEM_MB > 0 else ""))
        except SystemExit:
            error = "SystemExit was called (ignored in sandbox)"
        except BaseException:
            error = traceback.format_exc()
        finally:
            signal.alarm(0)

        # Deduplicated: matplotlib repeats the same warning per artist, and fifty
        # copies of one message would crowd out the traceback in the observation.
        seen, warns = set(), []
        for w in caught or []:
            try:
                text = f"{w.category.__name__}: {w.message}"
            except Exception:
                continue
            if text not in seen:
                seen.add(text)
                warns.append(text)

        figures = _capture_open_figures(turn)
        after = _snapshot()
        changed = sorted(p for p, m in after.items()
                         if before.get(p) != m and not p.startswith(FIG_DIR))
        images = sorted(p for p in changed if p.rsplit(".", 1)[-1].lower() in IMG_EXT)
        _reply({
            "stdout": out.getvalue()[-20000:],
            "stderr": err.getvalue()[-20000:],
            "error": error,
            "warnings": warns[:40],
            "files_changed": changed,
            "images_created": images,
            "figures_captured": figures,
        })


if __name__ == "__main__":
    main()
