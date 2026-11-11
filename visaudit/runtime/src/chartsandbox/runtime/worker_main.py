"""Persistent execution worker (the "kernel").

Runs as a child process. Reads one JSON message per line from stdin:
    {"code": "<python>", "timeout": <seconds>}
and replies with exactly one JSON line on the *real* fd 1 (sys.__stdout__):
    {"stdout", "stderr", "error", "files_changed", "images_created"}

State (the exec namespace) persists across messages — this is what makes
long-horizon, stateful trajectories possible. During exec, the user's stdout/
stderr are redirected to buffers so fd 1 stays clean for the protocol.

Determinism: forces the Agg backend and a fixed hash/PYTHONHASHSEED so renders
are reproducible. Hard network isolation is a Docker-mode concern; here we install
a best-effort socket guard.
"""
import contextlib
import io
import json
import os
import signal
import sys
import traceback

WORKDIR = sys.argv[1]
os.chdir(WORKDIR)

# --- deterministic, headless rendering -------------------------------------
os.environ.setdefault("MPLBACKEND", "Agg")
try:
    import matplotlib

    matplotlib.use("Agg")
except Exception:  # pragma: no cover - matplotlib always present in our env
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
                    "files_changed": [], "images_created": []})
            continue

        code = msg.get("code", "")
        timeout = int(msg.get("timeout", 30))
        before = _snapshot()
        out, err = io.StringIO(), io.StringIO()
        error = None

        signal.alarm(timeout)
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                exec(compile(code, "<agent_code>", "exec"), NS)
        except _Timeout:
            error = f"TimeoutError: execution exceeded {timeout}s"
        except SystemExit:
            error = "SystemExit was called (ignored in sandbox)"
        except BaseException:
            error = traceback.format_exc()
        finally:
            signal.alarm(0)

        after = _snapshot()
        changed = sorted(p for p, m in after.items() if before.get(p) != m)
        images = sorted(p for p in changed if p.rsplit(".", 1)[-1].lower() in IMG_EXT)
        _reply({
            "stdout": out.getvalue()[-20000:],
            "stderr": err.getvalue()[-20000:],
            "error": error,
            "files_changed": changed,
            "images_created": images,
        })


if __name__ == "__main__":
    main()
