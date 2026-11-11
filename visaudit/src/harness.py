"""Isolated execution of ChartNet plotting scripts.

Every script runs in its own temp cwd (all ChartNet scripts write 'output.png'),
with the Agg backend and a shared pre-warmed matplotlib config dir so the font
cache is built exactly once.

The interpreter is also seeded before the script's first line runs, so a script
that draws unseeded random numbers renders the same chart on every run. Without
that, two runs of the same code disagree and Recovery / Preservation pick up
noise that has nothing to do with the agent's edit.
"""
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# The repository runs plot scripts in its own pinned virtualenv. The shipped
# copy of this file (release packages) has no such venv beside it, so fall back
# to the interpreter that is running, and let CRB_PYTHON override both.
_VENV = ROOT / ".venv/bin/python"
VENV_PY = os.environ.get("CRB_PYTHON") or (str(_VENV) if _VENV.exists() else sys.executable)
MPL_CACHE = ROOT / "data/.mplcache"

SANDBOX_SEED = 42
SEED_ENTRY = "_chartbench_seed.py"

# Runs as the real entry point inside the sandbox and hands control to the
# script. Seeding here (rather than by editing the script) keeps the script's
# source untouched, and it happens *before* its first line, so a script with its
# own `np.random.seed(...)` simply overwrites ours and still wins.
SEED_WRAPPER = '''"""Seeds this interpreter, then runs the real entry point as __main__."""
import runpy
import sys

_seed = int(sys.argv[1])
_target = sys.argv[2]

import random
random.seed(_seed)

try:
    import numpy as _np
except Exception:            # numpy is not guaranteed to be importable
    _np = None

if _np is not None:
    _np.random.seed(_seed % (2 ** 32))       # legacy global RNG: np.random.*
    try:
        # ... and the modern factory, because seaborn's bootstrap CIs call
        # default_rng(None), which the legacy seed does not reach. Each call
        # still gets its own stream, just a reproducible one.
        _root = _np.random.SeedSequence(_seed)
        _default_rng = _np.random.default_rng

        def default_rng(seed=None):
            return _default_rng(_root.spawn(1)[0] if seed is None else seed)

        _np.random.default_rng = default_rng
    except Exception:
        pass

sys.argv = [_target]         # the script sees the argv it saw before
runpy.run_path(_target, run_name="__main__")
'''


def warm_cache():
    MPL_CACHE.mkdir(parents=True, exist_ok=True)
    code = "import matplotlib.pyplot as plt\nplt.plot([1, 2], [1, 2])\nplt.savefig('output.png')\n"
    run_script(code, timeout=120)


def run_script(code: str, timeout: int = 90, seed: int | None = SANDBOX_SEED) -> dict:
    return run_in_sandbox({"script.py": code}, "script.py", timeout=timeout, seed=seed)


def read_workspace_file(workspace, name):
    """Read a regular top-level artifact without following model-created links."""
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("artifact must be a workspace filename")
    path = Path(workspace) / name
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ValueError(f"refusing unsafe artifact {name}") from exc
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError(f"refusing non-regular artifact {name}")
        return stream.read()


def run_in_sandbox(files: dict, entry: str, timeout: int = 90, collect=(),
                   seed: int | None = SANDBOX_SEED, *, python_launcher=None) -> dict:
    """Run `entry` in an isolated temp cwd holding `files` (name -> text).

    Returns the same contract as run_script plus `files`: the text of every
    name in `collect` that the run produced (missing ones are simply absent).

    `seed` fixes the interpreter's random state (random, numpy, hash order);
    seed=None runs unseeded, which is only useful for measuring the wobble.

    `python_launcher(command, workspace, environment=env)` may return an OS-
    confined command and environment. If it rejects the platform/configuration,
    that error propagates; execution is never retried without the launcher.
    """
    if python_launcher is None:
        MPL_CACHE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        for name, text in files.items():
            if Path(name).name != name or name in (".", ".."):
                raise ValueError("input must be a workspace filename")
            (Path(td) / name).write_text(text, encoding="utf-8")
        if Path(entry).name != entry or entry not in files:
            raise ValueError("entry must name a workspace input file")
        script = Path(td) / entry
        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": td,
            "TMPDIR": td,
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(MPL_CACHE),
            "LANG": "en_US.UTF-8",
            "LC_ALL": "en_US.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        # -s -P rather than -I: same isolation (no user site, script dir off
        # sys.path, and the env below is the whole environment), but -I implies
        # -E, which would throw away PYTHONHASHSEED.
        cmd = [VENV_PY, "-s", "-P", str(script)]
        if seed is not None:
            # str/frozenset iteration order decides subplot order in some
            # scripts, and that order is a function of the hash seed.
            env["PYTHONHASHSEED"] = str(int(seed) % (2 ** 32))
            wrapper = Path(td) / SEED_ENTRY
            wrapper.write_text(SEED_WRAPPER, encoding="utf-8")
            cmd = [VENV_PY, "-s", "-P", str(wrapper), str(int(seed)), str(script)]
        if python_launcher is not None:
            cmd, env = python_launcher(cmd, Path(td), environment=env)
        t0 = time.time()
        stderr = ""
        try:
            if python_launcher is None:
                proc = subprocess.run(
                    cmd, cwd=td, capture_output=True, text=True, timeout=timeout, env=env,
                )
                stderr = (proc.stderr or "")[-2000:]
            else:
                proc = subprocess.Popen(
                    cmd, cwd=td, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, close_fds=True, start_new_session=True,
                )
                try:
                    _, error_output = proc.communicate(timeout=timeout)
                    stderr = (error_output or "")[-2000:]
                except subprocess.TimeoutExpired:
                    # This session belongs to this invocation, never a name-based
                    # process search. The leader has not yet been reaped.
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait(timeout=5)
                    raise
                finally:
                    proc.stdout.close()
                    proc.stderr.close()
            status = "ok" if proc.returncode == 0 else "error"
        except subprocess.TimeoutExpired:
            status = "timeout"
        dur = round(time.time() - t0, 2)

        png = None
        collected = {}
        try:
            png = read_workspace_file(td, "output.png")
            if png is None:
                others = sorted(Path(td).glob("*.png"))
                if others:
                    png = read_workspace_file(td, others[0].name)
            for name in collect:
                value = read_workspace_file(td, name)
                if value is not None:
                    collected[name] = value.decode("utf-8", errors="replace")
        except ValueError as exc:
            status = "error"
            stderr = (stderr + "\n" + str(exc))[-2000:]
        if status == "ok" and png is None:
            status = "no_output"
        return {"status": status, "duration": dur, "stderr": stderr, "png": png,
                "files": collected}
