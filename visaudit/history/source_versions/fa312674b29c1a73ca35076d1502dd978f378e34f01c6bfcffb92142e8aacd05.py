"""OS confinement for agent code; providers and scoring stay in the parent.

The supported backend is macOS Seatbelt. Other platforms fail closed. This is
filesystem/network/process-access confinement, not a resource-limit container.
Only the worker's workspace, Python installation and explicit runtime resources
are readable. Child processes inherit the policy. API credentials are omitted.
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
from pathlib import Path


class IsolationError(RuntimeError):
    pass


BACKEND = "macos-seatbelt/1"
_SEATBELT = Path("/usr/bin/sandbox-exec")
_PREFLIGHT_LOCK = threading.Lock()
_PREFLIGHT = None

# Old Kaleido's Chromium otherwise requires a cross-process Mach rendezvous.
# Single-process mode preserves the renderer while keeping general Mach lookup
# and service registration denied. The subprocess still inherits Seatbelt.
_BOOTSTRAP = """
import runpy, sys
import plotly.io as _pio
_pio.kaleido.scope.chromium_args += ('--single-process',)
sys.argv = sys.argv[1:]
sys.path.insert(0, str(__import__('pathlib').Path(sys.argv[0]).resolve().parent))
runpy.run_path(sys.argv[0], run_name='__main__')
"""


def _quoted(path):
    value = str(path)
    if any(c in value for c in ("\x00", "\n", "\r")):
        raise IsolationError("unsupported control character in isolation path")
    return json.dumps(value)


def _supported():
    if sys.platform != "darwin" or not _SEATBELT.is_file():
        raise IsolationError(
            "Agent isolation requires macOS /usr/bin/sandbox-exec; "
            "no supported backend is configured on this platform. Refusing an unconfined run.")


def seatbelt_profile(workspace, *, readable_files=()):
    _supported()
    workspace = Path(workspace).resolve(strict=True)
    if not workspace.is_dir() or workspace == Path("/"):
        raise IsolationError("isolation requires a dedicated workspace directory")
    prefix = Path(sys.prefix).resolve()
    base = Path(sys.base_prefix).resolve()
    read_dirs = [
        Path("/System/Library"), Path("/System/Cryptexes"),
        Path("/usr/lib"), Path("/usr/bin"), Path("/bin"),
        Path("/Library/Fonts"), base, prefix / "lib",
    ]
    if any(path == workspace or workspace in path.parents for path in read_dirs):
        raise IsolationError("workspace must not contain the Python installation or system libraries")
    literals = [Path("/"), prefix / "pyvenv.cfg", Path(sys.executable)]
    for path in readable_files:
        path = Path(path).resolve(strict=True)
        if not path.is_file():
            raise IsolationError("additional runtime grants must name individual files")
        literals.append(path)
    reads = "\n".join(f"    (subpath {_quoted(p)})" for p in read_dirs)
    reads += "\n" + "\n".join(f"    (literal {_quoted(p)})" for p in literals)
    return f'''(version 1)
(deny default)
(allow process-exec process-fork)
(allow signal (target self) (target children))
(allow process-info-pidinfo (target self))
(allow sysctl-read
    (sysctl-name-prefix "hw.")
    (sysctl-name "kern.osrelease") (sysctl-name "kern.ostype")
    (sysctl-name "kern.osversion") (sysctl-name "kern.osproductversion")
    (sysctl-name "kern.version") (sysctl-name "kern.hostname")
    (sysctl-name "kern.maxfilesperproc") (sysctl-name "kern.usrstack64")
    (sysctl-name "kern.hv_vmm_present") (sysctl-name "kern.secure_kernel")
    (sysctl-name "sysctl.proc_cputype"))
; Required by dyld/Chromium realpath; grants no contents or directory listings.
(allow file-read-metadata)
(allow file-read*
{reads})
(allow file-read* file-write* (subpath {_quoted(workspace)}))
(allow file-read* file-write-data
    (literal "/dev/null") (literal "/dev/random") (literal "/dev/urandom"))
(allow iokit-open (iokit-user-client-class "RootDomainUserClient"))
(allow mach-lookup
    (global-name "com.apple.system.notification_center")
    (global-name "com.apple.PowerManagement.control")
    (global-name "com.apple.fonts")
    (global-name "com.apple.FontObjectsServer"))
'''


def isolated_python(command, workspace, *, readable_files=(), environment=None):
    """Wrap a Python script command and return its confined command and clean env.

    The caller launches with cwd=workspace and start_new_session=True, and kills
    the process group on timeout/shutdown. Additional grants are individual
    trusted runtime files; dataset/solution files must never be granted.
    """
    _supported()
    command = list(command)
    if not command or Path(command[0]).resolve() != Path(sys.executable).resolve():
        raise IsolationError("isolated Python must use the current pinned interpreter")
    args = command[1:]
    while args and args[0].startswith("-"):
        if args.pop(0) not in ("-s", "-P", "-B", "-u"):
            raise IsolationError("isolation wrapper accepts Python script calls only")
    if not args:
        raise IsolationError("isolated Python needs a script entry point")
    workspace = Path(workspace).resolve(strict=True)
    cache = workspace / ".chartrepair-cache"
    if cache.is_symlink():
        raise IsolationError("runtime cache must not be a symlink")
    cache.mkdir(exist_ok=True)
    temp = cache / "tmp"
    if temp.is_symlink():
        raise IsolationError("runtime temp directory must not be a symlink")
    temp.mkdir(exist_ok=True)
    env = {
        "PATH": "/usr/bin:/bin", "TMPDIR": str(temp),
        "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(cache / "matplotlib"),
        "LANG": "en_US.UTF-8", "LC_ALL": "en_US.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "42",
        "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
    }
    if environment and "PYTHONHASHSEED" in environment:
        env["PYTHONHASHSEED"] = str(environment["PYTHONHASHSEED"])
    profile = seatbelt_profile(workspace, readable_files=readable_files)
    return [str(_SEATBELT), "-p", profile, command[0], "-s", "-P", "-B", "-u",
            "-c", _BOOTSTRAP, *args], env


def stop_process_group(proc):
    """Stop only the worker session created by this adapter, including Kaleido."""
    # communicate()/wait() may already have reaped the leader. Its PID can then
    # be reused; never signal that numeric group after reaping the process.
    if proc is None or proc.returncode is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def isolated_kernel_class(base_cls, worker_path):
    """Adapt the upstream protocol without editing the sibling repository."""
    worker_path = Path(worker_path).resolve(strict=True)

    class IsolatedKernel(base_cls):
        _chartrepair_isolated = True

        def _start(self):
            cmd, env = isolated_python(
                [sys.executable, str(worker_path), self.workspace], self.workspace,
                readable_files=(worker_path,))
            self.proc = subprocess.Popen(
                cmd, cwd=self.workspace, env=env, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1, close_fds=True, start_new_session=True)

        def _restart(self):
            self.shutdown()
            self._start()

        def shutdown(self):
            stop_process_group(self.proc)
            for stream in (self.proc.stdin, self.proc.stdout) if self.proc else ():
                if stream:
                    stream.close()
            self.proc = None

    return IsolatedKernel


def preflight_isolation():
    """Verify real OS denials and both renderers before any provider is called."""
    global _PREFLIGHT
    with _PREFLIGHT_LOCK:
        if _PREFLIGHT is not None:
            return dict(_PREFLIGHT)
        _supported()
        with tempfile.TemporaryDirectory(prefix="chartrepair-isolation-") as td:
            root = Path(td).resolve()
            workspace = root / "workspace"
            workspace.mkdir()
            forbidden = root / "solution.txt"
            forbidden.write_text("ISOLATION_CANARY", encoding="utf-8")
            (workspace / "outside-link").symlink_to(forbidden)
            (workspace / "table.csv").write_text("x,y\n1,3\n2,4\n", encoding="utf-8")
            probe = workspace / "probe.py"
            probe.write_text(f'''
import csv, os, pathlib, socket, subprocess, sys
for path in ({str(forbidden)!r}, '../solution.txt', 'outside-link'):
    try:
        pathlib.Path(path).read_text()
    except PermissionError:
        pass
    else:
        raise AssertionError('read escaped isolation: ' + path)
try:
    os.listdir('..')
except PermissionError:
    pass
else:
    raise AssertionError('parent directory listing escaped isolation')
result = subprocess.run(['/bin/cat', {str(forbidden)!r}], capture_output=True)
assert result.returncode != 0 and b'ISOLATION_CANARY' not in result.stdout
try:
    socket.socket().connect(('127.0.0.1', 9))
except PermissionError:
    pass
else:
    raise AssertionError('network was not denied by the OS')
assert not any('KEY' in key or 'TOKEN' in key for key in os.environ)
with open('table.csv') as f:
    rows = list(csv.DictReader(f))
import matplotlib.pyplot as plt
plt.plot([float(r['x']) for r in rows], [float(r['y']) for r in rows])
plt.savefig('matplotlib.png')
import plotly.graph_objects as go
go.Figure(go.Bar(x=[1, 2], y=[3, 4])).write_image('plotly.png')
assert pathlib.Path('matplotlib.png').stat().st_size > 100
assert pathlib.Path('plotly.png').stat().st_size > 100
print('ISOLATION_OK')
''', encoding="utf-8")
            command, env = isolated_python([sys.executable, str(probe)], workspace)
            proc = subprocess.Popen(command, cwd=workspace, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, close_fds=True, start_new_session=True)
            try:
                out, err = proc.communicate(timeout=45)
                if proc.returncode or out.strip() != "ISOLATION_OK":
                    raise IsolationError(f"OS isolation preflight failed: {err[-3000:]} {out[-500:]}")
            except subprocess.TimeoutExpired as exc:
                raise IsolationError("OS isolation preflight timed out") from exc
            finally:
                stop_process_group(proc)
        _PREFLIGHT = {
            "backend": BACKEND, "verified": True,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "checks": ["absolute_read", "parent_traversal", "symlink", "parent_listing",
                       "subprocess_read", "network", "environment", "csv",
                       "matplotlib_render", "plotly_render"],
            "kaleido_chromium_args_added": ["--single-process"],
        }
        return dict(_PREFLIGHT)


def install_kernel_isolation():
    """Patch the local import binding after a successful fail-closed preflight."""
    result = preflight_isolation()
    from chartsandbox import tools
    from chartsandbox.runtime import kernel
    if not getattr(tools.PersistentKernel, "_chartrepair_isolated", False):
        tools.PersistentKernel = isolated_kernel_class(kernel.PersistentKernel, kernel._WORKER)
    return result
