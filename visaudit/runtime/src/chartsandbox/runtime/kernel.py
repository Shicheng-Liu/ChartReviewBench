"""Parent-side handle to a persistent execution worker.

Sends code, reads one JSON reply, and enforces a hard wall-clock timeout on top
of the worker's own SIGALRM: if the worker goes unresponsive (e.g. a C-level hang
that SIGALRM can't interrupt) it is killed and restarted — state is lost and the
caller is told so.
"""
from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
from pathlib import Path

_WORKER = Path(__file__).parent / "worker_main.py"


class PersistentKernel:
    def __init__(self, workspace: str | Path, default_timeout: int = 30):
        self.workspace = str(Path(workspace).resolve())
        self.default_timeout = default_timeout
        self.proc: subprocess.Popen | None = None
        self._start()

    def _start(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, str(_WORKER), self.workspace],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def _restart(self) -> None:
        try:
            if self.proc:
                self.proc.kill()
        except Exception:
            pass
        self._start()

    def _readline(self, hard_timeout: float) -> str | None:
        assert self.proc and self.proc.stdout
        q: queue.Queue = queue.Queue(maxsize=1)

        def _read():
            try:
                q.put(self.proc.stdout.readline())  # type: ignore[union-attr]
            except Exception:
                q.put(None)

        threading.Thread(target=_read, daemon=True).start()
        try:
            return q.get(timeout=hard_timeout)
        except queue.Empty:
            return None

    def execute(self, code: str, timeout: int | None = None) -> dict:
        timeout = timeout or self.default_timeout
        empty = {"stdout": "", "stderr": "", "files_changed": [], "images_created": []}
        assert self.proc and self.proc.stdin
        try:
            self.proc.stdin.write(json.dumps({"code": code, "timeout": timeout}) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            self._restart()
            return {**empty, "error": "KernelError: worker unavailable (restarted)"}

        line = self._readline(timeout + 10)
        if line is None or line == "":
            self._restart()
            return {**empty, "error": f"TimeoutError: no response in {timeout}s (kernel restarted, state lost)"}
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return {**empty, "error": "KernelError: malformed reply"}

    def shutdown(self) -> None:
        try:
            if self.proc and self.proc.stdin:
                self.proc.stdin.close()
            if self.proc:
                self.proc.kill()
        except Exception:
            pass
