"""Conservative repeated-action evidence; this module never stops an episode.

Observe the full tool response after execution, before display/trajectory
truncation. Equal observations cannot prove that hidden process state is equal;
the policy therefore records evidence, not a claim that a repair is impossible.
"""
from collections import deque
import copy
import hashlib
import json
import os
from pathlib import Path
import stat


VERSION = "identical-tool-observation-workspace-v1"
PERIODS = (1, 2, 3)
MIN_CYCLES = 8
MIN_CALLS = 24
MAX_ENTRIES = 256
MAX_WORKSPACE_BYTES = 32 * 1024 * 1024
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_DEPTH = 32
MAX_OBSERVATION_BYTES = 16 * 1024 * 1024

POLICY = {
    "version": VERSION,
    "periods": list(PERIODS),
    "min_cycles": MIN_CYCLES,
    "min_calls": MIN_CALLS,
    "comparison": "canonical_full_tool_args_observation_and_workspace_content",
    "workspace_fingerprint": "relative_paths_types_permissions_and_file_bytes_sha256",
    "workspace_changes": "reset",
    "unverifiable_input": "reset_without_detection",
    "limits": {"workspace_entries": MAX_ENTRIES, "workspace_bytes": MAX_WORKSPACE_BYTES,
               "file_bytes": MAX_FILE_BYTES, "depth": MAX_DEPTH,
               "observation_bytes": MAX_OBSERVATION_BYTES},
}


def _signature(value, max_bytes):
    digest, size = hashlib.sha256(), 0
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False)
    try:
        for part in encoder.iterencode(value):
            data = part.encode("utf-8")
            size += len(data)
            if size > max_bytes:
                return None
            digest.update(data)
    except (TypeError, ValueError, UnicodeError, RecursionError):
        return None
    return digest.hexdigest()


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


class _Unverifiable(Exception):
    pass


def workspace_fingerprint(workspace, *, max_entries=MAX_ENTRIES,
                          max_bytes=MAX_WORKSPACE_BYTES, max_file_bytes=MAX_FILE_BYTES,
                          max_depth=MAX_DEPTH):
    """Hash a bounded regular directory tree, or return None without following links.

    Timestamps only detect concurrent changes during reading; they never enter
    the resulting content fingerprint. Every call reads the file bytes again.
    """
    manifest, counts = [], {"entries": 0, "files": 0, "bytes": 0}
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        return None

    def walk(fd, relative, depth):
        if depth > max_depth:
            raise _Unverifiable()
        before = os.fstat(fd)
        manifest.append([relative, "directory", stat.S_IMODE(before.st_mode)])
        entries = []
        with os.scandir(fd) as scan:
            for entry in scan:
                counts["entries"] += 1
                if counts["entries"] > max_entries:
                    raise _Unverifiable()
                entries.append((entry.name, entry.stat(follow_symlinks=False)))
        for name, expected in sorted(entries):
            child = str(Path(relative) / name)
            if stat.S_ISDIR(expected.st_mode):
                child_fd = os.open(name, os.O_RDONLY | directory | nofollow, dir_fd=fd)
                try:
                    if _identity(os.fstat(child_fd)) != _identity(expected):
                        raise _Unverifiable()
                    walk(child_fd, child, depth + 1)
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(expected.st_mode):
                if expected.st_size > max_file_bytes or counts["bytes"] + expected.st_size > max_bytes:
                    raise _Unverifiable()
                file_fd = os.open(name, os.O_RDONLY | nofollow | os.O_NONBLOCK, dir_fd=fd)
                try:
                    opened = os.fstat(file_fd)
                    if not stat.S_ISREG(opened.st_mode) or _identity(opened) != _identity(expected):
                        raise _Unverifiable()
                    digest, size = hashlib.sha256(), 0
                    while data := os.read(file_fd, min(1024 * 1024, max_file_bytes + 1 - size)):
                        size += len(data)
                        counts["bytes"] += len(data)
                        if size > max_file_bytes or counts["bytes"] > max_bytes:
                            raise _Unverifiable()
                        digest.update(data)
                    if size != opened.st_size or _identity(os.fstat(file_fd)) != _identity(opened):
                        raise _Unverifiable()
                    counts["files"] += 1
                    manifest.append([child, "file", stat.S_IMODE(opened.st_mode), digest.hexdigest()])
                finally:
                    os.close(file_fd)
            else:
                raise _Unverifiable()
        if _identity(os.fstat(fd)) != _identity(before):
            raise _Unverifiable()

    root_fd = None
    try:
        root_fd = os.open(workspace, os.O_RDONLY | directory | nofollow)
        walk(root_fd, ".", 0)
        digest = _signature(manifest, MAX_OBSERVATION_BYTES)
        return {"sha256": digest, **counts} if digest is not None else None
    except (OSError, _Unverifiable):
        return None
    finally:
        if root_fd is not None:
            os.close(root_fd)


class StagnationDetector:
    def __init__(self, workspace):
        self.workspace = Path(workspace)
        self._calls = 0
        self.reset()

    def reset(self):
        """Forget candidates after an external reset; retain the call counter."""
        self._workspace = None
        self._window = deque(maxlen=max(MIN_CALLS, max(PERIODS) * MIN_CYCLES))
        self._reported = False

    def observe(self, tool, args, observation):
        """Return hash-only evidence once per repeated sequence, otherwise None."""
        self._calls += 1
        signature = _signature([tool, args, observation], MAX_OBSERVATION_BYTES)
        workspace = workspace_fingerprint(self.workspace) if signature is not None else None
        if workspace is None:
            self.reset()
            return None
        if self._workspace != workspace["sha256"]:
            self.reset()
            self._workspace = workspace["sha256"]
        self._window.append(signature)
        window = list(self._window)
        for period in PERIODS:
            required = max(MIN_CALLS, period * MIN_CYCLES)
            required += (-required) % period
            if len(window) < required:
                continue
            candidate = window[-required:]
            if any(value != candidate[index % period] for index, value in enumerate(candidate)):
                continue
            if self._reported:
                return None
            self._reported = True
            return {
                "event": "stagnation", "policy": copy.deepcopy(POLICY),
                "period": period, "cycles": required // period, "window_calls": required,
                "first_call": self._calls - required + 1, "last_call": self._calls,
                "cycle_signatures_sha256": candidate[:period],
                "workspace_sha256": workspace["sha256"],
                "workspace_files": workspace["files"], "workspace_bytes": workspace["bytes"],
            }
        self._reported = False
        return None
