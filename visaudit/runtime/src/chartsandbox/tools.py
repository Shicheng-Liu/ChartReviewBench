"""The agent's action space — a small, model-agnostic tool set over one workspace.

A real LLM agent calls these via tool-calling using TOOL_SCHEMAS. The sandbox
enforces that every path stays inside the workspace, so the agent can never read
the oracle or escape the sandbox.
"""
from __future__ import annotations

import base64
from pathlib import Path

from .runtime import PersistentKernel

# JSON-schema-style tool declarations. Feed these to an LLM's tool-calling API;
# the same names are dispatched by the runner. Kept provider-neutral on purpose.
TOOL_SCHEMAS: list[dict] = [
    {
        "name": "execute_python",
        "description": "Run Python in the persistent workspace kernel. Variables/imports "
                       "persist across calls. Save charts to files (e.g. plt.savefig('out.png')).",
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python source to execute."}},
            "required": ["code"],
        },
    },
    {
        "name": "write_file",
        "description": "Create or overwrite a text file in the workspace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a text file from the workspace.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "list_files",
        "description": "List files in a workspace directory (default: root).",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "default": "."}},
        },
    },
    {
        "name": "view_image",
        "description": "Return an image file so a multimodal agent can look at its own output.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "finish",
        "description": "Declare the task complete. Ends the episode.",
        "input_schema": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
        },
    },
]

_IMG_MIME = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
             "gif": "image/gif", "webp": "image/webp"}


class Sandbox:
    """A writable workspace + its persistent kernel + safe file ops."""

    def __init__(self, workspace: str | Path, step_timeout: int = 30):
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.kernel = PersistentKernel(self.workspace, step_timeout)

    # -- path safety ---------------------------------------------------------
    def _safe(self, rel: str) -> Path:
        p = (self.workspace / rel).resolve()
        if p != self.workspace and self.workspace not in p.parents:
            raise ValueError(f"path {rel!r} escapes the workspace")
        return p

    # -- tools ---------------------------------------------------------------
    def execute_python(self, code: str) -> dict:
        return self.kernel.execute(code)

    def write_file(self, path: str, content: str) -> dict:
        try:
            p = self._safe(path)
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return {"ok": True, "path": path, "bytes": len(content)}

    def read_file(self, path: str, max_bytes: int = 100_000) -> dict:
        try:
            p = self._safe(path)
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        if not p.exists() or not p.is_file():
            return {"ok": False, "error": f"{path!r} not found"}
        return {"ok": True, "content": p.read_text(errors="replace")[:max_bytes]}

    def list_files(self, path: str = ".") -> dict:
        try:
            p = self._safe(path)
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        if not p.exists():
            return {"ok": False, "error": f"{path!r} not found"}
        out = []
        for c in sorted(p.iterdir()):
            out.append({"name": c.name, "is_dir": c.is_dir(),
                        "bytes": c.stat().st_size if c.is_file() else None})
        return {"ok": True, "files": out}

    def view_image(self, path: str) -> dict:
        try:
            p = self._safe(path)
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        if not p.exists():
            return {"ok": False, "error": f"{path!r} not found"}
        ext = p.suffix.lstrip(".").lower()
        try:
            from PIL import Image

            with Image.open(p) as im:
                w, h = im.size
        except Exception:
            w = h = None
        b64 = base64.b64encode(p.read_bytes()).decode()
        return {"ok": True, "path": path, "mime": _IMG_MIME.get(ext, "application/octet-stream"),
                "width": w, "height": h, "base64": b64}

    def close(self) -> None:
        self.kernel.shutdown()
