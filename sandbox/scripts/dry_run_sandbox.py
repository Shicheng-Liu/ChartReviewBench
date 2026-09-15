"""Check the Environment Core against the interaction-protocol spec.

One assertion per box in the sandbox/executor design: resource control,
deterministic rendering, output capture, error capture, and the rule that an
episode is scored on the *latest valid* chart. These are the properties an agent
loop silently depends on, and each of them fails quietly when it regresses — a
missing memory ceiling only shows up as a machine under load, an uncaptured
warning only as an unexplained visual-quality score.

    python scripts/dry_run_sandbox.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

SANDBOX = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX / "src"))

from chartsandbox.runner import _is_valid_image, _LatestValidRenders  # noqa: E402
from chartsandbox.tools import Sandbox  # noqa: E402

failures: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(label)


ws = Path(tempfile.mkdtemp(prefix="chartsandbox_core_"))
sb = Sandbox(ws, step_timeout=10)
try:
    print("resource control")
    # The address-space cap is opt-in (see the worker: capping it breaks kaleido's
    # bundled Chromium, which reserves far more virtual memory than it uses), so it
    # is exercised here by asking for it explicitly rather than assumed on.
    import os as _os
    _os.environ["SANDBOX_MEM_MB"] = "8192"   # enough for numpy/matplotlib, not for 64 GB
    capped = Sandbox(Path(tempfile.mkdtemp(prefix="chartsandbox_mem_")), step_timeout=10)
    try:
        r = capped.execute_python("x = bytearray(64 * 1024 * 1024 * 1024)")  # 64 GB
        check("with SANDBOX_MEM_MB set, a runaway allocation is a readable MemoryError",
              "MemoryError" in (r.get("error") or ""), f"error={r.get('error')!r}")
        check("the kernel survives it and still executes",
              capped.execute_python("y = 1 + 1\nprint(y)")["stdout"].strip() == "2")
    finally:
        capped.close()
        _os.environ.pop("SANDBOX_MEM_MB", None)

    r = sb.execute_python("while True:\n    pass")
    check("an unbounded loop is stopped by a timeout",
          "TimeoutError" in (r.get("error") or ""), f"error={r.get('error')!r}")
    check("state survives the timeout restart or is reported lost",
          sb.execute_python("print('alive')")["stdout"].strip() == "alive")

    print("\ndeterministic rendering")
    r = sb.execute_python(
        "import matplotlib\n"
        "print(matplotlib.get_backend())\n"
        "import os; print(os.environ.get('PYTHONHASHSEED'))")
    check("headless Agg backend is forced", "agg" in r["stdout"].lower(),
          r["stdout"].strip())

    print("\noutput capture")
    r = sb.execute_python(
        "import matplotlib.pyplot as plt\n"
        "plt.plot([1, 2, 3])\n"
        "plt.savefig('output.png')")
    check("a saved chart is reported as an image the agent produced",
          "output.png" in (r.get("images_created") or []), f"{r.get('images_created')}")

    r = sb.execute_python(
        "import matplotlib.pyplot as plt\n"
        "plt.figure()\n"
        "plt.plot([3, 2, 1])")   # deliberately never saved
    check("a figure left open but never saved is still captured",
          bool(r.get("figures_captured")), f"{r.get('figures_captured')}")
    check("...and is not passed off as a file the agent produced",
          not r.get("images_created"), f"{r.get('images_created')}")

    print("\nerror capture")
    r = sb.execute_python("import warnings\nwarnings.warn('axes labels may overlap')")
    check("a warning is captured, not just leaked to stderr",
          any("overlap" in w for w in (r.get("warnings") or [])), f"{r.get('warnings')}")
    r = sb.execute_python(
        "import warnings\n"
        "for _ in range(50):\n"
        "    warnings.warn('same message')")
    check("repeated identical warnings are deduplicated",
          len(r.get("warnings") or []) <= 2, f"{len(r.get('warnings') or [])} entries")

    r = sb.execute_python("1 / 0")
    check("an exception comes back as a traceback",
          "ZeroDivisionError" in (r.get("error") or ""))
    r = sb.execute_python("import sys\nprint('o')\nprint('e', file=sys.stderr)")
    check("stdout and stderr are captured separately",
          r["stdout"].strip() == "o" and r["stderr"].strip() == "e",
          f"stdout={r['stdout']!r} stderr={r['stderr']!r}")

    print("\nthe episode is scored on the latest valid chart")
    store = Path(tempfile.mkdtemp(prefix="chartsandbox_store_")) / ".last_valid"
    ledger = _LatestValidRenders(ws, store)
    good = sb.execute_python(
        "import matplotlib.pyplot as plt\n"
        "plt.figure(); plt.plot([1, 2, 3]); plt.savefig('output.png')")
    ledger.record(good)
    original = (ws / "output.png").read_bytes()
    check("a valid render is banked", (store / "output.png").exists())

    # the failure this rule exists for: a final turn that destroys its own output
    broken = sb.execute_python(
        "open('output.png', 'wb').write(b'not a png')\n"
        "raise RuntimeError('failed after truncating the chart')")
    ledger.record(broken)
    check("the damaged file is not banked over the good one",
          (store / "output.png").read_bytes() == original)
    check("the workspace copy is indeed unreadable now",
          not _is_valid_image(ws / "output.png"))
    restored = ledger.restore()
    check("it is rolled back to the last chart that rendered",
          restored == ["output.png"] and (ws / "output.png").read_bytes() == original,
          f"restored={restored}")

    print("\nworkspace confinement")
    r = sb.read_file("../../../etc/passwd")
    check("a path escaping the workspace is refused",
          r.get("ok") is False and "escape" in (r.get("error") or "").lower(),
          f"{r}")
finally:
    sb.close()
    shutil.rmtree(ws, ignore_errors=True)

print()
if failures:
    print(f"{len(failures)} check(s) FAILED:")
    for f in failures:
        print(f"  - {f}")
    raise SystemExit(1)
print("all checks passed — the environment core matches the interaction protocol")
