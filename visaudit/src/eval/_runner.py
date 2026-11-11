"""Executes a plotting script and captures its figure object.

Ships into the sandbox cwd next to the script. It wraps the save calls every
ChartNet script ends with (savefig / write_image / write_html), so the figure is
read out while it is still alive, then writes `figure.json` for the parent.
"""
import os
import json
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _extract  # noqa: E402

TARGET = sys.argv[1] if len(sys.argv) > 1 else "chart_code.py"
_state = {"done": False}


def _visibility_target():
    try:
        with open("_capture_target.json", "r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except FileNotFoundError:
        return None


def _note(exc):
    with open("capture_error.txt", "a", encoding="utf-8") as fh:
        fh.write("%s: %s\n%s\n" % (type(exc).__name__, exc, traceback.format_exc()))


def _emit(spec):
    if _state["done"]:
        return
    try:
        _extract.dump(spec)
        _state["done"] = True
    except Exception as exc:
        _note(exc)


def _grab_mpl(fig):
    if _state["done"]:
        return
    try:
        _emit(_extract.extract_matplotlib(fig, _visibility_target()))
    except Exception as exc:
        _note(exc)


def _grab_plotly(fig):
    if _state["done"]:
        return
    try:
        _emit(_extract.extract_plotly(fig))
    except Exception as exc:
        _note(exc)


def _patch():
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        orig_savefig = Figure.savefig

        def savefig(self, *a, **k):
            out = orig_savefig(self, *a, **k)
            _grab_mpl(self)
            return out

        Figure.savefig = savefig
    except Exception as exc:
        _note(exc)

    try:
        from plotly.basedatatypes import BaseFigure

        for name in ("write_image", "write_html", "show"):
            orig = getattr(BaseFigure, name, None)
            if orig is None:
                continue

            def make(orig_fn):
                def wrapper(self, *a, **k):
                    out = orig_fn(self, *a, **k)
                    _grab_plotly(self)
                    return out
                return wrapper

            setattr(BaseFigure, name, make(orig))

        import plotly.io as pio
        for name in ("write_image", "write_html", "to_image"):
            orig = getattr(pio, name, None)
            if orig is None:
                continue

            def make_io(orig_fn):
                def wrapper(fig, *a, **k):
                    out = orig_fn(fig, *a, **k)
                    _grab_plotly(fig)
                    return out
                return wrapper

            setattr(pio, name, make_io(orig))
    except Exception:
        pass  # plotly not installed for this script — fine


def _fallback():
    """Scripts that only call plt.show() (a no-op under Agg) still leave figures."""
    if _state["done"]:
        return
    try:
        import matplotlib.pyplot as plt
        nums = plt.get_fignums()
        if nums:
            _grab_mpl(plt.figure(nums[-1]))
    except Exception as exc:
        _note(exc)


def main():
    _patch()
    import runpy
    try:
        runpy.run_path(TARGET, run_name="__main__")
    except SystemExit:
        pass
    except BaseException as exc:  # capture whatever was drawn before the crash
        _note(exc)
        _fallback()
        raise
    _fallback()


main()
