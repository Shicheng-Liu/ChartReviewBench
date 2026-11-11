"""Cooperative process-lifetime cancellation without an episode budget."""
from contextlib import contextmanager
import signal
import threading

_event = threading.Event()
_reason = None


class Cancelled(RuntimeError):
    pass


def requested():
    return _event.is_set()


def request(reason="manual cancellation"):
    global _reason
    if not requested():
        _reason = str(reason)
    _event.set()


def reason():
    return _reason or "manual cancellation"


def check():
    if requested():
        raise Cancelled(reason())


def wait(seconds):
    """Wait between retries, returning promptly when cancellation is requested."""
    _event.wait(seconds)
    check()


@contextmanager
def install_signal_handlers():
    """Restore handlers on exit, but never clear a stop requested during this run."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    def stop(signum, frame):
        request(signal.Signals(signum).name)

    try:
        for sig in previous:
            signal.signal(sig, stop)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
