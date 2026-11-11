"""Cooperative process-lifetime cancellation without an episode budget."""
from contextlib import contextmanager
import math
import signal
import threading
from time import monotonic

_event = threading.Event()
_reason = None
_episode = threading.local()


class Cancelled(RuntimeError):
    pass


class Deferred(RuntimeError):
    """This attempt yields to the queue; other episodes keep running."""


@contextmanager
def episode_deadline(seconds):
    """Scope an optional first-pass scheduling deadline to this worker only."""
    if seconds is not None and (not math.isfinite(seconds) or seconds <= 0):
        raise ValueError("episode scheduling deadline must be positive and finite")
    previous = getattr(_episode, "deadline", None)
    _episode.deadline = None if seconds is None else monotonic() + seconds
    try:
        yield
    finally:
        _episode.deadline = previous


def remaining():
    deadline = getattr(_episode, "deadline", None)
    return None if deadline is None else max(0.0, deadline - monotonic())


def finish_episode_work():
    """A finished submission may complete scoring without a scheduling deadline."""
    _episode.deadline = None


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
    if remaining() == 0:
        raise Deferred("first-pass scheduling deadline reached")


def wait(seconds):
    """Wait between retries, returning promptly when cancellation is requested."""
    check()
    left = remaining()
    _event.wait(seconds if left is None else min(seconds, left))
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
