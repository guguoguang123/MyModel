"""Serialize first-time creation of a baseline shared by independent runs."""

from contextlib import contextmanager
import fcntl


@contextmanager
def baseline_lock(path, notify=None):
    """Hold a process lock only during baseline creation or reuse, not training."""
    with path.open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if notify is not None:
                notify("Waiting for the shared pre-finetune validation baseline; training will remain independent")
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
