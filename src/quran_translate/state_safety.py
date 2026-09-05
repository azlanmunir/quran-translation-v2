"""Small cross-process guards for local production state."""

from contextlib import contextmanager
import fcntl
from pathlib import Path


@contextmanager
def exclusive_lock(path: Path):
    """Never allow two workers to mutate the same run or episode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another worker owns {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
