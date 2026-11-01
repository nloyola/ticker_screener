"""
The lock that keeps two refreshes off one database.

The stock_analyzer admin pages and an operator running the screener by hand both refresh through 'run-refresh',
which holds an exclusive flock on 'screener.lock' beside the database file for the whole run. The kernel releases a
flock when its process dies, so a crashed run never leaves the lock held. The file itself stays: deleting it would
let a second run lock a new file while the first still holds the old one.
"""

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

LOCK_FILE_NAME = 'screener.lock'


class ScreenerLockHeld(Exception):
    """Another process holds the screener lock."""


def lock_path(db_path: str) -> Path:
    return Path(db_path).resolve().parent / LOCK_FILE_NAME


@contextmanager
def screener_lock(db_path: str) -> Iterator[Path]:
    """Holds the screener lock for the block; raises ScreenerLockHeld at once if another process holds it."""
    path = lock_path(db_path)
    with open(path, 'a') as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as err:
            raise ScreenerLockHeld(str(path)) from err
        try:
            yield path
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
