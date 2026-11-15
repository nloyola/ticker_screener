"""Which commit of the screener is running, recorded with each screened session."""

import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# The image's commit, set from the build's GIT_REVISION (Dockerfile); 'unknown' when the build passed none.
COMMIT_ENV = 'SCREENER_COMMIT'
UNKNOWN = 'unknown'

# the repo root, beside app/
REPO_DIR = Path(__file__).resolve().parent.parent


def git_commit() -> str | None:
    """
    The screener's commit: SCREENER_COMMIT when set, as in the image, which holds no .git; otherwise the checkout's
    HEAD, with '-dirty' when it has uncommitted changes, so a score from edited rules is never passed off as the
    commit's. None when neither says.
    """
    commit = os.environ.get(COMMIT_ENV, '').strip()
    if commit:
        return None if commit == UNKNOWN else commit
    try:
        head = _git('rev-parse', 'HEAD')
        dirty = _git('status', '--porcelain', '--untracked-files=no')
    except (OSError, subprocess.SubprocessError) as err:
        logger.info('Screener commit unknown: %s', err)
        return None
    return f'{head}-dirty' if dirty else head


def _git(*args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(REPO_DIR), *args], capture_output=True, text=True, check=True, timeout=10
    ).stdout.strip()
