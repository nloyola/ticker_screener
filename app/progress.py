"""
Machine-parsable progress lines on stdout.

The stock_analyzer admin pages run the screener and parse these lines to show a run's stage and progress, so the
format is a contract: see README.md § Progress lines before changing it.

    PROGRESS stage=fetch subsector=3/11
    PROGRESS stage=analyze status=start
"""

import re
import sys

_TOKEN = re.compile(r'[A-Za-z0-9_./-]+')


def format_progress(stage: str, **fields: str | int) -> str:
    """Returns 'PROGRESS stage=<stage>' followed by each field as key=value, in the order given."""
    pairs = [('stage', stage), *fields.items()]
    for key, value in pairs:
        for token in (key, str(value)):
            if not _TOKEN.fullmatch(token):
                raise ValueError(f'invalid progress token: {token!r}')
    return 'PROGRESS ' + ' '.join(f'{key}={value}' for key, value in pairs)


def report_progress(stage: str, **fields: str | int) -> None:
    """Prints a progress line to stdout and flushes it, so a reader sees it as soon as it is printed."""
    print(format_progress(stage, **fields), file=sys.stdout, flush=True)
