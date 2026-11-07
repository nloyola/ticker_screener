"""
Machine-parsable progress and flagged-ticker lines on stdout.

The stock_analyzer admin pages run the screener and parse these lines to show a run's stage and progress, and the
tickers it found to be dead or behind, so the format is a contract: see README.md § Progress lines and § Flagged
tickers before changing it.

    PROGRESS stage=fetch subsector=3/11
    PROGRESS stage=analyze status=start
    FLAGGED ticker=BLD reason=no_prices last=2026-07-02
"""

import logging
import re
import sys
from datetime import date

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r'[A-Za-z0-9_./-]+')

# why a ticker is flagged: the price API does not know it (a 404), its download holds no trading day in the window,
# or analyze left it unscored because its last price is older than its market's newest date, or because its price
# has been frozen (one close with zero volume) for its last trading days
NOT_FOUND = 'not_found'
NO_PRICES = 'no_prices'
STALE = 'stale'
FROZEN = 'frozen'
FLAG_REASONS = (NOT_FOUND, NO_PRICES, STALE, FROZEN)


def _line(word: str, pairs: list[tuple[str, str | int]]) -> str:
    for key, value in pairs:
        for token in (key, str(value)):
            if not _TOKEN.fullmatch(token):
                raise ValueError(f'invalid {word.lower()} token: {token!r}')
    return word + ' ' + ' '.join(f'{key}={value}' for key, value in pairs)


def format_progress(stage: str, **fields: str | int) -> str:
    """Returns 'PROGRESS stage=<stage>' followed by each field as key=value, in the order given."""
    return _line('PROGRESS', [('stage', stage), *fields.items()])


def report_progress(stage: str, **fields: str | int) -> None:
    """Prints a progress line to stdout and flushes it, so a reader sees it as soon as it is printed."""
    print(format_progress(stage, **fields), file=sys.stdout, flush=True)


def format_flagged(ticker: str, reason: str, last: date | None) -> str:
    """Returns 'FLAGGED ticker=<ticker> reason=<reason>', followed by 'last=<YYYY-MM-DD>' when 'last' is given."""
    if reason not in FLAG_REASONS:
        raise ValueError(f'unknown flag reason: {reason!r}')
    pairs: list[tuple[str, str | int]] = [('ticker', ticker), ('reason', reason)]
    if last is not None:
        pairs.append(('last', last.isoformat()))
    return _line('FLAGGED', pairs)


def report_flagged(ticker: str, reason: str, last: date | None) -> None:
    """
    Prints a flagged-ticker line to stdout and flushes it. A ticker the line's grammar cannot carry is logged as a
    warning instead, so an odd symbol never stops a run.
    """
    try:
        line = format_flagged(ticker, reason, last)
    except ValueError as err:
        logger.warning(f'not reported as flagged ({reason}): {ticker!r}: {err}')
        return
    print(line, file=sys.stdout, flush=True)
