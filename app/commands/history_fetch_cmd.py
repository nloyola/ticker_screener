import argparse
import logging
import sys
from typing import TYPE_CHECKING

from app.config import DB_NAME, HISTORY_DATABASE_FILE, PRICE_API
from app.container import Container
from app.history_fetch import HISTORY_YEARS, HistoryPlan, HistoryReport
from app.price_apis import PriceApiStop
from app.repositories.history_repo import BENCHMARK, TICKER

from .base_command import BaseCommand

if TYPE_CHECKING:
    from app.history_fetch import HistoryFetch

logger = logging.getLogger(__name__)

# the shell's exit status for a command stopped by SIGINT
EXIT_INTERRUPTED = 130


class HistoryFetchCmd(BaseCommand):
    _NAME = 'history-fetch'
    _DESCRIPTION = (
        f'fetches up to {HISTORY_YEARS} years of daily prices of every listed ticker and every benchmark from '
        'PRICE_API into HISTORY_DATABASE_FILE, skipping those fetched today already; exits 1 when a request failed'
    )

    REQUIRED_CONFIG = (DB_NAME, PRICE_API, HISTORY_DATABASE_FILE)

    def __init__(self) -> None:
        super().__init__(self._NAME, self._DESCRIPTION)

    def handle(self, args: argparse.Namespace, container: Container) -> None:
        history_fetch: HistoryFetch = container.history_fetch
        try:
            report = history_fetch.run(announce)
        except PriceApiStop as err:
            logger.error(f'history-fetch stopped: {err}')
            sys.exit(1)
        except KeyboardInterrupt:
            # every ticker stored so far is recorded: the next run today picks up from the first one that is not
            print('history-fetch: interrupted, run it again to resume', file=sys.stderr)
            sys.exit(EXIT_INTERRUPTED)
        summarize(report)
        if report.failed:
            sys.exit(1)


def announce(plan: HistoryPlan) -> None:
    """Prints the request count before the first request."""
    report = plan.report
    print(
        f'history-fetch: {report.requests} requests on {report.api}, {report.start} to {report.end}: '
        f'{sum(kind == TICKER for kind, _ in plan.todo)} tickers and '
        f'{sum(kind == BENCHMARK for kind, _ in plan.todo)} benchmarks; {report.done_today} fetched today already'
    )
    if report.not_covered:
        print(f'history-fetch: not covered by {report.api}, not requested: {", ".join(report.not_covered)}')
    if report.unsupported:
        print(f'history-fetch: unsupported exchange suffix, not requested: {", ".join(report.unsupported)}')
    sys.stdout.flush()


def summarize(report: HistoryReport) -> None:
    """Prints what the history holds after the fetch, and what it lacks."""
    print(
        f'history-fetch: {report.tickers_with_history} tickers and {report.benchmarks_with_history} benchmarks '
        f'have history on {report.api}'
    )
    if report.no_history:
        print(f'history-fetch: no history: {", ".join(report.no_history)}')
    if report.short_history:
        short = ', '.join(f'{ticker} (from {first})' for ticker, first in report.short_history.items())
        print(f'history-fetch: short history, starting after {report.start}: {short}')
    if report.failed:
        print(f'history-fetch: failed, run again to retry: {", ".join(report.failed)}')
