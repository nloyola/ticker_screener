import argparse
import logging
import os
import sys
from typing import TYPE_CHECKING

from app.config import DB_NAME, TIINGO_API_KEY
from app.container import Container
from app.screener_lock import ScreenerLockHeld, screener_lock

from .base_command import BaseCommand
from .ticker_fetcher_cmd import failure_rate_arg, strict_fetch_succeeded

if TYPE_CHECKING:
    from app.analyzer import Analyzer
    from app.ticker_fetcher import TickerFetcher

logger = logging.getLogger(__name__)

# EX_TEMPFAIL: another refresh holds the lock, so a caller can tell "busy, try later" from a failed run
EXIT_LOCK_HELD = os.EX_TEMPFAIL


class RunRefreshCmd(BaseCommand):
    _NAME = 'run-refresh'
    _DESCRIPTION = (
        'holding the screener lock, runs ticker-fetcher --strict and then analyze; '
        f'exits {EXIT_LOCK_HELD} if another refresh holds the lock'
    )

    REQUIRED_CONFIG = (DB_NAME, TIINGO_API_KEY)

    def __init__(self) -> None:
        super().__init__(self._NAME, self._DESCRIPTION)

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            '--max-failure-rate',
            type=failure_rate_arg,
            default=0.5,
            metavar='RATE',
            help='the highest fraction of failed tickers (0 to 1) that still runs analyze (default: 0.5)',
        )

    def handle(self, args: argparse.Namespace, container: Container) -> None:
        ticker_fetcher: TickerFetcher = container.ticker_fetcher
        analyzer: Analyzer = container.analyzer
        try:
            with screener_lock(container.db_path):
                stats = ticker_fetcher.fetch(strict=True)
                if not strict_fetch_succeeded(stats, args.max_failure_rate):
                    logger.error('run-refresh: fetch failed, analyze skipped')
                    sys.exit(1)
                ticker_fetcher.prune()
                analyzer.analyze()
        except ScreenerLockHeld as err:
            print(f'run-refresh: another refresh holds the lock: {err}', file=sys.stderr)
            sys.exit(EXIT_LOCK_HELD)
