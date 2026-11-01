import argparse
import logging
import sys
from typing import TYPE_CHECKING

from rich.console import Console

from app.container import Container

from .base_command import BaseCommand

if TYPE_CHECKING:
    from app.ticker_fetcher import FetchStats, TickerFetcher

console = Console()

logger = logging.getLogger(__name__)


def failure_rate_arg(value: str) -> float:
    rate = float(value)
    if not 0.0 <= rate <= 1.0:
        raise argparse.ArgumentTypeError(f'must be between 0 and 1: {value}')
    return rate


class TickerFetcherCmd(BaseCommand):
    _NAME = 'ticker-fetcher'
    _DESCRIPTION = 'fetches stock ticker price data'

    def __init__(self, container: Container) -> None:
        super().__init__(self._NAME, self._DESCRIPTION)
        self.ticker_fetcher: TickerFetcher = container.ticker_fetcher

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('--sector', help='filter sectors by name (case-insensitive)')
        parser.add_argument('--ticker', help='fetch a single ticker')
        parser.add_argument(
            '--strict',
            action='store_true',
            help='log each failed ticker, print a summary and exit 1 when nothing was fetched, '
            'every ticker failed or the failure rate exceeds --max-failure-rate',
        )
        parser.add_argument(
            '--max-failure-rate',
            type=failure_rate_arg,
            default=0.5,
            metavar='RATE',
            help='with --strict, the highest fraction of failed tickers (0 to 1) that still exits 0 (default: 0.5)',
        )

    def handle(self, args: argparse.Namespace) -> None:
        sector_filter = args.sector.lower() if args.sector else None
        ticker_filter = args.ticker if args.ticker else None
        stats = self.ticker_fetcher.fetch(sector_filter=sector_filter, ticker_filter=ticker_filter, strict=args.strict)
        if args.strict and not strict_fetch_succeeded(stats, args.max_failure_rate):
            sys.exit(1)


def strict_fetch_succeeded(stats: 'FetchStats', max_failure_rate: float) -> bool:
    """
    Prints the strict summary line and returns False, after logging an error, when nothing was fetched, every
    ticker failed or the failure rate exceeds max_failure_rate.
    """
    print(
        f'ticker-fetcher: {stats.attempted} attempted, {stats.failed} failed '
        f'({stats.failure_rate:.0%}, threshold {max_failure_rate:.0%})'
    )
    if stats.attempted == 0 or stats.failed == stats.attempted or stats.failure_rate > max_failure_rate:
        logger.error('ticker-fetcher failed')
        return False
    return True
