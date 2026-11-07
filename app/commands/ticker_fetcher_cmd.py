import argparse
import logging
import sys
from typing import TYPE_CHECKING

from app.config import DB_NAME, PRICE_API
from app.container import Container
from app.price_apis import PriceApiStop
from app.services.sector_service import normalize_ticker

from .base_command import BaseCommand

if TYPE_CHECKING:
    from app.ticker_fetcher import FetchStats, TickerFetcher

logger = logging.getLogger(__name__)


def ticker_arg(value: str) -> str:
    ticker = normalize_ticker(value)
    if not ticker:
        raise argparse.ArgumentTypeError('must not be blank')
    return ticker


def failure_rate_arg(value: str) -> float:
    rate = float(value)
    if not 0.0 <= rate <= 1.0:
        raise argparse.ArgumentTypeError(f'must be between 0 and 1: {value}')
    return rate


class TickerFetcherCmd(BaseCommand):
    _NAME = 'ticker-fetcher'
    _DESCRIPTION = 'fetches stock ticker price data'

    REQUIRED_CONFIG = (DB_NAME, PRICE_API)

    def __init__(self) -> None:
        super().__init__(self._NAME, self._DESCRIPTION)

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('--sector', help='filter sectors by name (case-insensitive)')
        parser.add_argument('--ticker', type=ticker_arg, help='fetch a single ticker (case-insensitive)')
        parser.add_argument(
            '--strict',
            action='store_true',
            help='log each failed ticker, print a summary and exit 1 when nothing was fetched, '
            'every ticker failed or the failure rate exceeds --max-failure-rate; otherwise delete the price rows '
            'older than 120 days (without --strict, old rows are never deleted)',
        )
        parser.add_argument(
            '--max-failure-rate',
            type=failure_rate_arg,
            default=0.5,
            metavar='RATE',
            help='with --strict, the highest fraction of failed tickers (0 to 1) that still exits 0 (default: 0.5)',
        )

    def handle(self, args: argparse.Namespace, container: Container) -> None:
        ticker_fetcher: TickerFetcher = container.ticker_fetcher
        sector_filter = args.sector.lower() if args.sector else None
        stats = fetch_or_exit(
            ticker_fetcher, sector_filter=sector_filter, ticker_filter=args.ticker, strict=args.strict
        )
        if args.strict:
            if not strict_fetch_succeeded(stats, args.max_failure_rate):
                sys.exit(1)
            # only a fetch judged a success may age out old rows: one that failed for every ticker keeps them all
            ticker_fetcher.prune()


def fetch_or_exit(ticker_fetcher: 'TickerFetcher', **kwargs) -> 'FetchStats':
    """Runs the fetch; exits 1, after logging why, when the price API stopped it before any price request."""
    try:
        return ticker_fetcher.fetch(**kwargs)
    except PriceApiStop as err:
        logger.error(f'ticker-fetcher stopped: {err}')
        sys.exit(1)


def strict_fetch_succeeded(stats: 'FetchStats', max_failure_rate: float) -> bool:
    """
    Prints the strict summary line and returns False, after logging an error, when nothing was fetched, every
    ticker failed or the failure rate exceeds max_failure_rate.

    The line starts 'ticker-fetcher: <N> attempted, <N> failed', which the dashboard's runner parses, and always ends
    with the skipped count, even at 0: 'ticker-fetcher: 1228 attempted, 2 failed (0%, threshold 50%), 15 skipped
    (not covered by Tiingo)'.
    """
    print(
        f'ticker-fetcher: {stats.attempted} attempted, {stats.failed} failed '
        f'({stats.failure_rate:.0%}, threshold {max_failure_rate:.0%}), {stats.skipped} skipped'
        + (f' (not covered by {stats.api})' if stats.api else '')
    )
    if stats.attempted == 0 or stats.failed == stats.attempted or stats.failure_rate > max_failure_rate:
        logger.error('ticker-fetcher failed')
        return False
    return True
