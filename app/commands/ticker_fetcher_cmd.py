import argparse
from typing import TYPE_CHECKING

from rich.console import Console

from app.container import Container

from .base_command import BaseCommand

if TYPE_CHECKING:
    from app.ticker_fetcher import TickerFetcher

console = Console()


class TickerFetcherCmd(BaseCommand):
    _NAME = 'ticker-fetcher'
    _DESCRIPTION = 'fetches stock ticker price data'

    def __init__(self, container: Container) -> None:
        super().__init__(self._NAME, self._DESCRIPTION)
        self.ticker_fetcher: TickerFetcher = container.ticker_fetcher

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('--sector', help='filter sectors by name (case-insensitive)')
        parser.add_argument('--ticker', help='fetch a single ticker')

    def handle(self, args: argparse.Namespace) -> None:
        sector_filter = args.sector.lower() if args.sector else None
        ticker_filter = args.ticker if args.ticker else None
        self.ticker_fetcher.fetch(sector_filter=sector_filter, ticker_filter=ticker_filter)
