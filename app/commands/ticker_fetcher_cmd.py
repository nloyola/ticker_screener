import argparse

from rich.console import Console

from app.container import Container

from .base_command import BaseCommand

console = Console()


class TickerFetcherCmd(BaseCommand):
    _NAME = 'ticker-fetcher'
    _DESCRIPTION = 'fetches stock ticker price data'

    def __init__(self, container: Container) -> None:
        super().__init__(self._NAME, self._DESCRIPTION)
        self.fetcher = container.ticker_fetcher

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('--sector', help='filter sectors by name (case-insensitive)')

    def handle(self, args: argparse.Namespace) -> None:
        sector_filter = args.sector.lower() if args.sector else None
        self.ticker_fetcher.fetch(sector_filter)
