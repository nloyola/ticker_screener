import argparse

from rich.console import Console

from app.container import Container

from .base_command import BaseCommand

console = Console()


class TickerSyncCmd(BaseCommand):
    _NAME = 'ticker-sync'
    _DESCRIPTION = 'fetches all ticker symbols from Tiingo'

    def __init__(self, container: Container) -> None:
        super().__init__(self._NAME, self._DESCRIPTION)
        self.ticker_sync = container.ticker_sync

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('--sector', help='filter sectors by name (case-insensitive)')

    def handle(self, args: argparse.Namespace) -> None:
        self.ticker_sync.sync()
