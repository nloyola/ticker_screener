import argparse

from app.config import DB_NAME, TIINGO_API_KEY
from app.container import Container

from .base_command import BaseCommand


class TickerSyncCmd(BaseCommand):
    _NAME = 'ticker-sync'
    _DESCRIPTION = 'fetches all ticker symbols from Tiingo'

    REQUIRED_CONFIG = (DB_NAME, TIINGO_API_KEY)

    def __init__(self) -> None:
        super().__init__(self._NAME, self._DESCRIPTION)

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('--sector', help='filter sectors by name (case-insensitive)')

    def handle(self, args: argparse.Namespace, container: Container) -> None:
        container.ticker_sync.sync()
