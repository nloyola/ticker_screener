import argparse

from rich.console import Console

from app.analyzer import Analyzer
from app.container import Container

from .base_command import BaseCommand

console = Console()


class AnalyzerCommand(BaseCommand):
    _NAME = 'analyze'
    _DESCRIPTION = 'analyzes the price data for the stock tickers in the DB'

    def __init__(self, container: Container) -> None:
        super().__init__(self._NAME, self._DESCRIPTION)
        self.analyzer = Analyzer(container)

    def handle(self, args: argparse.Namespace) -> None:
        self.analyzer.analyze()
