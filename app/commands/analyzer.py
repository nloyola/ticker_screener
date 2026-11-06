import argparse

from rich.console import Console

from app.container import Container

from .base_command import BaseCommand

console = Console()


class AnalyzerCommand(BaseCommand):
    _NAME = 'analyze'
    _DESCRIPTION = 'analyzes the price data for the stock tickers in the DB'

    def __init__(self) -> None:
        super().__init__(self._NAME, self._DESCRIPTION)

    def handle(self, args: argparse.Namespace, container: Container) -> None:
        container.analyzer.analyze()
