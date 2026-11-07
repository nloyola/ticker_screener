import argparse

from app.container import Container

from .base_command import BaseCommand


class AnalyzerCommand(BaseCommand):
    _NAME = 'analyze'
    _DESCRIPTION = 'analyzes the price data for the stock tickers in the DB'

    def __init__(self) -> None:
        super().__init__(self._NAME, self._DESCRIPTION)

    def handle(self, args: argparse.Namespace, container: Container) -> None:
        container.analyzer.analyze()
