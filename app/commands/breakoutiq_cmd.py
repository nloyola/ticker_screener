import argparse
from typing import TYPE_CHECKING

from app.breakoutiq.run import summarize
from app.config import DB_NAME, HISTORY_DATABASE_FILE
from app.container import Container

from .base_command import BaseCommand

if TYPE_CHECKING:
    from app.breakoutiq.run import Analysis


class BreakoutIQCmd(BaseCommand):
    _NAME = 'breakoutiq-analysis'
    _DESCRIPTION = (
        "runs every section of BreakoutIQ's analysis of the signal archive and stores the run in analysis_run, with "
        'its results in analysis_result; reads the regime of backfilled signals from HISTORY_DATABASE_FILE'
    )

    REQUIRED_CONFIG = (DB_NAME, HISTORY_DATABASE_FILE)

    def __init__(self) -> None:
        super().__init__(self._NAME, self._DESCRIPTION)

    def handle(self, args: argparse.Namespace, container: Container) -> None:
        analysis: Analysis = container.breakoutiq_analysis
        summarize(analysis.run(with_history=True))
