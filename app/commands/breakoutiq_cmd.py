import argparse
import logging
import sys
from typing import TYPE_CHECKING

from app.breakoutiq.dataset import MissingHistory
from app.breakoutiq.run import summarize
from app.config import DB_NAME, HISTORY_DATABASE_FILE
from app.container import Container

from .base_command import BaseCommand

if TYPE_CHECKING:
    from app.breakoutiq.run import Analysis

logger = logging.getLogger(__name__)


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
        try:
            summary = analysis.run()
        except MissingHistory as err:
            logger.error(f'breakoutiq-analysis: {err}; set HISTORY_DATABASE_FILE to the file history-fetch wrote')
            sys.exit(1)
        summarize(summary)
