import argparse
import sys
from typing import TYPE_CHECKING

from app.config import DB_NAME, HISTORY_DATABASE_FILE
from app.container import Container
from app.history_backfill import summarize
from app.screener_lock import ScreenerLockHeld, screener_lock

from .base_command import BaseCommand
from .run_refresh_cmd import EXIT_LOCK_HELD

if TYPE_CHECKING:
    from app.history_backfill import HistoryBackfill


class BackfillHistoryCmd(BaseCommand):
    _NAME = 'backfill-history'
    _DESCRIPTION = (
        'holding the screener lock, replays every session the price history in HISTORY_DATABASE_FILE can fully '
        f'score and archives its signals as backfill; exits {EXIT_LOCK_HELD} if a refresh holds the lock'
    )

    REQUIRED_CONFIG = (DB_NAME, HISTORY_DATABASE_FILE)

    def __init__(self) -> None:
        super().__init__(self._NAME, self._DESCRIPTION)

    def handle(self, args: argparse.Namespace, container: Container) -> None:
        history_backfill: HistoryBackfill = container.history_backfill
        try:
            # a refresh archives and links signals while it runs
            with screener_lock(container.db_path):
                result = history_backfill.run()
        except ScreenerLockHeld as err:
            print(f'backfill-history: a refresh holds the lock: {err}', file=sys.stderr)
            sys.exit(EXIT_LOCK_HELD)
        summarize(result)
