import argparse
import sys
from typing import TYPE_CHECKING

from app.container import Container
from app.screener_lock import ScreenerLockHeld, screener_lock

from .base_command import BaseCommand
from .run_refresh_cmd import EXIT_LOCK_HELD

if TYPE_CHECKING:
    from app.backfill import Backfill


class BackfillCmd(BaseCommand):
    _NAME = 'backfill-signals'
    _DESCRIPTION = (
        'holding the screener lock, replays every session the stored prices can fully score and archives its '
        f'signals as backfill; exits {EXIT_LOCK_HELD} if a refresh holds the lock'
    )

    def __init__(self) -> None:
        super().__init__(self._NAME, self._DESCRIPTION)

    def handle(self, args: argparse.Namespace, container: Container) -> None:
        backfill: Backfill = container.backfill
        try:
            # a refresh replaces price_data and benchmark_price while it runs
            with screener_lock(container.db_path):
                backfill.run()
        except ScreenerLockHeld as err:
            print(f'backfill-signals: a refresh holds the lock: {err}', file=sys.stderr)
            sys.exit(EXIT_LOCK_HELD)
