import argparse
import sys
from typing import TYPE_CHECKING

from app.container import Container
from app.screener_lock import ScreenerLockHeld, screener_lock

from .base_command import BaseCommand
from .run_refresh_cmd import EXIT_LOCK_HELD

if TYPE_CHECKING:
    from app.signal_outcome_updates import SignalOutcomeUpdates


class ComputeOutcomesCmd(BaseCommand):
    _NAME = 'compute-outcomes'
    _DESCRIPTION = (
        "holding the screener lock, stores every signal's newly complete outcomes and decided breakout failure from "
        f'its stored price path, as run-refresh does last; exits {EXIT_LOCK_HELD} if a refresh holds the lock'
    )

    def __init__(self) -> None:
        super().__init__(self._NAME, self._DESCRIPTION)

    def handle(self, args: argparse.Namespace, container: Container) -> None:
        updates: SignalOutcomeUpdates = container.signal_outcome_updates
        try:
            # a refresh rewrites the paths the outcomes are computed from while it runs
            with screener_lock(container.db_path):
                updates.update()
        except ScreenerLockHeld as err:
            print(f'compute-outcomes: a refresh holds the lock: {err}', file=sys.stderr)
            sys.exit(EXIT_LOCK_HELD)
