"""
Storing each archived record's outcomes (signal_outcome) and a signal's breakout failure and retest (signal_track),
computed from its price path by compute_outcomes (app/signal_outcomes.py): its own, or for a continuation its
breakout's from the continuation's session on. run-refresh runs it after the paths are updated, and compute-outcomes
runs it on demand: both store the same rows, since the outcomes depend on the stored path alone.
"""

import logging
from dataclasses import dataclass, replace

from app.container import Container
from app.models import CONTINUATION_KIND, SIGNAL_KIND
from app.repositories.signal_outcome_repo import SignalOutcomeRepo
from app.repositories.signal_path_repo import SignalPathRepo
from app.signal_outcomes import breakout_level, compute_outcomes
from app.signal_paths import continuation_horizons, kind_horizons

logger = logging.getLogger(__name__)


@dataclass
class OutcomeUpdate:
    """
    What one update did: signals it computed, horizons it completed, failures it decided, of which failed, and
    retests it decided, of which retested.
    """

    signals: int = 0
    completed: int = 0
    decided: int = 0
    failed: int = 0
    no_level: int = 0
    benchmarks_filled: int = 0
    retests_decided: int = 0
    retested: int = 0


class SignalOutcomeUpdates:
    def __init__(self, container: Container) -> None:
        self.signal_outcome_repo: SignalOutcomeRepo = container.signal_outcome_repo
        self.signal_path_repo: SignalPathRepo = container.signal_path_repo

    def update(self) -> OutcomeUpdate:
        """
        For every record with a path whose outcomes are not all decided, computes its outcomes from the path and
        stores each horizon newly complete at the horizons its kind is measured at, its exit session now in the path,
        and a signal's breakout failure and its retest, each once decided; no other kind has either. A continuation k
        sessions after its breakout is measured on its breakout's path from index k: entry the open at k + 1, exit the
        close at k + h, for each h with k + h at most 120. One whose session is not its breakout path's index k is
        left, with a warning. A complete horizon, a decided failure and a decided retest are never recomputed or
        stored again, except that a horizon
        completed while the path lacked the benchmark's prices (a run whose benchmark download failed) gets the
        benchmark's and the excess return once a later run's path has them. A horizon its path does not reach yet
        stays as it is: without a row (pending), or ended by the path update.

        The failure and the retest are judged against the exact level the signal's run broke, or, for a signal
        archived before the screener kept it, the level rebuilt from its rounded highs (breakout_level in
        app/signal_outcomes.py). A signal archived with neither gets its horizons but no failure or retest, with a
        warning. Logs one summary line and returns the counts.
        """
        result = OutcomeUpdate()
        for s in self.signal_outcome_repo.undecided():
            path = self.signal_path_repo.path(s.path_id)
            if s.kind == CONTINUATION_KIND:
                horizons = continuation_horizons(s.offset)
                path = [replace(p, session_index=p.session_index - s.offset) for p in path[s.offset :]]
                if not path:
                    continue  # its breakout's path does not reach it yet
                if path[0].session_date != s.session_date:
                    logger.warning(
                        f'continuation {s.id}: its session {s.session_date} is not session {s.offset} of its '
                        f'breakout {s.breakout_id}, outcomes left undecided'
                    )
                    continue
            else:
                horizons = kind_horizons(s.kind)
            level = None
            if s.kind == SIGNAL_KIND and not (s.failure_decided and s.retest_decided):
                level = breakout_level(s.breakout_level, s.high_20d, s.high_60d, s.breakout_60d)
                if level is None:
                    logger.warning(f'signal {s.id}: archived without its breakout level, failure left undecided')
                    result.no_level += 1
            outcomes = compute_outcomes(path, s.close, level)
            measured = [o for o in outcomes.horizons if o.complete and o.horizon in horizons]
            completed = [o for o in measured if o.horizon not in s.complete]
            benchmarks = [o for o in measured if o.horizon in s.benchmark_due and o.benchmark_return_pct is not None]
            self.signal_outcome_repo.store(s.id, completed, outcomes.failure, benchmarks, outcomes.retest)
            result.signals += 1
            result.completed += len(completed)
            result.benchmarks_filled += len(benchmarks)
            if not s.failure_decided and outcomes.failure is not None and outcomes.failure.failed is not None:
                result.decided += 1
                result.failed += int(outcomes.failure.failed)
            if not s.retest_decided and outcomes.retest is not None and outcomes.retest.retested is not None:
                result.retests_decided += 1
                result.retested += int(outcomes.retest.retested)

        logger.info(
            f'signal outcomes: computed {result.signals} signals, completed {result.completed} horizons, '
            f'decided {result.decided} breakout failures ({result.failed} failed) and {result.retests_decided} '
            f'retests ({result.retested} retested), '
            f'{result.no_level} without a breakout level, filled {result.benchmarks_filled} benchmark returns'
        )
        return result
