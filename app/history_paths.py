"""
The price paths of the backfilled signals, built from the price history (HISTORY_DATABASE_FILE) rather than from
price_data, whose 400-day window holds only the newest of them. backfill-history writes them after it archives the
signals, and then stores their outcomes from those paths, as run-refresh does from its own.
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING

from app.analyzer import BACKFILL
from app.container import Container
from app.markets import BENCHMARKS, MARKETS, last_complete_session, market_of
from app.models import PathSession
from app.repositories.history_repo import BENCHMARK, TICKER
from app.signal_paths import PATH_LENGTHS, build_path, ended_horizons

if TYPE_CHECKING:
    from app.repositories.history_repo import HistoryRepo
    from app.repositories.signal_path_repo import SignalPathRepo, TrackedSignal

logger = logging.getLogger(__name__)

# the reason a backfilled signal's horizons beyond its path are ended: its ticker's history stops before its market's
# history does, as when the ticker stopped trading for a while, so its path cannot reach them
HISTORY_ENDED = 'history_ended'


@dataclass
class HistoryPathUpdate:
    """
    What one update did: paths written, of which frozen, signals ended, and signals whose session is not in their
    ticker's history.
    """

    written: int = 0
    frozen: int = 0
    ended: int = 0
    missing_session: int = 0


class HistoryPaths:
    def __init__(self, container: Container) -> None:
        self.history_repo: HistoryRepo = container.history_repo
        self.signal_path_repo: SignalPathRepo = container.signal_path_repo

    def update(self, now: datetime) -> HistoryPathUpdate:
        """
        Writes the path of every backfill record not yet frozen whose ticker has a history, of every kind with a path
        of its own (all but a continuation), from that history and its market's benchmark's, as SignalPaths.update
        writes a path from price_data: whole, from the record's session to its market's last complete session at
        'now' (timezone aware), or to its kind's PATH_LENGTHS after it, when it is frozen and no later run rewrites it.

        A path that stops at its market's last session in the history is left for run-refresh to carry on from
        price_data. One whose ticker's history stops earlier keeps that path, and each horizon it does not reach is
        marked ended (HISTORY_ENDED) with its last session, as SignalPaths.update ends a signal whose ticker the run
        got no prices for, and so is each horizon of its continuations that the path does not reach. A signal whose
        session is not in its ticker's history keeps the path it has, if any.

        A live signal, and a backfill signal whose ticker has no history, is left to run-refresh. The paths of one
        ticker are written in one transaction. Logs one summary line and returns the counts.
        """
        result = HistoryPathUpdate()
        stored = self.history_repo.coverage(TICKER)
        signals = [s for s in self.signal_path_repo.unfrozen(MARKETS, source=BACKFILL) if s.ticker in stored]
        by_ticker: dict[str, list[TrackedSignal]] = {}
        for s in signals:
            by_ticker.setdefault(s.ticker, []).append(s)

        last_sessions = {market: last_complete_session(market, now) for market in MARKETS}
        # each market's last session in the history, at most its last complete session: a path reaching it is not
        # ended, since run-refresh carries it on
        history_ends: dict[str, date] = {}
        for ticker, coverage in stored.items():
            market = market_of(ticker)
            last = min(coverage.last, last_sessions[market])
            history_ends[market] = max(history_ends.get(market, last), last)
        markets = {market_of(ticker) for ticker in by_ticker}
        benchmarks = {
            market: {p.date: (p.open, p.close) for p in self.history_repo.prices(BENCHMARK, BENCHMARKS[market])}
            for market in markets
        }

        for ticker, ticker_signals in by_ticker.items():
            market = market_of(ticker)
            rows = self.history_repo.prices(TICKER, ticker)
            paths: list[tuple[int, list[PathSession], bool]] = []
            ends: list[tuple[int, list[int], str, date]] = []
            continuation_ends: list[tuple[int, int, str, date]] = []
            for s in ticker_signals:
                length = PATH_LENGTHS[s.kind]
                path = build_path(s.session_date, rows, benchmarks[market], last_sessions[market], length)
                if path is None:
                    result.missing_session += 1
                    continue
                last = path[-1]
                frozen = last.session_index == length
                paths.append((s.id, path, frozen))
                result.frozen += int(frozen)
                if not frozen and last.session_date < history_ends[market]:
                    ends.append((s.id, ended_horizons(s.kind, last.session_index), HISTORY_ENDED, last.session_date))
                    continuation_ends.append((s.id, last.session_index, HISTORY_ENDED, last.session_date))
            # the ended horizons after the paths: writing a path clears the ended horizons of it and its continuations
            self.signal_path_repo.write_all(paths)
            self.signal_path_repo.end_all(ends)
            self.signal_path_repo.end_continuations(continuation_ends)
            result.written += len(paths)
            result.ended += len(ends)

        if result.missing_session:
            logger.warning(
                f'history paths: {result.missing_session} backfill signals whose session is not in the history'
            )
        logger.info(
            f'history paths: wrote {result.written} of {len(signals)} backfill signals not frozen, froze '
            f'{result.frozen}, ended {result.ended}, kept {result.missing_session} whose session is not in the history'
        )
        return result
