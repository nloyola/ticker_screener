"""
Replaying the screener over the price history (HISTORY_DATABASE_FILE): every session the history can fully score,
for every listed ticker that can be scored on it, with its signals archived in the live database as backfill.
"""

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from app.analyzer import (
    BACKFILL,
    SIGNAL_STATUSES,
    Analyzer,
    Scored,
    rule_settings,
    subsector_medians,
    utc_now,
    with_status,
    with_subsector_medians,
)
from app.backfill import quiet, replayable_sessions
from app.container import Container
from app.markets import BENCHMARKS, MARKETS, UnsupportedTicker, market_of
from app.provenance import git_commit
from app.replay import TickerReplay, ticker_replay
from app.repositories.history_repo import BENCHMARK, TICKER

if TYPE_CHECKING:
    from app.backfill import Backfill
    from app.repositories.history_repo import HistoryRepo
    from app.services.sector_service import SectorService

logger = logging.getLogger(__name__)


@dataclass
class HistoryBackfillResult:
    """What a history backfill replayed and archived, and the listed tickers it could not replay."""

    sessions: int = 0
    first: date | None = None
    last: date | None = None
    tickers: int = 0
    signals: int = 0
    relinked: int = 0
    # listed tickers with no stored history, and those with an exchange suffix no market has
    no_history: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    seconds: float = 0.0


class HistoryBackfill:
    """
    Replays every session of the history database that a run could fully score (replayable_sessions in
    app/backfill.py, over the days the listed tickers' histories hold), for every ticker the sectors page lists with
    a history, and archives the signals as backfill, as Backfill does from price_data. The values are those
    Analyzer.score_as_of gives on the history for each session, against the history's benchmarks, but each
    indicator is computed once over the ticker's whole history (app/replay.py) rather than once per session.

    The universe is today's: each ticker in the subsector that lists it first, as the fetcher stores it, and only the
    tickers listed now. Tickers are replayed subsector by subsector, since the subsector comparison needs every
    ticker of the subsector scored on the session, so only one subsector's histories are held at a time.

    A signal already archived for the ticker, session and version, live or backfill, is left as it was, so a second
    backfill adds nothing. Each ticker's signals are archived oldest first, each linked to the ticker's previous one,
    and a signal archived before the backfill, on a session after one it archived, is relinked (Backfill.relink).
    """

    def __init__(self, container: Container, now: Callable[[], datetime] = utc_now) -> None:
        self.history_repo: HistoryRepo = container.history_repo
        self.sector_service: SectorService = container.sector_service
        self.analyzer: Analyzer = container.analyzer
        self.backfill: Backfill = container.backfill
        # the clock that decides which sessions are complete, timezone aware
        self.now = now

    def run(self) -> HistoryBackfillResult:
        """Replays the history, prints a line per subsector and a summary, and returns what it did."""
        started = time.monotonic()
        result = HistoryBackfillResult()
        universe = self._universe(result)
        stored = self.history_repo.coverage(TICKER)
        result.no_history = [ticker for ticker in universe if ticker not in stored]
        universe = {ticker: subsector for ticker, subsector in universe.items() if ticker in stored}

        sessions = replayable_sessions(self._market_dates(universe), self.now())
        result.sessions = len(sessions)
        if sessions:
            result.first, result.last = min(sessions), max(sessions)
        # each market's replayable sessions, as day ordinals
        replayable = {
            market: np.array([s.toordinal() for s, markets in sessions.items() if market in markets], dtype=np.int64)
            for market in MARKETS
        }
        benchmarks = {
            BENCHMARKS[market]: {p.date: p.close for p in self.history_repo.prices(BENCHMARK, BENCHMARKS[market])}
            for market in MARKETS
        }

        by_subsector: dict[int, list[str]] = {}
        for ticker, subsector_id in universe.items():
            by_subsector.setdefault(subsector_id, []).append(ticker)
        settings = json.dumps(rule_settings(), sort_keys=True)
        commit = git_commit()
        archived_tickers: set[str] = set()
        for done, (subsector_id, tickers) in enumerate(by_subsector.items(), start=1):
            signals = self._subsector_signals(subsector_id, tickers, replayable, benchmarks)
            result.tickers += len(tickers)
            stored_count = self._archive(signals, settings, commit, archived_tickers)
            result.signals += stored_count
            print(
                f'backfill-history {done}/{len(by_subsector)}: subsector {subsector_id}, '
                f'{len(tickers)} ticker{"" if len(tickers) == 1 else "s"}, '
                f'archived {stored_count} new signal{"" if stored_count == 1 else "s"}',
                flush=True,
            )

        result.relinked = self.backfill.relink(archived_tickers, self._dates)
        result.seconds = time.monotonic() - started
        return result

    def _universe(self, result: HistoryBackfillResult) -> dict[str, int]:
        """Every listed ticker of a known market with the subsector that lists it first, as the fetcher stores it."""
        universe: dict[str, int] = {}
        for sector in self.sector_service.get_all():
            for subsector in sector.subsectors:
                for ticker in subsector.tickers:
                    if ticker in universe or ticker in result.unsupported:
                        continue
                    try:
                        market_of(ticker)
                    except UnsupportedTicker:
                        result.unsupported.append(ticker)
                        continue
                    universe[ticker] = subsector.id
        return universe

    def _market_dates(self, universe: dict[str, int]) -> dict[str, pd.Index]:
        """Each market's session dates: the days any of its tickers in 'universe' has a price in the history."""
        return {
            market: pd.Index(self.history_repo.days(TICKER, [t for t in universe if market_of(t) == market]))
            for market in MARKETS
        }

    def _dates(self, ticker: str) -> pd.Index:
        return pd.Index([p.date for p in self.history_repo.prices(TICKER, ticker)])

    def _subsector_signals(
        self,
        subsector_id: int,
        tickers: list[str],
        replayable: dict[str, np.ndarray],
        benchmarks: dict[str, dict[date, float]],
    ) -> list[Scored]:
        """
        The signals of the subsector's tickers on every replayable session, oldest first: each ticker scored on
        each session of its market it can be scored on, compared with the median returns of the subsector's tickers
        scored on that session, and kept when its status makes it a signal.
        """
        replays: list[tuple[str, TickerReplay, np.ndarray]] = []
        # the rows meeting the core conditions, the only ones that can be signals: (replay index, row)
        candidates: list[tuple[int, int]] = []
        for ticker in tickers:
            market = market_of(ticker)
            replay = ticker_replay(ticker, subsector_id, self.history_repo.prices(TICKER, ticker))
            ordinals = np.array([d.toordinal() for d in replay.dates], dtype=np.int64)
            included = replay.scorable & np.isin(ordinals, replayable[market])
            replays.append((market, replay, ordinals))
            candidates.extend((len(replays) - 1, int(row)) for row in np.flatnonzero(included & replay.core))
        if not candidates:
            return []

        # the subsector's medians, only on the sessions a candidate needs them
        wanted = np.unique([replays[i][2][row] for i, row in candidates])
        returns: dict[int, list[tuple[int, float | None, float | None]]] = {}
        for market, replay, ordinals in replays:
            included = replay.scorable & np.isin(ordinals, replayable[market]) & np.isin(ordinals, wanted)
            for row in np.flatnonzero(included):
                r1, r3 = replay.return_1m[row], replay.return_3m[row]
                returns.setdefault(int(ordinals[row]), []).append(
                    (subsector_id, None if np.isnan(r1) else float(r1), None if np.isnan(r3) else float(r3))
                )
        medians = {day: subsector_medians(values) for day, values in returns.items()}

        signals: list[Scored] = []
        for i, row in candidates:
            market, replay, ordinals = replays[i]
            criteria = replay.criteria_at(row, benchmarks[BENCHMARKS[market]])
            criteria = with_status(with_subsector_medians(criteria, medians[int(ordinals[row])]))
            if criteria.status in SIGNAL_STATUSES:
                session = replay.dates[row]
                signals.append(Scored(criteria, market, session, float(replay.closes[row]), replay.dates[: row + 1]))
        signals.sort(key=lambda s: (s.session_date, s.criteria.ticker))
        return signals

    def _archive(self, signals: list[Scored], settings: str, commit: str | None, tickers: set[str]) -> int:
        """
        Archives 'signals', oldest first, one session at a time, so each links to the ticker's previous one; adds
        the tickers of those archived to 'tickers'. Returns how many it archived.
        """
        archived = 0
        by_session: dict[date, list[Scored]] = {}
        for s in signals:
            by_session.setdefault(s.session_date, []).append(s)
        for session_signals in by_session.values():
            with quiet('app.analyzer'):
                stored = self.analyzer.archive_signals(session_signals, None, settings, commit, source=BACKFILL)
            if stored:
                archived += stored
                tickers.update(s.criteria.ticker for s in session_signals)
        return archived


def summarize(result: HistoryBackfillResult) -> None:
    """Prints what a history backfill replayed and archived, and the listed tickers it could not replay."""
    sessions = f'{result.sessions} session{"" if result.sessions == 1 else "s"}'
    if result.first is not None:
        sessions += f' ({result.first} to {result.last})'
    print(
        f'backfill-history: replayed {sessions} of {result.tickers} ticker{"" if result.tickers == 1 else "s"}, '
        f'archived {result.signals} new signal{"" if result.signals == 1 else "s"}, '
        f'relinked {result.relinked} later signal{"" if result.relinked == 1 else "s"}, '
        f'in {result.seconds:.0f}s'
    )
    if result.no_history:
        print(f'backfill-history: no history, not replayed: {", ".join(result.no_history)}')
    if result.unsupported:
        print(f'backfill-history: unsupported exchange suffix, not replayed: {", ".join(result.unsupported)}')
