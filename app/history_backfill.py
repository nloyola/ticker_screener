"""
Replaying the screener over the price history (HISTORY_DATABASE_FILE): every session the history can fully score,
for every listed ticker that can be scored on it, with its signals archived in the live database as backfill.
"""

import json
import logging
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from app.analyzer import (
    BACKFILL,
    SCREENER_VERSION,
    Analyzer,
    Scored,
    archive_kind,
    describe_records,
    rule_settings,
    subsector_medians,
    utc_now,
    with_status,
    with_subsector_medians,
)
from app.backfill import quiet, replayable_sessions
from app.container import Container
from app.history_fetch import SHORT_HISTORY_SLACK, history_start
from app.history_paths import HistoryPathUpdate
from app.markets import BENCHMARKS, MARKET_CLOSES, MARKETS, US, UnsupportedTicker, market_of
from app.models import NEVER_SCORED, NO_HISTORY, SHORT_HISTORY, SIGNAL_KIND, UNSUPPORTED, BackfillRun
from app.provenance import git_commit
from app.replay import TickerReplay, ticker_replay
from app.repositories.history_repo import BENCHMARK, TICKER
from app.signal_outcome_updates import OutcomeUpdate

if TYPE_CHECKING:
    from app.backfill import Backfill
    from app.history_paths import HistoryPaths
    from app.repositories.backfill_run_repo import BackfillRunRepo
    from app.repositories.history_repo import HistoryRepo
    from app.services.sector_service import SectorService
    from app.signal_outcome_updates import SignalOutcomeUpdates

logger = logging.getLogger(__name__)

# the time zone of a run's universe date, the day the sectors page was read
UNIVERSE_ZONE = MARKET_CLOSES[US][0]


@dataclass
class HistoryBackfillResult:
    """
    What a history backfill replayed and archived, the listed tickers it could not replay or score, and the paths and
    outcomes it stored for the backfill signals. 'run' is the backfill_run row it stored.
    """

    sessions: int = 0
    first: date | None = None
    last: date | None = None
    tickers: int = 0
    signals: int = 0
    # the records archived of each kind (KINDS in app/models.py), the signals among them
    records: Counter[str] = field(default_factory=Counter)
    relinked: int = 0
    # listed tickers with no stored history, and those with an exchange suffix no market has
    no_history: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    # replayed tickers scored on no session, and those whose history starts late, with its first day
    never_scored: list[str] = field(default_factory=list)
    short_history: dict[str, date] = field(default_factory=dict)
    paths: HistoryPathUpdate = field(default_factory=HistoryPathUpdate)
    outcomes: OutcomeUpdate = field(default_factory=OutcomeUpdate)
    seconds: float = 0.0
    run: BackfillRun | None = None


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

    Then every backfill signal's path not yet frozen is written from the history (HistoryPaths), and the outcomes of
    every signal are stored from the paths (SignalOutcomeUpdates), as run-refresh stores them from its own.

    Last, the run's coverage and biases are stored in backfill_run: its sessions, universe date, ticker counts, the
    listed tickers it never scored and why, the tickers whose history starts late (more than SHORT_HISTORY_SLACK after
    the start the fetch asked for on the day it fetched them), and the price API their histories came from.
    """

    def __init__(self, container: Container, now: Callable[[], datetime] = utc_now) -> None:
        self.history_repo: HistoryRepo = container.history_repo
        self.sector_service: SectorService = container.sector_service
        self.analyzer: Analyzer = container.analyzer
        self.backfill: Backfill = container.backfill
        self.history_paths: HistoryPaths = container.history_paths
        self.signal_outcome_updates: SignalOutcomeUpdates = container.signal_outcome_updates
        self.backfill_run_repo: BackfillRunRepo = container.backfill_run_repo
        # the clock that decides which sessions are complete, timezone aware
        self.now = now

    def run(self) -> HistoryBackfillResult:
        """
        Replays the history, printing a line per subsector, stores the backfill signals' paths and outcomes and the
        run's coverage, and returns what it did.
        """
        started_at = self.now()
        started = time.monotonic()
        result = HistoryBackfillResult()
        universe = self._universe(result)
        listed = len(universe) + len(result.unsupported)
        stored = self.history_repo.coverage(TICKER)
        result.no_history = [ticker for ticker in universe if ticker not in stored]
        universe = {ticker: subsector for ticker, subsector in universe.items() if ticker in stored}
        fetches = self.history_repo.fetches(TICKER)
        for ticker in universe:
            fetched = fetches.get(ticker)
            if fetched is not None and stored[ticker].first > history_start(fetched[0]) + SHORT_HISTORY_SLACK:
                result.short_history[ticker] = stored[ticker].first

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
        scored: set[str] = set()
        for done, (subsector_id, tickers) in enumerate(by_subsector.items(), start=1):
            records = self._subsector_records(subsector_id, tickers, replayable, benchmarks, scored)
            result.tickers += len(tickers)
            stored = self._archive(records, settings, commit, archived_tickers)
            result.records.update(stored)
            result.signals += stored[SIGNAL_KIND]
            print(
                f'backfill-history {done}/{len(by_subsector)}: subsector {subsector_id}, '
                f'{len(tickers)} ticker{"" if len(tickers) == 1 else "s"}, archived {describe_records(stored)}',
                flush=True,
            )

        result.relinked = self.backfill.relink(archived_tickers, self._dates)
        result.paths = self.history_paths.update(self.now())
        result.outcomes = self.signal_outcome_updates.update()
        result.never_scored = [ticker for ticker in universe if ticker not in scored]
        price_api = ','.join(sorted({fetches[ticker][1] for ticker in universe if ticker in fetches})) or None
        result.run = self._store_run(result, started_at, listed, len(scored), price_api, commit)
        result.seconds = time.monotonic() - started
        return result

    def _store_run(
        self,
        result: HistoryBackfillResult,
        started_at: datetime,
        listed: int,
        scored: int,
        price_api: str | None,
        commit: str | None,
    ) -> BackfillRun:
        """Stores the run's coverage and biases in backfill_run and returns the row stored."""
        left_out = {ticker: NO_HISTORY for ticker in result.no_history}
        left_out.update({ticker: UNSUPPORTED for ticker in result.unsupported})
        for ticker in result.never_scored:
            left_out[ticker] = SHORT_HISTORY if ticker in result.short_history else NEVER_SCORED
        run = BackfillRun(
            started_at=started_at,
            finished_at=self.now(),
            universe_date=started_at.astimezone(UNIVERSE_ZONE).date(),
            first_session=result.first,
            last_session=result.last,
            sessions=result.sessions,
            price_api=price_api,
            tickers_listed=listed,
            tickers_replayed=result.tickers,
            tickers_scored=scored,
            left_out=left_out,
            short_history=result.short_history,
            signals=result.signals,
            relinked=result.relinked,
            screener_version=SCREENER_VERSION,
            git_commit=commit,
        )
        return replace(run, id=self.backfill_run_repo.add(run))

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

    def _subsector_records(
        self,
        subsector_id: int,
        tickers: list[str],
        replayable: dict[str, np.ndarray],
        benchmarks: dict[str, dict[date, float]],
        scored: set[str],
    ) -> list[Scored]:
        """
        The sessions of the subsector's tickers the signal archive keeps (archive_kind), on every replayable session,
        oldest first; adds the tickers scored on at least one session to 'scored'. Each ticker is scored on each
        session of its market it can be scored on, compared with the median returns of the subsector's tickers scored
        on that session, and kept when its status and values make it a record of some kind.
        """
        replays: list[tuple[str, TickerReplay, np.ndarray]] = []
        # the sessions that can be archived (TickerReplay.archivable): (replay index, row)
        candidates: list[tuple[int, int]] = []
        for ticker in tickers:
            market = market_of(ticker)
            replay = ticker_replay(ticker, subsector_id, self.history_repo.prices(TICKER, ticker))
            ordinals = np.array([d.toordinal() for d in replay.dates], dtype=np.int64)
            included = replay.scorable & np.isin(ordinals, replayable[market])
            if included.any():
                scored.add(ticker)
            replays.append((market, replay, ordinals))
            rows = np.flatnonzero(included & replay.archivable())
            candidates.extend((len(replays) - 1, int(row)) for row in rows)
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

        records: list[Scored] = []
        for i, row in candidates:
            market, replay, ordinals = replays[i]
            criteria = replay.criteria_at(row, benchmarks[BENCHMARKS[market]])
            criteria = with_status(with_subsector_medians(criteria, medians[int(ordinals[row])]))
            if archive_kind(criteria) is not None:
                session = replay.dates[row]
                records.append(Scored(criteria, market, session, float(replay.closes[row]), replay.dates[: row + 1]))
        records.sort(key=lambda s: (s.session_date, s.criteria.ticker))
        return records

    def _archive(self, records: list[Scored], settings: str, commit: str | None, tickers: set[str]) -> Counter[str]:
        """
        Archives 'records', oldest first, one session at a time, so each signal links to the ticker's previous one
        and each continuation to its breakout, archived on an earlier session; adds the tickers of the signals
        archived to 'tickers'. Returns how many it archived of each kind.
        """
        archived: Counter[str] = Counter()
        non_primary = self.analyzer.issuer_repo.non_primary()
        exchanges = self.analyzer.security_state_repo.exchanges()
        by_session: dict[date, list[Scored]] = {}
        for s in records:
            by_session.setdefault(s.session_date, []).append(s)
        for session_records in by_session.values():
            with quiet('app.analyzer'):
                stored = self.analyzer.archive_signals(
                    session_records,
                    None,
                    settings,
                    commit,
                    source=BACKFILL,
                    non_primary=non_primary,
                    exchanges=exchanges,
                )
            archived.update(stored)
            if stored[SIGNAL_KIND]:
                tickers.update(s.criteria.ticker for s in session_records if archive_kind(s.criteria) == SIGNAL_KIND)
        return archived


def summarize(result: HistoryBackfillResult) -> None:
    """
    Prints what a history backfill replayed and archived, the paths and outcomes it stored, and the listed tickers
    it could not replay.
    """
    sessions = f'{result.sessions} session{"" if result.sessions == 1 else "s"}'
    if result.first is not None:
        sessions += f' ({result.first} to {result.last})'
    print(
        f'backfill-history: replayed {sessions} of {result.tickers} ticker{"" if result.tickers == 1 else "s"}, '
        f'archived {describe_records(result.records)}, '
        f'relinked {result.relinked} later signal{"" if result.relinked == 1 else "s"}, '
        f'in {result.seconds:.0f}s'
    )
    paths, outcomes = result.paths, result.outcomes
    print(
        f'backfill-history: wrote {paths.written} signal path{"" if paths.written == 1 else "s"} from the history, '
        f"froze {paths.frozen}, ended {paths.ended} whose ticker's history stops early; "
        f'completed {outcomes.completed} horizon{"" if outcomes.completed == 1 else "s"}, '
        f'decided {outcomes.decided} breakout failure{"" if outcomes.decided == 1 else "s"}'
    )
    if result.no_history:
        print(f'backfill-history: no history, not replayed: {", ".join(result.no_history)}')
    if result.unsupported:
        print(f'backfill-history: unsupported exchange suffix, not replayed: {", ".join(result.unsupported)}')
    if result.never_scored:
        print(f'backfill-history: scored on no session: {", ".join(result.never_scored)}')
    if result.short_history:
        late = ', '.join(f'{ticker} ({first})' for ticker, first in result.short_history.items())
        print(f'backfill-history: history starts late: {late}')
    if result.run is not None:
        print(f'backfill-history: recorded as backfill run {result.run.id}')
