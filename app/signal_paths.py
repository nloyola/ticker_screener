"""
Each archived record's own price path (table signal_price), kept after every run-refresh.

price_data is rewritten by every run, because the price APIs restate adjusted prices after a split or dividend, and
the dashboard deletes a ticker's rows when it removes the ticker. So each signal, near-miss and watchlist record keeps
a copy of its ticker's prices from its session onward, as long as its kind's longest horizon (PATH_LENGTHS), which the
outcomes are computed from: see SignalPaths.update. A continuation has no path of its own: its outcomes are measured
on its breakout's path from the continuation's session on (continuation_horizons).
"""

import logging
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING

from app.analyzer import VOLUME_AVG_DAYS
from app.container import Container
from app.markets import BENCHMARKS, last_complete_session
from app.models import NEAR_MISS_KIND, SIGNAL_KIND, WATCHLIST_KIND, PathSession, PriceData
from app.price_apis import DailyPrice
from app.repositories.benchmark_price_repo import BenchmarkPriceRepo
from app.repositories.price_data_repo import PriceDataRepo

if TYPE_CHECKING:
    # under TYPE_CHECKING: the repository reads this module's horizons
    from app.repositories.signal_path_repo import SignalPathRepo, TrackedSignal

logger = logging.getLogger(__name__)

# a signal's path runs from its session (index 0) to the PATH_SESSIONS-th session after it, the longest horizon, and
# is frozen once it reaches it
PATH_SESSIONS = 120
# the horizons, in sessions after the record's session, a record's performance is measured at (signal_outcome)
HORIZONS = (5, 10, 20, 40, 60, 120)
# each kind's path length, its longest horizon: a near-miss is compared with the signals over 60 sessions and a
# watchlist record over 20, which keeps the archive's growth to what the comparisons need (issue #65, Step 1)
PATH_LENGTHS = {SIGNAL_KIND: PATH_SESSIONS, NEAR_MISS_KIND: 60, WATCHLIST_KIND: 20}


def kind_horizons(kind: str) -> tuple[int, ...]:
    """The horizons a record of 'kind' with a path of its own is measured at: those its path reaches."""
    return tuple(h for h in HORIZONS if h <= PATH_LENGTHS[kind])


def continuation_horizons(offset: int) -> tuple[int, ...]:
    """
    The horizons of a continuation 'offset' sessions after its breakout: those its breakout's path reaches from it,
    offset + h at most PATH_SESSIONS.
    """
    return tuple(h for h in HORIZONS if offset + h <= PATH_SESSIONS)


# the reason a signal is ended when the run gives none for its ticker: no price_data row is left for it, as after the
# dashboard removed it from every subsector
REMOVED = 'removed'


def build_path(
    session_date: date,
    rows: Sequence[PriceData] | Sequence[DailyPrice],
    benchmark: Mapping[date, tuple[float | None, float]],
    last_session: date,
    length: int = PATH_SESSIONS,
) -> list[PathSession] | None:
    """
    The path of a record on 'session_date' from its ticker's price rows (price_data's or the history's, oldest first)
    and its benchmark's open and close by date: the rows from the record's session up to 'length' sessions after it,
    its kind's PATH_LENGTHS, leaving out rows dated after 'last_session', the market's last complete session, and rows
    with no close. Each session carries its prior average volume (prior_average_volume). None when the rows do not
    hold the record's session, so the whole path cannot be written from them.
    """
    usable = [r for r in rows if r.close is not None and r.date <= last_session]
    start = next((i for i, r in enumerate(usable) if r.date == session_date), None)
    if start is None:
        return None
    path: list[PathSession] = []
    for index, r in enumerate(usable[start : start + length + 1]):
        bench_open, bench_close = benchmark.get(r.date, (None, None))
        before = usable[start + index - VOLUME_AVG_DAYS : start + index] if start + index >= VOLUME_AVG_DAYS else []
        path.append(
            PathSession(
                session_index=index,
                session_date=r.date,
                open=r.open,
                high=r.high,
                low=r.low,
                close=r.close,
                volume=r.volume,
                benchmark_open=bench_open,
                benchmark_close=bench_close,
                avg_volume_20d=prior_average_volume([b.volume for b in before]),
            )
        )
    return path


def prior_average_volume(volumes: Sequence[int | None]) -> float | None:
    """
    The mean of 'volumes', the VOLUME_AVG_DAYS sessions before a path session, a missing one skipped as analyze
    skips it: None when there are fewer sessions than that, as at the start of the ticker's rows, or none traded a
    known volume.
    """
    known = [v for v in volumes if v is not None]
    if len(volumes) < VOLUME_AVG_DAYS or not known:
        return None
    return sum(known) / len(known)


def benchmark_may_fill(path: Sequence[PathSession], newest_benchmark: date | None) -> bool:
    """
    Whether 'path' lacks a benchmark price its outcomes need (the entry session's open or close, or a horizon exit's
    close) that a later benchmark download can still bring: the benchmark's stored closes, whose newest date is
    'newest_benchmark' (None when it has none), end before the path's last session, as when the run's benchmark
    download failed and kept the closes it had. A gap inside the benchmark's stored range is one no download fills.
    """
    if newest_benchmark is not None and newest_benchmark >= path[-1].session_date:
        return False
    entry = path[1] if len(path) > 1 else None
    needed = (
        [] if entry is None else [entry.benchmark_open if entry.benchmark_open is not None else entry.benchmark_close]
    )
    needed += [path[h].benchmark_close for h in HORIZONS if h < len(path)]
    return any(v is None for v in needed)


@dataclass
class PathUpdate:
    """What one update did: paths rewritten, of which frozen, signals ended, and signals whose session is gone."""

    rewritten: int = 0
    frozen: int = 0
    ended: int = 0
    missing_session: int = 0


class SignalPaths:
    def __init__(self, container: Container) -> None:
        self.signal_path_repo: SignalPathRepo = container.signal_path_repo
        self.price_data_repo: PriceDataRepo = container.price_data_repo
        self.benchmark_price_repo: BenchmarkPriceRepo = container.benchmark_price_repo

    def update(self, markets: Collection[str], flagged: Mapping[str, str], started_at: datetime) -> PathUpdate:
        """
        After a run that fetched 'markets', updates the path of each of their records not yet frozen, every kind but
        a continuation, which has none of its own: each kind's path is as long as its PATH_LENGTHS, and "signal" below
        means a record of any of them. A signal of another market is left as it is: this run did not refresh its
        prices.

        A signal whose ticker the run flagged ('flagged' maps a ticker to its reason: not_found, no_prices, stale or
        frozen), or whose ticker has no price_data row left (REMOVED), keeps its path as it is, and each horizon its
        path does not reach is marked ended, with that reason and the path's last session; with no path yet, the
        last session is the signal session. A continuation of a signal so ended has the horizons its breakout's path
        does not reach ended too.

        Any other signal's path is rewritten whole from price_data and benchmark_price as this run left them, so every
        value in it shares one adjustment basis, from the signal session to its market's last complete session for a
        run started at 'started_at', or its kind's PATH_LENGTHS after the signal session, when the path is frozen,
        unless benchmark_may_fill says a later download can still fill a benchmark price its outcomes need. Its
        ended horizons, and its continuations', if an earlier run ended it, are deleted, since its ticker has prices
        again. A signal whose
        session is no longer in price_data keeps its path, with a warning.

        Logs one summary line and returns the counts.
        """
        result = PathUpdate()
        signals = self.signal_path_repo.unfrozen(markets)
        by_ticker: dict[str, list[TrackedSignal]] = {}
        for s in signals:
            by_ticker.setdefault(s.ticker, []).append(s)
        benchmarks = {market: self.benchmark_price_repo.opens_and_closes(BENCHMARKS[market]) for market in markets}
        newest_benchmark = {market: max(benchmarks[market], default=None) for market in markets}
        last_sessions = {market: last_complete_session(market, started_at) for market in markets}

        for ticker, ticker_signals in by_ticker.items():
            rows = [] if ticker in flagged else list(self.price_data_repo.all_for_ticker(ticker))
            if ticker in flagged or not rows:
                reason = flagged.get(ticker, REMOVED)
                for s in ticker_signals:
                    self._end(s, reason)
                    result.ended += 1
                continue
            for s in ticker_signals:
                length = PATH_LENGTHS[s.kind]
                path = build_path(s.session_date, rows, benchmarks[s.market], last_sessions[s.market], length)
                if path is None:
                    logger.warning(
                        f'signal {s.id} ({ticker} on {s.session_date}): its session is not in price_data, path kept'
                    )
                    result.missing_session += 1
                    continue
                frozen = path[-1].session_index == length and not benchmark_may_fill(path, newest_benchmark[s.market])
                self.signal_path_repo.write(s.id, path, frozen)
                result.rewritten += 1
                result.frozen += int(frozen)

        logger.info(
            f'signal paths: rewrote {result.rewritten} of {len(signals)} not frozen, froze {result.frozen}, '
            f'ended {result.ended}, kept {result.missing_session} whose session is not in price_data'
        )
        return result

    def _end(self, s: 'TrackedSignal', reason: str) -> None:
        last_index = s.last_index if s.last_index is not None else 0
        last_date = s.last_date if s.last_date is not None else s.session_date
        self.signal_path_repo.end_all([(s.id, ended_horizons(s.kind, last_index), reason, last_date)])
        self.signal_path_repo.end_continuations([(s.id, last_index, reason, last_date)])


def ended_horizons(kind: str, last_index: int) -> list[int]:
    """The horizons of a record of 'kind' whose path stops at 'last_index' that the path does not reach."""
    return [h for h in kind_horizons(kind) if h > last_index]
