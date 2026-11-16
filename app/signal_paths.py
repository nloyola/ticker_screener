"""
Each archived signal's own price path (table signal_price), kept after every run-refresh.

price_data is rewritten by every run, because the price APIs restate adjusted prices after a split or dividend, and
the dashboard deletes a ticker's rows when it removes the ticker. So each signal keeps a copy of its ticker's prices
from its signal session onward, which the outcomes are computed from: see SignalPaths.update.
"""

import logging
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime

from app.container import Container
from app.markets import BENCHMARKS, last_complete_session
from app.models import PathSession, PriceData
from app.price_apis import DailyPrice
from app.repositories.benchmark_price_repo import BenchmarkPriceRepo
from app.repositories.price_data_repo import PriceDataRepo
from app.repositories.signal_path_repo import SignalPathRepo, TrackedSignal

logger = logging.getLogger(__name__)

# a path runs from the signal session (index 0) to the PATH_SESSIONS-th session after it, the longest horizon, and
# is frozen once it reaches it
PATH_SESSIONS = 120
# the horizons, in sessions after the signal session, a signal's performance is measured at (signal_outcome)
HORIZONS = (5, 10, 20, 40, 60, 120)
# the reason a signal is ended when the run gives none for its ticker: no price_data row is left for it, as after the
# dashboard removed it from every subsector
REMOVED = 'removed'


def build_path(
    session_date: date,
    rows: Sequence[PriceData] | Sequence[DailyPrice],
    benchmark: Mapping[date, tuple[float | None, float]],
    last_session: date,
) -> list[PathSession] | None:
    """
    The path of a signal on 'session_date' from its ticker's price rows (price_data's or the history's, oldest first)
    and its benchmark's open and close by date: the rows from the signal session up to PATH_SESSIONS sessions after
    it, leaving out rows dated after 'last_session', the market's last complete session, and rows with no close.
    None when the rows do not hold the signal session, so the whole path cannot be written from them.
    """
    usable = [r for r in rows if r.close is not None and r.date <= last_session]
    start = next((i for i, r in enumerate(usable) if r.date == session_date), None)
    if start is None:
        return None
    path: list[PathSession] = []
    for index, r in enumerate(usable[start : start + PATH_SESSIONS + 1]):
        bench_open, bench_close = benchmark.get(r.date, (None, None))
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
            )
        )
    return path


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
        After a run that fetched 'markets', updates the path of each of their signals not yet frozen. A signal of
        another market is left as it is: this run did not refresh its prices.

        A signal whose ticker the run flagged ('flagged' maps a ticker to its reason: not_found, no_prices, stale or
        frozen), or whose ticker has no price_data row left (REMOVED), keeps its path as it is, and each horizon its
        path does not reach is marked ended, with that reason and the path's last session; with no path yet, the
        last session is the signal session.

        Any other signal's path is rewritten whole from price_data and benchmark_price as this run left them, so every
        value in it shares one adjustment basis, from the signal session to its market's last complete session for a
        run started at 'started_at', or PATH_SESSIONS sessions after the signal session, when the path is frozen,
        unless benchmark_may_fill says a later download can still fill a benchmark price its outcomes need. Its
        ended horizons, if an earlier run ended it, are deleted, since its ticker has prices again. A signal whose
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
                path = build_path(s.session_date, rows, benchmarks[s.market], last_sessions[s.market])
                if path is None:
                    logger.warning(
                        f'signal {s.id} ({ticker} on {s.session_date}): its session is not in price_data, path kept'
                    )
                    result.missing_session += 1
                    continue
                frozen = path[-1].session_index == PATH_SESSIONS and not benchmark_may_fill(
                    path, newest_benchmark[s.market]
                )
                self.signal_path_repo.write(s.id, path, frozen)
                result.rewritten += 1
                result.frozen += int(frozen)

        logger.info(
            f'signal paths: rewrote {result.rewritten} of {len(signals)} not frozen, froze {result.frozen}, '
            f'ended {result.ended}, kept {result.missing_session} whose session is not in price_data'
        )
        return result

    def _end(self, s: TrackedSignal, reason: str) -> None:
        last_index = s.last_index if s.last_index is not None else 0
        last_date = s.last_date if s.last_date is not None else s.session_date
        self.signal_path_repo.end(s.id, [h for h in HORIZONS if h > last_index], reason, last_date)
