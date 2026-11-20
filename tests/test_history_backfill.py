"""
backfill-history (app/history_backfill.py) replays the screener over the price history, with each indicator computed
once over a ticker's whole history (app/replay.py), and archives the same signals, with the same values, as
Analyzer.score_as_of gives on the history session by session; then it writes each backfill signal's path from the
history (app/history_paths.py) and stores its outcomes from it.
"""

import argparse
import os
import sqlite3
from datetime import date, datetime, time, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from app.analyzer import BREAKOUT_SESSION, SCREENER_VERSION, SIGNAL_STATUSES, Analyzer
from app.backfill import Backfill, market_dates, replayable_sessions
from app.commands.backfill_history_cmd import BackfillHistoryCmd
from app.db import init_db
from app.history_backfill import HistoryBackfill
from app.history_paths import HISTORY_ENDED, HistoryPaths
from app.markets import BENCHMARKS, CANADA, US
from app.models import NEVER_SCORED, NO_HISTORY, SHORT_HISTORY, UNSUPPORTED
from app.price_apis import DailyPrice
from app.replay import flat_days, run_lengths, unscored
from app.repositories.backfill_run_repo import BackfillRunRepo
from app.repositories.benchmark_price_repo import BenchmarkPriceRepo
from app.repositories.history_repo import BENCHMARK, TICKER, HistoryRepo
from app.repositories.price_data_repo import PriceDataRepo
from app.repositories.screened_session_repo import ScreenedSessionRepo
from app.repositories.sector_repo import SectorRepo
from app.repositories.signal_outcome_repo import SignalOutcomeRepo
from app.repositories.signal_path_repo import SignalPathRepo
from app.repositories.signal_repo import _ARCHIVED, _ARCHIVED_AT, SignalRepo
from app.repositories.stock_criteria_repo import StockCriteriaRepo, row_values
from app.repositories.subsector_repo import SubsectorRepo
from app.screener_lock import screener_lock
from app.services.sector_service import SectorService
from app.signal_outcome_updates import SignalOutcomeUpdates
from app.signal_outcomes import COST_BPS
from app.signal_paths import HORIZONS, PATH_SESSIONS

NEW_YORK = ZoneInfo('America/New_York')
# a Monday
FIRST = date(2024, 1, 1)
SESSIONS = 420
FETCHED = date(2026, 11, 16)


def weekdays(first: date, n: int) -> list[date]:
    days: list[date] = []
    day = first
    while len(days) < n:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


DAYS = weekdays(FIRST, SESSIONS)
# a US holiday Canada trades on, and a Canadian one the US trades on
US_DAYS = [d for i, d in enumerate(DAYS) if i != 300]
CANADA_DAYS = [d for i, d in enumerate(DAYS) if i != 310]
EVENING = datetime.combine(DAYS[-1], time(20), tzinfo=NEW_YORK)

# subsector id -> its tickers, as the sectors page lists them: AAA is listed twice, and counts in subsector 1, the
# first to list it; NOHIST has no history, BAD.XX no market, and HHH is the only ticker of its subsector
LISTED = {
    1: ['AAA', 'BBB', 'CCC', 'DDD', 'RY.TO', 'NOHIST', 'BAD.XX'],
    2: ['EEE', 'FFF', 'GGG', 'SHOP.TO', 'AAA', 'TD.TO'],
    3: ['HHH'],
}


SPIKE_SHARE = 0.12
SPIKE_JUMP = 0.03


def walk(
    rng: np.random.Generator, days: list[date], start: float = 50.0, shares: float = 1_000_000.0
) -> list[DailyPrice]:
    """
    A rising random walk with a volume spike on about one day in eight, each spike day's return 3% higher and every
    other day's lower by as much in all, so breakouts clear the minimum margin and their runs end. It trades about
    'shares' a day, well over the liquidity floor at the default.
    """
    returns = rng.normal(0.002, 0.02, len(days))
    spikes = rng.random(len(days)) < SPIKE_SHARE
    closes = start * np.exp(
        np.cumsum(returns + np.where(spikes, SPIKE_JUMP, -SPIKE_JUMP * SPIKE_SHARE / (1 - SPIKE_SHARE)))
    )
    volumes = (shares * rng.lognormal(0, 0.3, len(days)) * np.where(spikes, 3.5, 1.0)).astype(int) + 1
    return [
        DailyPrice(
            date=day,
            close=round(float(close), 4),
            volume=int(volume),
            high=round(float(close * (1 + rng.uniform(0, 0.02))), 4),
            low=round(float(close * (1 - rng.uniform(0, 0.02))), 4),
            open=round(float(close), 4),
        )
        for day, close, volume in zip(days, closes, volumes, strict=True)
    ]


def history_prices() -> dict[str, list[DailyPrice]]:
    rng = np.random.default_rng(50)
    prices = {ticker: walk(rng, US_DAYS) for ticker in ('AAA', 'BBB', 'CCC')}
    # EEE trades about 10,000 shares a day, under the $5M liquidity floor
    prices['EEE'] = walk(rng, US_DAYS, shares=10_000.0)
    prices['FFF'] = walk(rng, US_DAYS)
    prices['RY.TO'] = walk(rng, CANADA_DAYS)
    prices['SHOP.TO'] = walk(rng, CANADA_DAYS)
    prices['TD.TO'] = walk(rng, CANADA_DAYS)
    # DDD misses three sessions, stale on each of them
    prices['DDD'] = [p for i, p in enumerate(walk(rng, US_DAYS)) if i not in (280, 330, 331)]
    # GGG listed 120 sessions late
    prices['GGG'] = walk(rng, US_DAYS[120:])
    # HHH freezes for 12 sessions, trades again, then stops trading 3 sessions before the end, its last close padded
    hhh = walk(rng, US_DAYS)
    frozen = [
        DailyPrice(p.date, hhh[289].close, 0, hhh[289].close, hhh[289].close, hhh[289].close) for p in hhh[290:302]
    ]
    last = hhh[-4]
    padded = [DailyPrice(p.date, last.close, 0, last.close, last.close, last.close) for p in hhh[-3:]]
    prices['HHH'] = hhh[:290] + frozen + hhh[302:-3] + padded
    # history of a ticker no longer listed
    prices['ZZZ'] = walk(rng, US_DAYS)
    return prices


@pytest.fixture
def path(tmp_path) -> str:
    path = str(tmp_path / 'stock_data.db')
    init_db(path)
    with sqlite3.connect(path) as con:
        con.execute("INSERT INTO sector (id, sector) VALUES (1, 'Tech')")
        for subsector_id, tickers in LISTED.items():
            con.execute(
                'INSERT INTO subsector (id, sector_id, subsector, tickers) VALUES (?, 1, ?, ?)',
                (subsector_id, f'sub{subsector_id}', ','.join(tickers)),
            )
    return path


@pytest.fixture
def prices() -> dict[str, list[DailyPrice]]:
    return history_prices()


@pytest.fixture
def benchmarks() -> dict[str, list[DailyPrice]]:
    rng = np.random.default_rng(7)
    return {BENCHMARKS[US]: walk(rng, US_DAYS, 400.0), BENCHMARKS[CANADA]: walk(rng, CANADA_DAYS, 30.0)}


@pytest.fixture
def container(path, tmp_path, prices, benchmarks) -> SimpleNamespace:
    container = SimpleNamespace(db_path=path, history_db_path=str(tmp_path / 'history.db'))
    container.history_repo = HistoryRepo(container)
    for ticker, rows in prices.items():
        container.history_repo.store(TICKER, ticker, rows, FETCHED, 'eodhd')
    for ticker, rows in benchmarks.items():
        container.history_repo.store(BENCHMARK, ticker, rows, FETCHED, 'eodhd')
    container.sector_repo = SectorRepo(container)
    container.subsector_repo = SubsectorRepo(container)
    container.sector_service = SectorService(container)
    container.price_data_repo = PriceDataRepo(container)
    container.benchmark_price_repo = BenchmarkPriceRepo(container)
    container.stock_criteria_repo = StockCriteriaRepo(container)
    container.screened_session_repo = ScreenedSessionRepo(container)
    container.signal_repo = SignalRepo(container)
    container.analyzer = Analyzer(container, now=lambda: EVENING)
    container.backfill = Backfill(container, now=lambda: EVENING)
    container.signal_path_repo = SignalPathRepo(container)
    container.signal_outcome_repo = SignalOutcomeRepo(container)
    container.signal_outcome_updates = SignalOutcomeUpdates(container)
    container.history_paths = HistoryPaths(container)
    container.backfill_run_repo = BackfillRunRepo(container)
    container.history_backfill = HistoryBackfill(container, now=lambda: EVENING)
    return container


def signals(path: str) -> list[dict]:
    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        return [dict(row) for row in con.execute('SELECT * FROM signal ORDER BY ticker, session_date, id')]


def as_of_frame(prices: dict[str, list[DailyPrice]]) -> pd.DataFrame:
    """The listed tickers' histories as Analyzer.load_price_data frames price_data, each in its first subsector."""
    subsector_of: dict[str, int] = {}
    for subsector_id, tickers in LISTED.items():
        for ticker in tickers:
            subsector_of.setdefault(ticker, subsector_id)
    df = pd.DataFrame(
        [
            {
                'subsector_id': subsector_of[ticker],
                'ticker': ticker,
                'date': p.date,
                'close': p.close,
                'high': p.high,
                'low': p.low,
                'volume': p.volume,
            }
            for ticker, rows in prices.items()
            if ticker in subsector_of
            for p in rows
        ]
    )
    df.set_index(['ticker', 'date'], inplace=True)
    df.sort_index(inplace=True)
    df['volume'] = df['volume'].astype('Int64')
    return df[['subsector_id', 'close', 'high', 'low', 'volume']]


def test_backfill_archives_the_signals_score_as_of_gives_on_every_session(path, container, prices, benchmarks, capsys):
    result = container.history_backfill.run()

    frame = as_of_frame(prices)
    closes = {ticker: {p.date: p.close for p in rows} for ticker, rows in benchmarks.items()}
    sessions = replayable_sessions(market_dates(frame), EVENING)
    expected = {}
    floored = 0
    for session, markets in sessions.items():
        for s in container.analyzer.score_as_of(session, frame, markets, closes):
            sc = s.criteria
            floored += sc.ticker == 'EEE' and sc.breakout_phase == BREAKOUT_SESSION and sc.status is None
            if s.criteria.status in SIGNAL_STATUSES:
                values = row_values(s.criteria, session)
                expected[s.criteria.ticker, session.isoformat()] = (s.close, *(values[i] for i in _ARCHIVED_AT))

    archived = signals(path)
    actual = {(s['ticker'], s['session_date']): (s['close'], *(s[c] for c in _ARCHIVED)) for s in archived}
    assert actual == expected
    assert {s['source'] for s in archived} == {'backfill'}
    # the run covers every case: both kinds of signal, in both markets, with the subsector comparison filled in
    assert {s['status'] for s in archived} == set(SIGNAL_STATUSES)
    assert {s['market'] for s in archived} == {US, CANADA}
    assert any(s['rs_sector_1m'] is not None for s in archived)
    assert any(s['ema_21w'] is not None for s in archived)
    assert 'ZZZ' not in {s['ticker'] for s in archived}
    # EEE breaks out, but below the liquidity floor: never archived
    assert floored > 0
    assert 'EEE' not in {s['ticker'] for s in archived}

    assert (result.sessions, result.first, result.last) == (len(sessions), min(sessions), max(sessions))
    assert result.tickers == 11
    assert result.signals == len(archived)
    assert (result.no_history, result.unsupported) == (['NOHIST'], ['BAD.XX'])

    # each ticker's signals link to the one before it
    for ticker in {s['ticker'] for s in archived}:
        own = [s for s in archived if s['ticker'] == ticker]
        assert [s['previous_signal_id'] for s in own] == [None, *(s['id'] for s in own[:-1])]


def test_a_second_backfill_adds_nothing(path, container):
    first = container.history_backfill.run()
    archived = signals(path)
    assert first.signals > 0

    again = container.history_backfill.run()

    assert (again.signals, again.relinked) == (0, 0)
    assert signals(path) == archived


def test_the_unscored_rows_are_those_a_run_skips():
    closes = np.array([1.0, 2.0, 2.0, 2.0, 3.0, 3.0])
    volumes = np.array([5, 5, 0, 0, 0, 0])
    assert run_lengths(np.array([True, False, True, True])).tolist() == [1, 0, 1, 2]
    # the second day traded: its close is not part of the stretch of the still days after it
    assert flat_days(closes, volumes).tolist() == [0, 0, 1, 2, 1, 2]
    assert not unscored(closes, volumes).any()


def test_command_exits_tempfail_while_a_refresh_holds_the_lock(path, container, capsys):
    with screener_lock(path), pytest.raises(SystemExit) as exit_info:
        BackfillHistoryCmd().handle(argparse.Namespace(), container)
    assert exit_info.value.code == os.EX_TEMPFAIL
    assert 'a refresh holds the lock' in capsys.readouterr().err


def test_command_prints_a_summary(container, capsys):
    BackfillHistoryCmd().handle(argparse.Namespace(), container)

    out = capsys.readouterr().out
    assert 'backfill-history: replayed' in out
    assert 'no history, not replayed: NOHIST' in out
    assert 'signal paths from the history' in out
    assert 'history starts late: AAA (2024-01-01)' in out
    assert 'recorded as backfill run 1' in out


def rows_of(path: str, sql: str, *params) -> list[tuple]:
    with sqlite3.connect(path) as con:
        return con.execute(sql, params).fetchall()


def test_every_backfill_signal_120_sessions_old_has_six_completed_outcomes(path, container, prices):
    result = container.history_backfill.run()

    archived = signals(path)
    assert result.paths.written == len(archived)
    old = 0
    for s in archived:
        days = [p.date.isoformat() for p in prices[s['ticker']]]
        after = len(days) - 1 - days.index(s['session_date'])
        complete = rows_of(
            path, "SELECT horizon FROM signal_outcome WHERE signal_id = ? AND status = 'complete' ORDER BY 1", s['id']
        )
        frozen = rows_of(path, 'SELECT path_frozen_at FROM signal_track WHERE signal_id = ?', s['id'])[0][0]
        stored = rows_of(path, 'SELECT session_date FROM signal_price WHERE signal_id = ? ORDER BY 1', s['id'])
        assert [d for (d,) in stored] == days[days.index(s['session_date']) :][: PATH_SESSIONS + 1]
        if after >= PATH_SESSIONS:
            old += 1
            assert [h for (h,) in complete] == list(HORIZONS)
            assert frozen is not None
        else:
            # a path reaching the history's last session is left for run-refresh to carry on, nothing ended
            assert [h for (h,) in complete] == [h for h in HORIZONS if h <= after]
            assert frozen is None
    assert old > 0
    assert old < len(archived)
    assert rows_of(path, "SELECT COUNT(*) FROM signal_outcome WHERE status = 'ended'") == [(0,)]
    assert (
        result.outcomes.completed
        == rows_of(path, "SELECT COUNT(*) FROM signal_outcome WHERE status = 'complete'")[0][0]
    )


def test_a_spot_checked_signals_outcomes_match_its_prices_in_the_history(path, container, prices, benchmarks):
    container.history_backfill.run()

    # the oldest signal, with every horizon in its path, of GGG, listed 120 sessions late
    s = min(signals(path), key=lambda s: s['session_date'])
    assert (s['ticker'], s['session_date']) == ('GGG', '2024-12-19')
    rows = prices['GGG']
    start = [p.date.isoformat() for p in rows].index(s['session_date'])
    bench = {p.date: p for p in benchmarks[BENCHMARKS[US]]}
    entry = rows[start + 1]
    for horizon in HORIZONS:
        exit_ = rows[start + horizon]
        held = rows[start + 1 : start + horizon + 1]
        gross = (exit_.close / entry.open - 1) * 100
        net = gross - COST_BPS / 100
        benchmark = (bench[exit_.date].close / bench[entry.date].open - 1) * 100
        expected = (
            entry.date.isoformat(),
            entry.open,
            exit_.date.isoformat(),
            exit_.close,
            gross,
            net,
            benchmark,
            net - benchmark,
            (max(p.high for p in held) / entry.open - 1) * 100,
            (min(p.low for p in held) / entry.open - 1) * 100,
        )
        actual = rows_of(
            path,
            'SELECT entry_date, entry_price, exit_date, exit_price, gross_return_pct, net_return_pct, '
            'benchmark_return_pct, excess_return_pct, mfe_pct, mae_pct FROM signal_outcome '
            "WHERE signal_id = ? AND horizon = ? AND status = 'complete'",
            s['id'],
            horizon,
        )
        assert actual == [pytest.approx(expected)]


def test_a_signal_whose_tickers_history_stops_early_is_ended(path, container, prices):
    # FFF's history stops 15 sessions before its market's
    container.history_repo.store(TICKER, 'FFF', prices['FFF'][:-15], FETCHED, 'eodhd')
    last = prices['FFF'][-16].date

    container.history_backfill.run()

    days = [p.date.isoformat() for p in prices['FFF'][:-15]]
    young = [s for s in signals(path) if s['ticker'] == 'FFF' and len(days) - 1 - days.index(s['session_date']) < 120]
    assert len(young) == 3
    for s in young:
        reached = len(days) - 1 - days.index(s['session_date'])
        ended = rows_of(
            path,
            'SELECT horizon, ended_reason, last_session_date FROM signal_outcome '
            "WHERE signal_id = ? AND status = 'ended' ORDER BY horizon",
            s['id'],
        )
        assert ended == [(h, HISTORY_ENDED, last.isoformat()) for h in HORIZONS if h > reached]
    # every other ticker's history reaches its market's last session: nothing else is ended
    others = rows_of(
        path,
        "SELECT COUNT(*) FROM signal_outcome o JOIN signal s ON s.id = o.signal_id WHERE o.status = 'ended' "
        "AND s.ticker <> 'FFF'",
    )
    assert others == [(0,)]


def test_live_signals_and_tickers_without_history_are_left_to_run_refresh(path, container):
    with sqlite3.connect(path) as con:
        for signal_id, ticker, source in ((9001, 'AAA', 'live'), (9002, 'NOHIST', 'backfill')):
            con.execute(
                'INSERT INTO signal (id, ticker, market, time_zone, session_date, close, status, screener_version, '
                "rule_settings, source) VALUES (?, ?, 'US', 'America/New_York', ?, 1.0, 'confirmed', '0.0.1', '{}', ?)",
                (signal_id, ticker, DAYS[250].isoformat(), source),
            )

    container.history_backfill.run()

    assert rows_of(path, 'SELECT COUNT(*) FROM signal_price WHERE signal_id IN (9001, 9002)') == [(0,)]


def test_a_second_backfill_leaves_the_paths_and_outcomes_as_they_were(path, container, prices):
    container.history_repo.store(TICKER, 'CCC', prices['CCC'][:-50], FETCHED, 'eodhd')
    container.history_backfill.run()
    assert rows_of(path, "SELECT COUNT(*) FROM signal_outcome WHERE status = 'ended'")[0][0] > 0
    paths = rows_of(path, 'SELECT * FROM signal_price ORDER BY signal_id, session_index')
    outcomes_sql = (
        'SELECT signal_id, horizon, status, entry_price, exit_price, net_return_pct, ended_reason, last_session_date, '
        'completed_at FROM signal_outcome ORDER BY signal_id, horizon'
    )
    outcomes = rows_of(path, outcomes_sql)
    tracks = rows_of(path, 'SELECT * FROM signal_track ORDER BY signal_id')

    again = container.history_backfill.run()

    assert (again.outcomes.completed, again.outcomes.decided) == (0, 0)
    assert rows_of(path, 'SELECT * FROM signal_price ORDER BY signal_id, session_index') == paths
    assert rows_of(path, outcomes_sql) == outcomes
    assert rows_of(path, 'SELECT * FROM signal_track ORDER BY signal_id') == tracks


# The listed tickers the test histories have, all fetched on FETCHED, so all start late: 10 years before FETCHED is
# 2016, and they start in 2024
REPLAYED = ['AAA', 'BBB', 'CCC', 'DDD', 'RY.TO', 'EEE', 'FFF', 'GGG', 'SHOP.TO', 'TD.TO', 'HHH']


def test_the_run_is_recorded_with_its_range_universe_and_the_tickers_it_left_out(path, container, prices):
    # NEW listed 30 sessions before the end, too late to be scored; IDLE fetched 10 years after FIRST, so its history
    # starts on time, but never trades
    with sqlite3.connect(path) as con:
        con.execute("INSERT INTO subsector (id, sector_id, subsector, tickers) VALUES (4, 1, 'sub4', 'NEW,IDLE')")
    container.history_repo.store(TICKER, 'NEW', prices['AAA'][-30:], FETCHED, 'eodhd')
    idle = [DailyPrice(day, 10.0, 0, 10.0, 10.0, 10.0) for day in US_DAYS]
    container.history_repo.store(TICKER, 'IDLE', idle, date(2034, 1, 1), 'tiingo')

    result = container.history_backfill.run()

    [run] = container.backfill_run_repo.all()
    assert run == result.run
    assert (run.first_session, run.last_session, run.sessions) == (result.first, result.last, result.sessions)
    assert run.sessions > 0
    assert (run.started_at, run.finished_at, run.universe_date) == (EVENING, EVENING, DAYS[-1])
    assert (run.tickers_listed, run.tickers_replayed, run.tickers_scored) == (15, 13, 11)
    assert run.left_out == {
        'NOHIST': NO_HISTORY,
        'BAD.XX': UNSUPPORTED,
        'NEW': SHORT_HISTORY,
        'IDLE': NEVER_SCORED,
    }
    assert run.short_history == {
        **{ticker: prices[ticker][0].date for ticker in REPLAYED},
        'NEW': prices['AAA'][-30].date,
    }
    assert run.short_history['GGG'] == US_DAYS[120]
    assert run.price_api == 'eodhd,tiingo'
    assert (run.signals, run.relinked) == (result.signals, result.relinked)
    assert run.signals > 0
    assert run.screener_version == SCREENER_VERSION


def test_every_run_is_recorded_a_second_one_archiving_nothing(container):
    first = container.history_backfill.run()
    again = container.history_backfill.run()

    runs = container.backfill_run_repo.all()
    assert [r.id for r in runs] == [first.run.id, again.run.id]
    assert [r.signals for r in runs] == [first.signals, 0]
    assert runs[1].price_api == 'eodhd'
    assert runs[1].left_out == {'NOHIST': NO_HISTORY, 'BAD.XX': UNSUPPORTED}
