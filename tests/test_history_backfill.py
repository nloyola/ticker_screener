"""
backfill-history (app/history_backfill.py) replays the screener over the price history, with each indicator computed
once over a ticker's whole history (app/replay.py), and archives the same signals, with the same values, as
Analyzer.score_as_of gives on the history session by session.
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

from app.analyzer import SIGNAL_STATUSES, Analyzer
from app.backfill import Backfill, market_dates, replayable_sessions
from app.commands.backfill_history_cmd import BackfillHistoryCmd
from app.db import init_db
from app.history_backfill import HistoryBackfill
from app.markets import BENCHMARKS, CANADA, US
from app.price_apis import DailyPrice
from app.replay import flat_days, run_lengths, unscored
from app.repositories.benchmark_price_repo import BenchmarkPriceRepo
from app.repositories.history_repo import BENCHMARK, TICKER, HistoryRepo
from app.repositories.price_data_repo import PriceDataRepo
from app.repositories.screened_session_repo import ScreenedSessionRepo
from app.repositories.sector_repo import SectorRepo
from app.repositories.signal_repo import _ARCHIVED, _ARCHIVED_AT, SignalRepo
from app.repositories.stock_criteria_repo import StockCriteriaRepo, row_values
from app.repositories.subsector_repo import SubsectorRepo
from app.screener_lock import screener_lock
from app.services.sector_service import SectorService

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


def walk(rng: np.random.Generator, days: list[date], start: float = 50.0) -> list[DailyPrice]:
    """A rising random walk with a volume spike on about one day in eight, so breakouts happen often."""
    closes = start * np.exp(np.cumsum(rng.normal(0.002, 0.02, len(days))))
    spikes = rng.random(len(days)) < 0.12
    volumes = (rng.lognormal(7, 0.3, len(days)) * np.where(spikes, 3.5, 1.0)).astype(int) + 1
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
    prices = {ticker: walk(rng, US_DAYS) for ticker in ('AAA', 'BBB', 'CCC', 'EEE', 'FFF')}
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
    for session, markets in sessions.items():
        for s in container.analyzer.score_as_of(session, frame, markets, closes):
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
