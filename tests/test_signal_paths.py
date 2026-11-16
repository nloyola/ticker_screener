"""
Each signal's own price path (app/signal_paths.py): rewritten whole from every run's prices until it reaches the
120-session horizon, then frozen, and kept as it is, with its remaining horizons ended, once its ticker stops getting
prices.
"""

import logging
import sqlite3
from datetime import date, datetime, time, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from app.db import init_db
from app.markets import CANADA, US
from app.models import PriceData
from app.price_apis import DailyPrice
from app.repositories.benchmark_price_repo import BenchmarkPriceRepo
from app.repositories.price_data_repo import PriceDataRepo
from app.repositories.signal_path_repo import SignalPathRepo
from app.signal_paths import HORIZONS, PATH_SESSIONS, REMOVED, SignalPaths, build_path

NEW_YORK = ZoneInfo('America/New_York')
# a Monday
FIRST = date(2026, 1, 5)


def weekdays(first: date, n: int) -> list[date]:
    days: list[date] = []
    day = first
    while len(days) < n:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


DAYS = weekdays(FIRST, 200)


def evening(day: date) -> datetime:
    """A run started once both markets have published 'day's candles."""
    return datetime.combine(day, time(20), tzinfo=NEW_YORK)


def rows(ticker: str, days: list[date], close=lambda i: 100.0 + i) -> list[PriceData]:
    return [
        PriceData(
            id=0,
            subsector_id=1,
            ticker=ticker,
            date=d,
            close=close(i),
            volume=1000 + i,
            high=close(i) + 1,
            low=close(i) - 1,
            open=close(i) - 0.5,
        )
        for i, d in enumerate(days)
    ]


@pytest.fixture
def path(tmp_path) -> str:
    path = str(tmp_path / 'stock_data.db')
    init_db(path)
    return path


@pytest.fixture
def container(path) -> SimpleNamespace:
    container = SimpleNamespace(db_path=path)
    container.price_data_repo = PriceDataRepo(container)
    container.benchmark_price_repo = BenchmarkPriceRepo(container)
    container.signal_path_repo = SignalPathRepo(container)
    container.signal_paths = SignalPaths(container)
    return container


def add_signal(path: str, signal_id: int, ticker: str, session_date: date, market: str = US) -> None:
    with sqlite3.connect(path) as con:
        con.execute(
            'INSERT INTO signal (id, ticker, market, time_zone, session_date, close, status, screener_version, '
            "rule_settings, source) VALUES (?, ?, ?, 'America/New_York', ?, 1.0, 'confirmed', '1.0.0', '{}', 'live')",
            (signal_id, ticker, market, session_date.isoformat()),
        )


def store_prices(container: SimpleNamespace, ticker: str, days: list[date], close=lambda i: 100.0 + i) -> None:
    container.price_data_repo.replace_for_ticker(ticker, rows(ticker, days, close))


def store_benchmark(container: SimpleNamespace, ticker: str, days: list[date]) -> None:
    container.benchmark_price_repo.replace(
        ticker,
        [
            DailyPrice(date=d, close=500.0 + i, volume=0, high=None, low=None, open=499.0 + i)
            for i, d in enumerate(days)
        ],
    )


def path_of(path: str, signal_id: int) -> list[tuple]:
    with sqlite3.connect(path) as con:
        return con.execute(
            'SELECT session_index, session_date, open, high, low, close, volume, benchmark_open, benchmark_close '
            'FROM signal_price WHERE signal_id = ? ORDER BY session_index',
            (signal_id,),
        ).fetchall()


def frozen_at(path: str, signal_id: int) -> str | None:
    with sqlite3.connect(path) as con:
        row = con.execute('SELECT path_frozen_at FROM signal_track WHERE signal_id = ?', (signal_id,)).fetchone()
    assert row is not None, 'no signal_track row'
    return row[0]


def outcomes(path: str, signal_id: int) -> list[tuple]:
    with sqlite3.connect(path) as con:
        return con.execute(
            'SELECT horizon, status, ended_reason, last_session_date FROM signal_outcome WHERE signal_id = ? '
            'ORDER BY horizon',
            (signal_id,),
        ).fetchall()


def update(container: SimpleNamespace, day: date, flagged=None, markets=(US,)):
    return container.signal_paths.update(markets=markets, flagged=flagged or {}, started_at=evening(day))


def test_path_runs_from_the_signal_session_to_the_latest_session(path, container):
    # 30 sessions of prices, the signal on the 11th: its path holds the signal session and the 19 after it
    store_prices(container, 'AAA', DAYS[:30])
    store_benchmark(container, 'SPY', DAYS[:30])
    add_signal(path, 1, 'AAA', DAYS[10])

    result = update(container, DAYS[29])

    stored = path_of(path, 1)
    assert [r[0] for r in stored] == list(range(20))
    assert [r[1] for r in stored] == [d.isoformat() for d in DAYS[10:30]]
    # the signal session's prices and the benchmark's open and close that day
    assert stored[0] == (0, DAYS[10].isoformat(), 109.5, 111.0, 109.0, 110.0, 1010, 509.0, 510.0)
    assert frozen_at(path, 1) is None
    assert outcomes(path, 1) == []
    assert (result.rewritten, result.frozen, result.ended) == (1, 0, 0)


def test_a_split_inside_the_path_leaves_no_jump(path, container):
    # before the 2-for-1 split on session 20 the API reported closes near 200; the run after it restates them all
    store_prices(container, 'AAA', DAYS[:20], close=lambda i: 200.0 + 2 * i)
    add_signal(path, 1, 'AAA', DAYS[5])
    update(container, DAYS[19])
    assert path_of(path, 1)[-1][5] == 238.0

    store_prices(container, 'AAA', DAYS[:25], close=lambda i: 100.0 + i)
    update(container, DAYS[24])

    closes = [r[5] for r in path_of(path, 1)]
    assert closes == [100.0 + i for i in range(5, 25)]
    assert max(b / a for a, b in zip(closes, closes[1:], strict=False)) < 1.02


def test_a_path_freezes_at_the_120_session_horizon_and_is_never_rewritten(path, container):
    store_prices(container, 'AAA', DAYS[:140])
    add_signal(path, 1, 'AAA', DAYS[10])

    result = update(container, DAYS[139])

    stored = path_of(path, 1)
    assert [r[0] for r in stored] == list(range(PATH_SESSIONS + 1))
    assert stored[-1][1] == DAYS[10 + PATH_SESSIONS].isoformat()
    assert frozen_at(path, 1) is not None
    assert (result.rewritten, result.frozen) == (1, 1)

    # a later restatement, or the ticker's removal, leaves the frozen path alone
    store_prices(container, 'AAA', DAYS[:150], close=lambda i: 50.0 + i)
    update(container, DAYS[149])
    container.price_data_repo.replace_for_ticker('AAA', [])
    update(container, DAYS[149])
    assert path_of(path, 1) == stored
    assert outcomes(path, 1) == []


def test_a_removed_ticker_keeps_its_path_and_its_horizons_beyond_it_end(path, container):
    store_prices(container, 'AAA', DAYS[:40])
    add_signal(path, 1, 'AAA', DAYS[10])
    add_signal(path, 2, 'AAA', DAYS[30])
    update(container, DAYS[39])
    kept = path_of(path, 1), path_of(path, 2)

    # the dashboard deletes the price_data rows of a ticker it removes from every subsector
    container.price_data_repo.replace_for_ticker('AAA', [])
    result = update(container, DAYS[40])

    assert (path_of(path, 1), path_of(path, 2)) == kept
    # signal 1's path reaches session 29, so its 5, 10 and 20-session horizons are left to the outcomes
    last = DAYS[39].isoformat()
    assert outcomes(path, 1) == [(h, 'ended', REMOVED, last) for h in (40, 60, 120)]
    assert outcomes(path, 2) == [(h, 'ended', REMOVED, last) for h in HORIZONS if h > 9]
    assert result.ended == 2


def test_a_ticker_the_run_flagged_keeps_its_path_and_ends_with_the_runs_reason(path, container):
    store_prices(container, 'AAA', DAYS[:30])
    add_signal(path, 1, 'AAA', DAYS[10])
    update(container, DAYS[29])
    kept = path_of(path, 1)

    # a failed download keeps the old rows; were the path rewritten from them it would not change, but it must not be
    store_prices(container, 'AAA', DAYS[:31], close=lambda i: 1.0 + i)
    update(container, DAYS[30], flagged={'AAA': 'not_found'})

    assert path_of(path, 1) == kept
    assert outcomes(path, 1) == [(h, 'ended', 'not_found', DAYS[29].isoformat()) for h in (20, 40, 60, 120)]


def test_ended_horizons_are_cleared_once_the_ticker_has_prices_again(path, container):
    store_prices(container, 'AAA', DAYS[:30])
    add_signal(path, 1, 'AAA', DAYS[10])
    update(container, DAYS[29])
    update(container, DAYS[30], flagged={'AAA': 'stale'})
    assert outcomes(path, 1) != []

    store_prices(container, 'AAA', DAYS[:32])
    update(container, DAYS[31])

    assert len(path_of(path, 1)) == 22
    assert outcomes(path, 1) == []


def test_ending_never_touches_a_complete_horizon(path, container):
    store_prices(container, 'AAA', DAYS[:30])
    add_signal(path, 1, 'AAA', DAYS[10])
    update(container, DAYS[29])
    with sqlite3.connect(path) as con:
        con.execute(
            'INSERT INTO signal_outcome (signal_id, horizon, status, entry_date, entry_price, exit_date, exit_price, '
            'cost_bps, gross_return_pct, net_return_pct, mfe_pct, mae_pct, completed_at) '
            "VALUES (1, 5, 'complete', ?, 110.5, ?, 115.0, 20, 4.07, 3.87, 5.0, -1.0, '2026-02-01 00:00:00')",
            (DAYS[11].isoformat(), DAYS[15].isoformat()),
        )
    store_prices(container, 'AAA', DAYS[:31])
    update(container, DAYS[30])
    container.price_data_repo.replace_for_ticker('AAA', [])
    update(container, DAYS[31])

    assert outcomes(path, 1)[0] == (5, 'complete', None, None)


def test_a_signal_with_no_path_yet_ends_on_its_session(path, container):
    add_signal(path, 1, 'GONE', DAYS[10])

    update(container, DAYS[20])

    assert path_of(path, 1) == []
    assert outcomes(path, 1) == [(h, 'ended', REMOVED, DAYS[10].isoformat()) for h in HORIZONS]


def test_a_market_the_run_did_not_fetch_is_left_alone(path, container):
    store_prices(container, 'SHOP.TO', DAYS[:30])
    add_signal(path, 1, 'SHOP.TO', DAYS[10], market=CANADA)
    add_signal(path, 2, 'GONE.TO', DAYS[10], market=CANADA)

    update(container, DAYS[29], markets=(US,))

    assert path_of(path, 1) == []
    assert outcomes(path, 2) == []


def test_a_session_not_yet_final_stays_out_of_the_path(path, container):
    store_prices(container, 'AAA', DAYS[:30])
    add_signal(path, 1, 'AAA', DAYS[10])

    container.signal_paths.update(
        markets=(US,), flagged={}, started_at=datetime.combine(DAYS[29], time(11), tzinfo=NEW_YORK)
    )

    assert path_of(path, 1)[-1][1] == DAYS[28].isoformat()


def test_a_signal_whose_session_left_price_data_keeps_its_path(path, container, caplog):
    store_prices(container, 'AAA', DAYS[:30])
    add_signal(path, 1, 'AAA', DAYS[10])
    update(container, DAYS[29])
    kept = path_of(path, 1)

    store_prices(container, 'AAA', DAYS[20:31])
    with caplog.at_level(logging.WARNING):
        result = update(container, DAYS[30])

    assert path_of(path, 1) == kept
    assert outcomes(path, 1) == []
    assert result.missing_session == 1
    assert f'signal 1 (AAA on {DAYS[10]}): its session is not in price_data, path kept' in caplog.text


def test_build_path_skips_rows_without_a_close():
    prices = rows('AAA', DAYS[:5])
    prices[2] = PriceData(id=0, subsector_id=1, ticker='AAA', date=DAYS[2], close=None, volume=None)

    built = build_path(DAYS[1], prices, {}, DAYS[4])

    assert [(p.session_index, p.session_date) for p in built] == [(0, DAYS[1]), (1, DAYS[3]), (2, DAYS[4])]
    assert built[0].benchmark_close is None
    assert build_path(DAYS[2], prices, {}, DAYS[4]) is None
