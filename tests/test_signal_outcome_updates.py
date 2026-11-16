"""
Storing each signal's outcomes after its path is updated (app/signal_outcome_updates.py): every newly complete
horizon and the breakout failure once decided, never touching a complete row, the same rows whether run-refresh or
compute-outcomes stores them.
"""

import argparse
import logging
import sqlite3
from datetime import date, datetime, time, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from app.commands.compute_outcomes_cmd import ComputeOutcomesCmd
from app.db import init_db
from app.models import PriceData
from app.price_apis import DailyPrice
from app.repositories.benchmark_price_repo import BenchmarkPriceRepo
from app.repositories.price_data_repo import PriceDataRepo
from app.repositories.signal_outcome_repo import SignalOutcomeRepo
from app.repositories.signal_path_repo import SignalPathRepo
from app.signal_outcome_updates import SignalOutcomeUpdates
from app.signal_outcomes import compute_outcomes
from app.signal_paths import HORIZONS, SignalPaths

NEW_YORK = ZoneInfo('America/New_York')
FIRST = date(2026, 1, 5)  # a Monday


def weekdays(first: date, n: int) -> list[date]:
    days: list[date] = []
    day = first
    while len(days) < n:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


DAYS = weekdays(FIRST, 200)
SIGNAL_DAY = DAYS[10]


def rising(i: int) -> float:
    return 100.0 + i


def make_container(path: str) -> SimpleNamespace:
    c = SimpleNamespace(db_path=path)
    c.price_data_repo = PriceDataRepo(c)
    c.benchmark_price_repo = BenchmarkPriceRepo(c)
    c.signal_path_repo = SignalPathRepo(c)
    c.signal_paths = SignalPaths(c)
    c.signal_outcome_repo = SignalOutcomeRepo(c)
    c.signal_outcome_updates = SignalOutcomeUpdates(c)
    return c


@pytest.fixture
def path(tmp_path) -> str:
    path = str(tmp_path / 'stock_data.db')
    init_db(path)
    return path


@pytest.fixture
def container(path) -> SimpleNamespace:
    return make_container(path)


def add_signal(path: str, signal_id: int, ticker: str, close: float, high_20d=105.0, high_60d=None) -> None:
    with sqlite3.connect(path) as con:
        con.execute(
            'INSERT INTO signal (id, ticker, market, time_zone, session_date, close, status, screener_version, '
            'rule_settings, source, high_20d, high_60d, breakout_60d) '
            "VALUES (?, ?, 'US', 'America/New_York', ?, ?, 'confirmed', '1.0.0', '{}', 'live', ?, ?, ?)",
            (signal_id, ticker, SIGNAL_DAY.isoformat(), close, high_20d, high_60d, int(high_60d is not None)),
        )


def store_prices(c: SimpleNamespace, ticker: str, n: int, close=rising, benchmark_sessions: int | None = None) -> None:
    c.price_data_repo.replace_for_ticker(
        ticker,
        [
            PriceData(
                id=0,
                subsector_id=1,
                ticker=ticker,
                date=d,
                close=close(i),
                volume=1000,
                high=close(i) + 1,
                low=close(i) - 1,
                open=close(i) - 0.5,
            )
            for i, d in enumerate(DAYS[:n])
        ],
    )
    c.benchmark_price_repo.replace(
        'SPY',
        [
            DailyPrice(date=d, close=500.0 + i, volume=0, high=None, low=None, open=499.0 + i)
            for i, d in enumerate(DAYS[: n if benchmark_sessions is None else benchmark_sessions])
        ],
    )


def run(c: SimpleNamespace, n: int, flagged=None):
    """A run on the n-th day: the paths from the prices stored, then the outcomes, as run-refresh does."""
    c.signal_paths.update(
        markets=('US',), flagged=flagged or {}, started_at=datetime.combine(DAYS[n - 1], time(20), tzinfo=NEW_YORK)
    )
    return c.signal_outcome_updates.update()


def outcome_rows(path: str, with_times: bool = True) -> list[tuple]:
    times = ', completed_at, updated_at' if with_times else ''
    with sqlite3.connect(path) as con:
        return con.execute(
            'SELECT signal_id, horizon, status, entry_date, entry_price, exit_date, exit_price, cost_bps, '
            'gross_return_pct, net_return_pct, benchmark_return_pct, excess_return_pct, mfe_pct, mae_pct, '
            f'ended_reason, last_session_date{times} FROM signal_outcome ORDER BY signal_id, horizon'
        ).fetchall()


def track(path: str, signal_id: int) -> tuple:
    with sqlite3.connect(path) as con:
        return con.execute(
            'SELECT breakout_level, breakout_failed, failure_date, failure_decided_at IS NOT NULL '
            'FROM signal_track WHERE signal_id = ?',
            (signal_id,),
        ).fetchone()


def test_a_run_stores_each_horizon_its_path_reaches_as_the_computation_gives_it(path, container):
    # the signal session is DAYS[10] and prices run to DAYS[29]: the path reaches session 19, so the 5- and
    # 10-session horizons are complete and the 20-session one is not
    store_prices(container, 'AAA', 30)
    add_signal(path, 1, 'AAA', close=110.0)

    result = run(container, 30)

    rows = outcome_rows(path, with_times=False)
    assert [(r[1], r[2]) for r in rows] == [(5, 'complete'), (10, 'complete')]
    expected = compute_outcomes(container.signal_path_repo.path(1), 110.0, 105.0)
    for row, o in zip(rows, expected.horizons, strict=False):
        assert row[3:14] == (
            o.entry_date.isoformat(),
            o.entry_price,
            o.exit_date.isoformat(),
            o.exit_price,
            o.cost_bps,
            o.gross_return_pct,
            o.net_return_pct,
            o.benchmark_return_pct,
            o.excess_return_pct,
            o.mfe_pct,
            o.mae_pct,
        )
    # entry is session 11's open (110.5), the 5-session exit session 15's close (115)
    assert rows[0][4] == 110.5 and rows[0][6] == 115.0
    # every close after the signal is above the 20-day high of 105, and the 10 sessions are in: not failed
    assert track(path, 1) == (105.0, 0, None, 1)
    assert (result.signals, result.completed, result.decided, result.failed) == (1, 2, 1, 0)


def test_rerunning_changes_no_complete_row_and_completes_the_next_horizon(path, container):
    store_prices(container, 'AAA', 30)
    add_signal(path, 1, 'AAA', close=110.0)
    run(container, 30)
    before = outcome_rows(path)
    decided = track(path, 1)

    # a restatement halves every price (a split after the signal), and ten more sessions reach the 20-session
    # horizon: the path is rewritten on the new basis, but the complete rows and the decided failure stay as they were
    store_prices(container, 'AAA', 40, close=lambda i: (100.0 + i) / 2)
    result = run(container, 40)

    after = outcome_rows(path)
    assert after[:2] == before
    assert [(r[1], r[2]) for r in after] == [(5, 'complete'), (10, 'complete'), (20, 'complete')]
    assert track(path, 1) == decided
    assert (result.completed, result.decided) == (1, 0)

    # with nothing new, a rerun changes nothing at all
    run(container, 40)
    assert outcome_rows(path) == after


def test_a_horizon_completed_without_the_benchmark_gets_it_once_the_path_has_it(path, container):
    # the run's benchmark download failed, keeping closes up to DAYS[19]: the 10-session exit, DAYS[20], has no
    # benchmark close, so that horizon completes without the benchmark's return; the 5-session one has it
    store_prices(container, 'AAA', 30, benchmark_sessions=20)
    add_signal(path, 1, 'AAA', close=110.0)
    run(container, 30)
    before = outcome_rows(path)
    assert before[0][10] is not None and before[1][10] is None and before[1][11] is None

    # the next run's download brings the benchmark: the row gets the benchmark's return and the excess over it,
    # and nothing else in it changes, even after a restatement of the ticker's prices
    store_prices(container, 'AAA', 30, close=lambda i: (100.0 + i) / 2)
    result = run(container, 30)

    after = outcome_rows(path)
    expected = compute_outcomes(container.signal_path_repo.path(1), 55.0, 52.5).horizons[1]
    assert after[1][10] == expected.benchmark_return_pct
    assert after[1][11] == pytest.approx(before[1][9] - expected.benchmark_return_pct)
    assert after[1][:10] + after[1][12:17] == before[1][:10] + before[1][12:17]
    assert after[0] == before[0]
    assert (result.completed, result.benchmarks_filled) == (0, 1)

    # filled once: a rerun changes nothing
    run(container, 30)
    assert outcome_rows(path) == after


def test_a_complete_signal_gets_a_late_benchmark_too(path, container):
    # every horizon complete and the failure decided, but the 120-session exit lacks the benchmark: the path waits
    # to freeze, and the next run's benchmark fills the last row though the signal had nothing else left
    store_prices(container, 'AAA', 131, benchmark_sessions=130)
    add_signal(path, 1, 'AAA', close=110.0)
    run(container, 131)
    assert [r[10] is None for r in outcome_rows(path)] == [False] * 5 + [True]

    store_prices(container, 'AAA', 131)
    result = run(container, 131)

    assert all(r[10] is not None and r[11] is not None for r in outcome_rows(path))
    assert result.benchmarks_filled == 1


def test_the_command_stores_the_same_rows_as_the_run(path, container, tmp_path):
    store_prices(container, 'AAA', 30)
    store_prices(container, 'BBB', 30, close=lambda i: 90.0 if i == 12 else 100.0 + i)
    add_signal(path, 1, 'AAA', close=110.0)
    add_signal(path, 2, 'BBB', close=110.0, high_20d=104.0, high_60d=108.0)
    container.signal_paths.update(
        markets=('US',), flagged={}, started_at=datetime.combine(DAYS[29], time(20), tzinfo=NEW_YORK)
    )
    copy = str(tmp_path / 'copy.db')
    with sqlite3.connect(path) as src, sqlite3.connect(copy) as dst:
        src.backup(dst)  # the database is in WAL mode: a file copy can miss what the WAL holds

    container.signal_outcome_updates.update()
    ComputeOutcomesCmd().handle(argparse.Namespace(), make_container(copy))

    assert outcome_rows(copy, with_times=False) == outcome_rows(path, with_times=False)
    assert [track(copy, i)[:3] for i in (1, 2)] == [track(path, i)[:3] for i in (1, 2)]


def test_a_close_below_the_level_decides_the_failure_once(path, container):
    # session 2 after the signal (DAYS[12]) closes at 90, below the 60-day high of 108 the breakout cleared
    store_prices(container, 'AAA', 14, close=lambda i: 90.0 if i == 12 else 100.0 + i)
    add_signal(path, 1, 'AAA', close=110.0, high_20d=104.0, high_60d=108.0)

    result = run(container, 14)

    assert track(path, 1) == (108.0, 1, DAYS[12].isoformat(), 1)
    assert (result.decided, result.failed) == (1, 1)
    # no horizon is reached yet: no outcome row
    assert outcome_rows(path) == []

    # a restatement that lifts every close above the level leaves the decided failure as it was
    store_prices(container, 'AAA', 30, close=lambda i: 200.0 + i)
    run(container, 30)
    assert track(path, 1) == (108.0, 1, DAYS[12].isoformat(), 1)


def test_an_undecided_failure_keeps_the_level_on_the_paths_basis(path, container):
    # 5 sessions after the signal: the failure window is not over, and the level is stored rescaled to the path
    store_prices(container, 'AAA', 16, close=lambda i: (100.0 + i) / 2)
    add_signal(path, 1, 'AAA', close=110.0)

    run(container, 16)

    assert track(path, 1) == (52.5, None, None, 0)


def test_an_ended_signal_completes_the_horizons_its_path_reaches(path, container):
    store_prices(container, 'AAA', 30)
    add_signal(path, 1, 'AAA', close=110.0)
    container.signal_paths.update(
        markets=('US',), flagged={}, started_at=datetime.combine(DAYS[29], time(20), tzinfo=NEW_YORK)
    )

    run(container, 30, flagged={'AAA': 'not_found'})

    statuses = {r[1]: (r[2], r[14]) for r in outcome_rows(path, with_times=False)}
    assert statuses == {
        5: ('complete', None),
        10: ('complete', None),
        **{h: ('ended', 'not_found') for h in HORIZONS if h > 19},
    }


def test_a_signal_without_its_breakout_level_gets_horizons_but_no_failure(path, container, caplog):
    store_prices(container, 'AAA', 30)
    add_signal(path, 1, 'AAA', close=110.0, high_20d=None)

    with caplog.at_level(logging.WARNING):
        result = run(container, 30)

    assert [(r[1], r[2]) for r in outcome_rows(path)] == [(5, 'complete'), (10, 'complete')]
    assert track(path, 1) == (None, None, None, 0)
    assert result.no_level == 1
    assert 'signal 1: archived without its breakout level' in caplog.text


def test_a_signal_with_no_path_is_left_alone(path, container):
    add_signal(path, 1, 'AAA', close=110.0)

    result = container.signal_outcome_updates.update()

    assert result.signals == 0
    assert outcome_rows(path) == []
