"""
The outcome computation (app/signal_outcomes.py) on hand-built paths, each checked against values worked out by hand
in the comments beside them.
"""

from datetime import date, timedelta

import pytest

from app.models import PathSession
from app.signal_outcomes import (
    COST_BPS,
    FAILURE_SESSIONS,
    BreakoutFailure,
    breakout_level,
    compute_outcomes,
)
from app.signal_paths import HORIZONS

FIRST = date(2026, 3, 2)


def day(index: int) -> date:
    """Session 'index''s date: consecutive days are enough here, the computation never reads the calendar."""
    return FIRST + timedelta(days=index)


def session(index, close, open=None, high=None, low=None, bench_open=None, bench_close=None) -> PathSession:
    return PathSession(
        session_index=index,
        session_date=day(index),
        open=open,
        high=high,
        low=low,
        close=close,
        volume=1000,
        benchmark_open=bench_open,
        benchmark_close=bench_close,
    )


def rising(n: int) -> list[PathSession]:
    """
    Sessions 0 to n-1: session 0 closes at 100; session i >= 1 opens at 100 + i, trades from 99 + i to 102 + i and
    closes at 101 + i, while the benchmark opens at 50 and closes at 50 + i / 2.
    """
    path = [session(0, 100.0, 99.0, 100.5, 98.5, 50.0, 50.0)]
    for i in range(1, n):
        path.append(session(i, 101.0 + i, 100.0 + i, 102.0 + i, 99.0 + i, 50.0, 50.0 + i / 2))
    return path


def by_horizon(outcomes):
    return {o.horizon: o for o in outcomes.horizons}


def test_breakout_level_is_the_60_day_high_when_that_breakout_passed():
    assert breakout_level(high_20d=95.0, high_60d=98.0, breakout_60d=True) == 98.0
    assert breakout_level(high_20d=95.0, high_60d=98.0, breakout_60d=False) == 95.0


def test_rising_path_matches_the_hand_computation():
    outcomes = compute_outcomes(rising(11), signal_close=100.0, level=98.0)
    h = by_horizon(outcomes)

    # entry: session 1's open, 101. Exit at 5: session 5's close, 106.
    five = h[5]
    assert five.complete
    assert (five.entry_date, five.entry_price) == (day(1), 101.0)
    assert (five.exit_date, five.exit_price) == (day(5), 106.0)
    assert five.cost_bps == COST_BPS == 20
    assert five.gross_return_pct == pytest.approx(4.950495, abs=1e-6)  # 106 / 101 - 1
    assert five.net_return_pct == pytest.approx(4.750495, abs=1e-6)  # less 0.20
    assert five.benchmark_return_pct == pytest.approx(5.0)  # 50 open to 52.5 close
    assert five.excess_return_pct == pytest.approx(-0.249505, abs=1e-6)
    assert five.mfe_pct == pytest.approx(5.940594, abs=1e-6)  # highest high 107 (session 5)
    assert five.mae_pct == pytest.approx(-0.990099, abs=1e-6)  # lowest low 100 (session 1)

    ten = h[10]
    assert ten.complete
    assert (ten.exit_date, ten.exit_price) == (day(10), 111.0)
    assert ten.gross_return_pct == pytest.approx(9.900990, abs=1e-6)  # 111 / 101 - 1
    assert ten.net_return_pct == pytest.approx(9.700990, abs=1e-6)
    assert ten.benchmark_return_pct == pytest.approx(10.0)  # 50 to 55
    assert ten.excess_return_pct == pytest.approx(-0.299010, abs=1e-6)
    assert ten.mfe_pct == pytest.approx(10.891089, abs=1e-6)  # 112
    assert ten.mae_pct == pytest.approx(-0.990099, abs=1e-6)  # still 100

    # every close of sessions 1 to 10 (102 to 111) is at or above 98
    assert outcomes.failure == BreakoutFailure(level=98.0, failed=False, failure_date=None)


def test_a_horizon_is_complete_only_once_its_exit_session_is_in_the_path():
    assert not by_horizon(compute_outcomes(rising(5), 100.0, 98.0))[5].complete  # sessions 0 to 4
    assert by_horizon(compute_outcomes(rising(6), 100.0, 98.0))[5].complete  # session 5 is in

    full = by_horizon(compute_outcomes(rising(121), 100.0, 98.0))
    assert all(full[h].complete for h in HORIZONS)
    assert full[120].gross_return_pct == pytest.approx(118.811881, abs=1e-6)  # 221 / 101 - 1


def test_ticker_ending_before_the_20_session_horizon():
    outcomes = compute_outcomes(rising(15), 100.0, 98.0)  # its last session is the 14th
    h = by_horizon(outcomes)
    assert [o.horizon for o in outcomes.horizons] == list(HORIZONS)
    assert h[5].complete and h[10].complete
    for horizon in (20, 40, 60, 120):
        assert h[horizon] == type(h[horizon])(horizon=horizon, complete=False)
        assert h[horizon].entry_price is None and h[horizon].gross_return_pct is None
    # its 10 failure sessions were all in the path, so the failure is decided
    assert outcomes.failure.failed is False


def test_failure_is_undecided_until_its_sessions_are_in_the_path():
    assert compute_outcomes(rising(FAILURE_SESSIONS), 100.0, 98.0).failure.failed is None  # sessions 0 to 9
    assert compute_outcomes(rising(1), 100.0, 98.0).failure.failed is None
    only_signal = compute_outcomes(rising(1), 100.0, 98.0)
    assert not any(o.complete for o in only_signal.horizons)


def test_gap_down_through_the_breakout_level_fails_on_that_session():
    path = [
        session(0, 100.0, 97.0, 100.5, 96.5, 50.0, 50.0),
        # dips below 98 during the day but closes above it: no failure
        session(1, 100.5, 101.0, 102.0, 97.0, 50.0, 50.0),
        # gaps down through 98 and closes below it: fails here
        session(2, 96.0, 95.0, 96.5, 94.0, 50.0, 49.5),
        session(3, 97.0, 96.0, 97.5, 95.5, 49.5, 49.6),
        session(4, 99.0, 97.0, 99.5, 96.5, 49.6, 49.8),
        session(5, 99.5, 99.0, 100.0, 98.5, 49.8, 50.0),
    ]
    outcomes = compute_outcomes(path, signal_close=100.0, level=98.0)
    assert outcomes.failure == BreakoutFailure(level=98.0, failed=True, failure_date=day(2))

    five = by_horizon(outcomes)[5]
    assert five.gross_return_pct == pytest.approx(-1.485149, abs=1e-6)  # 99.5 / 101 - 1
    assert five.mae_pct == pytest.approx(-6.930693, abs=1e-6)  # lowest low 94 (session 2)
    assert five.mfe_pct == pytest.approx(0.990099, abs=1e-6)  # highest high 102 (session 1)
    assert five.benchmark_return_pct == pytest.approx(0.0)  # 50 open to 50 close


def test_split_inside_the_path_compares_the_rescaled_level():
    # The signal was archived before a 2:1 split, at a close of 100 above a 98 breakout level; the run that wrote
    # the path holds every price adjusted for the split, half the archived basis, so the level is 49 on its basis.
    path = [
        session(0, 50.0, 49.0, 50.25, 48.75, 50.0, 50.0),
        session(1, 51.0, 50.5, 51.5, 50.0, 50.0, 50.5),
        session(2, 49.5, 50.5, 51.0, 49.25, 50.5, 50.5),  # at or above 49: no failure
        session(3, 48.5, 49.25, 49.5, 48.25, 50.5, 50.0),  # below 49: fails here, not at session 1
        session(4, 50.0, 48.5, 50.5, 48.5, 50.0, 50.5),
        session(5, 52.0, 50.0, 52.5, 49.75, 50.5, 51.0),
    ]
    outcomes = compute_outcomes(path, signal_close=100.0, level=98.0)
    assert outcomes.failure == BreakoutFailure(level=49.0, failed=True, failure_date=day(3))

    # the returns read the path alone, on one basis, so the split leaves no jump
    five = by_horizon(outcomes)[5]
    assert five.entry_price == 50.5
    assert five.gross_return_pct == pytest.approx(2.970297, abs=1e-6)  # 52 / 50.5 - 1
    assert five.mfe_pct == pytest.approx(3.960396, abs=1e-6)  # 52.5
    assert five.mae_pct == pytest.approx(-4.455446, abs=1e-6)  # 48.25
    assert five.benchmark_return_pct == pytest.approx(2.0)  # 50 to 51


def test_entry_session_missing_an_open_enters_at_its_close():
    path = [
        session(0, 101.0, 100.0, 101.5, 99.5, 50.0, 50.0),
        # no open, for the ticker or the benchmark: enter at the closes, 102 and 51, and leave out this session's
        # high and low, which may have come before the entry
        session(1, 102.0, None, 110.0, 90.0, None, 51.0),
        session(2, 102.5, 102.0, 103.0, 101.0, 51.0, 51.2),
        session(3, 102.5, 102.5, None, 101.5, 51.2, 51.4),  # no high: its close counts
        session(4, 103.0, 102.5, 103.0, 102.0, 51.4, 51.6),
        session(5, 104.0, 103.0, 104.0, 102.5, 51.6, 52.0),
    ]
    five = by_horizon(compute_outcomes(path, signal_close=101.0, level=98.0))[5]
    assert (five.entry_date, five.entry_price) == (day(1), 102.0)
    assert five.gross_return_pct == pytest.approx(1.960784, abs=1e-6)  # 104 / 102 - 1
    assert five.benchmark_return_pct == pytest.approx(1.960784, abs=1e-6)  # 51 to 52
    assert five.excess_return_pct == pytest.approx(-0.2)  # the cost alone
    assert five.mfe_pct == pytest.approx(1.960784, abs=1e-6)  # highest high 104, not session 1's 110
    assert five.mae_pct == pytest.approx(-0.980392, abs=1e-6)  # lowest low 101, not session 1's 90


def test_missing_benchmark_leaves_its_return_and_the_excess_out():
    path = rising(6)
    path[5] = session(5, 106.0, 105.0, 107.0, 104.0, 50.0, None)
    five = by_horizon(compute_outcomes(path, 100.0, 98.0))[5]
    assert five.complete
    assert five.benchmark_return_pct is None and five.excess_return_pct is None
    assert five.net_return_pct == pytest.approx(4.750495, abs=1e-6)


def test_a_path_with_a_gap_in_its_indexes_is_refused():
    path = rising(6)
    del path[3]
    with pytest.raises(ValueError):
        compute_outcomes(path, 100.0, 98.0)
