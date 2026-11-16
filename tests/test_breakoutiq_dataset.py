"""
BreakoutIQ's analysis dataset: one row per signal and resolved horizon, with the derived mandatory strengths and the
regime, and the signals split by date into development, validation and test with a GAP_SESSIONS gap between them,
a signal whose own window reaches the next period in the gap too; one screener version's signals, read only when the
price history is there.
"""

import sqlite3
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.breakoutiq.dataset import (
    ABOVE,
    BELOW,
    DEVELOPMENT,
    GAP,
    GAP_SESSIONS,
    PERIODS,
    REGIME_SMA_DAYS,
    TEST,
    VALIDATION,
    MissingHistory,
    assign_periods,
    build_dataset,
    join_closes,
    load_dataset,
    regimes,
    window_ends,
)
from app.db import init_db
from app.repositories.analysis_data_repo import AnalysisDataRepo
from app.repositories.benchmark_price_repo import BenchmarkPriceRepo
from app.repositories.history_repo import HistoryRepo, init_history_db
from app.signal_paths import HORIZONS

SESSIONS = pd.bdate_range('2016-01-04', periods=2600)


def _signals(dates, market='US', source='backfill') -> pd.DataFrame:
    return pd.DataFrame({'session_date': pd.DatetimeIndex(dates), 'market': market, 'source': source})


def _sessions_between(first: pd.Timestamp, last: pd.Timestamp) -> int:
    """The sessions strictly between two dates."""
    return int(((SESSIONS > first) & (SESSIONS < last)).sum())


# periods


def test_the_periods_follow_the_dates_with_a_gap_between_them():
    # one signal a session, and three on every tenth, so the shares count signals and not dates
    dates = [d for i, d in enumerate(SESSIONS[300:2400]) for _ in range(3 if i % 10 == 0 else 1)]
    signals = _signals(dates)
    periods = assign_periods(signals, {'US': SESSIONS})

    assert set(periods) == {*PERIODS, GAP}
    spans = {p: signals.loc[periods == p, 'session_date'] for p in PERIODS}
    for earlier, later in ((DEVELOPMENT, VALIDATION), (VALIDATION, TEST)):
        assert spans[earlier].max() < spans[later].min()
        assert _sessions_between(spans[earlier].max(), spans[later].min()) >= GAP_SESSIONS
    # the gap holds only the GAP_SESSIONS sessions before each boundary: one session more and a signal is kept
    gaps = signals.loc[periods == GAP, 'session_date']
    assert gaps.nunique() == 2 * GAP_SESSIONS
    # every signal of a date is in one period, and the split is by count before the gaps take their share
    assert signals.assign(p=periods).groupby('session_date')['p'].nunique().max() == 1
    kept = len(signals) - len(gaps)
    assert (periods == TEST).sum() == pytest.approx(0.2 * len(signals), rel=0.02)
    assert ((periods == DEVELOPMENT).sum() + (periods == VALIDATION).sum() + (periods == TEST).sum()) == kept


def test_a_signal_on_the_last_session_whose_window_ends_before_the_boundary_is_kept():
    boundary = SESSIONS[1000]
    kept, dropped = SESSIONS[1000 - GAP_SESSIONS - 1], SESSIONS[1000 - GAP_SESSIONS]
    # development has 6 signals before the boundary: 60% of 10, so validation starts on the boundary
    signals = _signals([SESSIONS[10]] * 4 + [kept, dropped] + [boundary] * 2 + [SESSIONS[1500]] * 2)
    periods = assign_periods(signals, {'US': SESSIONS})

    assert list(periods) == [DEVELOPMENT] * 5 + [GAP] + [VALIDATION] * 2 + [TEST] * 2


def test_live_signals_are_always_in_test():
    dates = list(SESSIONS[100:1100])
    signals = pd.concat([_signals(dates), _signals([SESSIONS[400]], source='live')], ignore_index=True)
    periods = assign_periods(signals, {'US': SESSIONS})

    assert periods.iloc[-1] == TEST
    assert (signals.loc[periods.isin([DEVELOPMENT, VALIDATION]), 'session_date'] < SESSIONS[400]).all()
    assert not (signals.loc[periods == TEST, 'session_date'] < SESSIONS[400]).any()


def test_each_market_counts_the_gap_in_its_own_sessions():
    # Canada's calendar lacks every fifth session, so the same dates leave fewer of its sessions before the boundary
    canada = SESSIONS[np.arange(len(SESSIONS)) % 5 != 0]
    day = SESSIONS[1000 - GAP_SESSIONS - 5]
    signals = pd.concat(
        [
            _signals([SESSIONS[10]] * 5),
            _signals([day]),
            _signals([day], market='Canada'),
            _signals([SESSIONS[1000]] * 2 + [SESSIONS[1500]] * 2),
        ],
        ignore_index=True,
    )
    periods = assign_periods(signals, {'US': SESSIONS, 'Canada': canada})

    assert periods.iloc[5] == DEVELOPMENT
    assert periods.iloc[6] == GAP


# regime and benchmarks


def test_the_regime_compares_the_benchmark_with_its_sma():
    days = SESSIONS[: REGIME_SMA_DAYS + 50]
    closes = pd.Series(np.r_[np.linspace(100, 200, REGIME_SMA_DAYS + 20), np.linspace(199, 80, 30)], index=days)
    signals = _signals([days[10], days[REGIME_SMA_DAYS + 5], days[-1], days[-1] + pd.Timedelta(days=3)])

    assert list(regimes(signals, {'US': closes})) == [None, ABOVE, BELOW, BELOW]


def test_live_closes_join_the_history_on_its_basis():
    history = pd.Series([10.0, 11.0, 12.0], index=SESSIONS[:3])
    live = pd.Series([24.0, 26.0, 28.0], index=SESSIONS[2:5])

    joined = join_closes(history, live)

    assert list(joined) == [10.0, 11.0, 12.0, 13.0, 14.0]
    assert list(joined.index) == list(SESSIONS[:5])


# the dataset from the database


@pytest.fixture
def db(tmp_path) -> SimpleNamespace:
    path = str(tmp_path / 'stock_data.db')
    history = str(tmp_path / 'history.db')
    init_db(path)
    init_history_db(history)
    c = SimpleNamespace(db_path=path, history_db_path=history)
    c.analysis_data_repo = AnalysisDataRepo(c)
    c.benchmark_price_repo = BenchmarkPriceRepo(c)
    c.history_repo = HistoryRepo(c)
    return c


def _add_signal(path, signal_id, day, **values):
    columns = {
        'id': signal_id,
        'ticker': f'T{signal_id}',
        'market': 'US',
        'time_zone': 'America/New_York',
        'session_date': day.isoformat(),
        'close': 110.0,
        'status': 'confirmed',
        'screener_version': '1.0.0',
        'rule_settings': '{}',
        'source': 'backfill',
        'high_20d': 100.0,
        'high_60d': 105.0,
        'breakout_60d': 0,
        'dma_50': 88.0,
        'dma_50_prev_15d': 80.0,
        'volume_ratio': 2.5,
        'rsi': 64.0,
        'signal': 0.7,
        **values,
    }
    with sqlite3.connect(path) as con:
        con.execute(
            f'INSERT INTO signal ({", ".join(columns)}) VALUES ({", ".join("?" * len(columns))})',
            tuple(columns.values()),
        )


def _add_outcome(path, signal_id, horizon, status):
    with sqlite3.connect(path) as con:
        if status == 'complete':
            con.execute(
                'INSERT INTO signal_outcome (signal_id, horizon, status, entry_date, entry_price, exit_date, '
                'exit_price, cost_bps, gross_return_pct, net_return_pct, benchmark_return_pct, excess_return_pct, '
                "mfe_pct, mae_pct, completed_at) VALUES (?, ?, 'complete', '2020-01-02', 100, '2020-02-03', 105, 20, "
                "5.0, 4.8, 1.0, 3.8, 7.0, -2.0, datetime('now'))",
                (signal_id, horizon),
            )
        else:
            con.execute(
                'INSERT INTO signal_outcome (signal_id, horizon, status, ended_reason, last_session_date) '
                f"VALUES (?, ?, '{status}', ?, ?)",
                (signal_id, horizon, *(('history_ended', '2020-03-02') if status == 'ended' else (None, None))),
            )


def _add_benchmark(path, table, closes: pd.Series):
    """Stores SPY's closes in benchmark_price, or in the history's history_benchmark_price with its other prices."""
    live = table == 'benchmark_price'
    columns = ('ticker', 'date', 'close') if live else ('ticker', 'date', 'open', 'high', 'low', 'close', 'volume')
    rows = [
        ('SPY', d.date().isoformat(), c) if live else ('SPY', d.date().isoformat(), c, c, c, c, 1)
        for d, c in closes.items()
    ]
    with sqlite3.connect(path) as con:
        con.executemany(f'INSERT INTO {table} ({", ".join(columns)}) VALUES ({", ".join("?" * len(columns))})', rows)


def test_the_dataset_has_one_row_per_signal_and_resolved_horizon(db):
    day = date(2020, 1, 2)
    _add_signal(db.db_path, 1, day, breakout_60d=1, high_60d=104.0)
    _add_signal(db.db_path, 2, day)
    _add_signal(db.db_path, 3, day)  # no resolved horizon: not in the dataset
    for horizon in HORIZONS[:3]:
        _add_outcome(db.db_path, 1, horizon, 'complete')
    _add_outcome(db.db_path, 1, 40, 'ended')
    _add_outcome(db.db_path, 2, 5, 'complete')

    rows = load_dataset(db, with_history=False)

    assert list(zip(rows['signal_id'], rows['horizon'], strict=True)) == [(1, 5), (1, 10), (1, 20), (1, 40), (2, 5)]
    assert not rows.duplicated(['signal_id', 'horizon']).any()
    assert list(rows['censored']) == [False, False, False, True, False]
    assert rows.loc[rows['censored'], 'net_return_pct'].isna().all()
    first, second = rows.iloc[0], rows.iloc[4]
    # signal 1 broke its 60-day high, signal 2 its 20-day one
    assert first['breakout_level'] == 104.0
    assert first['breakout_margin_pct'] == pytest.approx((110 / 104 - 1) * 100)
    assert second['breakout_margin_pct'] == pytest.approx(10.0)
    assert first['sma50_distance_pct'] == pytest.approx(25.0)
    assert first['sma50_slope_pct'] == pytest.approx(10.0)
    assert first['volume_ratio'] == 2.5
    assert first['macd_signal'] == 0.7 and first['rsi'] == 64.0
    assert 'rule_settings' not in rows.columns
    assert first['regime'] is None  # no benchmark closes
    assert set(rows['period']) <= {*PERIODS, GAP}


def test_a_pending_horizon_is_left_out(db):
    _add_signal(db.db_path, 1, date(2020, 1, 2))
    _add_outcome(db.db_path, 1, 5, 'complete')
    _add_outcome(db.db_path, 1, 10, 'pending')

    assert list(load_dataset(db, with_history=False)['horizon']) == [5]


def test_the_regime_of_a_backfilled_signal_reads_the_history(db):
    days = SESSIONS[: REGIME_SMA_DAYS + 30]
    rising = pd.Series(np.linspace(100, 200, len(days)), index=days)
    # the history covers every day; benchmark_price only the last 20, as the live window would
    _add_benchmark(db.history_db_path, 'history_benchmark_price', rising)
    _add_benchmark(db.db_path, 'benchmark_price', rising[-20:] * 2)
    _add_signal(db.db_path, 1, days[REGIME_SMA_DAYS + 10].date())
    _add_outcome(db.db_path, 1, 5, 'complete')

    assert list(load_dataset(db, with_history=False)['regime']) == [None]
    assert list(load_dataset(db, with_history=True)['regime']) == [ABOVE]


def test_build_dataset_on_no_signals_is_empty():
    signals = pd.DataFrame(
        columns=[
            'signal_id',
            'session_date',
            'market',
            'source',
            'close',
            'high_20d',
            'high_60d',
            'breakout_60d',
            'dma_50',
            'dma_50_prev_15d',
        ]
    )
    outcomes = pd.DataFrame(
        columns=['signal_id', 'horizon', 'outcome_status', 'entry_date', 'exit_date', 'last_session_date']
    )
    path_sessions = pd.DataFrame(columns=['market', 'session_date'])

    assert build_dataset(signals, outcomes, {}, path_sessions).empty


def test_a_signal_whose_window_reaches_the_next_period_is_in_the_gap():
    # one signal a session: development ends well before validation, the calendar gap alone is met
    dates = SESSIONS[300:1300]
    signals = _signals(dates).assign(window_end=pd.NaT)
    periods = assign_periods(signals, {'US': SESSIONS})
    boundary = signals.loc[periods == VALIDATION, 'session_date'].min()
    last = signals.index[periods == DEVELOPMENT][-1]
    # its ticker missed sessions, so its known window ends on the next period's first date
    signals.loc[last, 'window_end'] = boundary

    moved = assign_periods(signals, {'US': SESSIONS})

    assert periods[last] == DEVELOPMENT and moved[last] == GAP
    assert (moved.drop(index=last) == periods.drop(index=last)).all()


def test_the_dataset_reads_one_screener_version(db):
    day = date(2020, 1, 2)
    _add_signal(db.db_path, 1, day)
    # the same ticker and session archived again by a later version, with its own values
    _add_signal(db.db_path, 2, day, ticker='T1', screener_version='1.1.0', volume_ratio=9.0)
    _add_outcome(db.db_path, 1, 5, 'complete')
    _add_outcome(db.db_path, 2, 5, 'complete')

    current = load_dataset(db, with_history=False)
    later = load_dataset(db, with_history=False, screener_version='1.1.0')

    assert list(current['signal_id']) == [1] and list(current['volume_ratio']) == [2.5]
    assert list(later['signal_id']) == [2] and list(later['volume_ratio']) == [9.0]


def test_a_missing_history_stops_the_dataset_and_creates_no_file(db, tmp_path):
    db.history_db_path = str(tmp_path / 'nowhere.db')
    _add_signal(db.db_path, 1, date(2020, 1, 2))
    _add_outcome(db.db_path, 1, 5, 'complete')

    with pytest.raises(MissingHistory, match='nowhere.db'):
        load_dataset(db, with_history=True)
    assert not (tmp_path / 'nowhere.db').exists()
    assert list(load_dataset(db, with_history=False)['signal_id']) == [1]


def test_window_ends_are_the_latest_exit_or_last_session():
    outcomes = pd.DataFrame(
        {
            'signal_id': [1, 1, 2],
            'outcome_status': ['complete', 'ended', 'complete'],
            'exit_date': pd.to_datetime(['2020-02-03', None, '2020-01-10']),
            'last_session_date': pd.to_datetime([None, '2020-03-02', None]),
        }
    )

    assert window_ends(outcomes).to_dict() == {1: pd.Timestamp('2020-03-02'), 2: pd.Timestamp('2020-01-10')}
