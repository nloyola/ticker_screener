"""
BreakoutIQ's baseline: on synthetic data with known returns the measures and their block-bootstrap intervals come out
as constructed, the splits partition the signals, a thin group shows counts but no measures, and on an archive the
counts equal its resolved outcomes.
"""

import sqlite3
from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.breakoutiq.baseline import ALL, MEASURES, PAYLOAD_VERSION, SECTION, SPLITS, UNKNOWN, baseline
from app.breakoutiq.dataset import load_dataset
from app.breakoutiq.stats import MIN_SIGNAL_DATES, block_bootstrap, block_mean, block_median, date_blocks
from app.db import init_db
from app.repositories.analysis_data_repo import AnalysisDataRepo
from app.repositories.benchmark_price_repo import BenchmarkPriceRepo
from app.repositories.history_repo import HistoryRepo, init_history_db

SESSIONS = pd.bdate_range('2018-01-01', periods=1000)


def _rows(net, dates=None, horizon=5, **columns) -> pd.DataFrame:
    """A dataset of complete outcomes with net returns 'net', gross 0.2 above, excess 2 below, MFE and MAE around."""
    net = np.asarray(net, dtype=float)
    dates = SESSIONS[: len(net)] if dates is None else pd.DatetimeIndex(dates)
    values = {
        'signal_id': np.arange(1, len(net) + 1),
        'session_date': dates,
        'horizon': horizon,
        'outcome_status': 'complete',
        'gross_return_pct': net + 0.2,
        'net_return_pct': net,
        'benchmark_return_pct': 2.0,
        'excess_return_pct': net - 2.0,
        'mfe_pct': np.abs(net) + 2.0,
        'mae_pct': -np.abs(net) - 1.0,
        'breakout_failed': 0,
        'status': 'confirmed',
        'source': 'backfill',
        'market': 'US',
        'regime': 'above',
    }
    return pd.DataFrame({**values, **columns})


def _payload(results, finding=ALL, horizon=5) -> dict:
    return next(r.payload for r in results if r.finding == finding and r.horizon == horizon)


# the measures on known returns


def test_the_measures_are_the_figures_constructed():
    # a repeating pattern of net returns, so every figure is known: three of four positive, mean 1.5, median 2
    net = np.tile([4.0, 2.0, 2.0, -2.0], 20)
    failed = np.tile([1, 0, 0, 0, 0], 16)
    results = baseline(_rows(net, breakout_failed=failed), resamples=200)
    m = _payload(results)['measures']

    assert m['positive_rate']['estimate'] == pytest.approx(0.75)
    # excess is net less 2: 2, 0, 0, -4, and an excess of zero does not outperform
    assert m['outperformance_rate']['estimate'] == pytest.approx(0.25)
    assert m['net_return_mean']['estimate'] == pytest.approx(1.5)
    assert m['net_return_median']['estimate'] == pytest.approx(2.0)
    assert m['gross_return_mean']['estimate'] == pytest.approx(1.7)
    assert m['excess_return_mean']['estimate'] == pytest.approx(-0.5)
    assert m['excess_return_median']['estimate'] == pytest.approx(0.0)
    assert m['mfe_mean']['estimate'] == pytest.approx(4.5)
    assert m['mae_median']['estimate'] == pytest.approx(-3.0)
    assert m['failure_rate']['estimate'] == pytest.approx(0.2)
    for name in MEASURES:
        assert m[name]['n'] == 80
        assert m[name]['ci_low'] <= m[name]['estimate'] <= m[name]['ci_high']


def test_identical_returns_give_an_interval_of_that_return():
    m = _payload(baseline(_rows([3.0] * 60), resamples=100))['measures']

    assert m['net_return_mean'] == {'estimate': 3.0, 'ci_low': 3.0, 'ci_high': 3.0, 'n': 60}
    assert m['positive_rate']['ci_low'] == m['positive_rate']['ci_high'] == 1.0


def test_the_interval_covers_the_true_mean_and_median_as_often_as_its_confidence_says():
    # each date's signals share a market shock, so the signals of a date are not independent; the shocks and the
    # noise are symmetric, so the true mean and median are both 1.0
    rng = np.random.default_rng(7)
    blocks = date_blocks(pd.Series(np.repeat(SESSIONS[:250], 4)), 5)
    covered, block_widths, naive_widths = np.zeros(2), [], []
    sims = 200
    for sim in range(sims):
        net = 1.0 + np.repeat(rng.normal(0, 3, 250), 4) + rng.normal(0, 1, 1000)
        low, high = block_bootstrap(blocks, [block_mean(net, blocks), block_median(net, blocks)], 200, seed=sim)
        covered += (low <= 1.0) & (1.0 <= high)
        block_widths.append(high[0] - low[0])
        naive_widths.append(2 * 1.96 * net.std(ddof=1) / np.sqrt(len(net)))

    assert ((0.9 <= covered / sims) & (covered / sims <= 0.985)).all()
    # an interval treating the 1000 signals as independent would be far narrower than the date blocks'
    assert np.mean(block_widths) > 1.6 * np.mean(naive_widths)


def test_the_blocks_keep_each_date_and_its_neighbours_together():
    dates = pd.Series(pd.DatetimeIndex([SESSIONS[0], SESSIONS[0], SESSIONS[4], SESSIONS[5], SESSIONS[12]]))

    assert list(date_blocks(dates, 5)) == [0, 0, 0, 1, 2]


@pytest.mark.parametrize('size', [1, 2, 7, 300, 5000])
def test_a_draws_mean_and_median_are_numpys_on_the_drawn_rows(size):
    # ties, missing observations and more rows than the median's rank bins, against the rows each draw copies
    rng = np.random.default_rng(size)
    values = np.round(rng.normal(0, 3, size), 1)
    values[rng.random(size) < 0.1] = np.nan
    blocks = np.sort(rng.integers(0, max(size // 5, 1), size))
    blocks = np.unique(blocks, return_inverse=True)[1]
    counts = np.vstack([rng.integers(0, 3, blocks.max() + 1) for _ in range(40)]).astype(float)

    means, medians = block_mean(values, blocks)(counts), block_median(values, blocks)(counts)

    for draw, row in enumerate(counts):
        drawn = np.repeat(values, row[blocks].astype(int))
        drawn = drawn[~np.isnan(drawn)]
        if len(drawn) == 0:
            assert np.isnan(means[draw]) and np.isnan(medians[draw])
        else:
            assert means[draw] == pytest.approx(drawn.mean())
            assert medians[draw] == pytest.approx(np.median(drawn))


def test_one_block_gives_no_interval():
    low, high = block_bootstrap(np.zeros(10, dtype=int), [lambda counts: counts.sum(axis=1)], 50)

    assert np.isnan(low).all() and np.isnan(high).all()


# counts, censoring and the minimum sample


def test_ended_horizons_are_counted_but_never_measured():
    rows = _rows([2.0] * 40)
    ended = rows.iloc[:5].assign(
        signal_id=range(100, 105),
        outcome_status='ended',
        gross_return_pct=np.nan,
        net_return_pct=np.nan,
        excess_return_pct=np.nan,
        mfe_pct=np.nan,
        mae_pct=np.nan,
        breakout_failed=1,
    )
    payload = _payload(baseline(pd.concat([rows, ended], ignore_index=True), resamples=50))

    assert (payload['signals'], payload['complete'], payload['censored']) == (45, 40, 5)
    assert payload['measures']['net_return_mean']['n'] == 40
    # the failure is decided for an ended signal too
    failure = payload['measures']['failure_rate']
    assert (failure['estimate'], failure['n']) == (pytest.approx(5 / 45), 45)


def test_a_group_on_too_few_dates_shows_counts_and_no_measures():
    # two signals a date on one date fewer than the minimum
    dates = np.repeat(SESSIONS[: MIN_SIGNAL_DATES - 1], 2)
    payload = _payload(baseline(_rows([1.0] * len(dates), dates), resamples=50))

    assert payload['shown'] is False
    assert payload['measures'] == {}
    assert (payload['signals'], payload['signal_dates'], payload['complete_signal_dates']) == (
        2 * (MIN_SIGNAL_DATES - 1),
        MIN_SIGNAL_DATES - 1,
        MIN_SIGNAL_DATES - 1,
    )


def test_an_undecided_failure_or_missing_benchmark_is_left_out_of_its_rate():
    rows = _rows([1.0] * 40, breakout_failed=[None] * 10 + [1] * 30)
    rows.loc[:19, 'excess_return_pct'] = np.nan
    m = _payload(baseline(rows, resamples=50))['measures']

    assert m['failure_rate']['n'] == 30 and m['failure_rate']['estimate'] == 1.0
    assert m['outperformance_rate']['n'] == 20


# the splits


def test_every_split_partitions_the_signals_at_each_horizon():
    n = 120
    rows = pd.concat(
        [
            _rows(
                np.linspace(-5, 5, n),
                horizon=horizon,
                status=np.where(np.arange(n) % 3 == 0, 'high_conviction', 'confirmed'),
                source=np.where(np.arange(n) < 100, 'backfill', 'live'),
                market=np.where(np.arange(n) % 4 == 0, 'Canada', 'US'),
                regime=np.where(np.arange(n) < 10, None, np.where(np.arange(n) % 2 == 0, 'above', 'below')),
            )
            for horizon in (5, 20)
        ],
        ignore_index=True,
    )
    results = baseline(rows, resamples=20)

    assert {r.section for r in results} == {SECTION}
    assert {r.payload_version for r in results} == {PAYLOAD_VERSION}
    for horizon in (5, 20):
        at = [r for r in results if r.horizon == horizon]
        assert at[0].finding == ALL and at[0].payload['signals'] == n
        for split in SPLITS:
            groups = [r.payload for r in at if r.payload['split'] == split]
            assert sum(g['signals'] for g in groups) == n
    assert _payload(results, 'regime:' + UNKNOWN)['signals'] == 10
    assert _payload(results, 'source:live', 20)['signals'] == 20
    assert _payload(results, 'status:high_conviction')['group'] == 'high_conviction'


# an archive


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


def _archive(path: str, signals: int) -> None:
    """'signals' signals, one a weekday, with every horizon resolved but the last ones' longer horizons pending."""
    rng = np.random.default_rng(3)
    with sqlite3.connect(path) as con:
        for i in range(signals):
            day = (date(2020, 1, 6) + timedelta(days=7 * (i // 5) + i % 5)).isoformat()
            con.execute(
                'INSERT INTO signal (id, ticker, market, time_zone, session_date, close, status, screener_version, '
                'rule_settings, source, high_20d, high_60d, breakout_60d, dma_50, dma_50_prev_15d) VALUES (?, ?, ?, '
                "'America/New_York', ?, 110, ?, '1.0.0', '{}', ?, 100, 105, 0, 90, 85)",
                (
                    i + 1,
                    f'T{i % 7}',
                    'Canada' if i % 6 == 0 else 'US',
                    day,
                    'high_conviction' if i % 4 == 0 else 'confirmed',
                    'live' if i > signals - 10 else 'backfill',
                ),
            )
            for horizon in (5, 10, 20, 40, 60, 120):
                if i > signals - horizon // 2:
                    status = 'pending'
                elif i % 9 == 0 and horizon >= 40:
                    status = 'ended'
                else:
                    status = 'complete'
                net = float(rng.normal(1, 5))
                if status == 'complete':
                    con.execute(
                        'INSERT INTO signal_outcome (signal_id, horizon, status, entry_date, entry_price, exit_date, '
                        'exit_price, cost_bps, gross_return_pct, net_return_pct, benchmark_return_pct, '
                        "excess_return_pct, mfe_pct, mae_pct, completed_at) VALUES (?, ?, 'complete', ?, 100, ?, "
                        "100, 20, ?, ?, 0.5, ?, ?, ?, datetime('now'))",
                        (i + 1, horizon, day, day, net + 0.2, net, net - 0.5, abs(net) + 1, -abs(net) - 1),
                    )
                elif status == 'ended':
                    con.execute(
                        'INSERT INTO signal_outcome (signal_id, horizon, status, ended_reason, last_session_date) '
                        "VALUES (?, ?, 'ended', 'history_ended', ?)",
                        (i + 1, horizon, day),
                    )
                else:
                    con.execute(
                        "INSERT INTO signal_outcome (signal_id, horizon, status) VALUES (?, ?, 'pending')",
                        (i + 1, horizon),
                    )


def test_on_an_archive_the_counts_equal_its_resolved_outcomes(db):
    _archive(db.db_path, 300)
    results = baseline(load_dataset(db, with_history=False), resamples=20)

    with sqlite3.connect(db.db_path) as con:
        archive = con.execute(
            "SELECT o.horizon, s.status, count(*), sum(o.status = 'complete'), sum(o.status = 'ended'), "
            'count(DISTINCT s.session_date) FROM signal_outcome o JOIN signal s ON s.id = o.signal_id '
            "WHERE o.status <> 'pending' GROUP BY o.horizon, s.status"
        ).fetchall()
    for horizon, status, signals, complete, ended, dates in archive:
        payload = _payload(results, f'status:{status}', horizon)
        assert (payload['signals'], payload['complete'], payload['censored'], payload['signal_dates']) == (
            signals,
            complete,
            ended,
            dates,
        )
    totals = {h: sum(row[2] for row in archive if row[0] == h) for h in {row[0] for row in archive}}
    assert {r.horizon: r.payload['signals'] for r in results if r.finding == ALL} == totals
