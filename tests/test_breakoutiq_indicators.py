"""
BreakoutIQ's ranking of the secondary indicators: on a synthetic dataset with one planted effect it finds that effect
and no other after correction, on datasets with no effect it finds none in the large majority of simulations even
when a date's signals share a shock, and each result keeps its bands, sample sizes, censored counts and raw and
adjusted p-values.
"""

import numpy as np
import pandas as pd
import pytest

from app.breakoutiq.indicators import (
    BANDS,
    COMPOSITES,
    FAILURE,
    FLAG,
    FLAGS,
    HORIZON_OUTCOMES,
    MAIN_OUTCOME,
    OUTCOMES,
    PAYLOAD_VERSION,
    SECONDARY,
    SECTION,
    SIGNAL_OUTCOMES,
    VALUE,
    VALUES,
    band_edges,
    indicator_frame,
    indicators,
)
from app.breakoutiq.stats import FDR, MIN_SIGNAL_DATES, cluster_codes, clustered_ols

SESSIONS = pd.bdate_range('2016-01-04', periods=600)
# the stored raw values the synthetic signals draw at random, besides those VALUES derives from close and the averages
_RAW = [v for v in VALUES if v not in ('macd_hist_pct',) and not v.endswith('_distance_pct')]


def _dataset(
    seed: int, signals_per_date: int = 4, dates: int = 300, horizons=(20,), date_flags=(), plant: dict | None = None
) -> pd.DataFrame:
    """
    A dataset of complete outcomes whose indicators are drawn at random, independent of each other and of the
    outcomes: every date's signals share a market shock in their excess return, and each ticker has its own drift,
    so neither the signals of a date nor those of a ticker are independent. The flags in 'date_flags' take one value
    per date, as a market-wide condition would. 'plant' adds an effect to the excess return: flag=(name, size) adds
    size to the signals passing the flag, value=(name, size) adds size times the value's standardized score.
    """
    rng = np.random.default_rng(seed)
    count = signals_per_date * dates
    session = np.repeat(SESSIONS[:dates], signals_per_date)
    tickers = rng.integers(0, 150, size=count)
    frame = pd.DataFrame(
        {
            'signal_id': np.arange(1, count + 1),
            'session_date': session,
            'ticker': [f'T{t}' for t in tickers],
            'close': 100.0,
            'macd_hist': rng.normal(0.5, 0.3, count),
            'dma_20': 100 / (1 + rng.normal(0.03, 0.02, count)),
            'dma_200': 100 / (1 + rng.normal(0.15, 0.1, count)),
            'ema_21': 100 / (1 + rng.normal(0.03, 0.02, count)),
            'ema_21w': 100 / (1 + rng.normal(0.08, 0.05, count)),
            'price_gt_50dma': 1,
        }
    )
    for name in _RAW:
        frame[name] = rng.normal(0, 1, count) if name != 'secondary_passed' else rng.integers(0, 8, count)
    for name in FLAGS:
        if name in date_flags:
            frame[name] = np.repeat(rng.integers(0, 2, dates), signals_per_date)
        else:
            frame[name] = rng.integers(0, 2, count)

    shock = np.repeat(rng.normal(0, 4, dates), signals_per_date)
    drift = rng.normal(0, 2, 150)[tickers]
    excess = shock + drift + rng.normal(0, 6, count)
    plant = plant or {}
    if 'flag' in plant:
        name, size = plant['flag']
        excess = excess + size * frame[name].to_numpy()
    if 'value' in plant:
        name, size = plant['value']
        values = frame[name].to_numpy(dtype=float)
        excess = excess + size * (values - values.mean()) / values.std()
    frame['mae_pct'] = -np.abs(rng.normal(4, 2, count))
    frame['breakout_failed'] = rng.integers(0, 2, count)
    frame['excess_return_pct'] = excess

    rows = pd.concat([frame.assign(horizon=h) for h in horizons], ignore_index=True)
    rows['outcome_status'] = 'complete'
    return rows


def _significant(results) -> set[str]:
    return {r.finding for r in results if r.payload['significant']}


def _payload(results, finding, horizon=20) -> dict:
    return next(r.payload for r in results if r.finding == finding and r.horizon == horizon)


def test_a_planted_flag_effect_is_found_and_nothing_else():
    results = indicators(_dataset(1, plant={'flag': ('consolidation_vol_declining', 4.0)}))

    assert _significant(results) == {f'{FLAG}:consolidation_vol_declining:{MAIN_OUTCOME}'}
    payload = _payload(results, f'{FLAG}:consolidation_vol_declining:{MAIN_OUTCOME}')
    assert payload['compared'] == ['fail', 'pass']
    assert payload['effect']['ci_low'] <= 4.0 <= payload['effect']['ci_high']
    assert payload['rank'] == 1
    assert payload['p_adjusted'] <= FDR


def test_a_planted_value_effect_is_found_and_nothing_else():
    results = indicators(_dataset(2, plant={'value': ('rs_market_3m', 4.0)}))

    assert _significant(results) == {f'{VALUE}:rs_market_3m:{MAIN_OUTCOME}'}
    payload = _payload(results, f'{VALUE}:rs_market_3m:{MAIN_OUTCOME}')
    assert payload['compared'] == [1, BANDS]
    means = [g['mean'] for g in payload['groups']]
    assert means == sorted(means)
    assert payload['effect']['estimate'] > 0


def test_with_no_effect_the_large_majority_of_simulations_find_none():
    # each date's signals share a shock, and three flags are market-wide, one value per date: unclustered errors
    # would find those flags often
    # BH holds a family's chance of any false finding to about FDR when nothing is real, so each family, the horizon's
    # and the failure's, is judged on its own
    simulations = 40
    runs = [
        indicators(_dataset(100 + i, date_flags=('dma_200_rising', 'price_gt_21wema', 'rsi_50_70')))
        for i in range(simulations)
    ]

    for horizon in (20, None):
        found = [any(r.payload['significant'] for r in results if r.horizon == horizon) for results in runs]
        assert sum(found) <= 0.15 * simulations, horizon


def test_clustering_by_date_widens_the_error_of_a_market_wide_flag():
    # a date's 4 signals share a shock of variance 16 in an excess return of variance 56, so their correlation is
    # about 0.29, and a flag set per date has its variance inflated by 1 + 3 x 0.29, its error by about 1.36
    rows = _dataset(7, date_flags=('dma_200_rising',))
    y = rows['excess_return_pct'].to_numpy()
    passing = rows['dma_200_rising'] == 1
    exog = pd.DataFrame({'const': 1.0, 'effect': passing.astype(float)})
    clustered = clustered_ols(y, exog, cluster_codes(rows))['effect']
    naive_se = np.std(y, ddof=1) * np.sqrt(1 / passing.sum() + 1 / (~passing).sum())

    assert clustered.se > 1.2 * naive_se


def test_every_indicator_has_a_result_per_outcome_and_horizon():
    results = indicators(_dataset(3, horizons=(5, 20)))

    expected = {
        (f'{kind}:{name}:{outcome}', horizon)
        for kind, names in ((VALUE, VALUES), (FLAG, FLAGS), (SECONDARY, COMPOSITES))
        for name in names
        for outcome, horizon in [*((o, h) for o in HORIZON_OUTCOMES for h in (5, 20)), (FAILURE, None)]
    }
    assert {(r.finding, r.horizon) for r in results} == expected
    assert len(results) == len(expected)
    for r in results:
        assert r.section == SECTION
        assert r.payload_version == PAYLOAD_VERSION
        assert r.payload['shown']
        assert r.payload['p_adjusted'] >= r.payload['p_value']
        assert r.payload['main'] == (r.payload['outcome'] == MAIN_OUTCOME)


def test_each_horizon_and_outcome_is_ranked_from_one():
    results = indicators(_dataset(4, horizons=(5, 20)))

    for horizon, outcomes in ((5, HORIZON_OUTCOMES), (20, HORIZON_OUTCOMES), (None, SIGNAL_OUTCOMES)):
        for outcome in outcomes:
            ranked = sorted(
                (r.payload for r in results if r.horizon == horizon and r.payload['outcome'] == outcome),
                key=lambda p: p['rank'],
            )
            assert [p['rank'] for p in ranked] == list(range(1, len(ranked) + 1))
            assert [p['p_value'] for p in ranked] == sorted(p['p_value'] for p in ranked)


def test_value_bands_are_quantiles_over_the_signals():
    rows = _dataset(5, horizons=(5, 20))
    frame = indicator_frame(rows)

    edges = band_edges(frame, 'rsi')
    assert len(edges) == BANDS + 1
    payload = _payload(indicators(rows), f'{VALUE}:rsi:{MAIN_OUTCOME}')
    assert payload['edges'] == edges
    sizes = [g['signals'] for g in payload['groups']]
    assert sum(sizes) == len(rows) // 2
    assert max(sizes) - min(sizes) <= 1
    # secondary_passed takes 8 values, tied across quantiles: fewer distinct edges, still bands
    assert 2 <= len(band_edges(frame, 'secondary_passed')) <= BANDS + 1


def test_a_composite_passes_as_the_strategy_judges_it():
    rows = _dataset(6)
    frame = indicator_frame(rows)
    flags = frame[['price_gt_20dma', 'price_gt_50dma', 'price_gt_200dma']]

    assert (frame['moving_average_alignment'] == (flags == 1).all(axis=1).astype(int)).all()


def test_ended_horizons_are_counted_in_their_group_but_never_measured():
    rows = _dataset(8)
    ended = rows['signal_id'] % 5 == 0
    rows.loc[ended, 'outcome_status'] = 'ended'
    rows.loc[ended, ['excess_return_pct', 'mae_pct']] = np.nan

    payload = _payload(indicators(rows), f'{FLAG}:macd_crossover:{MAIN_OUTCOME}')
    assert sum(g['censored'] for g in payload['groups']) == ended.sum()
    assert payload['effect']['n'] == (~ended).sum()
    # the failure is the signal's: read once per signal with a decided failure, ended horizon or not, never censored
    failure = _payload(indicators(rows), f'{FLAG}:macd_crossover:{FAILURE}', horizon=None)
    assert failure['effect']['n'] == rows['signal_id'].nunique()
    assert sum(g['censored'] for g in failure['groups']) == 0


def test_a_group_on_too_few_dates_is_neither_shown_nor_tested():
    rows = _dataset(9)
    thin = rows['session_date'].isin(SESSIONS[: MIN_SIGNAL_DATES - 1])
    rows['close_upper_quarter'] = np.where(thin, 1, 0)

    results = indicators(rows)
    payload = _payload(results, f'{FLAG}:close_upper_quarter:{MAIN_OUTCOME}')
    assert not payload['shown']
    assert payload['effect'] is None and payload['p_value'] is None and payload['rank'] is None
    assert [g['signal_dates'] for g in payload['groups']] == [300 - MIN_SIGNAL_DATES + 1, MIN_SIGNAL_DATES - 1]
    assert payload['groups'][1]['mean'] is None
    # left out of the family, so the others' adjustment is over one fewer test
    tested = [r for r in results if r.payload['p_value'] is not None]
    assert len(tested) == len(results) - len(OUTCOMES)


def test_the_failure_is_compared_once_over_the_signals_whatever_the_horizons():
    rows = _dataset(11, horizons=(5, 20, 60))

    results = indicators(rows)
    failures = [r for r in results if r.payload['outcome'] == FAILURE]

    assert {r.horizon for r in failures} == {None}
    assert len(failures) == len(VALUES) + len(FLAGS) + len(COMPOSITES)
    payload = _payload(results, f'{FLAG}:macd_crossover:{FAILURE}', horizon=None)
    assert sum(g['signals'] for g in payload['groups']) == rows['signal_id'].nunique()


def test_a_flag_with_one_side_only_is_not_compared():
    rows = _dataset(10)
    rows['dma_20_rising'] = 1

    payload = _payload(indicators(rows), f'{FLAG}:dma_20_rising:{MAIN_OUTCOME}')
    assert payload['compared'] is None
    assert not payload['shown']
    assert [g['group'] for g in payload['groups']] == ['pass']


@pytest.mark.parametrize('seed', [11, 12])
def test_the_effect_is_the_difference_in_means(seed):
    rows = _dataset(seed)
    payload = _payload(indicators(rows), f'{FLAG}:macd_hist_rising:mae')
    passing = rows['macd_hist_rising'] == 1

    assert payload['effect']['estimate'] == pytest.approx(
        rows.loc[passing, 'mae_pct'].mean() - rows.loc[~passing, 'mae_pct'].mean()
    )
    assert [g['mean'] for g in payload['groups']] == pytest.approx(
        [rows.loc[~passing, 'mae_pct'].mean(), rows.loc[passing, 'mae_pct'].mean()]
    )
