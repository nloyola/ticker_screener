"""
BreakoutIQ's test of the mandatory signals' strength: each strength has a result per horizon and outcome with its
bands, effect and sample sizes, a planted effect is found and nothing else, and every result says it measures
strength, not whether the rule is needed.
"""

import numpy as np
import pandas as pd

from app.breakoutiq.indicators import BANDS, FAILURE, HORIZON_OUTCOMES, MAIN_OUTCOME
from app.breakoutiq.mandatory import (
    NECESSITY_NOTE,
    PAYLOAD_VERSION,
    SECTION,
    STRENGTH,
    STRENGTHS,
    mandatory,
)
from app.breakoutiq.stats import FDR, MIN_SIGNAL_DATES

SESSIONS = pd.bdate_range('2016-01-04', periods=600)


def _dataset(seed: int, signals_per_date: int = 4, dates: int = 300, horizons=(20,), plant=None) -> pd.DataFrame:
    """
    A dataset of complete outcomes whose strengths are drawn at random, each past its rule as every signal's is,
    independent of each other and of the outcomes, with a shock shared by a date's signals and a drift per ticker.
    'plant' (name, size) adds size times the strength's standardized score to the excess return.
    """
    rng = np.random.default_rng(seed)
    count = signals_per_date * dates
    tickers = rng.integers(0, 150, size=count)
    frame = pd.DataFrame(
        {
            'signal_id': np.arange(1, count + 1),
            'session_date': np.repeat(SESSIONS[:dates], signals_per_date),
            'ticker': [f'T{t}' for t in tickers],
            'breakout_margin_pct': rng.exponential(1.5, count),
            'sma50_distance_pct': rng.exponential(6, count),
            'sma50_slope_pct': rng.exponential(3, count),
            'volume_ratio': 1.5 + rng.exponential(1, count),
        }
    )
    excess = np.repeat(rng.normal(0, 4, dates), signals_per_date) + rng.normal(0, 2, 150)[tickers]
    excess = excess + rng.normal(0, 6, count)
    if plant:
        name, size = plant
        values = frame[name].to_numpy(dtype=float)
        excess = excess + size * (values - values.mean()) / values.std()
    frame['excess_return_pct'] = excess
    frame['mae_pct'] = -np.abs(rng.normal(4, 2, count))
    frame['breakout_failed'] = rng.integers(0, 2, count)

    rows = pd.concat([frame.assign(horizon=h) for h in horizons], ignore_index=True)
    rows['outcome_status'] = 'complete'
    return rows


def _payload(results, finding, horizon=20) -> dict:
    return next(r.payload for r in results if r.finding == finding and r.horizon == horizon)


def test_each_strength_has_a_result_per_horizon_with_its_bands_effects_and_sample_sizes():
    rows = _dataset(1, horizons=(5, 20))
    results = mandatory(rows)

    expected = {
        *((f'{STRENGTH}:{name}:{outcome}', h) for name in STRENGTHS for outcome in HORIZON_OUTCOMES for h in (5, 20)),
        *((f'{STRENGTH}:{name}:{FAILURE}', None) for name in STRENGTHS),
    }
    assert {(r.finding, r.horizon) for r in results} == expected
    assert len(results) == len(expected)
    for r in results:
        p = r.payload
        assert r.section == SECTION
        assert r.payload_version == PAYLOAD_VERSION
        assert len(p['edges']) == BANDS + 1
        assert [g['group'] for g in p['groups']] == list(range(1, BANDS + 1))
        assert sum(g['signals'] for g in p['groups']) == rows['signal_id'].nunique()
        assert all(g['signal_dates'] >= MIN_SIGNAL_DATES and g['n'] > 0 for g in p['groups'])
        assert p['compared'] == [1, BANDS]
        assert p['shown']
        effect = p['effect']
        assert effect['ci_low'] <= effect['estimate'] <= effect['ci_high']
        assert effect['n'] > 0 and effect['signal_dates'] >= MIN_SIGNAL_DATES
        assert p['p_adjusted'] >= p['p_value']


def test_every_result_says_it_measures_strength_not_necessity():
    results = mandatory(_dataset(2))

    for r in results:
        assert r.payload['measures'] == 'strength'
        assert r.payload['tests_necessity'] is False
        assert r.payload['note'] == NECESSITY_NOTE
        assert 'not necessity' in r.payload['note']
        assert r.payload['rule'] == STRENGTHS[r.payload['indicator']]


def test_a_planted_strength_effect_is_found_and_nothing_else():
    results = mandatory(_dataset(3, plant=('volume_ratio', 4.0)))

    found = {r.finding for r in results if r.payload['significant']}
    assert found == {f'{STRENGTH}:volume_ratio:{MAIN_OUTCOME}'}
    payload = _payload(results, f'{STRENGTH}:volume_ratio:{MAIN_OUTCOME}')
    assert payload['effect']['estimate'] > 0
    assert payload['p_adjusted'] <= FDR
    assert payload['rank'] == 1


def test_ended_horizons_are_counted_in_their_band_but_never_measured():
    rows = _dataset(4)
    ended = rows['signal_id'] % 5 == 0
    rows.loc[ended, 'outcome_status'] = 'ended'
    rows.loc[ended, ['excess_return_pct', 'mae_pct']] = np.nan

    payload = _payload(mandatory(rows), f'{STRENGTH}:sma50_slope_pct:{MAIN_OUTCOME}')
    assert sum(g['censored'] for g in payload['groups']) == ended.sum()
    assert sum(g['n'] for g in payload['groups']) == (~ended).sum()
