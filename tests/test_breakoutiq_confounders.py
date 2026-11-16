"""
BreakoutIQ's control for confounders and redundancy: where one indicator only proxies for another, the regression
credits the true one and names the proxy redundant; an indicator that only proxies for its sector loses its effect
once the sector is held fixed; an effect present in one regime only shows in that regime's subgroup; and an indicator
missing on too many rows, or constant, is left out and says why. With no effect the large majority of simulations
find none.
"""

import json

import numpy as np
import pandas as pd

from app.breakoutiq.confounders import (
    ADJUSTED,
    CONSTANT,
    INDICATORS,
    MISSING,
    MODEL,
    PAYLOAD_VERSION,
    REDUNDANT_CORRELATION,
    REDUNDANT_VIF,
    SECTION,
    confounders,
)
from app.breakoutiq.indicators import VALUES
from app.breakoutiq.mandatory import STRENGTHS

SESSIONS = pd.bdate_range('2016-01-04', periods=600)
_SECTORS = ('Energy', 'Health', 'Technology')
# the stored raw values drawn at random, besides those VALUES derives from close and the averages
_RAW = [v for v in VALUES if v != 'macd_hist_pct' and not v.endswith('_distance_pct')]


def _dataset(seed: int, signals_per_date: int = 4, dates: int = 300, horizons=(20,)) -> pd.DataFrame:
    """
    A dataset of complete outcomes whose indicators are drawn at random, independent of each other and of the
    outcomes, each signal in a sector, a market and a regime, with a shock shared by a date's signals and a drift
    per ticker. The tests plant their effects on the returned frame's excess return.
    """
    rng = np.random.default_rng(seed)
    count = signals_per_date * dates
    tickers = rng.integers(0, 150, size=count)
    frame = pd.DataFrame(
        {
            'signal_id': np.arange(1, count + 1),
            'session_date': np.repeat(SESSIONS[:dates], signals_per_date),
            'ticker': [f'T{t}' for t in tickers],
            'sector': np.array(_SECTORS)[tickers % len(_SECTORS)],
            'market': np.where(tickers < 120, 'US', 'Canada'),
            'regime': np.repeat(np.where(np.arange(dates) % 50 < 30, 'above', 'below'), signals_per_date),
            'close': 100.0,
            'macd_hist': rng.normal(0.5, 0.3, count),
            'dma_20': 100 / (1 + rng.normal(0.03, 0.02, count)),
            'dma_200': 100 / (1 + rng.normal(0.15, 0.1, count)),
            'ema_21': 100 / (1 + rng.normal(0.03, 0.02, count)),
            'ema_21w': 100 / (1 + rng.normal(0.08, 0.05, count)),
            'breakout_margin_pct': rng.exponential(1.5, count),
            'sma50_distance_pct': rng.exponential(6, count),
            'sma50_slope_pct': rng.exponential(3, count),
            'volume_ratio': 1.5 + rng.exponential(1, count),
        }
    )
    for name in _RAW:
        frame[name] = rng.normal(0, 1, count) if name != 'secondary_passed' else rng.integers(0, 8, count)
    excess = np.repeat(rng.normal(0, 4, dates), signals_per_date) + rng.normal(0, 2, 150)[tickers]
    frame['excess_return_pct'] = excess + rng.normal(0, 6, count)
    frame['mae_pct'] = -np.abs(rng.normal(4, 2, count))
    frame['breakout_failed'] = rng.integers(0, 2, count)
    frame['outcome_status'] = 'complete'
    return pd.concat([frame.assign(horizon=h) for h in horizons], ignore_index=True)


def _standard(values: pd.Series) -> np.ndarray:
    values = values.to_numpy(dtype=float)
    return (values - values.mean()) / values.std()


def _payload(results, finding, horizon=20) -> dict:
    return next(r.payload for r in results if r.finding == finding and r.horizon == horizon)


def _covers(effect: dict, value: float) -> bool:
    return effect['ci_low'] <= value <= effect['ci_high']


def test_a_proxy_is_named_redundant_and_the_true_indicator_credited():
    rows = _dataset(6)
    rng = np.random.default_rng(11)
    # rs_market_1m only proxies for rs_market_3m: correlated about 0.95, with no effect of its own
    rows['rs_market_1m'] = 0.95 * rows['rs_market_3m'] + 0.3 * rng.normal(0, 1, len(rows))
    rows['excess_return_pct'] += 3.0 * _standard(rows['rs_market_3m'])

    results = confounders(rows)
    true = _payload(results, f'{ADJUSTED}:rs_market_3m')
    proxy = _payload(results, f'{ADJUSTED}:rs_market_1m')

    # alone, the proxy looks predictive
    assert proxy['unadjusted']['ci_low'] > 0
    # together, the true indicator gets the effect and the proxy none
    assert true['significant'] and _covers(true['adjusted'], 3.0)
    assert not proxy['significant'] and _covers(proxy['adjusted'], 0.0)
    assert true['rank'] == 1
    # the proxy is named redundant, with the true one, and the true one is not
    assert proxy['redundant'] and proxy['redundant_with'] == ['rs_market_3m']
    assert true['redundant'] is False
    assert [c['indicator'] for c in proxy['correlated']] == ['rs_market_3m']
    assert proxy['correlated'][0]['r'] >= REDUNDANT_CORRELATION
    assert true['high_vif'] and true['vif'] >= REDUNDANT_VIF
    # no other indicator is significant, nor redundant
    others = [r.payload for r in results if r.finding.startswith(ADJUSTED) and r.horizon == 20]
    assert {p['indicator'] for p in others if p['significant']} == {'rs_market_3m'}
    assert {p['indicator'] for p in others if p['redundant']} == {'rs_market_1m'}


def test_an_indicator_that_proxies_for_its_sector_loses_its_effect_once_the_sector_is_fixed():
    rows = _dataset(2)
    technology = (rows['sector'] == 'Technology').to_numpy(dtype=float)
    rows['rsi'] = rows['rsi'] + 1.5 * technology
    rows['excess_return_pct'] += 6.0 * technology

    payload = _payload(confounders(rows), f'{ADJUSTED}:rsi')

    assert payload['unadjusted']['ci_low'] > 0
    assert not payload['significant'] and _covers(payload['adjusted'], 0.0)
    assert payload['controls'] == ['market', 'sector', 'regime']


def test_an_effect_in_one_regime_shows_in_that_regime_only():
    rows = _dataset(3, dates=400)
    above = (rows['regime'] == 'above').to_numpy(dtype=float)
    rows['excess_return_pct'] += 4.0 * above * _standard(rows['volume_ratio'])

    payload = _payload(confounders(rows), f'{ADJUSTED}:volume_ratio')
    by_regime = {g['group']: g for g in payload['by_regime']}

    assert set(by_regime) == {'above', 'below'}
    assert by_regime['above']['shown'] and by_regime['above']['ci_low'] > 2.0
    assert _covers(by_regime['below'], 0.0)
    assert [g['group'] for g in payload['by_sector']] == list(_SECTORS)
    assert payload['sign_agreement']['sector'] == 1.0


def test_a_sparse_or_constant_indicator_is_left_out_and_says_why():
    rows = _dataset(4)
    rows.loc[rows.index % 2 == 0, 'rs_sector_3m'] = np.nan
    rows['consolidation_range_pct'] = 5.0

    results = confounders(rows)
    model = _payload(results, MODEL)

    assert model['left_out'] == {'consolidation_range_pct': CONSTANT, 'rs_sector_3m': MISSING}
    assert model['missing_share']['rs_sector_3m'] == 0.5
    # the sparse indicator does not drop rows from the regression
    assert model['n'] == len(rows) and model['missing'] == 0
    for name, reason in model['left_out'].items():
        payload = _payload(results, f'{ADJUSTED}:{name}')
        assert not payload['shown'] and payload['left_out'] == reason and payload['adjusted'] is None


def test_every_indicator_has_a_result_per_horizon_and_the_payloads_store_as_json():
    rows = _dataset(5, horizons=(5, 20))
    rows.loc[rows.index[:40], 'outcome_status'] = 'ended'
    rows.loc[rows.index[:40], 'excess_return_pct'] = np.nan

    results = confounders(rows)

    assert {r.section for r in results} == {SECTION}
    assert {r.payload_version for r in results} == {PAYLOAD_VERSION}
    for horizon in (5, 20):
        findings = [r.finding for r in results if r.horizon == horizon]
        assert findings == [MODEL, *(f'{ADJUSTED}:{name}' for name in INDICATORS)]
    assert set(INDICATORS) == set(VALUES) | set(STRENGTHS)
    model = _payload(results, MODEL, horizon=5)
    assert model['censored'] == 40 and model['n'] == len(rows) // 2 - 40
    assert model['shown'] and len(model['correlations']['matrix']) == len(model['indicators'])
    json.dumps([r.payload for r in results], allow_nan=False)


def test_with_no_effect_the_large_majority_of_simulations_find_none():
    # a date's signals share a shock, a ticker's a drift, and the regime is set per date: the clustered errors keep
    # the family's false discoveries near the FDR
    simulations = 40
    found = [
        any(r.payload['significant'] for r in confounders(_dataset(200 + i)) if r.finding.startswith(ADJUSTED))
        for i in range(simulations)
    ]

    assert sum(found) <= 0.15 * simulations
