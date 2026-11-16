"""
The strategy's baseline: how the archived breakouts performed at each horizon, over every signal and split by status,
source, market and regime, each measure with a block-bootstrap confidence interval (stats.py).

The baseline describes the whole archive, so it covers every period, gap included; the sections that compare periods
leave the gap out. Returns are measured on complete outcomes only; ended (censored) horizons are counted, never
dropped. A group with complete outcomes on fewer than MIN_SIGNAL_DATES distinct signal dates keeps its counts but
shows no measures.
"""

import numpy as np
import pandas as pd

from app.breakoutiq.dataset import COMPLETE, ENDED
from app.breakoutiq.stats import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CONFIDENCE,
    MIN_SIGNAL_DATES,
    block_bootstrap,
    block_mean,
    block_median,
    date_blocks,
)
from app.models import AnalysisResult

SECTION = 'baseline'
PAYLOAD_VERSION = 1

# the finding over every signal, and the columns each split groups by; a signal with no regime is in UNKNOWN
ALL = 'all'
SPLITS = ('status', 'source', 'market', 'regime')
UNKNOWN = 'unknown'

# the measures, in payload order. Rates are fractions from 0 to 1, the rest percent:
# - positive_rate: the share of complete outcomes with a net return above zero
# - outperformance_rate: the share of complete outcomes with an excess return (none without the benchmark's return)
#   above zero
# - the mean and median of each of RETURNS' columns, over complete outcomes
# - failure_rate: the share of the horizon's resolved rows, complete or ended, with a decided breakout failure whose
#   breakout failed
RETURNS = {
    'gross_return': 'gross_return_pct',
    'net_return': 'net_return_pct',
    'excess_return': 'excess_return_pct',
    'mfe': 'mfe_pct',
    'mae': 'mae_pct',
}
MEASURES = (
    'positive_rate',
    'outperformance_rate',
    *(f'{name}_{stat}' for name in RETURNS for stat in ('mean', 'median')),
    'failure_rate',
)
# the measures that are a mean of their observations (every rate is one); the rest are a median
_USE_MEAN = np.array([not name.endswith('_median') for name in MEASURES])


def baseline(
    rows: pd.DataFrame, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED
) -> list[AnalysisResult]:
    """
    The baseline's results from the analysis dataset (dataset.build_dataset): one per horizon and group, the group
    ALL first, then each value of each split in SPLITS. The finding is ALL or '<split>:<value>'.
    """
    results = []
    rows = rows.assign(regime=rows['regime'].where(rows['regime'].notna(), UNKNOWN))
    for horizon, at_horizon in rows.groupby('horizon', sort=True):
        groups = [(ALL, None, at_horizon)]
        for split in SPLITS:
            groups += [(f'{split}:{value}', split, group) for value, group in at_horizon.groupby(split, sort=True)]
        for finding, split, group in groups:
            payload = _payload(group, int(horizon), resamples, seed)
            payload = {'split': split, 'group': finding.split(':', 1)[1] if split else ALL, **payload}
            results.append(AnalysisResult(SECTION, finding, int(horizon), PAYLOAD_VERSION, payload))
    return results


def _payload(group: pd.DataFrame, horizon: int, resamples: int, seed: int) -> dict:
    """One group's counts at 'horizon', and its measures with their intervals when it meets the minimum sample."""
    complete = group['outcome_status'] == COMPLETE
    complete_dates = int(group.loc[complete, 'session_date'].nunique())
    shown = complete_dates >= MIN_SIGNAL_DATES
    payload = {
        'signals': int(group['signal_id'].nunique()),
        'signal_dates': int(group['session_date'].nunique()),
        'complete': int(complete.sum()),
        'complete_signal_dates': complete_dates,
        'censored': int((group['outcome_status'] == ENDED).sum()),
        'shown': shown,
        'min_signal_dates': MIN_SIGNAL_DATES,
        'confidence': CONFIDENCE,
        'block_sessions': horizon,
        'resamples': resamples,
        'measures': {},
    }
    if shown:
        payload['measures'] = measures(group, horizon, resamples, seed)
    return payload


def measures(group: pd.DataFrame, block_sessions: int, resamples: int, seed: int) -> dict:
    """
    Each of MEASURES on 'group' as {'estimate', 'ci_low', 'ci_high', 'n'}: the figure on the whole group, its interval
    from a bootstrap over blocks of 'block_sessions' sessions, and the observations it rests on. A figure with no
    observations is None.
    """
    values = _values(group)
    blocks = date_blocks(group['session_date'], block_sessions)
    figures = [
        (block_mean if use_mean else block_median)(row, blocks) for row, use_mean in zip(values, _USE_MEAN, strict=True)
    ]
    estimate = [_estimate(row, use_mean) for row, use_mean in zip(values, _USE_MEAN, strict=True)]
    low, high = block_bootstrap(blocks, figures, resamples, seed)
    counts = (~np.isnan(values)).sum(axis=1)
    return {
        name: {
            'estimate': _number(estimate[i]),
            'ci_low': _number(low[i]),
            'ci_high': _number(high[i]),
            'n': int(counts[i]),
        }
        for i, name in enumerate(MEASURES)
    }


def _values(group: pd.DataFrame) -> np.ndarray:
    """One row per measure, one column per dataset row: the observation, or NaN where the measure does not read it."""
    complete = (group['outcome_status'] == COMPLETE).to_numpy()

    def on_complete(series: pd.Series) -> np.ndarray:
        return np.where(complete, pd.to_numeric(series, errors='coerce').to_numpy(dtype=float), np.nan)

    net = on_complete(group['net_return_pct'])
    excess = on_complete(group['excess_return_pct'])
    failed = pd.to_numeric(group['breakout_failed'], errors='coerce').to_numpy(dtype=float)
    rows = {
        'positive_rate': np.where(np.isnan(net), np.nan, net > 0),
        'outperformance_rate': np.where(np.isnan(excess), np.nan, excess > 0),
        'failure_rate': failed,
    }
    for name, column in RETURNS.items():
        rows[f'{name}_mean'] = rows[f'{name}_median'] = on_complete(group[column])
    return np.vstack([rows[name] for name in MEASURES]).astype(float)


def _estimate(row: np.ndarray, use_mean: bool) -> float:
    """The mean or median of a measure's observations, NaN without any."""
    observed = row[~np.isnan(row)]
    if len(observed) == 0:
        return np.nan
    return float(observed.mean() if use_mean else np.median(observed))


def _number(value: float) -> float | None:
    return None if np.isnan(value) else float(value)
