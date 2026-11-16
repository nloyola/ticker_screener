"""
The secondary indicators ranked: which of them predict a better main outcome or less downside.

Each raw value is cut into quantile bands, and its highest band compared with its lowest; each flag, and each of the
strategy's secondary indicators that combines several flags, is compared passing against failing. A comparison is a
regression of the outcome on the group, so its effect is the difference in the outcome's mean (in a rate, a
difference in shares), with standard errors clustered two-way by signal date and by ticker (stats.clustered_ols).
The returns (HORIZON_OUTCOMES) are compared at each horizon, every comparison of a horizon one family. The breakout
failure is the signal's, decided over its first sessions whatever the horizon, so it is compared once over every
signal with a decided failure (SIGNAL_OUTCOMES), its results stored with no horizon, its comparisons one family of
their own. Each family is corrected by Benjamini-Hochberg at stats.FDR, and each result keeps both its raw and its
adjusted p-value.

Like the baseline, the ranking describes the whole archive, gap included: it chooses no rule, which only the
candidate search does, on development alone. Returns are compared on complete outcomes only; ended (censored)
horizons are counted in each group, never dropped. A comparison whose two groups do not each hold observations
(complete outcomes for a return, decided failures for the failure) on MIN_SIGNAL_DATES distinct signal dates is not
shown, and is not tested.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.analyzer import SECONDARY_INDICATORS
from app.breakoutiq.dataset import COMPLETE, ENDED, pct_change
from app.breakoutiq.stats import (
    CLUSTERS,
    CONFIDENCE,
    FDR,
    MIN_SIGNAL_DATES,
    benjamini_hochberg,
    cluster_codes,
    clustered_ols,
)
from app.models import AnalysisResult

SECTION = 'indicators'
PAYLOAD_VERSION = 1

# the kinds of indicator, the first part of a finding's name ('<kind>:<indicator>:<outcome>')
VALUE = 'value'
FLAG = 'flag'
SECONDARY = 'secondary'

# the raw values, each read from a column of the dataset or derived from them, in percent where named so:
# - the MACD histogram over the close, so tickers of every price compare
# - the close's distance above the 20- and 200-day SMAs and the daily and weekly 21-period EMAs (the 50-day SMA's is
#   a mandatory strength, which the mandatory section tests)
# - secondary_passed: how many of the seven secondary indicators the signal passed
VALUES = (
    'rsi',
    'macd_hist_pct',
    'macd_days_above',
    'sma20_distance_pct',
    'sma200_distance_pct',
    'ema21_distance_pct',
    'ema21w_distance_pct',
    'close_position',
    'consolidation_range_pct',
    'consolidation_vol_ratio',
    'return_1m_pct',
    'return_3m_pct',
    'rs_market_1m',
    'rs_market_3m',
    'rs_sector_1m',
    'rs_sector_3m',
    'secondary_passed',
)
_DERIVED = {
    'macd_hist_pct': ('macd_hist', 'close'),
    'sma20_distance_pct': ('close', 'dma_20'),
    'sma200_distance_pct': ('close', 'dma_200'),
    'ema21_distance_pct': ('close', 'ema_21'),
    'ema21w_distance_pct': ('close', 'ema_21w'),
}

# the secondary flags. The close above a rising 50-day SMA is a mandatory condition every signal passes, so its two
# flags are left out, and so is breakout proximity, which defines the watchlist rather than a breakout's quality
FLAGS = (
    'rsi_50_70',
    'macd_above_signal',
    'macd_crossover',
    'macd_hist_rising',
    'price_gt_20dma',
    'dma_20_rising',
    'price_gt_200dma',
    'dma_200_rising',
    'price_gt_21ema',
    'ema_21_rising',
    'price_gt_21wema',
    'ema_21w_rising',
    'close_upper_quarter',
    'consolidation_tight',
    'consolidation_vol_declining',
    'outperforms_market_1m',
    'outperforms_market_3m',
    'outperforms_sector_1m',
    'outperforms_sector_3m',
)

# the strategy's secondary indicators (SECONDARY_INDICATORS) that combine several flags; the rest are one flag each
# (RSI, consolidation quality, closing strength), already compared as that flag
COMPOSITES = ('macd', 'relative_strength', 'moving_average_alignment', 'ema_21_alignment')

# the outcomes compared: the main one, the net excess return over the benchmark, and the downside measures, the MAE
# and the breakout failure. Each maps to its column, and whether it is read on complete outcomes only (the returns)
# or on every signal with a decided failure (the failure)
MAIN_OUTCOME = 'excess_return'
FAILURE = 'failure'


@dataclass(frozen=True)
class Outcome:
    column: str
    complete_only: bool


OUTCOMES = {
    MAIN_OUTCOME: Outcome('excess_return_pct', True),
    'mae': Outcome('mae_pct', True),
    FAILURE: Outcome('breakout_failed', False),
}
# the outcomes compared at each horizon, and those compared once over the signals, with no horizon
HORIZON_OUTCOMES = (MAIN_OUTCOME, 'mae')
SIGNAL_OUTCOMES = (FAILURE,)

# the quantile bands a raw value is cut into, over the signals (each once); values tied across a quantile give fewer
BANDS = 5

# how a comparison's two groups are labelled, the lower one first
_FLAG_GROUPS = ('fail', 'pass')


def indicators(rows: pd.DataFrame) -> list[AnalysisResult]:
    """
    The section's results from the analysis dataset (dataset.build_dataset): per horizon, one per indicator and each
    of HORIZON_OUTCOMES, then, with no horizon, one per indicator and each of SIGNAL_OUTCOMES; VALUES first, then
    FLAGS, then COMPOSITES. The finding is '<kind>:<indicator>:<outcome>'.
    """
    frame = indicator_frame(rows)
    edges = {name: band_edges(frame, name) for name in VALUES}

    def compare(context: 'Horizon', at: pd.DataFrame, outcomes: tuple[str, ...]) -> list[dict]:
        payloads = []
        for kind, names in ((VALUE, VALUES), (FLAG, FLAGS), (SECONDARY, COMPOSITES)):
            for name in names:
                if kind == VALUE:
                    groups, labels = bands(at[name], edges[name]), None
                else:
                    groups, labels = _flag_groups(at[name]), _FLAG_GROUPS
                for outcome in outcomes:
                    payload = comparison(context, groups.to_numpy(dtype=float), labels, outcome)
                    payload = {'kind': kind, 'indicator': name, 'outcome': outcome, **payload}
                    if kind == VALUE:
                        payload['edges'] = edges[name]
                    payloads.append(payload)
        return payloads

    return compare_all(SECTION, PAYLOAD_VERSION, frame, compare)


def compare_all(
    section: str,
    payload_version: int,
    frame: pd.DataFrame,
    compare: Callable[['Horizon', pd.DataFrame, tuple[str, ...]], list[dict]],
) -> list[AnalysisResult]:
    """
    A comparing section's results: 'compare' run on each horizon's rows for HORIZON_OUTCOMES, then once on one row
    per signal for SIGNAL_OUTCOMES, with no horizon. Each call's payloads are one family, corrected and ranked, and
    stored as '<kind>:<indicator>:<outcome>'.
    """
    families = [
        (int(horizon), compare(Horizon.of(at), at, HORIZON_OUTCOMES))
        for horizon, at in frame.groupby('horizon', sort=True)
    ]
    if not frame.empty:
        signals = frame.drop_duplicates('signal_id')
        families.append((None, compare(Horizon.of(signals, per_signal=True), signals, SIGNAL_OUTCOMES)))
    results = []
    for horizon, payloads in families:
        correct(payloads)
        rank(payloads)
        results += [
            AnalysisResult(section, f'{p["kind"]}:{p["indicator"]}:{p["outcome"]}', horizon, payload_version, p)
            for p in payloads
        ]
    return results


@dataclass(frozen=True)
class Horizon:
    """
    A horizon's rows as the comparisons read them, each array aligned with the rows; or, 'per_signal', one row per
    signal, for the outcomes that do not depend on the horizon, with no row censored.
    """

    dates: np.ndarray  # each row's signal date, as a code
    signals: np.ndarray  # each row's signal, as a code
    clusters: np.ndarray  # each row's CLUSTERS, as codes (rows x 2)
    ended: np.ndarray  # whether the row's horizon ended (censored)
    outcomes: dict[str, np.ndarray]  # each of OUTCOMES' observations, NaN on a row it does not read

    @classmethod
    def of(cls, rows: pd.DataFrame, per_signal: bool = False) -> 'Horizon':
        complete = (rows['outcome_status'] == COMPLETE).to_numpy()
        outcomes = {}
        for name, spec in OUTCOMES.items():
            values = pd.to_numeric(rows[spec.column], errors='coerce').to_numpy(dtype=float)
            outcomes[name] = np.where(complete, values, np.nan) if spec.complete_only else values
        return cls(
            dates=pd.factorize(rows['session_date'])[0],
            signals=pd.factorize(rows['signal_id'])[0],
            clusters=cluster_codes(rows),
            ended=np.zeros(len(rows), dtype=bool) if per_signal else (rows['outcome_status'] == ENDED).to_numpy(),
            outcomes=outcomes,
        )


def indicator_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """'rows' with the derived VALUES and the COMPOSITES added: each composite 1 when it passes, else 0, an empty flag
    counting as not passed, as SECONDARY_INDICATORS judges it."""
    frame = value_frame(rows)
    flags = frame[[*FLAGS, 'price_gt_50dma']]
    for name in COMPOSITES:
        frame[name] = [int(SECONDARY_INDICATORS[name](row)) for row in flags.itertuples(index=False)]
    return frame


def value_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """'rows' with the VALUES derived from the close and the averages added."""
    frame = rows.copy()
    for name, (value, base) in _DERIVED.items():
        frame[name] = pct_change(frame[value], frame[base])
    return frame


def band_edges(frame: pd.DataFrame, name: str) -> list[float]:
    """
    The edges of 'name''s quantile bands over the signals in 'frame', each signal once: BANDS + 1 edges, lowest
    first, fewer when values are tied across a quantile, and none when the signals hold fewer than two distinct
    values.
    """
    values = pd.to_numeric(frame.drop_duplicates('signal_id')[name], errors='coerce').dropna()
    if values.nunique() < 2:
        return []
    edges = np.unique(np.quantile(values, np.linspace(0, 1, BANDS + 1)))
    return [float(e) for e in edges] if len(edges) >= 2 else []


def bands(values: pd.Series, edges: list[float]) -> pd.Series:
    """Each row's band, 1 for the lowest, aligned with 'values'; NaN without a value. A band holds the values above
    its lower edge up to its upper one, the lowest band its lower edge too."""
    if not edges:
        return pd.Series(np.nan, index=values.index)
    bands = pd.cut(pd.to_numeric(values, errors='coerce'), edges, labels=False, include_lowest=True)
    return bands + 1


def _flag_groups(values: pd.Series) -> pd.Series:
    """Each row's group, 1 failing and 2 passing, as bands are numbered; NaN for an empty flag."""
    return pd.to_numeric(values, errors='coerce') + 1


def comparison(context: Horizon, groups: np.ndarray, labels: tuple[str, ...] | None, outcome: str) -> dict:
    """
    One indicator's comparison on 'outcome' at a horizon: every group's counts and mean, and the effect of the
    highest group over the lowest, with its clustered interval and raw p-value when both meet the minimum sample.
    'groups' numbers each row's group from 1 (NaN in none), and 'labels' names group N at N - 1, or None to name
    each by its number.
    """
    y = context.outcomes[outcome]
    observed = ~np.isnan(y) & ~np.isnan(groups)
    numbers = [int(g) for g in np.unique(groups[~np.isnan(groups)])]

    group_payloads = []
    for number in numbers:
        in_group = groups == number
        seen = in_group & observed
        dates = _distinct(context.dates, seen)
        group_payloads.append(
            {
                'group': labels[number - 1] if labels else number,
                'signals': _distinct(context.signals, in_group),
                'n': int(seen.sum()),
                'signal_dates': dates,
                'censored': int((in_group & context.ended).sum()),
                'mean': float(y[seen].mean()) if dates >= MIN_SIGNAL_DATES else None,
            }
        )

    payload = {
        'main': outcome == MAIN_OUTCOME,
        'groups': group_payloads,
        'compared': None,
        'shown': False,
        'effect': None,
        'p_value': None,
        'p_adjusted': None,
        'significant': False,
        'rank': None,
        'min_signal_dates': MIN_SIGNAL_DATES,
        'confidence': CONFIDENCE,
        'fdr': FDR,
        'clusters': list(CLUSTERS),
    }
    if len(numbers) < 2:
        return payload
    low, high = group_payloads[0], group_payloads[-1]
    payload['compared'] = [low['group'], high['group']]
    if min(low['signal_dates'], high['signal_dates']) < MIN_SIGNAL_DATES:
        return payload

    tested = observed & ((groups == numbers[0]) | (groups == numbers[-1]))
    exog = pd.DataFrame({'const': 1.0, 'effect': (groups[tested] == numbers[-1]).astype(float)})
    fit = clustered_ols(y[tested], exog, context.clusters[tested], of=['effect'])
    if fit is None:
        return payload
    effect = fit['effect']
    payload['shown'] = True
    payload['effect'] = {
        'estimate': effect.estimate,
        'se': effect.se,
        'ci_low': effect.ci_low,
        'ci_high': effect.ci_high,
        'n': int(tested.sum()),
        'signal_dates': _distinct(context.dates, tested),
    }
    payload['p_value'] = effect.p_value
    return payload


def _distinct(codes: np.ndarray, mask: np.ndarray) -> int:
    """How many distinct codes the rows in 'mask' hold."""
    return len(np.unique(codes[mask]))


def correct(payloads: list[dict]) -> None:
    """Sets each tested payload's adjusted p-value over the family 'payloads', and whether it is significant at
    FDR."""
    tested = [p for p in payloads if p['p_value'] is not None]
    for payload, adjusted in zip(tested, benjamini_hochberg([p['p_value'] for p in tested]), strict=True):
        payload['p_adjusted'] = adjusted
        payload['significant'] = adjusted <= FDR


def rank(payloads: list[dict], effect: str = 'effect') -> None:
    """Ranks the tested payloads of each outcome, 1 for the strongest evidence: the smallest raw p-value, then the
    largest effect (the payload's 'effect' key) relative to its standard error."""
    for outcome in OUTCOMES:
        tested = [p for p in payloads if p['outcome'] == outcome and p['p_value'] is not None]
        tested.sort(key=lambda p: (p['p_value'], -abs(p[effect]['estimate'] / p[effect]['se'])))
        for position, payload in enumerate(tested, start=1):
            payload['rank'] = position
