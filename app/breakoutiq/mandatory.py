"""
The strength of the mandatory signals, and the near-misses beside them: whether signals that passed a mandatory rule by
more had a better main outcome or less downside, and whether the sessions that failed one rule did worse than the
signals.

Every signal in the archive passed all four mandatory rules, so the strengths can only compare how strongly signals
passed each rule. Each strength is cut into quantile bands and its highest band compared with its lowest, as the
indicator ranking compares a raw value (indicators.comparison), on the same outcomes, with errors clustered by
signal date and ticker, the returns at each horizon and the breakout failure once over the signals, each horizon's
comparisons, and the failure's, one family corrected by Benjamini-Hochberg at the run's fdr (indicators.compare_all).

The near-misses, sessions that met three of the four rules, or all four short of the minimum margin
(dataset.near_miss_dataset, one per event), are compared with the signals by the rule they failed (dataset.CONDITIONS):
at each horizon a near-miss's path reaches (NEAR_MISS_HORIZONS), the near-misses failing it against every signal, on
the returns (indicators.HORIZON_OUTCOMES), a near-miss having no breakout to fail. Each horizon's near-miss comparisons
are one family of their own. Every result says what the section can and cannot show (NECESSITY_NOTE). Like the
ranking, it describes the whole archive, gap included, and chooses no rule.
"""

import pandas as pd

from app.analyzer import BREAKOUT_WINDOWS, SMA_DAYS, SMA_RISING_DAYS, VOLUME_RATIO_MIN
from app.breakoutiq import dataset as data
from app.breakoutiq.indicators import (
    HORIZON_OUTCOMES,
    Horizon,
    band_edges,
    bands,
    compare_all,
    comparison,
    correct,
    rank,
)
from app.breakoutiq.thresholds import DEFAULTS, Thresholds
from app.models import NEAR_MISS_KIND, AnalysisResult
from app.signal_paths import kind_horizons

SECTION = 'mandatory'
PAYLOAD_VERSION = 1

# the kinds of finding of the section, the first part of its name ('strength:<strength>:<outcome>',
# 'near_miss:<condition>:<outcome>')
STRENGTH = 'strength'
NEAR_MISS = 'near_miss'

# each mandatory strength (dataset.build_dataset's columns), with the rule every signal passed, which it measures
# how strongly the signal passed
STRENGTHS = {
    'breakout_margin_pct': f'the close above the highest high of the {" or ".join(map(str, BREAKOUT_WINDOWS))} '
    'sessions before it',
    'sma50_distance_pct': f'the close above the {SMA_DAYS}-day SMA',
    'sma50_slope_pct': f'the {SMA_DAYS}-day SMA above its value {SMA_RISING_DAYS} sessions earlier',
    'volume_ratio': f'the volume at least {VOLUME_RATIO_MIN} times its average',
}

# each mandatory condition a near-miss can fail (dataset.CONDITIONS), with the rule it failed
CONDITIONS = {
    data.BREAKOUT: STRENGTHS['breakout_margin_pct'],
    data.VOLUME: STRENGTHS['volume_ratio'],
    data.ABOVE_SMA50: STRENGTHS['sma50_distance_pct'],
    data.SMA50_RISING: STRENGTHS['sma50_slope_pct'],
    data.MARGIN: 'the breakout by the minimum margin',
}

# the horizons near-misses are compared at: those their paths reach
NEAR_MISS_HORIZONS = kind_horizons(NEAR_MISS_KIND)

# how a near-miss comparison's two groups are labelled: the signals, then the near-misses that failed the rule
_NEAR_MISS_GROUPS = ('signal', 'near_miss')

# what every result of the section states, so no reader takes a strength's effect, or a near-miss's, for the rule's
NECESSITY_NOTE = (
    'Every signal passed every mandatory rule, so the strengths compare how strongly signals passed one. They measure '
    'strength, not necessity: they cannot show whether the rule is needed. The near-misses met the other three rules '
    '(or all four, short of the minimum margin) and failed this one: near-misses that did about as well as the signals '
    'suggest the cut-off could move, and ones that did worse that it keeps out weaker sessions. Neither proves the '
    'rule is needed: a near-miss differs from a signal in more than the rule it failed, and only sessions that met '
    'the other rules are compared.'
)


def mandatory(
    rows: pd.DataFrame, thresholds: Thresholds = DEFAULTS, near_misses: pd.DataFrame | None = None
) -> list[AnalysisResult]:
    """
    The section's results from the analysis dataset (dataset.build_dataset): per horizon, one per strength and each
    of indicators.HORIZON_OUTCOMES, then, with no horizon, one per strength and each of indicators.SIGNAL_OUTCOMES;
    STRENGTHS in order. The finding is 'strength:<strength>:<outcome>'. Then the near-misses' comparisons
    (near_miss_comparisons) with the near-miss dataset 'near_misses', none when it is None.
    """
    return strengths(rows, thresholds) + (
        near_miss_comparisons(rows, near_misses, thresholds) if near_misses is not None else []
    )


def strengths(rows: pd.DataFrame, thresholds: Thresholds = DEFAULTS) -> list[AnalysisResult]:
    """The strengths' results, as mandatory describes them."""
    edges = {name: band_edges(rows, name) for name in STRENGTHS}

    def compare(context: Horizon, at: pd.DataFrame, outcomes: tuple[str, ...]) -> list[dict]:
        payloads = []
        for name, rule in STRENGTHS.items():
            groups = bands(at[name], edges[name]).to_numpy(dtype=float)
            for outcome in outcomes:
                payload = comparison(context, groups, None, outcome, thresholds)
                payloads.append(
                    {
                        'kind': STRENGTH,
                        'indicator': name,
                        'outcome': outcome,
                        'rule': rule,
                        'measures': 'strength',
                        'tests_necessity': False,
                        'note': NECESSITY_NOTE,
                        **payload,
                        'edges': edges[name],
                    }
                )
        return payloads

    return compare_all(SECTION, PAYLOAD_VERSION, rows, compare, thresholds)


def near_miss_comparisons(
    rows: pd.DataFrame, near_misses: pd.DataFrame, thresholds: Thresholds = DEFAULTS
) -> list[AnalysisResult]:
    """
    The near-misses compared with the signals, from the analysis dataset 'rows' and the near-miss dataset
    'near_misses' (dataset.near_miss_dataset): per horizon of NEAR_MISS_HORIZONS the signals hold, one per condition
    of CONDITIONS, in order, and each of indicators.HORIZON_OUTCOMES, every signal against the near-misses that
    failed the condition, the effect the near-misses' mean less the signals'. Both groups are always reported, an
    empty one with no observations. Each horizon's comparisons are one family. The finding is
    'near_miss:<condition>:<outcome>'.
    """
    results = []
    for horizon in sorted(set(rows['horizon']).intersection(NEAR_MISS_HORIZONS)):
        signals = _compared(rows[rows['horizon'] == horizon], 1)
        failed = near_misses[near_misses['horizon'] == horizon]
        payloads = []
        for condition, rule in CONDITIONS.items():
            parts = [signals, _compared(failed[failed['condition'] == condition], 2)]
            at = pd.concat([p for p in parts if not p.empty], ignore_index=True)
            context = Horizon.of(at)
            groups = at['_group'].to_numpy(dtype=float)
            for outcome in HORIZON_OUTCOMES:
                payload = comparison(context, groups, _NEAR_MISS_GROUPS, outcome, thresholds, every_group=True)
                payloads.append(
                    {
                        'kind': NEAR_MISS,
                        'indicator': condition,
                        'outcome': outcome,
                        'rule': rule,
                        'measures': NEAR_MISS,
                        'tests_necessity': False,
                        'note': NECESSITY_NOTE,
                        **payload,
                    }
                )
        correct(payloads, thresholds.fdr)
        rank(payloads)
        results += [
            AnalysisResult(SECTION, f'{p["kind"]}:{p["indicator"]}:{p["outcome"]}', int(horizon), PAYLOAD_VERSION, p)
            for p in payloads
        ]
    return results


# the columns a near-miss comparison reads of each row (indicators.Horizon)
_COMPARED = ('signal_id', 'session_date', 'ticker', 'outcome_status', 'excess_return_pct', 'mae_pct')


def _compared(rows: pd.DataFrame, group: int) -> pd.DataFrame:
    """The columns of 'rows' a near-miss comparison reads, each row in 'group', with no breakout failure to read."""
    return rows.loc[:, list(_COMPARED)].assign(breakout_failed=float('nan'), _group=float(group))
