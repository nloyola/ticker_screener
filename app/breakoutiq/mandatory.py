"""
The strength of the mandatory signals: whether signals that passed a mandatory rule by more had a better main outcome
or less downside.

Every signal in the archive passed all four mandatory rules, so this section can only compare how strongly signals
passed each rule, never signals that passed it with signals that did not: it cannot show whether a rule is needed,
and every result says so. Each strength is cut into quantile bands and its highest band compared with its lowest, as
the indicator ranking compares a raw value (indicators.comparison), on the same outcomes, with errors clustered by
signal date and ticker, the returns at each horizon and the breakout failure once over the signals, each horizon's
comparisons, and the failure's, one family corrected by Benjamini-Hochberg at the run's fdr (indicators.compare_all).
Like the ranking, it describes the whole archive, gap included, and chooses no rule.
"""

import pandas as pd

from app.analyzer import BREAKOUT_WINDOWS, SMA_DAYS, SMA_RISING_DAYS, VOLUME_RATIO_MIN
from app.breakoutiq.indicators import Horizon, band_edges, bands, compare_all, comparison
from app.breakoutiq.thresholds import DEFAULTS, Thresholds
from app.models import AnalysisResult

SECTION = 'mandatory'
PAYLOAD_VERSION = 1

# the kind of every finding of the section, the first part of its name ('strength:<strength>:<outcome>')
STRENGTH = 'strength'

# each mandatory strength (dataset.build_dataset's columns), with the rule every signal passed, which it measures
# how strongly the signal passed
STRENGTHS = {
    'breakout_margin_pct': f'the close above the highest high of the {" or ".join(map(str, BREAKOUT_WINDOWS))} '
    'sessions before it',
    'sma50_distance_pct': f'the close above the {SMA_DAYS}-day SMA',
    'sma50_slope_pct': f'the {SMA_DAYS}-day SMA above its value {SMA_RISING_DAYS} sessions earlier',
    'volume_ratio': f'the volume at least {VOLUME_RATIO_MIN} times its average',
}

# what every result of the section states, so no reader takes a strength's effect for the rule's
NECESSITY_NOTE = (
    'Every signal passed this mandatory rule, so this compares how strongly signals passed it. It measures strength, '
    'not necessity: it cannot show whether the rule is needed.'
)


def mandatory(rows: pd.DataFrame, thresholds: Thresholds = DEFAULTS) -> list[AnalysisResult]:
    """
    The section's results from the analysis dataset (dataset.build_dataset): per horizon, one per strength and each
    of indicators.HORIZON_OUTCOMES, then, with no horizon, one per strength and each of indicators.SIGNAL_OUTCOMES;
    STRENGTHS in order. The finding is 'strength:<strength>:<outcome>'.
    """
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
