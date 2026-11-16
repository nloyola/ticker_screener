"""
The strength of the mandatory signals: whether signals that passed a mandatory rule by more had a better main outcome
or less downside, at each horizon.

Every signal in the archive passed all four mandatory rules, so this section can only compare how strongly signals
passed each rule, never signals that passed it with signals that did not: it cannot show whether a rule is needed,
and every result says so. Each strength is cut into quantile bands and its highest band compared with its lowest, as
the indicator ranking compares a raw value (indicators.comparison), on the same outcomes, with errors clustered by
signal date and ticker, and with every comparison of a horizon one family corrected by Benjamini-Hochberg at
stats.FDR. Like the ranking, it describes the whole archive, gap included, and chooses no rule.
"""

import pandas as pd

from app.analyzer import BREAKOUT_WINDOWS, SMA_DAYS, SMA_RISING_DAYS, VOLUME_RATIO_MIN
from app.breakoutiq.indicators import OUTCOMES, Horizon, band_edges, bands, comparison, correct, rank
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


def mandatory(rows: pd.DataFrame) -> list[AnalysisResult]:
    """
    The section's results from the analysis dataset (dataset.build_dataset): one per horizon, strength and outcome,
    STRENGTHS in order, each with every one of indicators.OUTCOMES. The finding is 'strength:<strength>:<outcome>'.
    """
    edges = {name: band_edges(rows, name) for name in STRENGTHS}
    results = []
    for horizon, at_horizon in rows.groupby('horizon', sort=True):
        context = Horizon.of(at_horizon)
        payloads = []
        for name, rule in STRENGTHS.items():
            groups = bands(at_horizon[name], edges[name]).to_numpy(dtype=float)
            for outcome in OUTCOMES:
                payload = comparison(context, groups, None, outcome)
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
        correct(payloads)
        rank(payloads)
        results += [
            AnalysisResult(SECTION, f'{STRENGTH}:{p["indicator"]}:{p["outcome"]}', int(horizon), PAYLOAD_VERSION, p)
            for p in payloads
        ]
    return results
