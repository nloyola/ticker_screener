"""
The thresholds BreakoutIQ's sections judge their results by: the minimum sample a result needs to be shown, the
bootstrap's draws and the confidence of every interval, the false discovery rate, when an indicator is left out of a
regression or named redundant, and how candidate rules are carried and validated, with the economic bar.

The named constants are the defaults, the values a run uses unless told otherwise. A run takes a Thresholds, which
breakoutiq-analysis builds from its options, so a test or a small archive can lower a bar without changing the code;
the run records the values it used in analysis_run.settings, and each payload the ones it was judged by.
"""

from dataclasses import dataclass, field, fields, replace
from types import MappingProxyType

from app.signal_paths import HORIZONS

# a result needs observations (complete outcomes for a return, decided failures for the failure) on at least this
# many distinct signal dates to be shown at all
MIN_SIGNAL_DATES = 30
# a bootstrap interval needs at least this many blocks of signal dates; with fewer, it is left out
MIN_BLOCKS = 10
# the confidence level of every interval, and the bootstrap's resamples
CONFIDENCE = 0.95
BOOTSTRAP_RESAMPLES = 2000
# the false discovery rate every family of tests is corrected to, by Benjamini-Hochberg
FDR = 0.10

# an indicator missing on more than this share of a horizon's complete outcomes is left out of its regression, so
# one sparse indicator does not drop most rows from it
MAX_MISSING_SHARE = 0.2
# two indicators correlated at least this much (in absolute value) are named as saying the same thing
REDUNDANT_CORRELATION = 0.8
# an indicator whose variance inflation factor is at least this is mostly explained by the others
REDUNDANT_VIF = 5.0
# a regression needs at least this many rows per column of its design (constant, fixed effects and indicators)
MIN_ROWS_PER_COLUMN = 10

# the economic bar: the mean net excess return, in percent, the signals a rule keeps must reach on test, per horizon
# in sessions, for the rule to be validated: a quarter of a percent for each week (5 sessions) a signal is held
MIN_NET_EXCESS_RETURN = MappingProxyType({5: 0.25, 10: 0.5, 20: 1.0, 40: 2.0, 60: 3.0, 120: 6.0})
# a rule must keep at least this share of the development signals, so a niche of a few signals is not a rule
MIN_KEPT_SHARE = 0.1
# the rules carried from development to validation, at most, per horizon
MAX_CANDIDATES = 20
# the indicators whose strongest single condition on development is combined in pairs
PAIR_POOL = 10


@dataclass(frozen=True)
class Thresholds:
    """The thresholds of one run, each defaulting to its named constant; ValueError when one is out of range."""

    min_signal_dates: int = MIN_SIGNAL_DATES
    min_blocks: int = MIN_BLOCKS
    confidence: float = CONFIDENCE
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES
    fdr: float = FDR
    max_missing_share: float = MAX_MISSING_SHARE
    redundant_correlation: float = REDUNDANT_CORRELATION
    redundant_vif: float = REDUNDANT_VIF
    min_rows_per_column: int = MIN_ROWS_PER_COLUMN
    min_net_excess_return: MappingProxyType = field(default=MIN_NET_EXCESS_RETURN)
    min_kept_share: float = MIN_KEPT_SHARE
    max_candidates: int = MAX_CANDIDATES
    pair_pool: int = PAIR_POOL

    def __post_init__(self) -> None:
        bar = {int(h): float(v) for h, v in dict(self.min_net_excess_return).items()}
        unknown, unset = sorted(set(bar) - set(HORIZONS)), sorted(set(HORIZONS) - set(bar))
        if unknown:
            raise ValueError(f'min_net_excess_return: no horizon {unknown}, only {list(HORIZONS)}')
        if unset:
            raise ValueError(f'min_net_excess_return: no bar for horizon {unset}')
        object.__setattr__(self, 'min_net_excess_return', MappingProxyType(dict(sorted(bar.items()))))
        for name in ('min_signal_dates', 'min_blocks', 'bootstrap_resamples', 'min_rows_per_column'):
            _at_least(name, getattr(self, name), 1)
        for name in ('max_candidates', 'pair_pool'):
            _at_least(name, getattr(self, name), 0)
        for name in ('confidence', 'fdr'):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f'{name} must lie strictly between 0 and 1, got {getattr(self, name)}')
        for name in ('max_missing_share', 'redundant_correlation', 'min_kept_share'):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'{name} must lie between 0 and 1, got {getattr(self, name)}')
        _at_least('redundant_vif', self.redundant_vif, 1)

    def with_bars(self, bars: dict[int, float]) -> 'Thresholds':
        """These thresholds with the economic bar of each horizon in 'bars' replaced."""
        return replace(self, min_net_excess_return={**self.min_net_excess_return, **bars})

    def settings(self) -> dict:
        """Every threshold by name, as a run stores it; the economic bar keyed by the horizon as text."""
        values = {name: getattr(self, name) for name in names()}
        values['min_net_excess_return'] = {str(h): bar for h, bar in self.min_net_excess_return.items()}
        return values


# what each threshold sets, as the command's help gives it
DESCRIPTIONS = {
    'min_signal_dates': 'the distinct signal dates a result needs to be shown',
    'min_blocks': 'the blocks of signal dates a bootstrap interval needs',
    'confidence': 'the confidence level of every interval',
    'bootstrap_resamples': "the bootstrap's draws",
    'fdr': 'the false discovery rate every family of tests is corrected to',
    'max_missing_share': 'the share of complete outcomes an indicator may miss and stay in a regression',
    'redundant_correlation': 'the correlation at which two indicators are named as saying the same thing',
    'redundant_vif': 'the variance inflation factor reported as high',
    'min_rows_per_column': 'the rows a regression needs per column of its design',
    'min_net_excess_return': 'the economic bar a validated rule must reach on test, in percent, per horizon',
    'min_kept_share': 'the share of the development signals a candidate rule must keep',
    'max_candidates': 'the candidate rules carried to validation per horizon, at most',
    'pair_pool': 'the indicators whose strongest condition is combined in pairs',
}


def names() -> list[str]:
    """The thresholds' names, in the order Thresholds declares them."""
    return [f.name for f in fields(Thresholds)]


def _at_least(name: str, value: float, low: float) -> None:
    if value < low:
        raise ValueError(f'{name} must be at least {low}, got {value}')


DEFAULTS = Thresholds()
