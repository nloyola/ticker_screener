"""
Candidate rule changes found on development, checked on validation and confirmed once on test: which extra
conditions on a breakout would have kept signals with a better main outcome, the net excess return over the
benchmark.

A condition keeps the signals whose raw value (a secondary indicator's, VALUES, or a mandatory strength, STRENGTHS)
lies above a threshold or at or below it, or whose flag or combined secondary indicator (FLAGS, COMPOSITES) passes. A
signal with no value fails the condition, as it would under the rule. The thresholds are the interior edges of the
value's quantile bands over the development signals alone. A rule is one condition, or two on different indicators
both required; the pairs combine the strongest single condition of each of the PAIR_POOL indicators with the
strongest evidence on development, the stronger of the two first.

A single condition is scored on a period by comparing the signals it keeps with those it drops: the difference in
their mean net excess return, with errors clustered two-way by signal date and ticker (stats.clustered_ols), on the
period's complete outcomes. A pair is scored on what its second condition adds to its first: among the signals the
first keeps, those the second keeps against those it drops. So a pair whose second condition is noise, or implied by
the first, adds nothing and is not carried, however strong its first condition. Ended (censored) horizons are counted
in each group, never dropped silently, and gap signals are in no period, so they are left out.

At each horizon, separately:
- development: every rule is scored, and the whole search is one family corrected by Benjamini-Hochberg at
  stats.FDR. A threshold's two directions (above it, at or below it) are one test of the same split, so only the
  direction with the larger effect is in the family. A rule is carried to validation when it keeps a better outcome
  (a positive effect), significantly after the correction, and keeps at least MIN_KEPT_SHARE of the signals. Only the
  best rule of each indicator, or pair of indicators, is carried, and at most MAX_CANDIDATES of them, by their
  development p-value.
- validation: each carried rule is scored, the carried rules one family. A rule survives when its effect is positive
  again and significant after the correction.
- test: each survivor is scored exactly once, through TestPeriod, which refuses a second score of the same rule, the
  survivors one family.

A rule is a candidate at one horizon: the same rule can be carried at several horizons, and is then scored on test
once at each, every horizon's test family corrected on its own, with no correction across horizons.

Every carried rule is a candidate, labelled PRELIMINARY until it holds, its effect positive and significant, on both
validation and test, and VALIDATED only if, on test too, the signals it keeps also reach MIN_NET_EXCESS_RETURN at the
horizon: the economic bar, on the mean of the signals the rule keeps rather than on its effect. Its payload says why
a preliminary rule stopped. Nothing here changes the live screener.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.breakoutiq.dataset import COMPLETE, DEVELOPMENT, ENDED, GAP, TEST, VALIDATION
from app.breakoutiq.indicators import (
    COMPOSITES,
    FLAG,
    FLAGS,
    MAIN_OUTCOME,
    OUTCOMES,
    SECONDARY,
    VALUE,
    VALUES,
    band_edges,
    indicator_frame,
)
from app.breakoutiq.mandatory import STRENGTH, STRENGTHS
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

SECTION = 'candidates'
PAYLOAD_VERSION = 1

# the findings: the search at a horizon as a whole, and each candidate rule ('rule:<rule>')
SEARCH = 'search'
RULE = 'rule'

# a candidate's labels
PRELIMINARY = 'preliminary'
VALIDATED = 'validated'

# why a candidate stayed preliminary: a period could not test it (a group below the minimum sample, or errors that
# could not be clustered), or tested it and did not confirm it, or it kept too small a return
NOT_TESTED_ON_VALIDATION = 'not tested on validation'
NOT_CONFIRMED_ON_VALIDATION = 'not confirmed on validation'
NOT_TESTED_ON_TEST = 'not tested on test'
NOT_CONFIRMED_ON_TEST = 'not confirmed on test'
BELOW_ECONOMIC_BAR = 'below the economic bar'

# the economic bar: the mean net excess return, in percent, the signals a rule keeps must reach on test, per horizon
# in sessions, for the rule to be validated: a quarter of a percent for each week (5 sessions) a signal is held
MIN_NET_EXCESS_RETURN = {5: 0.25, 10: 0.5, 20: 1.0, 40: 2.0, 60: 3.0, 120: 6.0}
# a rule must keep at least this share of the development signals, so a niche of a few signals is not a rule
MIN_KEPT_SHARE = 0.1
# the rules carried from development to validation, at most, per horizon
MAX_CANDIDATES = 20
# the indicators whose strongest single condition on development is combined in pairs
PAIR_POOL = 10

# the conditions' operators: the value above the threshold, at or below it, or the flag passing
ABOVE = '>'
AT_MOST = '<='
PASSES = 'pass'

# the indicators a condition can be on, each with its kind: the raw values and the strengths cut at thresholds, the
# flags and combined secondary indicators required to pass
CONDITION_INDICATORS = {
    **dict.fromkeys(VALUES, VALUE),
    **dict.fromkeys(STRENGTHS, STRENGTH),
    **dict.fromkeys(FLAGS, FLAG),
    **dict.fromkeys(COMPOSITES, SECONDARY),
}


@dataclass(frozen=True)
class Condition:
    """One condition of a rule: the indicator, the operator and, for a value, its threshold."""

    indicator: str
    op: str
    threshold: float | None = None

    @property
    def key(self) -> str:
        return self.indicator if self.op == PASSES else f'{self.indicator}{self.op}{self.threshold:.6g}'

    @property
    def text(self) -> str:
        return f'{self.indicator} passes' if self.op == PASSES else f'{self.indicator} {self.op} {self.threshold:.6g}'

    def keeps(self, frame: pd.DataFrame) -> np.ndarray:
        """Whether each row of 'frame' passes the condition; a row with no value does not."""
        values = pd.to_numeric(frame[self.indicator], errors='coerce').to_numpy(dtype=float)
        with np.errstate(invalid='ignore'):
            if self.op == PASSES:
                return values == 1
            if self.op == ABOVE:
                return values > self.threshold
            return values <= self.threshold

    def payload(self) -> dict:
        return {
            'indicator': self.indicator,
            'kind': CONDITION_INDICATORS[self.indicator],
            'op': self.op,
            'threshold': self.threshold,
        }


# a rule: one condition, or two on different indicators, all required
Rule = tuple[Condition, ...]


def rule_key(rule: Rule) -> str:
    return '&'.join(c.key for c in rule)


def candidates(rows: pd.DataFrame) -> list[AnalysisResult]:
    """
    The section's results from the analysis dataset (dataset.build_dataset): per horizon, the SEARCH finding, then
    one 'rule:<rule>' finding per candidate carried to validation, in their development order.
    """
    frame = indicator_frame(rows)
    development = frame[frame['period'] == DEVELOPMENT]
    conditions = single_conditions(development)
    results = []
    for horizon, at_horizon in frame.groupby('horizon', sort=True):
        search, payloads = _horizon(int(horizon), at_horizon.reset_index(drop=True), conditions)
        results.append(AnalysisResult(SECTION, SEARCH, int(horizon), PAYLOAD_VERSION, search))
        results += [AnalysisResult(SECTION, f'{RULE}:{p["rule"]}', int(horizon), PAYLOAD_VERSION, p) for p in payloads]
    return results


def single_conditions(development: pd.DataFrame) -> list[Condition]:
    """
    Every single condition, from the development rows alone: for each raw value and strength, above and at or below
    each interior edge of its quantile bands over the development signals; for each flag and combined secondary
    indicator, passing.
    """
    conditions = []
    for name, kind in CONDITION_INDICATORS.items():
        if kind in (VALUE, STRENGTH):
            for edge in band_edges(development, name)[1:-1]:
                conditions += [Condition(name, ABOVE, edge), Condition(name, AT_MOST, edge)]
        else:
            conditions.append(Condition(name, PASSES))
    return conditions


class Period:
    """A period's complete and censored rows at one horizon, as a rule is scored on them."""

    def __init__(self, rows: pd.DataFrame):
        y = pd.to_numeric(rows[OUTCOMES[MAIN_OUTCOME].column], errors='coerce').to_numpy(dtype=float)
        observed = (rows['outcome_status'] == COMPLETE).to_numpy() & ~np.isnan(y)
        self.rows = rows.reset_index(drop=True)
        self.y = y
        self.observed = observed
        self.ended = (rows['outcome_status'] == ENDED).to_numpy()
        self.dates = pd.factorize(rows['session_date'])[0]
        self.signals = pd.factorize(rows['signal_id'])[0]
        self.clusters = cluster_codes(rows) if len(rows) else np.empty((0, len(CLUSTERS)), dtype=int)

    def counts(self) -> dict:
        """The period's counts and the mean of every complete outcome, the outcome before any rule."""
        return {
            **_group(self, np.ones(len(self.y), dtype=bool)),
            'mean': _mean(self, np.ones(len(self.y), dtype=bool)),
        }

    def score(self, rule: Rule) -> dict:
        """
        The rule on this period: the counts and mean outcome of the signals it keeps and of those it drops, the share
        of the signals it keeps, and, when both groups meet the minimum sample, the effect of keeping (kept less
        dropped) with its clustered interval and raw p-value. For a pair, the dropped signals are those its first
        condition keeps ('within') and its second drops, so the effect is what the second condition adds.
        """
        within = rule[0].keeps(self.rows) if len(rule) > 1 else np.ones(len(self.y), dtype=bool)
        kept = within.copy()
        for condition in rule:
            kept &= condition.keeps(self.rows)
        dropped = within & ~kept
        groups = {'kept': _group(self, kept), 'dropped': _group(self, dropped)}
        for name, mask in (('kept', kept), ('dropped', dropped)):
            groups[name]['mean'] = _mean(self, mask) if groups[name]['signal_dates'] >= MIN_SIGNAL_DATES else None
        signals = _distinct(self.signals, np.ones(len(self.y), dtype=bool))
        score = {
            'within': rule_key(rule[:1]) if len(rule) > 1 else None,
            **groups,
            'kept_share': groups['kept']['signals'] / signals if signals else None,
            'shown': False,
            'effect': None,
            'p_value': None,
            'p_adjusted': None,
            'significant': False,
        }
        if min(groups['kept']['signal_dates'], groups['dropped']['signal_dates']) < MIN_SIGNAL_DATES:
            return score
        compared = within & self.observed
        exog = pd.DataFrame({'const': 1.0, 'kept': kept[compared].astype(float)})
        fit = clustered_ols(self.y[compared], exog, self.clusters[compared], of=['kept'])
        if fit is None:
            return score
        effect = fit['kept']
        score['shown'] = True
        score['effect'] = {
            'estimate': effect.estimate,
            'se': effect.se,
            'ci_low': effect.ci_low,
            'ci_high': effect.ci_high,
        }
        score['p_value'] = effect.p_value
        return score


class TestPeriod(Period):
    """
    One horizon's test period, which scores a rule at most once: a second score of the same rule raises. A run builds
    one per horizon, so a rule is scored on test at most once per horizon in a run.
    """

    __test__ = False  # not a pytest test class

    def __init__(self, rows: pd.DataFrame, horizon: int):
        super().__init__(rows)
        self.horizon = horizon
        self.scored: set[str] = set()

    def score(self, rule: Rule) -> dict:
        key = rule_key(rule)
        if key in self.scored:
            raise RuntimeError(f'rule {key} was already scored on the test period at horizon {self.horizon}')
        self.scored.add(key)
        return super().score(rule)


def _horizon(horizon: int, rows: pd.DataFrame, conditions: list[Condition]) -> tuple[dict, list[dict]]:
    """One horizon's search payload and its candidates' payloads."""
    periods = {
        DEVELOPMENT: Period(rows[rows['period'] == DEVELOPMENT]),
        VALIDATION: Period(rows[rows['period'] == VALIDATION]),
    }
    test = TestPeriod(rows[rows['period'] == TEST], horizon)
    bar = MIN_NET_EXCESS_RETURN[horizon]

    # development: the single conditions, then pairs of the strongest indicators' best conditions, one family
    searched = {(c,): periods[DEVELOPMENT].score((c,)) for c in conditions}
    pool = _best_per_indicators([(rule, score) for rule, score in searched.items() if _promising(score)], PAIR_POOL)
    # the pool holds one condition per indicator, best first, so each pair's first condition is its stronger
    pairs = [(a[0], b[0]) for i, (a, _) in enumerate(pool) for (b, _) in pool[i + 1 :]]
    singles = len(searched)
    searched |= {pair: periods[DEVELOPMENT].score(pair) for pair in pairs}
    family = _development_family(searched)
    _correct(family)
    carried = _best_per_indicators(
        [(rule, score) for rule, score in searched.items() if _promising(score) and score['significant']],
        MAX_CANDIDATES,
    )

    # validation: each carried rule, one family
    checked = [periods[VALIDATION].score(rule) for rule, _ in carried]
    _correct(checked)

    # test: each survivor once, one family
    survivors = [i for i, score in enumerate(checked) if _holds(score)]
    confirmed = {i: test.score(carried[i][0]) for i in survivors}
    _correct(list(confirmed.values()))

    payloads = []
    for position, ((rule, development), validation) in enumerate(zip(carried, checked, strict=True), start=1):
        on_test = confirmed.get(position - 1)
        label, reason = _label(validation, on_test, bar)
        payloads.append(
            {
                'rule': rule_key(rule),
                'conditions': [c.payload() for c in rule],
                'description': ' and '.join(c.text for c in rule),
                'outcome': MAIN_OUTCOME,
                'label': label,
                'reason': reason,
                'development_rank': position,
                'periods': {DEVELOPMENT: development, VALIDATION: validation, TEST: on_test},
                'scored_on_test': on_test is not None,
                'expected_excess_return': on_test['kept']['mean'] if on_test else None,
                'min_net_excess_return': bar,
                **_settings(),
            }
        )

    search = {
        'outcome': MAIN_OUTCOME,
        'periods': {
            DEVELOPMENT: periods[DEVELOPMENT].counts(),
            VALIDATION: periods[VALIDATION].counts(),
            TEST: test.counts(),
        },
        'gap': int(rows['signal_id'][rows['period'] == GAP].nunique()),
        'conditions': singles,
        'pairs': len(pairs),
        'searched': len(searched),
        'tested_on_development': sum(1 for s in searched.values() if s['p_value'] is not None),
        'development_family': len(family),
        'carried': len(carried),
        'survived_validation': len(survivors),
        'scored_on_test': len(test.scored),
        'validated': sum(1 for p in payloads if p['label'] == VALIDATED),
        'min_net_excess_return': bar,
        **_settings(),
    }
    return search, payloads


def _settings() -> dict:
    return {
        'min_kept_share': MIN_KEPT_SHARE,
        'max_candidates': MAX_CANDIDATES,
        'pair_pool': PAIR_POOL,
        'min_signal_dates': MIN_SIGNAL_DATES,
        'confidence': CONFIDENCE,
        'fdr': FDR,
        'clusters': list(CLUSTERS),
    }


def _promising(score: dict) -> bool:
    """A development score worth carrying further: tested, keeping a better outcome, and enough of the signals."""
    return (
        score['p_value'] is not None
        and score['effect']['estimate'] > 0
        and score['kept_share'] is not None
        and score['kept_share'] >= MIN_KEPT_SHARE
    )


def _holds(score: dict | None) -> bool:
    """A later period's score confirms the development one: a positive effect, significant after the correction."""
    return score is not None and score['significant'] and score['effect']['estimate'] > 0


def _label(validation: dict, test: dict | None, bar: float) -> tuple[str, str | None]:
    """A candidate's label, and why it is preliminary (None when validated)."""
    if not validation['shown']:
        return PRELIMINARY, NOT_TESTED_ON_VALIDATION
    if not _holds(validation):
        return PRELIMINARY, NOT_CONFIRMED_ON_VALIDATION
    assert test is not None, 'a rule that holds on validation is scored on test'
    if not test['shown']:
        return PRELIMINARY, NOT_TESTED_ON_TEST
    if not _holds(test):
        return PRELIMINARY, NOT_CONFIRMED_ON_TEST
    if test['kept']['mean'] < bar:
        return PRELIMINARY, BELOW_ECONOMIC_BAR
    return VALIDATED, None


def _development_family(searched: dict[Rule, dict]) -> list[dict]:
    """
    The development search's tested scores, corrected as one family: every tested pair and passing condition, and of
    the two directions of each threshold, the one with the larger effect.
    """
    family, splits = [], {}
    for rule, score in searched.items():
        if score['p_value'] is None:
            continue
        if len(rule) == 1 and rule[0].op != PASSES:
            split = (rule[0].indicator, rule[0].threshold)
            other = splits.get(split)
            if other is None or score['effect']['estimate'] > other['effect']['estimate']:
                splits[split] = score
            continue
        family.append(score)
    return family + list(splits.values())


def _best_per_indicators(scored: list[tuple[Rule, dict]], limit: int) -> list[tuple[Rule, dict]]:
    """
    At most 'limit' rules, the best of each set of indicators, best first: the smallest p-value, then the largest
    effect relative to its standard error.
    """
    ordered = sorted(scored, key=lambda item: (item[1]['p_value'], -_t(item[1])))
    best, seen = [], set()
    for rule, score in ordered:
        indicators = frozenset(c.indicator for c in rule)
        if indicators in seen:
            continue
        seen.add(indicators)
        best.append((rule, score))
        if len(best) == limit:
            break
    return best


def _t(score: dict) -> float:
    return abs(score['effect']['estimate'] / score['effect']['se'])


def _correct(scores: list[dict]) -> None:
    """Sets each tested score's adjusted p-value over the family 'scores', and whether it is significant at FDR."""
    tested = [s for s in scores if s['p_value'] is not None]
    for score, adjusted in zip(tested, benjamini_hochberg([s['p_value'] for s in tested]), strict=True):
        score['p_adjusted'] = adjusted
        score['significant'] = adjusted <= FDR


def _group(period: Period, mask: np.ndarray) -> dict:
    seen = mask & period.observed
    return {
        'signals': _distinct(period.signals, mask),
        'n': int(seen.sum()),
        'signal_dates': _distinct(period.dates, seen),
        'censored': int((mask & period.ended).sum()),
    }


def _mean(period: Period, mask: np.ndarray) -> float | None:
    seen = mask & period.observed
    return float(period.y[seen].mean()) if seen.any() else None


def _distinct(codes: np.ndarray, mask: np.ndarray) -> int:
    return len(np.unique(codes[mask]))
