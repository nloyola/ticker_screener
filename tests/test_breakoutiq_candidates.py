"""
BreakoutIQ's candidate rule search: a rule planted in a synthetic dataset comes out validated, a rule that only fits
development's noise stays preliminary, a rule that holds but keeps too small a return stays below the economic bar,
a rule that holds on validation only is not confirmed on test, a pair is scored on what its second condition adds, so
a noise partner of a real condition is never validated, a period that cannot test a rule says so, the search on
development reads neither validation nor test nor the gap, and no rule is ever scored on test more than once per
horizon in an analysis run.
"""

import numpy as np
import pandas as pd
import pytest

from app.breakoutiq.candidates import (
    ABOVE,
    BELOW_ECONOMIC_BAR,
    CONDITION_INDICATORS,
    NOT_CONFIRMED_ON_TEST,
    NOT_CONFIRMED_ON_VALIDATION,
    NOT_TESTED_ON_TEST,
    NOT_TESTED_ON_VALIDATION,
    PAYLOAD_VERSION,
    PRELIMINARY,
    RULE,
    SEARCH,
    SECTION,
    VALIDATED,
    Condition,
    Period,
    TestPeriod,
    candidates,
)
from app.breakoutiq.dataset import DEVELOPMENT, GAP, TEST, VALIDATION
from app.breakoutiq.indicators import FLAG, FLAGS, SECONDARY, VALUES
from app.breakoutiq.thresholds import DEFAULTS, MAX_CANDIDATES, MIN_NET_EXCESS_RETURN
from app.signal_paths import HORIZONS

SESSIONS = pd.bdate_range('2016-01-04', periods=600)
# the stored raw values drawn at random, besides those VALUES derives from close and the averages
_RAW = [v for v in VALUES if v != 'macd_hist_pct' and not v.endswith('_distance_pct')]
# each date's period by its position: development, a gap, validation, a gap, test
_PERIODS = np.select(
    [np.arange(600) < 340, np.arange(600) < 360, np.arange(600) < 460, np.arange(600) < 480],
    [DEVELOPMENT, GAP, VALIDATION, GAP],
    TEST,
)
# the planted rule: volume at least this times its average adds the planted effect to the excess return
_PLANTED_VOLUME = 2.5


def _dataset(seed: int, signals_per_date: int = 4, horizons=(20,)) -> pd.DataFrame:
    """
    A dataset of complete outcomes over 600 sessions, split into the periods by _PERIODS, whose indicators are drawn
    at random, independent of each other and of the outcomes: every date's signals share a market shock in their
    excess return, and each ticker has its own drift. The tests plant their rules on the returned frame.
    """
    rng = np.random.default_rng(seed)
    dates = len(SESSIONS)
    count = signals_per_date * dates
    tickers = rng.integers(0, 150, size=count)
    frame = pd.DataFrame(
        {
            'signal_id': np.arange(1, count + 1),
            'session_date': np.repeat(SESSIONS, signals_per_date),
            'period': np.repeat(_PERIODS, signals_per_date),
            'ticker': [f'T{t}' for t in tickers],
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
            'price_gt_50dma': 1,
        }
    )
    for name in _RAW:
        frame[name] = rng.normal(0, 1, count) if name != 'secondary_passed' else rng.integers(0, 8, count)
    for name in FLAGS:
        frame[name] = rng.integers(0, 2, count)
    excess = np.repeat(rng.normal(0, 4, dates), signals_per_date) + rng.normal(0, 2, 150)[tickers]
    frame['excess_return_pct'] = excess + rng.normal(0, 6, count)
    frame['mae_pct'] = -np.abs(rng.normal(4, 2, count))
    frame['breakout_failed'] = rng.integers(0, 2, count)
    frame['outcome_status'] = 'complete'
    return pd.concat([frame.assign(horizon=h) for h in horizons], ignore_index=True)


def _plant_volume(rows: pd.DataFrame, size: float = 4.0) -> pd.DataFrame:
    rows = rows.copy()
    rows['excess_return_pct'] += size * (rows['volume_ratio'] > _PLANTED_VOLUME)
    return rows


def _rules(results, horizon=20) -> list[dict]:
    return [
        r.payload for r in results if r.section == SECTION and r.finding.startswith(f'{RULE}:') and r.horizon == horizon
    ]


def _search(results, horizon=20) -> dict:
    return next(r.payload for r in results if r.finding == SEARCH and r.horizon == horizon)


def _indicators(payload: dict) -> set[str]:
    return {c['indicator'] for c in payload['conditions']}


def test_a_planted_rule_comes_out_validated():
    results = candidates(_plant_volume(_dataset(1)))
    rules = _rules(results)

    validated = [p for p in rules if p['label'] == VALIDATED]
    assert validated, 'the planted rule was not validated'
    best = rules[0]
    assert _indicators(best) == {'volume_ratio'}
    assert best['label'] == VALIDATED and best['reason'] is None
    (condition,) = best['conditions']
    assert condition['op'] == ABOVE
    assert condition['threshold'] == pytest.approx(_PLANTED_VOLUME, abs=0.4)
    # a pair adds its second condition to the planted one, which is noise: none is validated
    assert [p['rule'] for p in validated] == [best['rule']]
    pairs = [p for p in rules if len(p['conditions']) == 2]
    assert all(p['periods'][DEVELOPMENT]['within'] == p['rule'].split('&')[0] for p in pairs)

    for period in (DEVELOPMENT, VALIDATION, TEST):
        score = best['periods'][period]
        assert score['shown'] and score['significant'] and score['effect']['estimate'] > 0
        assert score['kept']['signals'] > 0 and score['dropped']['signals'] > 0
        assert score['kept']['signal_dates'] > 0 and score['kept']['n'] == score['kept']['signals']
    assert best['scored_on_test']
    assert best['expected_excess_return'] == best['periods'][TEST]['kept']['mean']
    assert best['expected_excess_return'] >= MIN_NET_EXCESS_RETURN[20]
    assert best['min_net_excess_return'] == MIN_NET_EXCESS_RETURN[20]


def test_a_rule_that_only_fits_development_noise_stays_preliminary():
    rows = _dataset(2)
    in_development = rows['period'] == DEVELOPMENT
    rows.loc[in_development, 'excess_return_pct'] += 4.0 * rows.loc[in_development, 'consolidation_tight']

    rules = _rules(candidates(rows))

    fitted = next(p for p in rules if _indicators(p) == {'consolidation_tight'})
    assert fitted['periods'][DEVELOPMENT]['significant']
    assert fitted['label'] == PRELIMINARY
    assert fitted['reason'] == NOT_CONFIRMED_ON_VALIDATION
    assert not fitted['scored_on_test'] and fitted['periods'][TEST] is None
    assert fitted['expected_excess_return'] is None
    assert not [p for p in rules if p['label'] == VALIDATED]


def test_a_rule_that_holds_below_the_economic_bar_stays_preliminary():
    rows = _plant_volume(_dataset(1))
    # every signal loses the same amount: the planted signals still beat the others, but keep too little, their
    # expected excess return 4.0 - 4.5 = -0.5, well below the bar of 1.0 whatever the seed's noise
    rows['excess_return_pct'] -= 4.5

    rules = _rules(candidates(rows))

    best = rules[0]
    assert _indicators(best) == {'volume_ratio'}
    assert best['periods'][VALIDATION]['significant'] and best['periods'][TEST]['significant']
    assert best['expected_excess_return'] < MIN_NET_EXCESS_RETURN[20]
    assert best['label'] == PRELIMINARY
    assert best['reason'] == BELOW_ECONOMIC_BAR


def test_a_lower_economic_bar_validates_the_rule_the_default_bar_holds_back():
    rows = _plant_volume(_dataset(1))
    rows['excess_return_pct'] -= 4.5
    kept = _rules(candidates(rows))[0]['expected_excess_return']
    lower = DEFAULTS.with_bars({20: kept - 0.01})

    best = _rules(candidates(rows, lower))[0]

    assert _indicators(best) == {'volume_ratio'}
    assert best['expected_excess_return'] == kept
    assert best['min_net_excess_return'] == kept - 0.01
    assert best['label'] == VALIDATED and best['reason'] is None


def test_no_rule_is_scored_on_test_more_than_once_per_run(monkeypatch):
    scored = []
    score = TestPeriod.score

    def counting(self, rule):
        scored.append(('&'.join(c.key for c in rule), self.horizon))
        return score(self, rule)

    monkeypatch.setattr(TestPeriod, 'score', counting)
    results = candidates(_plant_volume(_dataset(1, horizons=(10, 20))))

    assert scored, 'no survivor was scored on test'
    assert len(scored) == len(set(scored))
    assert {horizon for _, horizon in scored} == {10, 20}
    for horizon in (10, 20):
        search = _search(results, horizon)
        rules = _rules(results, horizon)
        assert search['scored_on_test'] == search['survived_validation']
        assert search['scored_on_test'] == sum(p['scored_on_test'] for p in rules)
        # only survivors of validation reach test
        assert all(p['periods'][VALIDATION]['significant'] for p in rules if p['scored_on_test'])


def test_the_test_period_refuses_a_second_score_of_a_rule():
    rows = _dataset(3)
    test = TestPeriod(rows[rows['period'] == TEST], 20)
    rule = (Condition('volume_ratio', ABOVE, 2.5),)

    test.score(rule)
    with pytest.raises(RuntimeError, match='already scored'):
        test.score(rule)


def test_the_development_search_reads_neither_validation_nor_test():
    rows = _plant_volume(_dataset(4))
    later = rows['period'].isin([VALIDATION, TEST, GAP])
    scrambled = rows.copy()
    scrambled.loc[later, 'excess_return_pct'] = np.random.default_rng(9).permutation(
        scrambled.loc[later, 'excess_return_pct'].to_numpy()
    )
    scrambled.loc[later, 'volume_ratio'] *= 3

    first, second = _rules(candidates(rows)), _rules(candidates(scrambled))

    assert [p['rule'] for p in first] == [p['rule'] for p in second]
    assert [p['periods'][DEVELOPMENT] for p in first] == [p['periods'][DEVELOPMENT] for p in second]


def test_the_search_reports_every_period_and_leaves_the_gap_out():
    rows = _plant_volume(_dataset(1))
    rows.loc[rows.index[:40], 'outcome_status'] = 'ended'
    rows.loc[rows.index[:40], 'excess_return_pct'] = np.nan

    results = candidates(rows)
    search = _search(results)

    assert {r.payload_version for r in results} == {PAYLOAD_VERSION}
    signals = rows.groupby('period')['signal_id'].nunique()
    for period in (DEVELOPMENT, VALIDATION, TEST):
        assert search['periods'][period]['signals'] == signals[period]
    assert search['gap'] == signals[GAP]
    assert search['periods'][DEVELOPMENT]['censored'] == 40
    assert search['periods'][DEVELOPMENT]['n'] == signals[DEVELOPMENT] - 40
    assert search['pairs'] > 0
    assert search['searched'] == search['conditions'] + search['pairs']
    assert 0 < search['carried'] <= MAX_CANDIDATES
    assert search['validated'] == sum(p['label'] == VALIDATED for p in _rules(results))
    # each carried rule is the best of its indicators
    carried = [frozenset(_indicators(p)) for p in _rules(results)]
    assert len(carried) == len(set(carried))


def test_with_no_effect_the_large_majority_of_simulations_validate_nothing():
    found = sum(any(p['label'] == VALIDATED for p in _rules(candidates(_dataset(100 + seed)))) for seed in range(20))
    assert found <= 2


def test_every_horizon_has_an_economic_bar():
    assert set(MIN_NET_EXCESS_RETURN) == set(HORIZONS)


def test_a_pair_is_scored_on_what_its_second_condition_adds():
    rows = _dataset(5)
    rows = rows[rows['period'] == DEVELOPMENT].reset_index(drop=True)
    volume, tight = Condition('volume_ratio', ABOVE, _PLANTED_VOLUME), Condition('consolidation_tight', 'pass')
    # only the planted volume matters: inside the signals it keeps, the noise flag adds nothing
    rows = _plant_volume(rows, size=6.0)

    single, pair = Period(rows).score((volume,)), Period(rows).score((volume, tight))

    assert single['significant'] is False and single['p_value'] < 1e-6
    assert pair['within'] == volume.key
    in_volume = volume.keeps(rows)
    assert pair['kept']['n'] + pair['dropped']['n'] == int(in_volume.sum())
    assert pair['p_value'] > 0.01
    assert abs(pair['effect']['estimate']) < 1.5


def test_a_signal_with_no_value_fails_a_condition():
    rows = pd.DataFrame({'volume_ratio': [3.0, np.nan, 1.0], 'consolidation_tight': [1, np.nan, 0]})

    assert Condition('volume_ratio', ABOVE, 2.0).keeps(rows).tolist() == [True, False, False]
    assert Condition('volume_ratio', '<=', 2.0).keeps(rows).tolist() == [False, False, True]
    assert Condition('consolidation_tight', 'pass').keeps(rows).tolist() == [True, False, False]


def test_a_period_that_cannot_test_a_rule_says_so():
    rows = _plant_volume(_dataset(1))
    # validation keeps only a few dates: too few for the minimum sample
    validation_dates = rows.loc[rows['period'] == VALIDATION, 'session_date'].drop_duplicates()
    rows = rows[~rows['session_date'].isin(validation_dates.iloc[20:])]

    rules = _rules(candidates(rows))

    assert rules and all(p['reason'] == NOT_TESTED_ON_VALIDATION for p in rules)
    assert all(not p['scored_on_test'] for p in rules)


def test_a_test_period_that_cannot_test_a_rule_says_so():
    rows = _plant_volume(_dataset(1))
    test_dates = rows.loc[rows['period'] == TEST, 'session_date'].drop_duplicates()
    rows = rows[~rows['session_date'].isin(test_dates.iloc[20:])]

    best = _rules(candidates(rows))[0]

    assert best['periods'][VALIDATION]['significant'] and best['scored_on_test']
    assert best['label'] == PRELIMINARY and best['reason'] == NOT_TESTED_ON_TEST


def test_each_threshold_is_one_test_in_the_development_family():
    search = _search(candidates(_plant_volume(_dataset(1))))

    passing = sum(1 for name, kind in CONDITION_INDICATORS.items() if kind in (FLAG, SECONDARY))
    splits = (search['conditions'] - passing) // 2
    assert search['development_family'] <= passing + splits + search['pairs']
    assert search['development_family'] < search['tested_on_development']


def test_a_rule_that_holds_on_validation_only_is_not_confirmed_on_test():
    rows = _dataset(1)
    # the planted effect is in development and validation, gone on test
    earlier = rows['period'].isin([DEVELOPMENT, VALIDATION])
    rows.loc[earlier, 'excess_return_pct'] += 4.0 * (rows.loc[earlier, 'volume_ratio'] > _PLANTED_VOLUME)

    best = _rules(candidates(rows))[0]

    assert _indicators(best) == {'volume_ratio'}
    assert best['periods'][VALIDATION]['significant'] and best['scored_on_test']
    assert best['periods'][TEST]['shown'] and not best['periods'][TEST]['significant']
    assert best['label'] == PRELIMINARY and best['reason'] == NOT_CONFIRMED_ON_TEST
