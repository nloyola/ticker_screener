"""
BreakoutIQ's models: on a planted effect the ridge and logistic models rank a session's signals better than the
statuses and the rules, are calibrated and validated, while with no effect the large majority of simulations validate
none; the trees are kept only when they beat the simple models out of sample by the economic bar, on a planted
interaction the simple models cannot fit, never on an effect they already fit, and their attributions name what they
read; a sample too small to compare is untested, never validated; a rule is compared only on the dates with no
development signal; the design standardizes on the training rows alone; and the section reads the rules the run
validated before it.
"""

import json

import numpy as np
import pandas as pd

from app.breakoutiq.candidates import (
    AT_MOST,
    BELOW_ECONOMIC_BAR,
    PRELIMINARY,
    RULE,
    SECTION,
    VALIDATED,
    Condition,
    rule_key,
)
from app.breakoutiq.dataset import DEVELOPMENT, TEST, VALIDATION
from app.breakoutiq.indicators import value_frame
from app.breakoutiq.predictive import (
    CATEGORIES,
    FEATURES,
    LINEAR,
    LOGISTIC,
    MODEL,
    NO_FOLD_TRAINED,
    NOT_BETTER,
    NOT_KEPT,
    SIMPLE_MODELS,
    TOO_FEW_DATES,
    TREES,
    UNTESTED,
    WALK_FORWARD,
    Design,
    TreeDesign,
    fit_linear,
    fit_logistic,
    linear,
    logistic,
    predictive,
    trees,
)
from app.breakoutiq.thresholds import DEFAULTS
from app.breakoutiq.walkforward import AFTER_DEVELOPMENT, EVERY_DATE, STATUS, walk_forward
from app.models import AnalysisResult
from tests.test_breakoutiq_walkforward import HORIZON, SESSIONS, _dataset

_RULE = (Condition('rsi', AT_MOST, 0.0),)


def _rows(seed: int, size: float = 0.0, signals_per_date: int = 4) -> pd.DataFrame:
    """
    The walk-forward test's random dataset with the strengths, sector, regime and period the models read, the oldest
    60% of dates in development, then 20% in validation, the rest in test; 'size' plants a linear effect of rsi.
    """
    rows = _dataset(seed, signals_per_date)
    rng = np.random.default_rng(seed + 1000)
    for name in ('breakout_margin_pct', 'sma50_distance_pct', 'sma50_slope_pct', 'volume_ratio'):
        rows[name] = rng.normal(0, 1, len(rows))
    rows['sector'] = rng.choice(['Energy', 'Tech', 'Health'], len(rows))
    rows['regime'] = rng.choice(['above', 'below'], len(rows))
    position = rows['session_date'].map({d: i / len(SESSIONS) for i, d in enumerate(SESSIONS)})
    rows['period'] = np.select([position < 0.6, position < 0.8], [DEVELOPMENT, VALIDATION], TEST)
    rows['excess_return_pct'] += size * rows['rsi']
    return rows


def _interaction(seed: int, size: float = 8.0) -> pd.DataFrame:
    """The random dataset with an interaction planted: rsi times the volume ratio, which no single indicator shows."""
    rows = _rows(seed)
    rows['excess_return_pct'] += size * rows['rsi'] * rows['volume_ratio']
    return rows


def _models(results: list[AnalysisResult]) -> dict[str, dict]:
    return {r.finding.split(':', 1)[1]: r.payload for r in results if r.finding.startswith(f'{MODEL}:')}


def _validated(rule, horizon: int = HORIZON) -> AnalysisResult:
    payload = {'conditions': [c.payload() for c in rule], 'label': VALIDATED}
    return AnalysisResult(SECTION, f'{RULE}:{rule_key(rule)}', horizon, 1, payload)


def test_on_a_planted_effect_both_models_beat_the_rules_and_are_validated():
    results = predictive(_rows(1, size=3.0), earlier=[_validated(_RULE)])

    assert [r.finding for r in results] == [WALK_FORWARD, *(f'{MODEL}:{m}' for m in (LINEAR, LOGISTIC, TREES))]
    assert all(r.horizon == HORIZON for r in results)
    evaluation = results[0].payload
    assert evaluation['rules'] == 1 and evaluation['features'] and evaluation['l2_penalty'] > 0
    models = _models(results)
    for name in SIMPLE_MODELS:
        payload = models[name]
        assert payload['label'] == VALIDATED, name
        assert payload['reason'] is None
        assert payload['beats_rules'] and payload['clears_bar']
        assert payload['top_mean'] >= DEFAULTS.min_net_excess_return[HORIZON]
        assert set(payload['baselines']) == {STATUS, f'{RULE}:{rule_key(_RULE)}'}
        for baseline in payload['baselines'].values():
            assert baseline['shown'] and baseline['ci_low'] > 0
            # the model's top-ranked signals do better than the baseline's
            assert payload['top_mean'] > baseline['top_mean']
        assert payload['n'] > 0 and payload['signal_dates'] > 0 and payload['signals'] > 0
    # the coefficients name the planted indicator, and it dominates the others
    for name in (LINEAR, LOGISTIC):
        columns = models[name]['coefficients']['columns']
        assert max(columns, key=lambda c: abs(columns[c])) == 'rsi'
        assert columns['rsi'] > 0


def test_trees_that_do_not_beat_the_simple_models_by_the_bar_are_not_kept():
    # the effect is linear, so the simple models already fit it: the trees rank well, beat the rules and clear the
    # bar, yet do not leave the simple models behind, so they stay preliminary
    trees = _models(predictive(_rows(1, size=3.0), earlier=[_validated(_RULE)]))[TREES]

    assert trees['beats_rules'] and trees['clears_bar']
    assert not trees['kept']
    assert (trees['label'], trees['reason']) == (PRELIMINARY, NOT_KEPT)
    assert set(trees['simple_models']) == set(SIMPLE_MODELS)
    for compared in trees['simple_models'].values():
        assert compared['shown'] and compared['window'] == EVERY_DATE
        assert not compared['by_bar'] and compared['estimate'] < DEFAULTS.min_net_excess_return[HORIZON]
    assert trees['coefficients'] is None
    # still, the attributions name the indicator they read
    columns = trees['attributions']['columns']
    assert max(columns, key=columns.get) == 'rsi'


def test_on_a_planted_interaction_the_trees_are_kept_and_validated():
    results = predictive(_interaction(1), earlier=[_validated(_RULE)])
    models = _models(results)

    for name in SIMPLE_MODELS:
        assert (models[name]['label'], models[name]['reason']) == (PRELIMINARY, NOT_BETTER), name
    trees = models[TREES]
    assert trees['kept'] and (trees['label'], trees['reason']) == (VALIDATED, None)
    bar = DEFAULTS.min_net_excess_return[HORIZON]
    for name, compared in trees['simple_models'].items():
        # by the economic bar, out of sample, against each simple model's own lift
        assert compared['by_bar'] and compared['ci_low'] > 0 and compared['estimate'] >= bar
        assert compared['lift'] == models[name]['ranking']['lift']
    # the walk-forward finding holds the same comparisons
    assert results[0].payload['comparisons'][TREES]['models'].keys() == set(SIMPLE_MODELS)
    assert results[0].payload['comparisons'][LINEAR]['models'] == {}

    attributions = trees['attributions']
    assert attributions['shown'] and attributions['repeats'] > 0
    assert attributions['signal_dates'] == trees['ranking']['signal_dates']
    ranked = sorted(attributions['columns'], key=attributions['columns'].get, reverse=True)
    assert set(ranked[:2]) == {'rsi', 'volume_ratio'}
    assert list(attributions['columns']) == [*FEATURES, *CATEGORIES]
    # nothing else carries the interaction
    assert all(abs(attributions['columns'][c]) < 0.5 for c in ranked[2:])


def test_the_trees_read_raw_values_and_an_unseen_level_as_missing_and_leave_out_what_they_cannot_split():
    train = value_frame(_rows(11).iloc[:400].reset_index(drop=True))
    train.loc[0, 'rsi'] = np.nan
    train['volume_ratio'] = np.nan  # missing on every training row: left out
    train['macd_days_above'] = 3.0  # constant: left out
    design = TreeDesign(train)
    x = design.matrix(train)

    assert design.columns == [*(f for f in FEATURES if f not in ('volume_ratio', 'macd_days_above')), *CATEGORIES]
    rsi = design.columns.index('rsi')
    assert np.isnan(x[0, rsi]) and x[1, rsi] == train.loc[1, 'rsi']
    sector = design.columns.index('sector')
    assert set(x[:, sector]) == {0.0, 1.0, 2.0}
    test = train.iloc[:2].copy()
    test['sector'] = 'Utilities'
    assert np.isnan(design.matrix(test)[:, sector]).all()
    assert np.isfinite(trees(train, test)).all()


def test_the_models_are_calibrated_on_a_planted_effect():
    models = _models(predictive(_rows(2, size=3.0)))

    linear_fit = models[LINEAR]['calibration']
    assert linear_fit['shown'] and len(linear_fit['bins']) == 5
    realized = [b['realized'] for b in linear_fit['bins']]
    assert realized == sorted(realized), 'a higher predicted bin realizes more'
    # ridge shrinks the predictions a little, so the slope sits near and slightly above one
    assert 0.7 < linear_fit['slope']['estimate'] < 1.5
    assert linear_fit['slope']['ci_low'] > 0

    logistic_fit = models[LOGISTIC]['calibration']
    assert logistic_fit['shown'] and logistic_fit['brier'] < logistic_fit['brier_base']
    for b in logistic_fit['bins']:
        assert 0 <= b['predicted'] <= 1 and 0 <= b['realized'] <= 1


def test_with_no_effect_the_large_majority_of_simulations_validate_no_model():
    validated, untested, kept = 0, 0, 0
    for seed in range(12):
        for payload in _models(predictive(_rows(200 + seed), earlier=[_validated(_RULE)])).values():
            validated += payload['label'] == VALIDATED
            untested += payload['label'] == UNTESTED
            if payload['label'] != VALIDATED:
                assert payload['reason'] in (NOT_BETTER, BELOW_ECONOMIC_BAR, NOT_KEPT)
            kept += payload.get('kept', False)
    assert untested == 0
    assert validated <= 2
    assert kept <= 1


def test_a_sample_too_small_to_compare_is_untested():
    rows = _rows(3, size=3.0, signals_per_date=2)
    rows = rows[rows['session_date'] < SESSIONS[150]]
    models = _models(predictive(rows))
    for payload in models.values():
        assert payload['label'] == UNTESTED and payload['reason'] == TOO_FEW_DATES
        assert not payload['beats_rules']
        assert all(not b['shown'] for b in payload['baselines'].values())

    lone = _models(predictive(_rows(4).iloc[:40]))  # ten dates: no fold at all
    for payload in lone.values():
        assert payload['label'] == UNTESTED and payload['reason'] == NO_FOLD_TRAINED
    assert lone[TREES]['attributions'] is None


def test_a_rule_is_compared_only_on_dates_with_no_development_signal():
    rows = _rows(5, size=3.0)
    result = walk_forward(rows, HORIZON, {LINEAR: linear}, [_RULE])
    against = result['comparisons'][LINEAR]['baselines']
    status, rule = against[STATUS], against[f'{RULE}:{rule_key(_RULE)}']

    assert (status['window'], rule['window']) == (EVERY_DATE, AFTER_DEVELOPMENT)
    tested = sorted(pd.Timestamp(d) for f in result['folds'] for d in pd.bdate_range(f['first'], f['last']))
    development = set(rows.loc[rows['period'] == DEVELOPMENT, 'session_date'])
    assert status['signal_dates'] == len(tested)
    assert rule['signal_dates'] == len([d for d in tested if d not in development])
    assert 0 < rule['signal_dates'] < status['signal_dates']


def test_the_design_is_set_from_the_training_rows_alone():
    train = value_frame(_rows(6).iloc[:400].reset_index(drop=True))
    train['volume_ratio'] = 2.0  # constant in training: left out
    train.loc[0, 'rsi'] = np.nan
    design = Design(train)

    assert 'volume_ratio' not in design.features and 'rsi' in design.features
    x = design.matrix(train)
    z = x[:, : len(design.features)]
    assert abs(z[:, design.features.index('macd_days_above')].mean()) < 1e-9
    assert z[0, design.features.index('rsi')] == 0, 'a missing value is the training mean'

    # a level unseen in training adds nothing, and a test row is scaled by the training statistics
    test = train.iloc[:2].copy()
    test['sector'] = 'Utilities'
    dummies = design.matrix(test)[:, len(design.features) :]
    sector = [i for i, c in enumerate(design.columns[len(design.features) :]) if c.startswith('sector=')]
    assert (dummies[:, sector] == 0).all()


def test_the_fits_match_their_closed_forms():
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, (500, 3))
    y = 1.0 + x @ np.array([2.0, 0.0, -1.0]) + rng.normal(0, 0.1, 500)
    # no penalty is least squares, and a penalty shrinks the slopes but not the intercept
    assert np.allclose(fit_linear(x, y, penalty=0), np.linalg.lstsq(np.column_stack([np.ones(500), x]), y)[0])
    shrunk = fit_linear(x, y, penalty=1.0)
    assert np.linalg.norm(shrunk[1:]) < np.linalg.norm(fit_linear(x, y, penalty=0)[1:])
    assert abs(shrunk[0] - 1.0) < 0.1

    beat = (x[:, 0] + rng.normal(0, 0.5, 500) > 0).astype(float)
    beta = fit_logistic(x, beat)
    assert beta[1] > 1 and abs(beta[2]) < 0.3


def test_a_fold_the_logistic_model_cannot_fit_scores_nothing():
    rows = value_frame(_rows(8))
    train = rows.iloc[:200].reset_index(drop=True)
    train['excess_return_pct'] = 1.0  # every signal beat the benchmark: one class
    test = rows.iloc[200:220].reset_index(drop=True)
    assert np.isnan(logistic(train, test)).all()
    assert np.isfinite(linear(train, test)).all()
    assert np.isfinite(trees(train, test)).all()


def test_the_section_reads_only_the_rules_validated_at_its_horizon():
    other = (Condition('rsi', AT_MOST, 1.0),)
    results = predictive(_rows(9), earlier=[_validated(_RULE), _validated(other, horizon=60)])
    assert results[0].payload['rules'] == 1
    assert set(_models(results)[LINEAR]['baselines']) == {STATUS, f'{RULE}:{rule_key(_RULE)}'}


def test_every_payload_is_plain_json_with_no_missing_number():
    for result in predictive(_rows(10, size=3.0), earlier=[_validated(_RULE)]):
        text = json.dumps(result.payload, allow_nan=False)
        assert json.loads(text) == result.payload
