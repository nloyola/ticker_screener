"""
BreakoutIQ's walk-forward framework: a model ranks a session's signals better than the statuses and the rules when a
nonlinear effect is planted, and not in the large majority of simulations with no effect; each fold trains only on
signals whose outcomes were over GAP_SESSIONS sessions before it; a ranking that cannot tell a session's signals apart
lifts it by exactly zero, tied scores sharing their places, many rankings at once judged alike; a fold with nothing to
train on is left out; and the rules validated in a run are read back from its results.
"""

import numpy as np
import pandas as pd

from app.analyzer import CONFIRMED, HIGH_CONVICTION
from app.breakoutiq.candidates import ABOVE, AT_MOST, PRELIMINARY, RULE, SECTION, VALIDATED, Condition, rule_key
from app.breakoutiq.dataset import GAP_SESSIONS
from app.breakoutiq.indicators import FLAGS, VALUES, band_edges, bands
from app.breakoutiq.walkforward import (
    FOLDS,
    INITIAL_TRAIN_SHARE,
    STATUS,
    folds,
    lift_matrix,
    session_lifts,
    validated_rules,
    walk_forward,
)
from app.models import AnalysisResult

SESSIONS = pd.bdate_range('2016-01-04', periods=1000)
HORIZON = 20
# the stored raw values drawn at random, besides those VALUES derives from close and the averages
_RAW = [v for v in VALUES if v != 'macd_hist_pct' and not v.endswith('_distance_pct')]
# a rule a run could have validated on the planted effect: it keeps the middle and high rsi alike
_RULE = (Condition('rsi', AT_MOST, 0.5),)


def _dataset(seed: int, signals_per_date: int = 4) -> pd.DataFrame:
    """
    Complete outcomes at HORIZON over 1000 sessions, with indicators and statuses drawn at random, independent of the
    outcomes: every date's signals share a market shock, and each ticker has its own drift.
    """
    rng = np.random.default_rng(seed)
    count = signals_per_date * len(SESSIONS)
    tickers = rng.integers(0, 150, size=count)
    dates = np.repeat(SESSIONS, signals_per_date)
    frame = pd.DataFrame(
        {
            'signal_id': np.arange(1, count + 1),
            'session_date': dates,
            'market': 'US',
            'ticker': [f'T{t}' for t in tickers],
            'status': np.where(rng.integers(0, 2, count) == 1, HIGH_CONVICTION, CONFIRMED),
            'close': 100.0,
            'macd_hist': rng.normal(0.5, 0.3, count),
            'dma_20': 100 / (1 + rng.normal(0.03, 0.02, count)),
            'dma_200': 100 / (1 + rng.normal(0.15, 0.1, count)),
            'ema_21': 100 / (1 + rng.normal(0.03, 0.02, count)),
            'ema_21w': 100 / (1 + rng.normal(0.08, 0.05, count)),
            'price_gt_50dma': 1,
            'horizon': HORIZON,
            'outcome_status': 'complete',
            'exit_date': dates + pd.offsets.BDay(HORIZON),
            'last_session_date': pd.NaT,
        }
    )
    for name in _RAW:
        frame[name] = rng.normal(0, 1, count) if name != 'secondary_passed' else rng.integers(0, 8, count)
    for name in FLAGS:
        frame[name] = rng.integers(0, 2, count)
    shock = np.repeat(rng.normal(0, 4, len(SESSIONS)), signals_per_date)
    frame['excess_return_pct'] = shock + rng.normal(0, 2, 150)[tickers] + rng.normal(0, 6, count)
    return frame


def _plant(rows: pd.DataFrame, size: float = 4.0) -> pd.DataFrame:
    """An inverted U: a middling rsi, neither low nor high, adds 'size' to the excess return."""
    rows = rows.copy()
    rows['excess_return_pct'] += size * (rows['rsi'].abs() < 0.5)
    return rows


def _band_means(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """A model: each test signal scored by the mean excess return of its rsi band over the training signals."""
    edges = band_edges(train, 'rsi')
    means = train.groupby(bands(train['rsi'], edges))['excess_return_pct'].mean()
    clipped = test['rsi'].clip(edges[0], edges[-1])
    return bands(clipped, edges).map(means).to_numpy(dtype=float)


def _run(rows: pd.DataFrame) -> dict:
    return walk_forward(rows, HORIZON, {'bands': _band_means}, [_RULE])


def test_a_model_ranks_a_planted_nonlinear_effect_better_than_the_rules():
    result = _run(_plant(_dataset(1)))
    comparison = result['comparisons']['bands']

    assert comparison['beats_rules']
    assert set(comparison['baselines']) == {STATUS, f'{RULE}:{rule_key(_RULE)}'}
    for against in comparison['baselines'].values():
        assert against['shown'] and against['ci_low'] > 0
    rankers = result['rankers']
    # the rule catches part of the effect, the statuses none of it, and the model more than either
    assert rankers['bands']['lift'] > rankers[f'{RULE}:{rule_key(_RULE)}']['lift'] > rankers[STATUS]['lift']
    assert abs(rankers[STATUS]['lift']) < 0.5
    assert [f['fold'] for f in rankers['bands']['folds']] == list(range(1, FOLDS + 1))


def test_with_no_effect_the_large_majority_of_simulations_find_no_model_better():
    beaten = sum(_run(_dataset(100 + seed))['comparisons']['bands']['beats_rules'] for seed in range(20))
    assert beaten <= 2


def test_each_fold_trains_only_on_outcomes_over_a_gap_before_it():
    rows = _dataset(2)
    rows.loc[rows['signal_id'] % 7 == 0, 'exit_date'] += pd.offsets.BDay(200)  # some windows run late
    found = folds(rows)
    dates = pd.to_datetime(rows['session_date'])
    distinct = np.sort(dates.unique())
    first_test = int(np.ceil(len(distinct) * INITIAL_TRAIN_SHARE))

    assert len(found) == FOLDS
    tested = np.concatenate([np.sort(dates[f.test].unique()) for f in found])
    assert (tested == distinct[first_test:]).all(), 'the folds cover the dates after the initial training, in order'
    for fold in found:
        train = rows[fold.train]
        assert train['session_date'].max() < fold.first
        between = np.busday_count(train['session_date'].max().date(), fold.first.date()) - 1
        assert between >= GAP_SESSIONS
        assert (pd.to_datetime(train['exit_date']) < fold.first).all()
        # every signal old enough and over in time trains
        ready = (
            np.busday_count(dates.dt.date.to_numpy(dtype='datetime64[D]'), fold.first.date()) - 1 >= GAP_SESSIONS
        ) & (pd.to_datetime(rows['exit_date']) < fold.first)
        assert (fold.train == ready.to_numpy()).all()


def test_the_folds_count_a_markets_own_sessions_when_given():
    rows = _dataset(3)
    calendar = pd.DatetimeIndex(SESSIONS[::2])  # a market open every other business day
    plain, sparse = folds(rows), folds(rows, {'US': calendar})
    for a, b in zip(plain, sparse, strict=True):
        assert b.train.sum() < a.train.sum()
        last = rows[b.train]['session_date'].max()
        assert calendar.searchsorted(b.first) - calendar.searchsorted(last, side='right') >= GAP_SESSIONS


def test_a_ranking_that_cannot_tell_signals_apart_lifts_nothing():
    rows = _dataset(4)
    lifts = session_lifts(rows, np.ones(len(rows)))
    assert len(lifts) == len(SESSIONS)
    assert np.allclose(lifts['lift'], 0)


def test_tied_scores_share_the_places_they_span():
    test = pd.DataFrame(
        {
            'session_date': pd.Timestamp('2020-01-02'),
            'outcome_status': 'complete',
            'excess_return_pct': [9.0, 3.0, 0.0, -6.0],
        }
    )
    # the top half is two places: the first signal, and the two tied for second share the other
    lifts = session_lifts(test, np.array([3.0, 2.0, 2.0, np.nan]))
    assert lifts['top_mean'].item() == (9.0 + 0.5 * 3.0 + 0.5 * 0.0) / 2
    assert lifts['lift'].item() == lifts['top_mean'].item() - 1.5


def test_censored_horizons_and_lone_signals_are_never_ranked():
    test = pd.DataFrame(
        {
            'session_date': pd.to_datetime(['2020-01-02'] * 3 + ['2020-01-03'] * 2),
            'outcome_status': ['complete', 'complete', 'ended', 'complete', 'ended'],
            'excess_return_pct': [4.0, 0.0, np.nan, 5.0, np.nan],
        }
    )
    lifts = session_lifts(test, np.array([1.0, 0.0, 9.0, 1.0, 9.0]))
    assert lifts['session_date'].tolist() == [pd.Timestamp('2020-01-02')]
    assert lifts['signals'].item() == 2 and lifts['lift'].item() == 2.0


def test_a_fold_with_nothing_to_train_on_is_reported_and_left_out():
    rows = _dataset(5)
    rows.loc[rows['session_date'] < SESSIONS[600], 'outcome_status'] = 'ended'
    rows.loc[rows['session_date'] < SESSIONS[600], 'excess_return_pct'] = np.nan
    result = _run(rows)
    untrained = [f['fold'] for f in result['folds'] if not f['trained']]
    assert untrained and untrained[0] == 1
    ranked = {f['fold'] for f in result['rankers']['bands']['folds']}
    assert ranked.isdisjoint(untrained)


def test_too_few_dates_are_not_compared():
    rows = _dataset(6, signals_per_date=2)
    rows = rows[rows['session_date'] < SESSIONS[150]]
    result = _run(rows)
    against = result['comparisons']['bands']['baselines'][STATUS]
    assert not result['comparisons']['bands']['beats_rules']
    assert against['signal_dates'] < 30 and not against['shown'] and against['estimate'] is None


def test_the_validated_rules_are_read_from_a_runs_results():
    def rule(conditions, label, horizon=HORIZON):
        payload = {'conditions': [c.payload() for c in conditions], 'label': label}
        return AnalysisResult(SECTION, f'{RULE}:{rule_key(conditions)}', horizon, 1, payload)

    pair = (Condition('volume_ratio', ABOVE, 2.5), Condition('rsi_50_70', 'pass'))
    results = [
        AnalysisResult(SECTION, 'search', HORIZON, 1, {}),
        rule(pair, VALIDATED),
        rule(_RULE, PRELIMINARY),
        rule(_RULE, VALIDATED, horizon=60),
    ]
    assert validated_rules(results, HORIZON) == [pair]
    assert validated_rules(results, 60) == [_RULE]


def test_the_lift_matrix_judges_each_ranking_as_session_lifts_does():
    rows = _dataset(12)[:400].reset_index(drop=True)
    rows.loc[::7, 'outcome_status'] = 'ended'
    rng = np.random.default_rng(12)
    scores = rng.normal(0, 1, (len(rows), 3))
    scores[::5, 1] = np.nan
    scores[:, 2] = 1.0

    matrix = lift_matrix(rows, scores)
    for column in range(scores.shape[1]):
        assert np.allclose(matrix[:, column], session_lifts(rows, scores[:, column])['lift'].to_numpy())
    assert np.allclose(matrix[:, 2], 0)
