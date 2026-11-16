"""
Simple predictive models of a signal's main outcome, the net excess return over the benchmark, judged walk-forward
(walkforward.py) against the strategy's statuses and the rules a run validated.

Two simple, inspectable models (SIMPLE_MODELS), and one more complex, each fitted on a fold's complete training
outcomes:
- LINEAR: a ridge regression of the net excess return.
- LOGISTIC: an L2-regularized logistic regression of whether the signal beat the benchmark net of costs (a net
  excess return above zero).
- TREES: gradient-boosted regression trees of the net excess return (scikit-learn's HistGradientBoostingRegressor,
  TREE_ITERATIONS trees at most TREE_MAX_DEPTH deep, each leaf holding at least TREE_MIN_LEAF signals), which can fit
  an interaction or a threshold the simple models cannot. They read every indicator as it is, a missing value as
  missing, and the regime, the sector and the market as categories, a level not seen in training read as missing.

The trees are kept only when they leave the simple models behind: when, against each simple model on every date both
ranked, the interval of their lift less its lift lies above zero and the difference reaches the economic bar. Trees
that are not kept are never more than PRELIMINARY (NOT_KEPT), however they fare against the rules, so a proposal
comes from them only when the simple models leave signal on the table. Their attributions say what they read: for
each indicator and category, the lift lost, out of sample, when its values are shuffled among a fold's test signals
(ATTRIBUTION_REPEATS times, averaged over every date ranked), read from the very trees each fold of the
walk-forward fitted.

The simple models read the same design: every indicator the confounders section regresses (confounders.INDICATORS: each
secondary indicator's raw value but secondary_passed, and each mandatory strength), standardized by the training rows'
mean and standard deviation, a missing value set to the training mean, and one column per level of each of CATEGORIES
(the regime, the sector and the market) seen in training, a level not seen in training adding nothing. An indicator
constant or missing on every training row is left out. The intercept is not penalized, every other coefficient by
L2_PENALTY, so the penalty treats every standardized indicator alike. A model ranks a session's signals by its
prediction, higher better; a fold whose training rows cannot fit it (fewer than two rows, or for LOGISTIC a single
class) scores nothing, so it ranks no signal above another.

At each horizon the section reports, per model:
- its ranking quality: its lift per fold and over every fold, and the mean net excess return of the signals it ranks
  highest (top_mean), beside each baseline's, and its comparison with each baseline (walkforward.walk_forward);
- its calibration, on its pooled out-of-sample predictions of complete outcomes: CALIBRATION_BINS quantile bins of the
  prediction, each with its predicted and realized mean, and, for LINEAR and TREES, the slope of the realized outcome on
  the prediction (1 when calibrated, 0 when it says nothing), with errors clustered by date and ticker, for LOGISTIC its
  Brier score beside that of predicting each fold's training share of signals that beat the benchmark. Calibration is
  across sessions, so the market's shared move weighs on it, while the ranking is within each session;
- for a simple model, its coefficients fitted on every complete outcome of the horizon, per standard deviation,
  descriptive; for TREES, their comparison with each simple model, whether they are kept, and their attributions;
- its label: UNTESTED when a comparison with a baseline (or for TREES, a simple model) could not be made (too few
  signal dates, or no fold trained), PRELIMINARY when it does not beat the rules, when its top-ranked signals' mean
  net excess return is below the economic bar, or when it is TREES not kept, VALIDATED only when it beats every
  baseline and its top-ranked signals reach the bar.

Every figure carries its sample size. Ended (censored) horizons are counted, never ranked or fitted. A model's output
is a ranking proposal: nothing here changes the live screener.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor

from app.breakoutiq.candidates import BELOW_ECONOMIC_BAR, PRELIMINARY, VALIDATED
from app.breakoutiq.confounders import INDICATORS
from app.breakoutiq.dataset import COMPLETE, ENDED
from app.breakoutiq.indicators import MAIN_OUTCOME, OUTCOMES, value_frame
from app.breakoutiq.stats import CLUSTERS, cluster_codes, clustered_ols
from app.breakoutiq.thresholds import DEFAULTS, Thresholds
from app.breakoutiq.walkforward import Ranker, folds, lift_matrix, validated_rules, walk_forward
from app.models import AnalysisResult

SECTION = 'models'
PAYLOAD_VERSION = 1

# the findings: the walk-forward evaluation at a horizon as a whole, and each model ('model:<name>')
WALK_FORWARD = 'walk_forward'
MODEL = 'model'

# the models
LINEAR = 'linear'
LOGISTIC = 'logistic'
TREES = 'trees'
DESCRIPTIONS = {
    LINEAR: 'ridge regression of the net excess return',
    LOGISTIC: 'L2-regularized logistic regression of beating the benchmark net of costs',
    TREES: 'gradient-boosted regression trees of the net excess return',
}
# the simple models, which the trees must leave behind to be kept
SIMPLE_MODELS = (LINEAR, LOGISTIC)

# the indicators each model reads, standardized, and the categories it reads as one column per level
FEATURES = tuple(INDICATORS)
CATEGORIES = ('regime', 'sector', 'market')
# the L2 penalty on every coefficient but the intercept, per row: the mean loss plus L2_PENALTY / 2 times the sum of
# the squared coefficients (for LINEAR the loss is half the squared error)
L2_PENALTY = 0.1
# the quantile bins of a model's predictions its calibration is read in
CALIBRATION_BINS = 5
# the trees: how many, how deep, the fewest signals a leaf holds, each tree's shrinkage and the L2 penalty on the
# leaves' values
TREE_ITERATIONS = 100
TREE_MAX_DEPTH = 3
TREE_MIN_LEAF = 20
TREE_LEARNING_RATE = 0.1
TREE_L2_PENALTY = 1.0
# how many times each indicator or category is shuffled for the trees' attributions, and the shuffles' seed
ATTRIBUTION_REPEATS = 5
ATTRIBUTION_SEED = 20260

# a model's label when no comparison with a baseline could be made
UNTESTED = 'untested'
# why a model is untested, or preliminary
NO_FOLD_TRAINED = 'no fold trained'
TOO_FEW_DATES = 'too few signal dates'
NOT_BETTER = 'does not beat the rules'
NOT_KEPT = 'does not beat the simple models by the economic bar'

# a level of a category a row has no value for
_UNKNOWN = 'unknown'


class Design:
    """
    The design a model is fitted on, set from its training rows: the indicators kept, their means and standard
    deviations, and each category's levels.
    """

    def __init__(self, train: pd.DataFrame):
        values = _values(train)
        self.means = values.mean()
        self.sds = values.std()
        self.features = [f for f in FEATURES if np.isfinite(self.sds[f]) and self.sds[f] > 0]
        self.levels = {c: sorted(_levels(train[c]).unique().tolist()) for c in CATEGORIES}

    @property
    def columns(self) -> list[str]:
        return [*self.features, *(f'{c}={level}' for c in CATEGORIES for level in self.levels[c])]

    def matrix(self, frame: pd.DataFrame) -> np.ndarray:
        """'frame' as the design's columns: the indicators standardized, a missing one 0, then the levels' dummies."""
        values = _values(frame)[self.features]
        z = ((values - self.means[self.features]) / self.sds[self.features]).fillna(0.0).to_numpy(dtype=float)
        dummies = [
            (_levels(frame[c]).to_numpy()[:, None] == np.array(self.levels[c], dtype=object)[None, :]).astype(float)
            for c in CATEGORIES
        ]
        return np.column_stack([z, *dummies]) if len(self.columns) else np.empty((len(frame), 0))


class TreeDesign:
    """
    The design the trees are fitted on, set from their training rows: every indicator as it is, a missing value
    missing, then each category as the index of its level among those seen in training, a level not seen in training
    missing. An indicator with fewer than two distinct values in training is left out: no tree could split on it.
    """

    def __init__(self, train: pd.DataFrame):
        values = _values(train)
        self.features = [f for f in FEATURES if values[f].nunique() > 1]
        self.levels = {c: sorted(_levels(train[c]).unique().tolist()) for c in CATEGORIES}

    @property
    def columns(self) -> list[str]:
        return [*self.features, *CATEGORIES]

    @property
    def categorical(self) -> list[bool]:
        return [False] * len(self.features) + [True] * len(CATEGORIES)

    def matrix(self, frame: pd.DataFrame) -> np.ndarray:
        codes = [
            _levels(frame[c]).map({level: i for i, level in enumerate(self.levels[c])}).to_numpy(dtype=float)
            for c in CATEGORIES
        ]
        return np.column_stack([_values(frame)[self.features].to_numpy(dtype=float), *codes])


def fit_trees(design: TreeDesign, x: np.ndarray, y: np.ndarray) -> HistGradientBoostingRegressor:
    """The trees fitted on 'x', the design's matrix, to 'y', with no early stopping, so a fit is reproducible."""
    return HistGradientBoostingRegressor(
        max_iter=TREE_ITERATIONS,
        max_depth=TREE_MAX_DEPTH,
        min_samples_leaf=TREE_MIN_LEAF,
        learning_rate=TREE_LEARNING_RATE,
        l2_regularization=TREE_L2_PENALTY,
        categorical_features=design.categorical,
        early_stopping=False,
        random_state=0,
    ).fit(x, y)


def fit_linear(x: np.ndarray, y: np.ndarray, penalty: float = L2_PENALTY) -> np.ndarray:
    """The ridge coefficients, intercept first: the mean of half the squared error plus penalty / 2 times |w|^2."""
    a = np.column_stack([np.ones(len(x)), x])
    shrink = np.diag([0.0, *([penalty] * x.shape[1])])
    return np.linalg.solve(a.T @ a / len(a) + shrink, a.T @ y / len(a))


def fit_logistic(x: np.ndarray, y: np.ndarray, penalty: float = L2_PENALTY) -> np.ndarray:
    """The logistic coefficients, intercept first: the mean log loss plus penalty / 2 times |w|^2."""
    a = np.column_stack([np.ones(len(x)), x])
    mask = np.array([0.0, *([1.0] * x.shape[1])])

    def loss(beta: np.ndarray) -> tuple[float, np.ndarray]:
        logits = a @ beta
        value = np.mean(np.logaddexp(0, logits) - y * logits) + penalty / 2 * np.sum(mask * beta**2)
        gradient = a.T @ (_sigmoid(logits) - y) / len(a) + penalty * mask * beta
        return float(value), gradient

    share = np.clip(y.mean(), 1e-6, 1 - 1e-6)
    start = np.zeros(a.shape[1])
    start[0] = np.log(share / (1 - share))
    return minimize(loss, start, jac=True, method='L-BFGS-B').x


def linear(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """The LINEAR ranker: each test signal's predicted net excess return."""
    return _rank(train, test, LINEAR)


def logistic(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """The LOGISTIC ranker: each test signal's predicted probability of beating the benchmark net of costs."""
    return _rank(train, test, LOGISTIC)


def trees(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """The TREES ranker: each test signal's predicted net excess return."""
    return _rank(train, test, TREES)


class FoldTrees:
    """The TREES ranker, keeping each fold's design, fitted trees and test rows, for their attributions."""

    def __init__(self):
        self.fits: list[tuple[TreeDesign, HistGradientBoostingRegressor, pd.DataFrame]] = []

    def __call__(self, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        fitted = _fit(TREES, _complete(train))
        if fitted is None:
            return np.full(len(test), np.nan)
        design, model = fitted
        self.fits.append((design, model, test))
        return model.predict(design.matrix(test))


MODELS: dict[str, Ranker] = {LINEAR: linear, LOGISTIC: logistic, TREES: trees}


def predictive(
    rows: pd.DataFrame, thresholds: Thresholds = DEFAULTS, earlier: list[AnalysisResult] | None = None
) -> list[AnalysisResult]:
    """
    The section's results from the analysis dataset (dataset.build_dataset), against the rules 'earlier' (the run's
    results before this section) validated: per horizon, the WALK_FORWARD finding, then one 'model:<name>' per model.
    """
    results = []
    for horizon in sorted(int(h) for h in rows['horizon'].unique()):
        raw = rows[rows['horizon'] == horizon].reset_index(drop=True)
        at = value_frame(raw)
        rules = validated_rules(earlier or [], horizon)
        predictions: dict[str, pd.DataFrame] = {}
        fold_trees = FoldTrees()
        evaluation = walk_forward(
            raw,
            horizon,
            {**MODELS, TREES: fold_trees},
            rules,
            thresholds,
            predictions=predictions,
            versus={TREES: SIMPLE_MODELS},
        )
        evaluation = {**evaluation, 'rules': len(rules), **_settings(thresholds)}
        results.append(AnalysisResult(SECTION, WALK_FORWARD, horizon, PAYLOAD_VERSION, evaluation))
        for name in MODELS:
            payload = _model(name, at, evaluation, predictions[name], horizon, thresholds)
            if name == TREES:
                payload = _trees(payload, fold_trees, evaluation, horizon, thresholds)
            results.append(AnalysisResult(SECTION, f'{MODEL}:{name}', horizon, PAYLOAD_VERSION, payload))
    return results


def _model(
    name: str,
    rows: pd.DataFrame,
    evaluation: dict,
    predicted: pd.DataFrame,
    horizon: int,
    thresholds: Thresholds,
) -> dict:
    """One model's payload at one horizon."""
    ranking = evaluation['rankers'][name]
    comparison = evaluation['comparisons'][name]
    bar = thresholds.min_net_excess_return[horizon]
    trained = any(f['trained'] for f in evaluation['folds'])
    label, reason = _label(trained, comparison['baselines'], comparison['beats_rules'], ranking['top_mean'], bar)
    observed = _observed(predicted)
    return {
        'model': name,
        'description': DESCRIPTIONS[name],
        'outcome': MAIN_OUTCOME,
        'label': label,
        'reason': reason,
        'signals': int(predicted['signal_id'].nunique()),
        'n': int(observed.sum()),
        'signal_dates': int(predicted.loc[observed, 'session_date'].nunique()),
        'censored': int((predicted['outcome_status'] == ENDED).sum()),
        'ranking': ranking,
        'baselines': {
            baseline: {
                **comparison['baselines'][baseline],
                'lift': evaluation['rankers'][baseline]['lift'],
                'top_mean': evaluation['rankers'][baseline]['top_mean'],
                'ranked_dates': evaluation['rankers'][baseline]['signal_dates'],
            }
            for baseline in comparison['baselines']
        },
        'beats_rules': comparison['beats_rules'],
        'top_mean': ranking['top_mean'],
        'min_net_excess_return': bar,
        'clears_bar': ranking['top_mean'] is not None and ranking['top_mean'] >= bar,
        'calibration': _calibration(name, rows, predicted, thresholds),
        'coefficients': _coefficients(name, rows),
        **_settings(thresholds),
    }


def _trees(payload: dict, fold_trees: FoldTrees, evaluation: dict, horizon: int, thresholds: Thresholds) -> dict:
    """
    The TREES payload with their comparison with each simple model, whether they are kept, their attributions, and
    their label again: never more than PRELIMINARY unless kept, and UNTESTED when a comparison could not be made.
    """
    bar = thresholds.min_net_excess_return[horizon]
    against = {
        name: {
            **compared,
            'lift': evaluation['rankers'][name]['lift'],
            'top_mean': evaluation['rankers'][name]['top_mean'],
            'ranked_dates': evaluation['rankers'][name]['signal_dates'],
            'by_bar': compared['beats'] and compared['estimate'] >= bar,
        }
        for name, compared in evaluation['comparisons'][TREES]['models'].items()
    }
    kept = all(c['by_bar'] for c in against.values())
    trained = any(f['trained'] for f in evaluation['folds'])
    shown = {**payload['baselines'], **against}
    label, reason = _label(trained, shown, payload['beats_rules'], payload['top_mean'], bar, kept)
    return {
        **payload,
        'label': label,
        'reason': reason,
        'simple_models': against,
        'kept': kept,
        'attributions': _attributions(fold_trees.fits, thresholds) if trained else None,
    }


def _attributions(
    fits: list[tuple[TreeDesign, HistGradientBoostingRegressor, pd.DataFrame]], thresholds: Thresholds
) -> dict:
    """
    What the trees read, out of sample, from each fold's 'fits' (FoldTrees.fits): for each indicator and category, the
    lift lost when its values are shuffled among the fold's test signals, ATTRIBUTION_REPEATS times, averaged over
    every date the trees ranked.
    """
    rng = np.random.default_rng(ATTRIBUTION_SEED)
    columns = [*FEATURES, *CATEGORIES]
    lost = dict.fromkeys(columns, 0.0)
    dates = 0
    for design, model, test in fits:
        x = design.matrix(test)
        base = lift_matrix(test, model.predict(x)[:, None])[:, 0]
        dates += len(base)
        # a column the fold's trees left out loses nothing when shuffled
        for j, column in enumerate(design.columns):
            shuffled = np.tile(x, (ATTRIBUTION_REPEATS, 1))
            shuffled[:, j] = np.concatenate([rng.permutation(x[:, j]) for _ in range(ATTRIBUTION_REPEATS)])
            scores = model.predict(shuffled).reshape(ATTRIBUTION_REPEATS, len(test)).T
            lost[column] += float((base[:, None] - lift_matrix(test, scores)).sum()) / ATTRIBUTION_REPEATS
    shown = dates >= thresholds.min_signal_dates
    return {
        'method': 'permutation',
        'measure': 'lift lost when shuffled',
        'repeats': ATTRIBUTION_REPEATS,
        'signal_dates': dates,
        'shown': shown,
        'columns': {c: lost[c] / dates for c in columns} if shown else {},
    }


def _label(
    trained: bool,
    compared: dict[str, dict],
    beats_rules: bool,
    top_mean: float | None,
    bar: float,
    kept: bool = True,
) -> tuple[str, str | None]:
    """
    A model's label from its comparisons, and why it is untested or preliminary (None when validated); 'kept' is False
    only for trees that do not leave the simple models behind.
    """
    if not trained:
        return UNTESTED, NO_FOLD_TRAINED
    if not all(c['shown'] for c in compared.values()):
        return UNTESTED, TOO_FEW_DATES
    if not kept:
        return PRELIMINARY, NOT_KEPT
    if not beats_rules:
        return PRELIMINARY, NOT_BETTER
    if top_mean is None or top_mean < bar:
        return PRELIMINARY, BELOW_ECONOMIC_BAR
    return VALIDATED, None


def _calibration(name: str, rows: pd.DataFrame, predicted: pd.DataFrame, thresholds: Thresholds) -> dict:
    """
    A model's calibration on its out-of-sample predictions of complete outcomes: the bins, and for LINEAR the slope,
    for LOGISTIC the Brier scores.
    """
    at = predicted[_observed(predicted)].reset_index(drop=True)
    y = at[OUTCOMES[MAIN_OUTCOME].column].to_numpy(dtype=float)
    realized = (y > 0).astype(float) if name == LOGISTIC else y
    score = at['score'].to_numpy(dtype=float)
    dates = int(at['session_date'].nunique())
    result = {
        'n': len(at),
        'signal_dates': dates,
        'shown': dates >= thresholds.min_signal_dates,
        'realized': 'share that beat the benchmark' if name == LOGISTIC else 'net excess return',
        'bins': _bins(at, score, realized, thresholds),
    }
    if name != LOGISTIC:
        result['slope'] = _slope(at, score, realized, thresholds) if result['shown'] else None
    else:
        base = _training_shares(rows, at)
        result['brier'] = float(np.mean((score - realized) ** 2)) if result['shown'] else None
        result['brier_base'] = float(np.mean((base - realized) ** 2)) if result['shown'] else None
    return result


def _bins(at: pd.DataFrame, score: np.ndarray, realized: np.ndarray, thresholds: Thresholds) -> list[dict]:
    """The predictions' quantile bins, lowest first: each one's sample, and its predicted and realized means."""
    if not len(at):
        return []
    codes = pd.qcut(pd.Series(score).rank(method='first'), CALIBRATION_BINS, labels=False, duplicates='drop')
    bins = []
    for code in sorted(codes.unique()):
        mask = (codes == code).to_numpy()
        dates = int(at.loc[mask, 'session_date'].nunique())
        shown = dates >= thresholds.min_signal_dates
        bins.append(
            {
                'bin': int(code) + 1,
                'n': int(mask.sum()),
                'signal_dates': dates,
                'shown': shown,
                'predicted': float(score[mask].mean()) if shown else None,
                'realized': float(realized[mask].mean()) if shown else None,
            }
        )
    return bins


def _slope(at: pd.DataFrame, score: np.ndarray, realized: np.ndarray, thresholds: Thresholds) -> dict | None:
    """The slope of the realized outcome on the prediction, with errors clustered by date and ticker."""
    if np.std(score) == 0:
        return None
    fit = clustered_ols(
        realized,
        pd.DataFrame({'const': 1.0, 'predicted': score}),
        cluster_codes(at),
        of=['predicted'],
        confidence=thresholds.confidence,
    )
    if fit is None:
        return None
    slope = fit['predicted']
    return {'estimate': slope.estimate, 'se': slope.se, 'ci_low': slope.ci_low, 'ci_high': slope.ci_high}


def _training_shares(rows: pd.DataFrame, at: pd.DataFrame) -> np.ndarray:
    """Each prediction's fold's training share of complete outcomes that beat the benchmark, aligned with 'at'."""
    shares = {}
    for fold in folds(rows):
        train = rows[fold.train]
        y = pd.to_numeric(train[OUTCOMES[MAIN_OUTCOME].column], errors='coerce')
        y = y[(train['outcome_status'] == COMPLETE) & y.notna()]
        shares[fold.number] = float((y > 0).mean()) if len(y) else np.nan
    return at['fold'].map(shares).to_numpy(dtype=float)


def _coefficients(name: str, rows: pd.DataFrame) -> dict | None:
    """
    The model fitted on every complete outcome of the horizon: its sample and coefficients, by column; None for the
    trees, which have none.
    """
    if name == TREES:
        return None
    complete = _complete(rows)
    beta = _fit(name, complete)
    if beta is None:
        return None
    design, coefficients = beta
    return {
        'n': len(complete),
        'signal_dates': int(complete['session_date'].nunique()),
        'scale': 'per standard deviation',
        'intercept': float(coefficients[0]),
        'columns': dict(zip(design.columns, (float(c) for c in coefficients[1:]), strict=True)),
    }


def _rank(train: pd.DataFrame, test: pd.DataFrame, name: str) -> np.ndarray:
    fitted = _fit(name, _complete(train))
    if fitted is None:
        return np.full(len(test), np.nan)
    if name == TREES:
        design, model = fitted
        return model.predict(design.matrix(test))
    design, beta = fitted
    logits = beta[0] + design.matrix(test) @ beta[1:]
    return logits if name == LINEAR else _sigmoid(logits)


def _fit(name: str, complete: pd.DataFrame) -> tuple | None:
    """
    The model fitted on 'complete' (complete outcomes only), with its design: for the trees the fitted trees, else the
    coefficients; None when it cannot be fitted.
    """
    if len(complete) < 2:
        return None
    y = complete[OUTCOMES[MAIN_OUTCOME].column].to_numpy(dtype=float)
    if name == TREES:
        tree_design = TreeDesign(complete)
        return tree_design, fit_trees(tree_design, tree_design.matrix(complete), y)
    design = Design(complete)
    x = design.matrix(complete)
    if name == LINEAR:
        return design, fit_linear(x, y)
    beat = (y > 0).astype(float)
    if beat.min() == beat.max():
        return None
    return design, fit_logistic(x, beat)


def _complete(rows: pd.DataFrame) -> pd.DataFrame:
    y = pd.to_numeric(rows[OUTCOMES[MAIN_OUTCOME].column], errors='coerce')
    return rows[(rows['outcome_status'] == COMPLETE) & y.notna()].reset_index(drop=True)


def _observed(predicted: pd.DataFrame) -> np.ndarray:
    y = pd.to_numeric(predicted[OUTCOMES[MAIN_OUTCOME].column], errors='coerce')
    return ((predicted['outcome_status'] == COMPLETE) & y.notna()).to_numpy()


def _values(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[list(FEATURES)].apply(pd.to_numeric, errors='coerce').astype(float)


def _levels(values: pd.Series) -> pd.Series:
    return values.astype(object).where(values.notna(), _UNKNOWN).astype(str)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0, -x))


def _settings(thresholds: Thresholds) -> dict:
    return {
        'features': list(FEATURES),
        'categories': list(CATEGORIES),
        'l2_penalty': L2_PENALTY,
        'calibration_bins': CALIBRATION_BINS,
        'tree_iterations': TREE_ITERATIONS,
        'tree_max_depth': TREE_MAX_DEPTH,
        'tree_min_leaf': TREE_MIN_LEAF,
        'tree_learning_rate': TREE_LEARNING_RATE,
        'tree_l2_penalty': TREE_L2_PENALTY,
        'attribution_repeats': ATTRIBUTION_REPEATS,
        'min_signal_dates': thresholds.min_signal_dates,
        'confidence': thresholds.confidence,
        'clusters': list(CLUSTERS),
    }
