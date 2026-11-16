"""
Walk-forward evaluation: how well a model ranks a session's signals by their net excess return over the benchmark,
trained only on what was known before, against the strategy's statuses and the validated candidate rules.

The signal dates of one horizon's rows are cut in order: the oldest INITIAL_TRAIN_SHARE of them only ever train, and
the rest into FOLDS consecutive test folds of about equal dates. Each fold trains on the signals dated before its
first date whose outcome was over before it: at least GAP_SESSIONS sessions of the signal's market lie strictly
between the two dates, the same gap the dataset's periods keep (dataset.assign_periods), and the signal's known outcome
window (its exit, or for an ended horizon its last session) ends before the fold's first date. A fold left with no
complete outcome to train on is reported, and left out of every comparison.

A ranker scores a fold's test signals, higher better, from the fold's training rows: a model fits them, while the
baselines ignore them, the statuses ranking a high-conviction signal above a confirmed one (status_ranker) and a rule
ranking the signals it keeps above those it drops (rule_ranker). A signal with no score ranks last.

A ranking is judged within each session, since a model's output is a ranking of one session's signals and the
market's move is shared by all of them: on every test date with complete outcomes on at least two signals, its lift
is the mean net excess return of the TOP_SHARE of them it ranks highest, less the mean of them all. Tied scores share
the places they span, so a ranking that cannot tell a session's signals apart lifts it by exactly zero. Ended
(censored) horizons are counted, never ranked.

A model is compared with each baseline on the dates both ranked: the mean over those dates of its lift less the
baseline's, with a confidence interval bootstrapped by blocks of signal dates as long as the horizon
(stats.block_bootstrap), since signals a few sessions apart share most of their window. A comparison on fewer than
the run's min_signal_dates dates is not shown. A model beats a baseline when the interval lies above zero, and beats
the rules when it beats every baseline, which needs no correction for the several comparisons. A model can be
compared with other models the same way, on every date both ranked.

The folds cover the dataset's periods without regard to them, and each fold reports its signals by period. A rule was
found on development, so it is in sample there: a model is compared with a rule only on the dates none of whose
signals is in development (AFTER_DEVELOPMENT), and with the statuses, which nothing was fitted to, on every date both
ranked (EVERY_DATE). Rows with no period are all compared. Nothing here changes the live screener.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.analyzer import HIGH_CONVICTION
from app.breakoutiq.candidates import RULE, SECTION, VALIDATED, Condition, Rule, rule_key
from app.breakoutiq.dataset import COMPLETE, DEVELOPMENT, ENDED, GAP_SESSIONS, PERIODS
from app.breakoutiq.indicators import MAIN_OUTCOME, OUTCOMES, indicator_frame
from app.breakoutiq.stats import block_bootstrap, block_mean, date_blocks
from app.breakoutiq.thresholds import DEFAULTS, Thresholds
from app.models import AnalysisResult

# the share of the signal dates, oldest first, that only ever train
INITIAL_TRAIN_SHARE = 0.4
# the test folds the rest of the signal dates are cut into
FOLDS = 5
# the share of a session's signals a ranking's top holds
TOP_SHARE = 0.5

# what a ranker is
MODEL = 'model'
STATUS = 'status'
RULE_RANKER = 'rule'

# the dates a model is compared with a baseline on: every date both ranked, or those with no development signal
EVERY_DATE = 'every date'
AFTER_DEVELOPMENT = 'after development'

# a ranker: from a fold's training rows and its test rows, each test row's score, higher better, NaN for none
Ranker = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


@dataclass(frozen=True)
class Fold:
    """One test fold: its number from 1, its first and last date, and its training and test rows as masks."""

    number: int
    first: pd.Timestamp
    last: pd.Timestamp
    train: np.ndarray
    test: np.ndarray


def folds(rows: pd.DataFrame, calendars: dict[str, pd.DatetimeIndex] | None = None) -> list[Fold]:
    """
    The test folds of one horizon's rows (dataset.build_dataset's columns), oldest first, their masks aligned with
    'rows'. A market's sessions are its calendar in 'calendars' when given, else every business day (holidays
    included, which can only count more sessions than the market had; the window end check still holds then).
    """
    dates = np.sort(pd.to_datetime(rows['session_date']).unique())
    first_test = int(np.ceil(len(dates) * INITIAL_TRAIN_SHARE))
    if len(dates) - first_test < FOLDS or first_test == 0:
        return []
    session_dates = pd.to_datetime(rows['session_date']).to_numpy()
    window_end = _window_ends(rows)
    result = []
    for number, block in enumerate(np.array_split(dates[first_test:], FOLDS), start=1):
        first, last = block[0], block[-1]
        before = session_dates < first
        between = _sessions_between(rows, session_dates, first, calendars)
        train = before & (between >= GAP_SESSIONS) & ~(window_end >= first)
        test = (session_dates >= first) & (session_dates <= last)
        result.append(Fold(number, pd.Timestamp(first), pd.Timestamp(last), train, test))
    return result


def status_ranker(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """The strategy's statuses: a high-conviction signal (1) above a confirmed one (0)."""
    return (test['status'] == HIGH_CONVICTION).to_numpy(dtype=float)


def rule_ranker(rule: Rule) -> Ranker:
    """A rule: the signals it keeps (1) above those it drops (0)."""

    def rank(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        kept = np.ones(len(test), dtype=bool)
        for condition in rule:
            kept &= condition.keeps(test)
        return kept.astype(float)

    return rank


def baselines(rules: list[Rule]) -> dict[str, tuple[str, Ranker]]:
    """The baselines a model is compared with, by name and with their kind: the statuses, then each rule."""
    return {
        STATUS: (STATUS, status_ranker),
        **{f'{RULE}:{rule_key(rule)}': (RULE_RANKER, rule_ranker(rule)) for rule in rules},
    }


def validated_rules(results: list[AnalysisResult], horizon: int) -> list[Rule]:
    """The candidate rules (candidates.py) labelled validated at 'horizon' in a run's results, in their order."""
    return [
        tuple(Condition(c['indicator'], c['op'], c['threshold']) for c in r.payload['conditions'])
        for r in results
        if r.section == SECTION
        and r.horizon == horizon
        and r.finding.startswith(f'{RULE}:')
        and r.payload['label'] == VALIDATED
    ]


def session_lifts(test: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """
    The ranking 'scores' gives 'test' (one fold's rows at one horizon, aligned with them), judged per session: one
    row per date with complete outcomes on at least two signals, with the signals ranked, the mean net excess return
    of the TOP_SHARE ranked highest (top_mean), of them all (mean) and the difference (lift), oldest first.
    """
    y, score, sessions = _judged(test, scores)
    records = []
    for date, at in sessions:
        top_mean = _top_mean(y[at], score[at])
        mean = float(y[at].mean())
        records.append((pd.Timestamp(date), int(at.sum()), top_mean, mean, top_mean - mean))
    return pd.DataFrame(records, columns=['session_date', 'signals', 'top_mean', 'mean', 'lift'])


def lift_matrix(test: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    """
    The lift of each ranking in 'scores' (one column per ranking, one row per row of 'test') on each date
    session_lifts judges, oldest first: one row per date, one column per ranking.
    """
    y, score, sessions = _judged(test, scores)
    lifts = np.empty((len(sessions), score.shape[1]))
    for i, (_, at) in enumerate(sessions):
        lifts[i] = _top_means(y[at], score[at]) - y[at].mean()
    return lifts


def walk_forward(
    rows: pd.DataFrame,
    horizon: int,
    models: dict[str, Ranker],
    rules: list[Rule],
    thresholds: Thresholds = DEFAULTS,
    calendars: dict[str, pd.DatetimeIndex] | None = None,
    predictions: dict[str, pd.DataFrame] | None = None,
    versus: dict[str, tuple[str, ...]] | None = None,
) -> dict:
    """
    Every model in 'models', by name, against the statuses and 'rules' (validated_rules), walk-forward on the
    dataset's rows (dataset.build_dataset) at 'horizon': the folds, each ranker's lift per fold and over them all, and
    each model's comparison with each baseline, judged by 'thresholds'.

    When 'predictions' is given, it is filled with each model's out-of-sample scores, by name: one row per test row
    of a trained fold, with its fold, signal, date, ticker, market, outcome status, main outcome and score.

    'versus' names, for a model, the other models it is compared with as well, on every date both ranked; each
    model's comparison holds them under 'models', none for a model 'versus' does not name.
    """
    at = indicator_frame(rows[rows['horizon'] == horizon].reset_index(drop=True))
    rankers = {name: (MODEL, model) for name, model in models.items()} | baselines(rules)
    lifts: dict[str, list[pd.DataFrame]] = {name: [] for name in rankers}
    scored: dict[str, list[pd.DataFrame]] = {name: [] for name in models}
    fold_payloads = []
    for fold in folds(at, calendars):
        train, test = at[fold.train].reset_index(drop=True), at[fold.test].reset_index(drop=True)
        trained = bool((train['outcome_status'] == COMPLETE).any())
        fold_payloads.append(_fold_payload(fold, train, test, trained))
        if not trained:
            continue
        for name, (_, ranker) in rankers.items():
            scores = np.asarray(ranker(train, test), dtype=float)
            lifts[name].append(session_lifts(test, scores).assign(fold=fold.number))
            if name in scored:
                scored[name].append(_scored(test, scores, fold.number))
    by_ranker = {name: pd.concat(parts, ignore_index=True) if parts else _no_lifts() for name, parts in lifts.items()}
    if predictions is not None:
        for name, parts in scored.items():
            predictions[name] = pd.concat(parts, ignore_index=True) if parts else _scored(at.iloc[:0], np.array([]), 0)
    development = _development_dates(at)
    return {
        'horizon': horizon,
        'outcome': MAIN_OUTCOME,
        'top_share': TOP_SHARE,
        'folds': fold_payloads,
        'rankers': {name: {'kind': kind, **_ranker_payload(by_ranker[name])} for name, (kind, _) in rankers.items()},
        'comparisons': {
            name: {
                **_comparisons(by_ranker[name], by_ranker, baselines(rules), development, horizon, thresholds),
                'models': {
                    other: _compare(by_ranker[name], by_ranker[other], EVERY_DATE, horizon, thresholds)
                    for other in (versus or {}).get(name, ())
                },
            }
            for name in models
        },
        'initial_train_share': INITIAL_TRAIN_SHARE,
        'fold_count': FOLDS,
        'gap_sessions': GAP_SESSIONS,
        'min_signal_dates': thresholds.min_signal_dates,
        'confidence': thresholds.confidence,
    }


def _judged(test: pd.DataFrame, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tuple]]:
    """
    The complete outcomes of 'test' and their 'scores' (a missing score lowest), with each date judged, oldest first,
    and the mask of its rows: the dates with complete outcomes on at least two signals.
    """
    y = pd.to_numeric(test[OUTCOMES[MAIN_OUTCOME].column], errors='coerce').to_numpy(dtype=float)
    observed = (test['outcome_status'] == COMPLETE).to_numpy() & ~np.isnan(y)
    score = np.where(np.isnan(scores), -np.inf, scores)
    dates = pd.to_datetime(test['session_date']).to_numpy()[observed]
    sessions = [(date, dates == date) for date in np.unique(dates)]
    return y[observed], score[observed], [(date, at) for date, at in sessions if at.sum() >= 2]


def _top_mean(y: np.ndarray, score: np.ndarray) -> float:
    """The mean of 'y' over the TOP_SHARE of its rows 'score' ranks highest, tied rows sharing the places they span."""
    return float(_top_means(y, score[:, None])[0])


def _top_means(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """_top_mean of each ranking in 'scores', one column per ranking."""
    k = TOP_SHARE * len(y)
    cut = -np.sort(-scores, axis=0)[int(np.ceil(k)) - 1]
    above, tied = scores > cut, scores == cut
    weights = above + tied * (k - above.sum(axis=0)) / tied.sum(axis=0)
    return (weights * y[:, None]).sum(axis=0) / k


def _window_ends(rows: pd.DataFrame) -> np.ndarray:
    """Each row's known outcome window end: its exit when complete, its last session when ended, else NaT."""
    if 'exit_date' not in rows or 'last_session_date' not in rows:
        return np.full(len(rows), np.datetime64('NaT'), dtype='datetime64[ns]')
    ends = pd.to_datetime(rows['exit_date']).where(rows['outcome_status'] == COMPLETE)
    ends = ends.fillna(pd.to_datetime(rows['last_session_date']).where(rows['outcome_status'] == ENDED))
    return ends.to_numpy(dtype='datetime64[ns]')


def _sessions_between(
    rows: pd.DataFrame, session_dates: np.ndarray, boundary: np.datetime64, calendars: dict | None
) -> np.ndarray:
    """The sessions of each row's market strictly between its date and 'boundary' (0 for a row on or after it)."""
    if calendars is None:
        days = session_dates.astype('datetime64[D]')
        return np.maximum(np.busday_count(days, np.datetime64(boundary, 'D')) - 1, 0)
    between = np.zeros(len(rows), dtype=int)
    for market, index in rows.groupby('market').indices.items():
        calendar = calendars.get(market, pd.DatetimeIndex([]))
        counted = calendar.searchsorted(boundary, side='left') - calendar.searchsorted(
            session_dates[index], side='right'
        )
        between[index] = np.maximum(counted, 0)
    return between


def _fold_payload(fold: Fold, train: pd.DataFrame, test: pd.DataFrame, trained: bool) -> dict:
    periods = test.drop_duplicates('signal_id')['period'] if 'period' in test else pd.Series(dtype=object)
    return {
        'fold': fold.number,
        'first': fold.first.date().isoformat(),
        'last': fold.last.date().isoformat(),
        'trained': trained,
        'train_signals': int(train['signal_id'].nunique()),
        'train_signal_dates': int(train['session_date'].nunique()),
        'test_signals': int(test['signal_id'].nunique()),
        'test_signal_dates': int(test['session_date'].nunique()),
        'censored': int((test['outcome_status'] == ENDED).sum()),
        'periods': {p: int((periods == p).sum()) for p in PERIODS if (periods == p).any()},
    }


def _ranker_payload(lifts: pd.DataFrame) -> dict:
    """A ranker's lift and top mean over every fold and per fold, with the dates each is over."""

    def summary(frame: pd.DataFrame) -> dict:
        shown = len(frame) > 0
        return {
            'signal_dates': len(frame),
            'lift': float(frame['lift'].mean()) if shown else None,
            'top_mean': float(frame['top_mean'].mean()) if shown else None,
            'mean': float(frame['mean'].mean()) if shown else None,
        }

    return {
        **summary(lifts),
        'folds': [{'fold': int(number), **summary(frame)} for number, frame in lifts.groupby('fold', sort=True)],
    }


def _comparisons(
    model: pd.DataFrame,
    by_ranker: dict[str, pd.DataFrame],
    against: dict[str, tuple[str, Ranker]],
    development: np.ndarray,
    horizon: int,
    thresholds: Thresholds,
) -> dict:
    """
    A model's comparison with each baseline, a rule's on the dates with no development signal ('development' lists
    those that have one), and whether it beats them all.
    """
    compared = {}
    for name, (kind, _) in against.items():
        if kind == RULE_RANKER:
            window = AFTER_DEVELOPMENT
            outside = ~model['session_date'].isin(development).to_numpy(dtype=bool)
            compared[name] = _compare(model[outside], by_ranker[name], window, horizon, thresholds)
        else:
            compared[name] = _compare(model, by_ranker[name], EVERY_DATE, horizon, thresholds)
    return {'baselines': compared, 'beats_rules': all(c['beats'] for c in compared.values())}


def _compare(model: pd.DataFrame, baseline: pd.DataFrame, window: str, horizon: int, thresholds: Thresholds) -> dict:
    """
    The model's lift less the baseline's, over the dates both ranked (the model's lifts already cut to 'window'), with
    its block-bootstrap interval.
    """
    both = model.merge(baseline, on=['fold', 'session_date'], suffixes=('_model', '_baseline'))
    difference = (both['lift_model'] - both['lift_baseline']).to_numpy(dtype=float)
    result = {
        'window': window,
        'signal_dates': len(both),
        'shown': False,
        'estimate': None,
        'ci_low': None,
        'ci_high': None,
    }
    if len(both) < thresholds.min_signal_dates:
        return {**result, 'beats': False}
    blocks = date_blocks(both['session_date'], horizon)
    low, high = block_bootstrap(blocks, [block_mean(difference, blocks)], thresholds)
    shown_interval = not np.isnan(low[0])
    return {
        **result,
        'shown': True,
        'estimate': float(difference.mean()),
        'ci_low': float(low[0]) if shown_interval else None,
        'ci_high': float(high[0]) if shown_interval else None,
        'beats': bool(shown_interval and low[0] > 0),
    }


def _development_dates(rows: pd.DataFrame) -> np.ndarray:
    """The dates with a development signal, none when the rows have no period."""
    if 'period' not in rows:
        return np.array([], dtype='datetime64[ns]')
    return pd.to_datetime(rows.loc[rows['period'] == DEVELOPMENT, 'session_date']).unique()


def _scored(test: pd.DataFrame, scores: np.ndarray, fold: int) -> pd.DataFrame:
    """A model's scores of one fold's test rows, with what judging them needs."""
    columns = ['signal_id', 'session_date', 'ticker', 'market', 'outcome_status', OUTCOMES[MAIN_OUTCOME].column]
    return test[columns].reset_index(drop=True).assign(fold=fold, score=scores)


def _no_lifts() -> pd.DataFrame:
    return pd.DataFrame(columns=['session_date', 'signals', 'top_mean', 'mean', 'lift', 'fold'])
