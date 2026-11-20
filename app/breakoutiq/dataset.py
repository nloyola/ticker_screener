"""
The analysis dataset: one row per event and resolved horizon, holding the outcome, every raw value and flag the
signal was archived with, the derived strengths of the mandatory conditions, the market regime on the signal's
session and the period the signal falls in.

An event is one breakout of one issuer, and its signal (an archived record of kind signal) is the only record of it
the dataset reads, so every section, counting distinct signals, counts each event once (the parent's GLOSSARY.md,
"Event"). The archive keeps each event once by construction: a breakout run has one breakout session, its Extended
sessions are continuations, which are left out, and a listing that is not its issuer's primary is never archived.

The near-misses the strength test compares with the signals have a dataset of their own (near_miss_dataset): one row
per near-miss event and resolved horizon, an event being a run of one ticker's near-misses on consecutive sessions of
its market, which its first session stands for.

build_dataset, near_miss_dataset and their helpers are pure functions of data frames; load_dataset and
load_near_misses read them from the database.

A resolved horizon is complete, with its returns, or ended: censored, with no returns, because the ticker's prices
stopped before the horizon's exit. Ended rows stay in the dataset (censored is True), never dropped, so every section
can report how many there were. Pending horizons are left out.
"""

import logging
import os

import numpy as np
import pandas as pd

from app.analyzer import SCREENER_VERSION
from app.container import Container
from app.markets import BENCHMARKS
from app.repositories.history_repo import BENCHMARK
from app.signal_paths import PATH_SESSIONS

# the chronological periods: candidate rules are found on development, checked on validation, confirmed on test
DEVELOPMENT = 'development'
VALIDATION = 'validation'
TEST = 'test'
PERIODS = (DEVELOPMENT, VALIDATION, TEST)
# a signal too close to the next period's first session for its outcome window to end before it: in no period
GAP = 'gap'
# the shares of the signals, oldest first, in development and validation; test has the rest
DEVELOPMENT_SHARE = 0.6
VALIDATION_SHARE = 0.2
# the sessions between a period's last signal and the next period's first: the longest horizon, so no outcome
# window crosses a boundary
GAP_SESSIONS = PATH_SESSIONS

# the market regime on a signal's session: the market's benchmark closed above or below its REGIME_SMA_DAYS-day SMA
REGIME_SMA_DAYS = 200
ABOVE = 'above'
BELOW = 'below'

COMPLETE = 'complete'
ENDED = 'ended'

# the mandatory condition a near-miss failed (failed_condition), in the order the screener counts them, then the
# minimum margin, which a near-miss meeting all four missed
BREAKOUT = 'breakout'
VOLUME = 'volume'
ABOVE_SMA50 = 'above_sma50'
SMA50_RISING = 'sma50_rising'
MARGIN = 'margin'
CONDITIONS = (BREAKOUT, VOLUME, ABOVE_SMA50, SMA50_RISING, MARGIN)

logger = logging.getLogger(__name__)


class MissingHistory(Exception):
    """The price history database the regime of a backfilled signal needs is not there."""


def load_dataset(container: Container, with_history: bool, screener_version: str = SCREENER_VERSION) -> pd.DataFrame:
    """
    The dataset from the database, of the signals 'screener_version' archived. Each market's benchmark closes come
    from benchmark_price and, when 'with_history' is True, from the price history database (HISTORY_DATABASE_FILE)
    too, which holds the years the regime of a backfilled signal needs; MissingHistory when that file is not there,
    rather than an empty history that would leave those regimes unknown.
    """
    archive = container.analysis_data_repo.archive(screener_version)
    benchmarks = load_benchmarks(container, with_history)
    return build_dataset(archive.signals, archive.outcomes, benchmarks, archive.path_sessions)


def load_near_misses(
    container: Container, with_history: bool, screener_version: str = SCREENER_VERSION
) -> pd.DataFrame:
    """
    The near-miss dataset (near_miss_dataset) from the database, of the near-misses 'screener_version' archived, each
    market's calendar its benchmark's closes, read as load_dataset reads them (MissingHistory likewise).
    """
    near_misses = container.analysis_data_repo.near_misses(screener_version)
    benchmarks = load_benchmarks(container, with_history)
    return near_miss_dataset(near_misses.records, near_misses.outcomes, benchmarks)


def load_benchmarks(container: Container, with_history: bool) -> dict[str, pd.Series]:
    """
    Each market's benchmark closes, indexed by date: benchmark_price's and, when 'with_history' is True, the price
    history database's before them (join_closes); MissingHistory when that file is not there.
    """
    if with_history and not os.path.isfile(container.history_db_path):
        raise MissingHistory(f'no price history database at {container.history_db_path}')
    benchmarks = {}
    for market, ticker in BENCHMARKS.items():
        live = _series(container.benchmark_price_repo.closes(ticker))
        history = (
            _series({p.date: p.close for p in container.history_repo.prices(BENCHMARK, ticker)})
            if with_history
            else _series({})
        )
        benchmarks[market] = join_closes(history, live)
    return benchmarks


def build_dataset(
    signals: pd.DataFrame,
    outcomes: pd.DataFrame,
    benchmarks: dict[str, pd.Series],
    path_sessions: pd.DataFrame,
) -> pd.DataFrame:
    """
    One row per signal and resolved horizon, oldest session first.

    'signals' and 'outcomes' are shaped as AnalysisDataRepo gives them, 'benchmarks' maps each market to its
    benchmark's closes indexed by date, and 'path_sessions' lists the sessions (market, session_date) of the signals'
    price paths. Each market's session calendar is its benchmark's dates, its path sessions and its signal sessions
    together.

    Adds to the signal's columns and the outcome's:
    - breakout_level and breakout_margin_pct: the level the signal cleared (the 60-day high when it broke it, else
      the 20-day high, as app/signal_outcomes.py's breakout_level) and the close's margin over it, the signal's own
      unrounded breakout_margin_pct where it has one, worked out from its rounded highs only for a signal archived
      before the screener stored it
    - sma50_distance_pct: the close's distance above the 50-day SMA
    - sma50_slope_pct: the 50-day SMA's change over the SMA_RISING_DAYS sessions the rising rule looks back
    - volume_ratio, already a column of the signal: the day's volume over its average
    - regime: ABOVE or BELOW, from the market's benchmark on the signal's session (or the latest before it), None
      without REGIME_SMA_DAYS closes by then
    - period: DEVELOPMENT, VALIDATION, TEST or GAP (assign_periods)
    - gap_date: the GAP_SESSIONS-th session of its market after the signal's (gap_dates)
    - censored: True for an ended horizon
    """
    signals = signals.copy()
    signals['session_date'] = pd.to_datetime(signals['session_date'])
    outcomes = outcomes.copy()
    for column in ('entry_date', 'exit_date', 'last_session_date'):
        outcomes[column] = pd.to_datetime(outcomes[column])

    # only the signals with a resolved horizon are in the dataset, and the periods split those
    signals = signals[signals['signal_id'].isin(outcomes['signal_id'])].reset_index(drop=True)

    level = np.where(signals['breakout_60d'] == 1, signals['high_60d'], signals['high_20d']).astype(float)
    signals['breakout_level'] = level
    signals['breakout_margin_pct'] = (
        signals['breakout_margin_pct'].astype(float).fillna(pct_change(signals['close'], level))
    )
    signals['sma50_distance_pct'] = pct_change(signals['close'], signals['dma_50'])
    signals['sma50_slope_pct'] = pct_change(signals['dma_50'], signals['dma_50_prev_15d'])

    signals['regime'] = regimes(signals, benchmarks)
    calendars = _calendars(signals, benchmarks, path_sessions)
    signals['window_end'] = signals['signal_id'].map(window_ends(outcomes))
    signals['period'] = assign_periods(signals, calendars)
    signals['gap_date'] = gap_dates(signals, calendars)
    signals = signals.drop(columns='window_end')

    rows = signals.merge(outcomes, on='signal_id', how='inner', validate='one_to_many')
    rows['censored'] = rows['outcome_status'] == ENDED
    return rows.sort_values(['session_date', 'signal_id', 'horizon'], kind='stable').reset_index(drop=True)


def near_miss_dataset(records: pd.DataFrame, outcomes: pd.DataFrame, benchmarks: dict[str, pd.Series]) -> pd.DataFrame:
    """
    One row per near-miss event and resolved horizon, oldest session first: the event's first near-miss, with
    'condition', the mandatory condition it failed (failed_condition), and its outcome's columns. 'records' and
    'outcomes' are shaped as AnalysisDataRepo.near_misses gives them, and 'benchmarks' as build_dataset takes them.

    A ticker failing a cut-off one day often fails it the next too, so its near-misses on consecutive sessions of its
    market are one event, as a breakout run is one signal; a session the ticker was not a near-miss ends the event.
    Each market's session calendar is its benchmark's dates and its near-miss sessions together. Like the signals, a
    near-miss stands for the event only with a resolved horizon; one with none is left out and ends no event.
    """
    records = records.copy()
    records['session_date'] = pd.to_datetime(records['session_date'])
    outcomes = outcomes.copy()
    for column in ('entry_date', 'exit_date', 'last_session_date'):
        outcomes[column] = pd.to_datetime(outcomes[column])

    records = records[event_starts(records, benchmarks)]
    records = records[records['signal_id'].isin(outcomes['signal_id'])].reset_index(drop=True)
    records['condition'] = [failed_condition(r) for r in records.itertuples(index=False)]
    rows = records.merge(outcomes, on='signal_id', how='inner', validate='one_to_many')
    rows['censored'] = rows['outcome_status'] == ENDED
    return rows.sort_values(['session_date', 'signal_id', 'horizon'], kind='stable').reset_index(drop=True)


def event_starts(records: pd.DataFrame, benchmarks: dict[str, pd.Series]) -> pd.Series:
    """
    Whether each record (columns ticker, market, session_date as Timestamps) begins an event, aligned with 'records':
    True unless the same ticker has a record on the session before its own in its market's calendar (its benchmark's
    dates and the records' sessions).
    """
    starts = pd.Series(True, index=records.index)
    for market, group in records.groupby('market'):
        calendar = pd.DatetimeIndex(benchmarks.get(market, _series({})).index)
        calendar = calendar.union(pd.DatetimeIndex(group['session_date'].unique())).unique().sort_values()
        position = pd.Series(calendar.searchsorted(group['session_date'].to_numpy()), index=group.index)
        for _, sessions in position.groupby(group['ticker']):
            ordered = sessions.sort_values()
            starts.loc[ordered.index] = ordered.diff().ne(1).to_numpy()
    return starts


def failed_condition(record) -> str:
    """
    The mandatory condition a near-miss failed (CONDITIONS), read from its flags: MARGIN when it met all four
    (core_criteria_met), or passes all four flags, else the first of the four it does not pass, an empty flag counting
    as not passed.
    """
    if record.core_criteria_met == 1:
        return MARGIN
    passes = {
        BREAKOUT: record.breakout_20d == 1 or record.breakout_60d == 1,
        VOLUME: record.breakout_confirmed == 1,
        ABOVE_SMA50: record.price_gt_50dma == 1,
        SMA50_RISING: record.dma_50_rising == 1,
    }
    return next((name for name, passed in passes.items() if not passed), MARGIN)


def assign_periods(signals: pd.DataFrame, calendars: dict[str, pd.DatetimeIndex]) -> pd.Series:
    """
    Each signal's period, aligned with 'signals' (columns session_date, market, source, and optionally window_end).

    The signals are split by session date, every signal of a date in the same period: a date is in development
    while the signals before it are fewer than DEVELOPMENT_SHARE of them all, then in validation while fewer than
    DEVELOPMENT_SHARE + VALIDATION_SHARE, then in test. Live signals are always out of sample: the date of the first
    live signal and every date after it are in test.

    A signal is then moved to GAP when fewer than GAP_SESSIONS sessions of its market's calendar lie strictly between
    its date and the next period's first date, since its outcome window, which ends on the GAP_SESSIONS-th session
    after its date, would reach into the next period. A ticker that misses sessions reaches its GAP_SESSIONS-th
    later, so a signal whose known window end (window_end, window_ends) is on or after the next period's first date
    is moved to GAP too. The last period has no gap.
    """
    if signals.empty:
        return pd.Series([], index=signals.index, dtype=object)
    if 'window_end' not in signals:
        signals = signals.assign(window_end=pd.NaT)
    counts = signals.groupby('session_date').size().sort_index()
    before = (counts.cumsum() - counts) / counts.sum()
    by_date = pd.Series(
        np.select(
            [before < DEVELOPMENT_SHARE, before < DEVELOPMENT_SHARE + VALIDATION_SHARE],
            [DEVELOPMENT, VALIDATION],
            TEST,
        ),
        index=counts.index,
    )
    live = signals.loc[signals['source'] == 'live', 'session_date']
    if not live.empty:
        by_date[by_date.index >= live.min()] = TEST

    periods = signals['session_date'].map(by_date)
    present = [p for p in PERIODS if (by_date == p).any()]
    for period, following in zip(present, present[1:], strict=False):
        boundary = by_date.index[by_date == following].min()
        for market, group in signals[periods == period].groupby('market'):
            calendar = calendars.get(market, pd.DatetimeIndex([]))
            between = calendar.searchsorted(boundary, side='left') - calendar.searchsorted(
                group['session_date'].to_numpy(), side='right'
            )
            crosses = (between < GAP_SESSIONS) | (group['window_end'] >= boundary).to_numpy()
            periods.loc[group.index[crosses]] = GAP
    return periods


def gap_dates(signals: pd.DataFrame, calendars: dict[str, pd.DatetimeIndex]) -> pd.Series:
    """
    Each signal's GAP_SESSIONS-th session of its market's calendar after its date, aligned with 'signals' (columns
    session_date and market): a later date has at least GAP_SESSIONS sessions strictly between it and the signal's
    exactly when it is after this one. NaT when the calendar ends first.
    """
    result = pd.Series(pd.NaT, index=signals.index, dtype='datetime64[ns]')
    for market, group in signals.groupby('market'):
        calendar = pd.DatetimeIndex(calendars.get(market, pd.DatetimeIndex([])))
        position = calendar.searchsorted(group['session_date'].to_numpy(), side='right') - 1 + GAP_SESSIONS
        inside = position < len(calendar)
        result.loc[group.index[inside]] = calendar[position[inside]]
    return result


def window_ends(outcomes: pd.DataFrame) -> pd.Series:
    """
    Each signal's latest known outcome window end, indexed by signal_id: the latest exit date of its complete
    horizons, or last session of its ended ones.
    """
    ends = outcomes['exit_date'].where(outcomes['outcome_status'] == COMPLETE, outcomes['last_session_date'])
    return pd.to_datetime(ends).groupby(outcomes['signal_id']).max()


def regimes(signals: pd.DataFrame, benchmarks: dict[str, pd.Series]) -> pd.Series:
    """
    The market regime on each signal's session, aligned with 'signals' (columns session_date and market): ABOVE when
    its market's benchmark closed above its REGIME_SMA_DAYS-day SMA on the latest benchmark date on or before the
    session, BELOW when it closed at or below it, None when the benchmark has fewer than REGIME_SMA_DAYS closes by
    then.
    """
    result = pd.Series([None] * len(signals), index=signals.index, dtype=object)
    for market, group in signals.groupby('market'):
        closes = benchmarks.get(market)
        if closes is None or closes.empty:
            continue
        closes = closes.sort_index()
        sma = closes.rolling(REGIME_SMA_DAYS).mean()
        state = np.where(sma.isna(), None, np.where(closes > sma, ABOVE, BELOW))
        at = closes.index.searchsorted(group['session_date'].to_numpy(), side='right') - 1
        result[group.index] = [state[i] if i >= 0 else None for i in at]
    return result


def join_closes(history: pd.Series, live: pd.Series) -> pd.Series:
    """
    One benchmark's closes from the price history and from benchmark_price: the history's, then the live closes
    dated after it. The two are adjusted on different bases, so the live closes are rescaled to the history's at the
    latest date both have; with no date in common they are joined as they are.
    """
    if history.empty:
        return live.sort_index()
    if live.empty:
        return history.sort_index()
    history, live = history.sort_index(), live.sort_index()
    common = history.index.intersection(live.index)
    if not len(common):
        logger.warning('benchmark history and live closes share no date: joined unscaled, the regime may be wrong')
    scale = history[common[-1]] / live[common[-1]] if len(common) else 1.0
    later = live[live.index > history.index[-1]] * scale
    return pd.concat([history, later])


def _calendars(
    signals: pd.DataFrame, benchmarks: dict[str, pd.Series], path_sessions: pd.DataFrame
) -> dict[str, pd.DatetimeIndex]:
    """Each market's sessions: its benchmark's dates, its signals' path sessions and its signals' sessions."""
    markets = set(benchmarks) | set(signals['market'])
    calendars = {}
    for market in markets:
        dates = [
            pd.DatetimeIndex(benchmarks.get(market, _series({})).index),
            pd.DatetimeIndex(pd.to_datetime(path_sessions.loc[path_sessions['market'] == market, 'session_date'])),
            pd.DatetimeIndex(signals.loc[signals['market'] == market, 'session_date']),
        ]
        calendars[market] = dates[0].union(dates[1]).union(dates[2]).sort_values()
    return calendars


def pct_change(value: pd.Series, base) -> pd.Series:
    """value over base less one, in percent; NaN where the base is missing or not positive."""
    base = pd.Series(base, index=value.index, dtype=float)
    return (value.astype(float) / base.where(base > 0) - 1) * 100


def _series(closes: dict) -> pd.Series:
    """Closes keyed by date as a series indexed by Timestamp."""
    return pd.Series(list(closes.values()), index=pd.DatetimeIndex(pd.to_datetime(list(closes.keys()))), dtype=float)
