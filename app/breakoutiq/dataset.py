"""
The analysis dataset: one row per archived signal and resolved horizon, holding the outcome, every raw value and flag
the signal was archived with, the derived strengths of the mandatory conditions, the market regime on the signal's
session and the period the signal falls in.

build_dataset and its helpers are pure functions of data frames; load_dataset reads them from the database.

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
    if with_history and not os.path.isfile(container.history_db_path):
        raise MissingHistory(f'no price history database at {container.history_db_path}')
    archive = container.analysis_data_repo.archive(screener_version)
    benchmarks = {}
    for market, ticker in BENCHMARKS.items():
        live = _series(container.benchmark_price_repo.closes(ticker))
        history = (
            _series({p.date: p.close for p in container.history_repo.prices(BENCHMARK, ticker)})
            if with_history
            else _series({})
        )
        benchmarks[market] = join_closes(history, live)
    return build_dataset(archive.signals, archive.outcomes, benchmarks, archive.path_sessions)


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
      the 20-day high, as app/signal_outcomes.py's breakout_level) and the close's margin over it
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
    signals['breakout_margin_pct'] = pct_change(signals['close'], level)
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
