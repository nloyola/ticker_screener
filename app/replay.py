"""
Scoring every session of one ticker's long price history at once, for the history backfill (app/history_backfill.py).

Analyzer.score_as_of scores a session from the rows dated on or before it, so replaying years of sessions with it
recomputes every indicator over a longer prefix for each session. Every indicator the analyzer reads is causal - its
value on a row depends only on that row and the rows before it - so computing each one once over the ticker's full
series and reading it on the session's row gives the value score_as_of computes from the prefix ending there. The
one exception is the 21-week EMA, whose last weekly close is the session's own close rather than its week's: it is
one EMA step from the previous week's value, taken with pandas' own EMA so the arithmetic is the same.

ticker_replay() returns the rows a run would score and, per row, the core criteria and returns that decide which rows
need their full criteria (TickerReplay.criteria_at), the subsector comparison and the status being left to the caller,
which knows every ticker scored on each session. tests/test_history_backfill.py checks the values against score_as_of.

The history database's rows are never missing a value (its columns are NOT NULL), so the analyzer's handling of
missing highs, lows and volumes never applies here.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD

from app.analyzer import (
    BREAKOUT_WINDOWS,
    CLOSE_POSITION_MIN,
    CONSOLIDATION_DAYS,
    CONSOLIDATION_TIGHT_PCT,
    EMA_MIN_PERIODS,
    EMA_RISING_DAYS,
    EMA_SPAN,
    FROZEN_DAYS,
    LONG_SMA_DAYS,
    MACD_CROSSOVER_DAYS,
    MACD_FAST_DAYS,
    MACD_SIGNAL_DAYS,
    MACD_SLOW_DAYS,
    MIN_ROWS,
    PADDED_MIN_DAYS,
    PADDED_WINDOW,
    RS_1M_DAYS,
    RS_3M_DAYS,
    RSI_BAND,
    RSI_DAYS,
    SHORT_SMA_DAYS,
    SMA_DAYS,
    SMA_RISING_DAYS,
    VOLUME_AVG_DAYS,
    VOLUME_RATIO_MIN,
    _above,
    _below,
    _benchmark_return_pct,
    _close_position,
    _difference,
    _distance_pct,
    _near,
    _positive,
    _round,
    ema,
)
from app.models import StockCriteria
from app.price_apis import DailyPrice


def run_lengths(flags: np.ndarray) -> np.ndarray:
    """At each row, how many rows in a row, ending on it, are True: 0 where it is False."""
    rows = np.arange(len(flags))
    last_false = np.maximum.accumulate(np.where(flags, -1, rows))
    return rows - last_false


def flat_days(closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """
    At each row, the trailing days ending on it whose close equals its close and whose volume is zero, as
    _flat_days in app/analyzer.py counts them over the rows up to it.
    """
    still = volumes == 0
    # a still day continues the stretch of the still day before it when both closed at the same price
    joined = still & np.concatenate(([False], still[:-1] & (closes[:-1] == closes[1:])))
    return np.where(still, 1 + run_lengths(joined), 0)


def unscored(closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """
    At each row, whether the rows up to it end in a padded stop (padded_stop) or a frozen stretch (frozen_since),
    which a run on that session would skip as stale or frozen.
    """
    flat = flat_days(closes, volumes)
    rows = np.arange(len(closes))
    last_trade = rows - flat
    traded = run_lengths(volumes > 0)
    at = np.clip(last_trade, 0, None)
    padded = (
        (flat >= PADDED_MIN_DAYS)
        & (last_trade + 1 >= PADDED_WINDOW)
        & (closes[at] == closes)
        & (traded[at] >= PADDED_WINDOW)
    )
    return padded | (flat >= FROZEN_DAYS)


def weekly_ema(dates: pd.Index, closes: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """
    At each row, the 21-week EMA of the weekly closes of the rows up to it (ema(weekly_closes(...)) in
    app/analyzer.py, whose last weekly close is the row's own), and that EMA's previous week's value, NaN where the
    analyzer's is empty. A week's value is one EMA step from the previous week's, taken by pandas over that pair.
    """
    periods = pd.DatetimeIndex(pd.to_datetime(dates)).to_period('W')
    # each row's week, counted from the ticker's first
    week = np.concatenate(([0], np.cumsum(periods[1:] != periods[:-1]))) if len(periods) else np.array([], int)
    week_closes = pd.Series(closes.to_numpy()).groupby(week).last()
    ema_weeks = week_closes.ewm(span=EMA_SPAN, adjust=False).mean().to_numpy()
    close_values = closes.to_numpy()

    current = close_values.astype(float).copy()
    later = week > 0
    if later.any():
        pairs = pd.DataFrame([ema_weeks[week[later] - 1], close_values[later]])
        current[later] = pairs.ewm(span=EMA_SPAN, adjust=False).mean().iloc[1].to_numpy()
    current[week < EMA_MIN_PERIODS - 1] = np.nan

    previous = np.full(len(week), np.nan)
    known = week >= EMA_MIN_PERIODS
    previous[known] = ema_weeks[week[known] - 1]
    return current, previous


def _value(values: np.ndarray, row: int) -> float | None:
    """The value at 'row', None before the first row or where it is missing: _at in app/analyzer.py."""
    if row < 0:
        return None
    value = values[row]
    return None if np.isnan(value) else float(value)


@dataclass(frozen=True)
class TickerReplay:
    """
    One ticker's history scored on every row. 'scorable' marks the rows a run on that session would score: those
    with MIN_ROWS rows up to them, not ending in a padded stop or a frozen stretch. 'core' marks those meeting the
    four core conditions, and 'return_1m' and 'return_3m' are each row's returns, as criteria_at gives them, NaN for
    none.
    """

    ticker: str
    subsector_id: int
    dates: pd.Index
    closes: np.ndarray
    scorable: np.ndarray
    core: np.ndarray
    return_1m: np.ndarray
    return_3m: np.ndarray
    series: Mapping[str, np.ndarray]

    def criteria_at(self, row: int, benchmark: Mapping[date, float]) -> StockCriteria:
        """
        The criteria Analyzer.analyze_ticker gives for the rows up to 'row', against the benchmark's closes by date:
        the subsector comparison and the status are left empty.
        """
        s = self.series
        price = float(self.closes[row])
        volume = int(s['volume'][row])
        high_20d, high_60d = (float(s[f'high_{days}d'][row]) for days in BREAKOUT_WINDOWS)
        avg_vol = float(s['avg_vol'][row])
        volume_ratio = volume / avg_vol if avg_vol > 0 else None
        dma_50 = float(s['sma'][row])
        dma_50_prior = float(s['sma'][row - SMA_RISING_DAYS])

        rsi_val = float(s['rsi'][row])
        macd_val = float(s['macd'][row])
        signal_val = float(s['macd_signal'][row])
        hist_val = _value(s['macd_hist'], row)
        hist_prior = _value(s['macd_hist'], row - 1)
        days_above = int(s['macd_days_above'][row])
        hist_rising = None if hist_val is None or hist_prior is None else int(0 < hist_val and hist_prior < hist_val)

        dma_20, dma_20_prior = _value(s['sma_short'], row), _value(s['sma_short'], row - SMA_RISING_DAYS)
        dma_200, dma_200_prior = _value(s['sma_long'], row), _value(s['sma_long'], row - SMA_RISING_DAYS)
        ema_21, ema_21_prior = _value(s['ema'], row), _value(s['ema'], row - EMA_RISING_DAYS)
        ema_21w, ema_21w_prior = _value(s['ema_weekly'], row), _value(s['ema_weekly_prior'], row)

        close_position = _close_position(price, float(s['high'][row]), float(s['low'][row]))
        top = float(s['consolidation_high'][row])
        consolidation_range_pct = (top - float(s['consolidation_low'][row])) / top * 100 if top > 0 else None
        earlier = float(s['consolidation_earlier_vol'][row])
        consolidation_vol_ratio = float(s['consolidation_later_vol'][row]) / earlier if earlier > 0 else None
        dist_20d, dist_60d = _distance_pct(price, high_20d), _distance_pct(price, high_60d)

        dates = self.dates[: row + 1]
        return_1m, return_3m = _value(self.return_1m, row), _value(self.return_3m, row)
        rs_market_1m = _difference(return_1m, _benchmark_return_pct(dates, RS_1M_DAYS, benchmark))
        rs_market_3m = _difference(return_3m, _benchmark_return_pct(dates, RS_3M_DAYS, benchmark))

        breakout_20d = price > high_20d
        breakout_60d = price > high_60d
        breakout_confirmed = volume_ratio is not None and volume_ratio >= VOLUME_RATIO_MIN
        price_above_50dma = price > dma_50
        dma_slope_up = dma_50 > dma_50_prior
        core = [breakout_20d or breakout_60d, breakout_confirmed, price_above_50dma, dma_slope_up]

        return StockCriteria(
            id=0,
            subsector_id=self.subsector_id,
            ticker=self.ticker,
            price=round(price, 2),
            core_criteria_met=all(core),
            core_criteria_score=sum(core),
            high_20d=round(high_20d, 2),
            high_60d=round(high_60d, 2),
            volume=volume,
            avg_vol_20d=int(avg_vol),
            volume_ratio=volume_ratio,
            dma_50=round(dma_50, 2),
            dma_50_prev_15d=round(dma_50_prior, 2),
            rsi=round(rsi_val, 2),
            macd=round(macd_val, 2),
            signal=round(signal_val, 2),
            breakout_20d=int(breakout_20d),
            breakout_60d=int(breakout_60d),
            breakout_confirmed=int(breakout_confirmed),
            price_gt_50dma=int(price_above_50dma),
            dma_50_rising=int(dma_slope_up),
            rsi_50_70=int(RSI_BAND[0] <= rsi_val <= RSI_BAND[1]),
            macd_above_signal=int(macd_val > signal_val),
            macd_hist=_round(hist_val),
            macd_hist_prev=_round(hist_prior),
            macd_days_above=days_above,
            macd_crossover=int(1 <= days_above <= MACD_CROSSOVER_DAYS),
            macd_hist_rising=hist_rising,
            dma_20=_round(dma_20),
            dma_20_prev_15d=_round(dma_20_prior),
            dma_200=_round(dma_200),
            dma_200_prev_15d=_round(dma_200_prior),
            price_gt_20dma=_above(price, dma_20),
            dma_20_rising=_above(dma_20, dma_20_prior),
            price_gt_200dma=_above(price, dma_200),
            dma_200_rising=_above(dma_200, dma_200_prior),
            ema_21=_round(ema_21),
            ema_21_prev_5d=_round(ema_21_prior),
            price_gt_21ema=_above(price, ema_21),
            ema_21_rising=_above(ema_21, ema_21_prior),
            ema_21w=_round(ema_21w),
            ema_21w_prev_1w=_round(ema_21w_prior),
            price_gt_21wema=_above(price, ema_21w),
            ema_21w_rising=_above(ema_21w, ema_21w_prior),
            close_position=close_position,
            close_upper_quarter=None if close_position is None else int(close_position >= CLOSE_POSITION_MIN),
            consolidation_range_pct=consolidation_range_pct,
            consolidation_tight=_below(consolidation_range_pct, CONSOLIDATION_TIGHT_PCT),
            consolidation_vol_ratio=consolidation_vol_ratio,
            consolidation_vol_declining=_below(consolidation_vol_ratio, 1.0),
            dist_high_20d_pct=dist_20d,
            dist_high_60d_pct=dist_60d,
            near_resistance=_near([dist_20d, dist_60d]),
            return_1m_pct=return_1m,
            return_3m_pct=return_3m,
            rs_market_1m=rs_market_1m,
            rs_market_3m=rs_market_3m,
            outperforms_market_1m=_positive(rs_market_1m),
            outperforms_market_3m=_positive(rs_market_3m),
        )


def _returns(closes: pd.Series, days: int) -> np.ndarray:
    """Each row's return over the close 'days' rows before it, in percent: NaN when too short or not positive."""
    start = closes.shift(days)
    return ((closes / start - 1) * 100).where(start > 0).to_numpy()


def ticker_replay(ticker: str, subsector_id: int, prices: Sequence[DailyPrice]) -> TickerReplay:
    """Every indicator of the ticker's history ('prices', oldest first) computed once over the full series."""
    dates = pd.Index([p.date for p in prices])
    close = pd.Series([p.close for p in prices], dtype=float)
    high = pd.Series([p.high for p in prices], dtype=float)
    low = pd.Series([p.low for p in prices], dtype=float)
    volume = np.array([p.volume for p in prices], dtype=np.int64)
    # whole share counts: every sum of them is exact in floating point, so their means are those the analyzer takes
    volume_f = pd.Series(volume, dtype=float)

    # each lookback excludes the row's own day, as the analyzer's do
    before = high.shift(1)
    series: dict[str, np.ndarray] = {
        f'high_{days}d': before.rolling(days).max().to_numpy() for days in BREAKOUT_WINDOWS
    }
    series['avg_vol'] = volume_f.shift(1).rolling(VOLUME_AVG_DAYS).mean().to_numpy()
    sma = close.rolling(window=SMA_DAYS).mean()
    series['sma'] = sma.to_numpy()

    series['rsi'] = RSIIndicator(close=close, window=RSI_DAYS).rsi().to_numpy()
    macd = MACD(close=close, window_slow=MACD_SLOW_DAYS, window_fast=MACD_FAST_DAYS, window_sign=MACD_SIGNAL_DAYS)
    macd_line, signal_line = macd.macd(), macd.macd_signal()
    series['macd'] = macd_line.to_numpy()
    series['macd_signal'] = signal_line.to_numpy()
    series['macd_hist'] = macd.macd_diff().to_numpy()
    series['macd_days_above'] = run_lengths((macd_line > signal_line).to_numpy())

    series['sma_short'] = close.rolling(window=SHORT_SMA_DAYS).mean().to_numpy()
    series['sma_long'] = close.rolling(window=LONG_SMA_DAYS).mean().to_numpy()
    series['ema'] = ema(close).to_numpy()
    series['ema_weekly'], series['ema_weekly_prior'] = weekly_ema(dates, close)

    series['high'] = high.to_numpy()
    series['low'] = low.to_numpy()
    series['volume'] = volume
    series['consolidation_high'] = before.rolling(CONSOLIDATION_DAYS).max().to_numpy()
    series['consolidation_low'] = low.shift(1).rolling(CONSOLIDATION_DAYS).min().to_numpy()
    half = CONSOLIDATION_DAYS // 2
    series['consolidation_later_vol'] = volume_f.shift(1).rolling(half).mean().to_numpy()
    series['consolidation_earlier_vol'] = volume_f.shift(1 + half).rolling(half).mean().to_numpy()

    closes = close.to_numpy()
    with np.errstate(invalid='ignore', divide='ignore'):
        volume_ratio = np.where(series['avg_vol'] > 0, volume / series['avg_vol'], np.nan)
    sma_values = series['sma']
    sma_prior = sma.shift(SMA_RISING_DAYS).to_numpy()
    breakout = (closes > series[f'high_{BREAKOUT_WINDOWS[0]}d']) | (closes > series[f'high_{BREAKOUT_WINDOWS[1]}d'])
    core = breakout & (volume_ratio >= VOLUME_RATIO_MIN) & (closes > sma_values) & (sma_values > sma_prior)

    rows = np.arange(len(closes))
    scorable = (rows + 1 >= MIN_ROWS) & ~unscored(closes, volume)
    return TickerReplay(
        ticker=ticker,
        subsector_id=subsector_id,
        dates=dates,
        closes=closes,
        scorable=scorable,
        core=scorable & core,
        return_1m=_returns(close, RS_1M_DAYS),
        return_3m=_returns(close, RS_3M_DAYS),
        series=series,
    )
