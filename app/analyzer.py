import json
import logging
import math
import statistics
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD

from app.container import Container
from app.markets import (
    BENCHMARKS,
    CANADA,
    MARKET_CLOSES,
    MARKETS,
    PUBLISH_DELAY,
    US,
    UnsupportedTicker,
    exchange_of,
    last_complete_session,
    market_of,
)
from app.models import (
    CONTINUATION_KIND,
    KINDS,
    NEAR_MISS_KIND,
    SIGNAL_KIND,
    WATCHLIST_KIND,
    ScreenedSession,
    Signal,
    StockCriteria,
)
from app.progress import FROZEN, STALE, report_flagged, report_progress
from app.provenance import git_commit

if TYPE_CHECKING:
    from app.price_apis import PriceApi
    from app.repositories.benchmark_price_repo import BenchmarkPriceRepo
    from app.repositories.issuer_repo import IssuerRepo
    from app.repositories.price_data_repo import PriceDataRepo
    from app.repositories.screened_session_repo import ScreenedSessionRepo
    from app.repositories.signal_repo import SignalRepo
    from app.repositories.stock_criteria_repo import StockCriteriaRepo

logger = logging.getLogger(__name__)

# The version of the screener's rules, bumped (semver) whenever a threshold, a window, an indicator's definition or a
# status rule changes, so a stored score can always be traced to the rules that produced it (AGENTS.md, "Screener
# version"). rule_settings() is the snapshot of the settings that travels with it.
SCREENER_VERSION = '1.0.0'

# The four mandatory conditions of the breakout strategy (GLOSSARY.md, "Core criteria"), scored on the last row.
# The breakout compares the close with the highest high of the BREAKOUT_WINDOWS trading days before it.
BREAKOUT_WINDOWS = (20, 60)
# The 120-day tier (GLOSSARY.md, "120-day tier"): a breakout whose close is also above the highest high of the
# TIER_WINDOW trading days before it, a longer lookback that BreakoutIQ can test as a higher tier. It is stored
# beside the breakout windows' flags but is never a core condition, and a ticker with fewer than TIER_WINDOW rows
# before the scored day has no such high and is not in the tier.
TIER_WINDOW = 120
# Volume confirmation: the day's volume is at least VOLUME_RATIO_MIN times the average of the VOLUME_AVG_DAYS days
# before it. The strategy calls 2 times or more stronger; the ratio is stored so the app can show it.
VOLUME_AVG_DAYS = 20
VOLUME_RATIO_MIN = 1.5
# Trend confirmation: the close is above the SMA_DAYS-day simple moving average, which is rising when it is above
# its value SMA_RISING_DAYS trading days earlier (the strategy's window is 10-20 days).
SMA_DAYS = 50
SMA_RISING_DAYS = 15

# The secondary indicators (GLOSSARY.md, "Secondary indicators"): stored with their flags, never counted in the core
# criteria score. An indicator a ticker has too few rows for is left empty, and so are the flags that need it.
# Momentum: the RSI_DAYS-day RSI inside RSI_BAND, ends included.
RSI_DAYS = 14
RSI_BAND = (50.0, 70.0)
# MACD: the MACD_FAST_DAYS-day EMA less the MACD_SLOW_DAYS-day one, with a MACD_SIGNAL_DAYS-day EMA of it as its
# signal line. Crossover: MACD crossed above its signal line within the last MACD_CROSSOVER_DAYS sessions, the
# scored day included, and has stayed above it since.
MACD_FAST_DAYS = 12
MACD_SLOW_DAYS = 26
MACD_SIGNAL_DAYS = 9
MACD_CROSSOVER_DAYS = 5
# Moving-average alignment: the close above the 20-, 50- and 200-day simple moving averages, each rising over
# SMA_RISING_DAYS like the 50-day one (SMA_DAYS, a core condition).
SHORT_SMA_DAYS = 20
LONG_SMA_DAYS = 200
# The 21-period EMAs, of the daily closes and of the weekly closes (each week's last close, the scored day's week
# ending on the scored day). The daily one is rising when above its value EMA_RISING_DAYS trading days earlier, the
# weekly one when above last week's.
EMA_SPAN = 21
EMA_RISING_DAYS = 5
# An EMA starts from the first close, whose weight in it falls below 1% only after 50 values (alpha = 2 / 22): one
# with fewer is left empty.
EMA_MIN_PERIODS = 50
# Closing strength: the close's position in the day's high-low range, 0 at the low and 1 at the high, strong at
# CLOSE_POSITION_MIN or more (the upper 25% of the range).
CLOSE_POSITION_MIN = 0.75
# Consolidation quality, over the CONSOLIDATION_DAYS trading days before the scored day (the strategy's window is
# 10-30 days): the range from its highest high to its lowest low as a percentage of the highest high, tight below
# CONSOLIDATION_TIGHT_PCT, and volume declining when its later half averages less than its earlier half.
CONSOLIDATION_DAYS = 20
CONSOLIDATION_TIGHT_PCT = 10.0
# Breakout proximity: the close at most PROXIMITY_PCT percent below the 20- or 60-day resistance (the highest high of
# BREAKOUT_WINDOWS), and not above it.
PROXIMITY_PCT = 3.0
# Relative strength: the close's return over RS_1M_DAYS and RS_3M_DAYS trading days (about 1 and 3 months), each
# compared with the return of the market's benchmark (BENCHMARKS in app/markets.py) between the same two dates, and
# with the median return of the tickers scored in the same run in the ticker's subsector, itself included. The
# subsector median needs SECTOR_MIN_TICKERS returns; with fewer it is left empty. The ticker outperforms when its
# return is above the one it is compared with.
RS_1M_DAYS = 21
RS_3M_DAYS = 63
SECTOR_MIN_TICKERS = 3

# The strategy's statuses (GLOSSARY.md, "Status"), set once a ticker's subsector comparison is known
# (with_status). A confirmed breakout is a breakout session (breakout_runs, below), and is high-conviction when it
# also passes at least HIGH_CONVICTION_MIN of the SECONDARY_INDICATORS. An extended ticker is in the run of an
# earlier breakout, closing above the level it broke (the strategy's "post-breakout"). A ticker on the pre-breakout
# watchlist is neither, but passes both trend conditions, closes near its resistance (PROXIMITY_PCT) and has a tight
# base (CONSOLIDATION_TIGHT_PCT).
HIGH_CONVICTION = 'high_conviction'
CONFIRMED = 'confirmed'
PRE_BREAKOUT = 'pre_breakout'
EXTENDED = 'extended'
HIGH_CONVICTION_MIN = 5

# The liquidity floor (GLOSSARY.md, "Liquidity floor"): a ticker whose average daily dollar volume, its close times
# its volume averaged over the DOLLAR_VOLUME_DAYS sessions before the scored one, is below its market's
# MIN_DOLLAR_VOLUME, or whose close is below its market's MIN_PRICE, gets no status, since a thin stock's breakout
# cannot be traded at its signal price. It is still scored, so its values stay visible and the floor can be tested.
# A market's floors are in its own currency: a Canadian ticker's in Canadian dollars. The average excludes the
# scored session, as every lookback does, and on a session of a breakout run it is measured before the run's breakout
# (liquidity_row), so neither a breakout day's own volume nor its run's later sessions lift a thin stock over the floor.
DOLLAR_VOLUME_DAYS = 20
MIN_DOLLAR_VOLUME = {US: 5_000_000.0, CANADA: 5_000_000.0}
MIN_PRICE = {US: 5.0, CANADA: 5.0}

# A breakout is one session (breakout_runs, GLOSSARY.md "Status"): the first on which a ticker meets the four core
# conditions with its close clearing the broken level, the prior 60-day high when the 60-day breakout passed and the
# prior 20-day high otherwise, by the minimum margin. That margin is the larger of MARGIN_FLOOR_PCT and
# MARGIN_RANGE_FRACTION times the ticker's average daily range, (high - low) / close in percent, over the RANGE_DAYS
# sessions before the scored one, so a move the stock's own noise makes is not a breakout. The sessions after it
# whose close stays above that level are EXTENDED, never a breakout again, until RUN_END_CLOSES closes in a row below
# the level end the run. During a run, a session meeting the conditions with the margin is a new breakout only when
# the resistance it clears was set at least PAUSE_SESSIONS sessions before it: a base, not a run of new highs.
MARGIN_FLOOR_PCT = 1.0
MARGIN_RANGE_FRACTION = 0.5
RANGE_DAYS = 14
RUN_END_CLOSES = 2
PAUSE_SESSIONS = 10
# A session's place in a breakout run (StockCriteria.breakout_phase): its breakout session, or an Extended one.
BREAKOUT_SESSION = 'breakout'
EXTENDED_SESSION = 'extended'
# The statuses that make a scored ticker a signal, archived in the signal table: a breakout session.
# A signal archived by the run that scored its session is LIVE, and one archived by replaying a past session from the
# stored prices (app/backfill.py) is BACKFILL.
SIGNAL_STATUSES = (CONFIRMED, HIGH_CONVICTION)
LIVE = 'live'
BACKFILL = 'backfill'
# A near-miss (archive_kind) meets at least this many of the four core conditions: three, or all four without the
# minimum margin. It says which sessions are archived beside the signals, not what any of them scores.
NEAR_MISS_CONDITIONS = 3


def _all_pass(sc: StockCriteria, *flags: str) -> bool:
    return all(getattr(sc, flag) == 1 for flag in flags)


def _any_passes(sc: StockCriteria, *flags: str) -> bool:
    return any(getattr(sc, flag) == 1 for flag in flags)


# The strategy's secondary indicators judged on the breakout day, each passed by its flags as the strategy words it,
# an empty flag counting as not passed. Breakout proximity is left out (it defines the watchlist, never a breakout's
# quality), and so are the retest and durability, which need the days after the breakout. Moving-average alignment
# includes the core close above the 50-day average, which every confirmed breakout passes; the rising averages and
# the declining base volume are the strategy's "preferably" and "ideally", not required.
SECONDARY_INDICATORS: dict[str, Callable[[StockCriteria], bool]] = {
    'rsi': lambda sc: _all_pass(sc, 'rsi_50_70'),
    'macd': lambda sc: _any_passes(sc, 'macd_crossover', 'macd_above_signal', 'macd_hist_rising'),
    'relative_strength': lambda sc: _any_passes(
        sc, 'outperforms_market_1m', 'outperforms_market_3m', 'outperforms_sector_1m', 'outperforms_sector_3m'
    ),
    'moving_average_alignment': lambda sc: _all_pass(sc, 'price_gt_20dma', 'price_gt_50dma', 'price_gt_200dma'),
    'consolidation_quality': lambda sc: _all_pass(sc, 'consolidation_tight'),
    'closing_strength': lambda sc: _all_pass(sc, 'close_upper_quarter'),
    'ema_21_alignment': lambda sc: _all_pass(
        sc, 'price_gt_21ema', 'ema_21_rising', 'price_gt_21wema', 'ema_21w_rising'
    ),
}

# The history every core condition needs: SMA_DAYS rows ending SMA_RISING_DAYS days before the scored day, and the
# longest breakout window before it. A ticker with fewer rows is not scored at all.
MIN_ROWS = max(SMA_DAYS + SMA_RISING_DAYS, max(BREAKOUT_WINDOWS) + 1)

# A ticker whose last FROZEN_DAYS trading days all have one close and zero volume is frozen: a price API (Tiingo)
# repeats the last close of a ticker that stopped trading, which would pass the stale check and be scored.
FROZEN_DAYS = 10

# A ticker that traded on each of the PADDED_WINDOW days before a flat stretch of at least PADDED_MIN_DAYS, repeating
# the close of its last trade, has stopped trading: Tiingo pads such a ticker's rows with its last close. Thinly
# traded tickers also have flat zero-volume runs, but in a year of the sectors page's prices none of 2 days or more
# followed more than 20 traded days, while a liquid ticker can show a single flat day mid-run (SUPN, CVR), hence
# the 2-day minimum. A longer flat stretch after thin trading is left to the FROZEN_DAYS rule.
PADDED_WINDOW = 30
PADDED_MIN_DAYS = 2


def rule_settings() -> dict[str, object]:
    """
    The snapshot of the rules SCREENER_VERSION names: every threshold and window the analyzer scores with, keyed by
    its constant's name, the secondary indicators a high-conviction breakout counts, and each market's benchmark
    (BENCHMARKS in app/markets.py), which relative strength is judged against, in a form json.dumps takes (a tuple
    as a list, a market's floors as a dict). tests/test_rule_settings.py fails when a constant is added here without
    being listed.
    """
    return {
        'BREAKOUT_WINDOWS': list(BREAKOUT_WINDOWS),
        'TIER_WINDOW': TIER_WINDOW,
        'VOLUME_AVG_DAYS': VOLUME_AVG_DAYS,
        'VOLUME_RATIO_MIN': VOLUME_RATIO_MIN,
        'SMA_DAYS': SMA_DAYS,
        'SMA_RISING_DAYS': SMA_RISING_DAYS,
        'RSI_DAYS': RSI_DAYS,
        'RSI_BAND': list(RSI_BAND),
        'MACD_FAST_DAYS': MACD_FAST_DAYS,
        'MACD_SLOW_DAYS': MACD_SLOW_DAYS,
        'MACD_SIGNAL_DAYS': MACD_SIGNAL_DAYS,
        'MACD_CROSSOVER_DAYS': MACD_CROSSOVER_DAYS,
        'SHORT_SMA_DAYS': SHORT_SMA_DAYS,
        'LONG_SMA_DAYS': LONG_SMA_DAYS,
        'EMA_SPAN': EMA_SPAN,
        'EMA_RISING_DAYS': EMA_RISING_DAYS,
        'EMA_MIN_PERIODS': EMA_MIN_PERIODS,
        'CLOSE_POSITION_MIN': CLOSE_POSITION_MIN,
        'CONSOLIDATION_DAYS': CONSOLIDATION_DAYS,
        'CONSOLIDATION_TIGHT_PCT': CONSOLIDATION_TIGHT_PCT,
        'PROXIMITY_PCT': PROXIMITY_PCT,
        'RS_1M_DAYS': RS_1M_DAYS,
        'RS_3M_DAYS': RS_3M_DAYS,
        'SECTOR_MIN_TICKERS': SECTOR_MIN_TICKERS,
        'BENCHMARKS': dict(BENCHMARKS),
        'HIGH_CONVICTION_MIN': HIGH_CONVICTION_MIN,
        'DOLLAR_VOLUME_DAYS': DOLLAR_VOLUME_DAYS,
        'MIN_DOLLAR_VOLUME': dict(MIN_DOLLAR_VOLUME),
        'MIN_PRICE': dict(MIN_PRICE),
        'MARGIN_FLOOR_PCT': MARGIN_FLOOR_PCT,
        'MARGIN_RANGE_FRACTION': MARGIN_RANGE_FRACTION,
        'RANGE_DAYS': RANGE_DAYS,
        'RUN_END_CLOSES': RUN_END_CLOSES,
        'PAUSE_SESSIONS': PAUSE_SESSIONS,
        'SECONDARY_INDICATORS': list(SECONDARY_INDICATORS),
        'MIN_ROWS': MIN_ROWS,
        'FROZEN_DAYS': FROZEN_DAYS,
        'PADDED_WINDOW': PADDED_WINDOW,
        'PADDED_MIN_DAYS': PADDED_MIN_DAYS,
    }


def _flat_days(closes: list, volumes: list) -> int:
    """The number of trailing days whose close equals the last close and whose volume is zero."""
    flat = 0
    for close, volume in zip(reversed(closes), reversed(volumes), strict=True):
        if pd.isna(close) or pd.isna(volume) or close != closes[-1] or volume != 0:
            break
        flat += 1
    return flat


def frozen_since(df: pd.DataFrame) -> date | None:
    """
    Returns the first day of the flat stretch at the end of 'df' (one ticker's rows, oldest first): the trailing days
    whose close equals the last close and whose volume is zero. Returns None when that stretch is shorter than
    FROZEN_DAYS, including when the last day traded any volume or its volume is missing.
    """
    dates = df.index.get_level_values('date')
    flat = _flat_days(df['close'].to_list(), df['volume'].to_list())
    if flat < FROZEN_DAYS:
        return None
    return dates[len(dates) - flat]


def padded_stop(df: pd.DataFrame) -> date | None:
    """
    Returns the last traded day of a ticker whose rows ('df', oldest first) end in a padded stop: a flat stretch of
    at least PADDED_MIN_DAYS (as frozen_since measures it) whose close is that of the day before it, which is the
    last of PADDED_WINDOW days that all traded volume. Returns None otherwise, including when 'df' holds fewer than
    PADDED_WINDOW rows before the stretch.
    """
    closes = df['close'].to_list()
    volumes = df['volume'].to_list()
    flat = _flat_days(closes, volumes)
    last_trade = len(closes) - flat - 1
    if flat < PADDED_MIN_DAYS or last_trade + 1 < PADDED_WINDOW or closes[last_trade] != closes[-1]:
        return None
    window = volumes[last_trade + 1 - PADDED_WINDOW : last_trade + 1]
    if any(pd.isna(volume) or volume <= 0 for volume in window):
        return None
    return df.index.get_level_values('date')[last_trade]


def ema(closes: pd.Series, span: int = EMA_SPAN) -> pd.Series:
    """The exponential moving average of 'closes', as charts draw it, NaN until it has EMA_MIN_PERIODS values."""
    values = closes.ewm(span=span, adjust=False).mean()
    values.iloc[: EMA_MIN_PERIODS - 1] = float('nan')
    return values


def weekly_closes(df: pd.DataFrame) -> pd.Series:
    """The last close of each calendar week (Monday to Sunday) of one ticker's rows, oldest first."""
    dates = pd.DatetimeIndex(pd.to_datetime(df.index.get_level_values('date')))
    closes = pd.Series(df['close'].to_numpy(), index=dates)
    return closes.groupby(dates.to_period('W')).last()


def _at(series: pd.Series, back: int = 0) -> float | None:
    """The value 'back' rows before the last one, None when the series is that short or the value is missing."""
    if back >= len(series):
        return None
    value = series.iloc[-1 - back]
    return None if pd.isna(value) else float(value)


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 2)


def _above(value: float | None, level: float | None) -> int | None:
    """1 when 'value' is above 'level', 0 when not, None when either is missing."""
    if value is None or level is None:
        return None
    return int(value > level)


def _close_position(close: float, high: float | None, low: float | None) -> float | None:
    """Where 'close' sits in the day's range, 0 at the low and 1 at the high: None without a range."""
    if high is None or low is None or high <= low:
        return None
    return (close - low) / (high - low)


def _consolidation(df: pd.DataFrame) -> tuple[float | None, float | None]:
    """
    The range of the CONSOLIDATION_DAYS rows before the last one, as a percentage of their highest high, and their
    later half's average volume over their earlier half's. Each is None when a row it needs lacks the value, or
    there are too few rows, or the earlier half traded no volume.
    """
    window = df.iloc[-1 - CONSOLIDATION_DAYS : -1]
    if len(window) < CONSOLIDATION_DAYS:
        return None, None
    range_pct = None
    if not (window['high'].isna().any() or window['low'].isna().any()):
        top = float(window['high'].max())
        range_pct = (top - float(window['low'].min())) / top * 100 if top > 0 else None
    vol_ratio = None
    volumes = window['volume']
    if not volumes.isna().any():
        half = CONSOLIDATION_DAYS // 2
        earlier = float(volumes.iloc[:half].mean())
        vol_ratio = float(volumes.iloc[half:].mean()) / earlier if earlier > 0 else None
    return range_pct, vol_ratio


def _below(value: float | None, limit: float) -> int | None:
    """1 when 'value' is below 'limit', 0 when not, None when it is missing."""
    return None if value is None else int(value < limit)


def _distance_pct(price: float, resistance: float) -> float | None:
    """How far 'price' is below 'resistance', as a percentage of it: negative above it."""
    return (resistance - price) / resistance * 100 if resistance > 0 else None


@dataclass(frozen=True)
class Tier:
    """
    The 120-day tier's values for one session (TIER_WINDOW): the prior TIER_WINDOW-day high, rounded as the other
    highs are, the close's distance below it in percent (negative above it), and whether the close is above it.
    All three are None when the ticker has too few rows for that high.
    """

    high: float | None
    distance_pct: float | None
    broken: int | None


def tier(price: float, high: float | None) -> Tier:
    """The 120-day tier's values for a close of 'price' against the prior TIER_WINDOW-day 'high', None if unknown."""
    if high is None:
        return Tier(None, None, None)
    return Tier(round(high, 2), _distance_pct(price, high), int(price > high))


def _near(distances: list[float | None]) -> int | None:
    """1 when any distance is from 0 to PROXIMITY_PCT, ends included, 0 when none is, None when all are missing."""
    known = [d for d in distances if d is not None]
    if not known:
        return None
    return int(any(0 <= d <= PROXIMITY_PCT for d in known))


def _return_pct(closes: pd.Series, days: int) -> float | None:
    """The last close's return over the close 'days' rows before it, in percent: None when too short or not positive."""
    end, start = _at(closes), _at(closes, days)
    if end is None or start is None or start <= 0:
        return None
    return (end / start - 1) * 100


def _benchmark_return_pct(dates: pd.Index, days: int, benchmark: Mapping[date, float]) -> float | None:
    """
    The benchmark's return, in percent, between the dates of the row 'days' before the last and the last: None when
    there are too few rows, or the benchmark has no positive close on either date.
    """
    if days >= len(dates):
        return None
    start, end = benchmark.get(dates[-1 - days]), benchmark.get(dates[-1])
    if start is None or end is None or start <= 0:
        return None
    return (end / start - 1) * 100


def _difference(value: float | None, other: float | None) -> float | None:
    return None if value is None or other is None else value - other


def _positive(value: float | None) -> int | None:
    """1 when 'value' is above zero, 0 when not, None when it is missing."""
    return None if value is None else int(value > 0)


def subsector_medians(returns: Iterable[tuple[int, float | None, float | None]]) -> dict[tuple[int, str], float]:
    """
    The median 1- and 3-month returns of each subsector, keyed by (subsector_id, 'return_1m_pct' or
    'return_3m_pct'), from 'returns', one (subsector_id, return_1m_pct, return_3m_pct) per ticker scored; left out
    where the subsector has fewer than SECTOR_MIN_TICKERS such returns.
    """
    by_subsector: dict[tuple[int, str], list[float]] = {}
    for subsector_id, return_1m, return_3m in returns:
        for field, value in (('return_1m_pct', return_1m), ('return_3m_pct', return_3m)):
            if value is not None:
                by_subsector.setdefault((subsector_id, field), []).append(value)
    return {key: statistics.median(values) for key, values in by_subsector.items() if len(values) >= SECTOR_MIN_TICKERS}


def with_subsector_medians(sc: StockCriteria, medians: Mapping[tuple[int, str], float]) -> StockCriteria:
    """'sc' with its 1- and 3-month returns compared with its subsector's medians (subsector_medians)."""
    rs_1m = _difference(sc.return_1m_pct, medians.get((sc.subsector_id, 'return_1m_pct')))
    rs_3m = _difference(sc.return_3m_pct, medians.get((sc.subsector_id, 'return_3m_pct')))
    return replace(
        sc,
        rs_sector_1m=rs_1m,
        rs_sector_3m=rs_3m,
        outperforms_sector_1m=_positive(rs_1m),
        outperforms_sector_3m=_positive(rs_3m),
    )


def with_sector_strength(items: list[StockCriteria]) -> list[StockCriteria]:
    """
    'items' with each one's 1- and 3-month returns compared with the median of those of the items in its subsector
    (subsector_id), itself included; left empty where the subsector has fewer than SECTOR_MIN_TICKERS such returns.
    """
    medians = subsector_medians((sc.subsector_id, sc.return_1m_pct, sc.return_3m_pct) for sc in items)
    return [with_subsector_medians(sc, medians) for sc in items]


@dataclass(frozen=True)
class CoreConditions:
    """
    The four core conditions (GLOSSARY.md, "Core criteria") at each of one ticker's rows, oldest first, each judged
    from the rows up to it: the close above the prior 20-day or 60-day high, the volume confirming it, the close above
    the 50-day SMA, and that SMA rising. Every value is False on a row with too few rows before it for its condition.
    """

    breakout_20d: np.ndarray
    breakout_60d: np.ndarray
    confirmed: np.ndarray
    above_sma: np.ndarray
    sma_rising: np.ndarray

    def met(self) -> np.ndarray:
        """Whether each row meets all four conditions."""
        return (self.breakout_20d | self.breakout_60d) & self.confirmed & self.above_sma & self.sma_rising

    def at(self, row: int) -> list[bool]:
        """The four conditions at 'row', in the order core_criteria_score counts them."""
        return [
            bool(self.breakout_20d[row] or self.breakout_60d[row]),
            bool(self.confirmed[row]),
            bool(self.above_sma[row]),
            bool(self.sma_rising[row]),
        ]


def core_conditions(closes: np.ndarray, highs: np.ndarray, volumes: np.ndarray) -> CoreConditions:
    """
    The four core conditions at each of one ticker's rows (oldest first). 'highs' has a missing high already replaced
    by its close, and 'volumes' is NaN where a volume is missing. Every lookback excludes the row it judges.
    """
    close = pd.Series(closes, dtype=float)
    before = pd.Series(highs, dtype=float).shift(1)
    breakout_20d, breakout_60d = ((close > before.rolling(days).max()).to_numpy() for days in BREAKOUT_WINDOWS)
    volume = pd.Series(volumes, dtype=float)
    # the average skips a missing volume
    average = volume.shift(1).rolling(VOLUME_AVG_DAYS, min_periods=1).mean()
    sma = close.rolling(SMA_DAYS).mean()
    return CoreConditions(
        breakout_20d=breakout_20d,
        breakout_60d=breakout_60d,
        confirmed=((average > 0) & (volume / average >= VOLUME_RATIO_MIN)).to_numpy(),
        above_sma=(close > sma).to_numpy(),
        sma_rising=(sma > sma.shift(SMA_RISING_DAYS)).to_numpy(),
    )


@dataclass(frozen=True)
class BreakoutRuns:
    """
    One ticker's breakout runs, at each of its rows (oldest first). 'phase' is BREAKOUT_SESSION on a breakout,
    EXTENDED_SESSION on a later session of its run that closes above the run's level, and None otherwise. 'level' is
    the broken level of the run the row is in, and 'start' the row of that run's breakout: NaN and -1 outside a run.
    """

    phase: np.ndarray
    level: np.ndarray
    start: np.ndarray


def _resistance_age(highs: np.ndarray, row: int, days: int) -> int:
    """How many sessions before 'row' the highest of the 'days' highs before it was first set: 1 for the day before."""
    return days - int(np.argmax(highs[row - days : row]))


def breakout_runs(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray, core: np.ndarray) -> BreakoutRuns:
    """
    One ticker's breakout runs from its rows (oldest first), each session judged only from the rows up to it, as the
    constants before MARGIN_FLOOR_PCT describe: 'core' marks the rows meeting the four core conditions
    (CoreConditions.met), and
    'highs' and 'lows' are as stored, NaN where missing. A missing high counts its close in the broken level, as
    analyze_ticker's breakout does, and a session missing its high or low is left out of the average daily range;
    with no range known, the margin is MARGIN_FLOOR_PCT.

    A run begins on a breakout, with the level that breakout cleared. It ends on the RUN_END_CLOSES-th close in a row
    below that level, and the next session meeting the conditions with the margin is a breakout again. A session of
    the run meeting them with the margin is a new breakout, beginning a new run at its own level, only when the
    resistance it clears was first set at least PAUSE_SESSIONS sessions before it.
    """
    count = len(closes)
    filled = np.where(np.isnan(highs), closes, highs)
    before = pd.Series(filled).shift(1)
    short, long = BREAKOUT_WINDOWS
    high_short, high_long = (before.rolling(days).max().to_numpy() for days in (short, long))
    with np.errstate(invalid='ignore', divide='ignore'):
        long_break = closes > high_long
        level = np.where(long_break, high_long, high_short)
        margin = (closes / level - 1) * 100
        ranges = (highs - lows) / closes * 100
    average_range = pd.Series(ranges).shift(1).rolling(RANGE_DAYS, min_periods=1).mean().to_numpy()
    qualifying = core & (margin >= np.fmax(MARGIN_FLOOR_PCT, MARGIN_RANGE_FRACTION * average_range))

    phase = np.full(count, None, dtype=object)
    run_level = np.full(count, np.nan)
    run_start = np.full(count, -1)
    began, current, below = -1, float('nan'), 0
    for row in range(int(np.argmax(qualifying)) if qualifying.any() else count, count):
        if began >= 0:
            below = below + 1 if closes[row] < current else 0
            if below >= RUN_END_CLOSES:
                began = -1
        window = long if long_break[row] else short
        if qualifying[row] and (began < 0 or _resistance_age(filled, row, window) >= PAUSE_SESSIONS):
            began, current, below = row, float(level[row]), 0
            phase[row] = BREAKOUT_SESSION
        elif began >= 0 and closes[row] > current:
            phase[row] = EXTENDED_SESSION
        if began >= 0:
            run_level[row], run_start[row] = current, began
    return BreakoutRuns(phase, run_level, run_start)


@dataclass(frozen=True)
class BreakoutPosition:
    """
    Where a breakout or extended session stands in its run (GLOSSARY.md, "Status"): 'margin_pct' is the close's
    margin over the level the run broke, in percent of that level; 'invalidation_pct' the distance from the close
    down to that level, the invalidation level, in percent of the close, the size of a stop placed there; and
    'sessions' the sessions since the run's breakout, 0 on the breakout itself. On an extended session all three are
    measured against the run's level, not a new one.
    """

    margin_pct: float | None = None
    invalidation_pct: float | None = None
    sessions: int | None = None


def breakout_position(runs: BreakoutRuns, closes: np.ndarray, row: int) -> BreakoutPosition:
    """The position of 'row' in its run (breakout_runs): every value None unless a breakout or extended session."""
    if runs.phase[row] not in (BREAKOUT_SESSION, EXTENDED_SESSION):
        return BreakoutPosition()
    close, level = float(closes[row]), float(runs.level[row])
    return BreakoutPosition(
        margin_pct=(close / level - 1) * 100,
        invalidation_pct=(close - level) / close * 100,
        sessions=row - int(runs.start[row]),
    )


def average_dollar_volume(closes: np.ndarray, volumes: np.ndarray, row: int) -> float | None:
    """
    The mean of close times volume over the DOLLAR_VOLUME_DAYS rows before 'row' (oldest first), the liquidity
    floor's measure, or None when there are fewer rows or one of them has no volume (NaN). The sum is exactly rounded
    (math.fsum), so the history replay, which reads the same rows out of a longer series, gives the same value.
    """
    if row < DOLLAR_VOLUME_DAYS:
        return None
    window = slice(row - DOLLAR_VOLUME_DAYS, row)
    total = math.fsum((closes[window] * volumes[window]).tolist())
    return None if math.isnan(total) else total / DOLLAR_VOLUME_DAYS


def liquidity_row(runs: BreakoutRuns, row: int) -> int:
    """
    The row whose average_dollar_volume the liquidity floor judges 'row' by: on a breakout or extended session its
    run's breakout session, so the heavy volume of a breakout and of the run after it never lifts a thin stock over
    the floor while the run lasts; on any other session 'row' itself.
    """
    if runs.phase[row] in (BREAKOUT_SESSION, EXTENDED_SESSION):
        return int(runs.start[row])
    return row


def above_liquidity_floor(sc: StockCriteria) -> bool:
    """
    Whether 'sc' clears its market's liquidity floor: an average dollar volume of at least MIN_DOLLAR_VOLUME and a
    close of at least MIN_PRICE. A ticker with no average dollar volume does not clear it.
    """
    market = market_of(sc.ticker)
    return (
        sc.avg_dollar_volume is not None
        and sc.avg_dollar_volume >= MIN_DOLLAR_VOLUME[market]
        and sc.price is not None
        and sc.price >= MIN_PRICE[market]
    )


def with_status(sc: StockCriteria) -> StockCriteria:
    """
    'sc' with the number of SECONDARY_INDICATORS it passes and its status, None when it has none. A ticker below its
    market's liquidity floor (above_liquidity_floor) has none, nor a breakout position, which only a breakout or
    extended session with its status keeps, as when a later run clears a status (StockCriteriaRepo.clear_status).
    Otherwise only a breakout session (breakout_phase) is confirmed or high-conviction; a later session of its run is
    extended; any other session passing both trend conditions, closing near its resistance and with a tight base is on
    the pre-breakout watchlist.
    """
    passed = sum(passes(sc) for passes in SECONDARY_INDICATORS.values())
    if not above_liquidity_floor(sc):
        return replace(
            sc,
            secondary_passed=passed,
            status=None,
            breakout_margin_pct=None,
            invalidation_distance_pct=None,
            sessions_since_breakout=None,
        )
    status = None
    if sc.breakout_phase == BREAKOUT_SESSION:
        status = HIGH_CONVICTION if passed >= HIGH_CONVICTION_MIN else CONFIRMED
    elif sc.breakout_phase == EXTENDED_SESSION:
        status = EXTENDED
    elif _all_pass(sc, 'price_gt_50dma', 'dma_50_rising', 'near_resistance', 'consolidation_tight'):
        status = PRE_BREAKOUT
    return replace(sc, secondary_passed=passed, status=status)


def archive_kind(sc: StockCriteria) -> str | None:
    """
    The kind of record the signal archive keeps for 'sc' (KINDS in app/models.py, GLOSSARY.md "Signal archive"), the
    first that fits of: a signal, a breakout session (SIGNAL_STATUSES); a continuation, an extended session; a
    near-miss, a session outside every breakout run meeting at least NEAR_MISS_CONDITIONS of the four core conditions,
    which outside a run means three, or all four without the minimum margin; and a session on the pre-breakout
    watchlist. None for any other session, and for every session below its market's liquidity floor, which has no
    status. A continuation is archived only once its breakout is (Analyzer.archive_signals).
    """
    if not above_liquidity_floor(sc):
        return None
    if sc.status in SIGNAL_STATUSES:
        return SIGNAL_KIND
    if sc.status == EXTENDED:
        return CONTINUATION_KIND
    if not sc.in_breakout_run and (sc.core_criteria_score or 0) >= NEAR_MISS_CONDITIONS:
        return NEAR_MISS_KIND
    if sc.status == PRE_BREAKOUT:
        return WATCHLIST_KIND
    return None


# each kind's name in a summary line, singular and plural
_KIND_NAMES = {
    SIGNAL_KIND: ('signal', 'signals'),
    NEAR_MISS_KIND: ('near-miss', 'near-misses'),
    WATCHLIST_KIND: ('watchlist record', 'watchlist records'),
    CONTINUATION_KIND: ('continuation', 'continuations'),
}


def describe_records(stored: Mapping[str, int]) -> str:
    """How many new records of each kind 'stored' counts, every kind named: '2 new signals, 9 near-misses, ...'."""
    parts = [f'{stored.get(kind, 0)} {_KIND_NAMES[kind][stored.get(kind, 0) != 1]}' for kind in KINDS]
    return f'{parts[0].replace(" ", " new ", 1)}, {", ".join(parts[1:-1])} and {parts[-1]}'


def _days_above(macd: pd.Series, signal: pd.Series) -> int:
    """How many sessions in a row, ending on the last, MACD has been above its signal line: 0 when it is not."""
    days = 0
    for above in reversed((macd > signal).to_list()):
        if not above:
            break
        days += 1
    return days


def trading_days_between(dates: pd.Index, previous: date, session: date) -> int:
    """
    The trading days after 'previous' up to and including 'session', counted from 'dates', the ticker's stored price
    dates. When 'previous' is older than its oldest stored row, the stretch before that row, which the price window
    no longer holds, is counted in weekdays, holidays included.
    """
    days = sum(1 for d in dates if previous < d <= session)
    oldest = min(dates)
    if previous < oldest:
        days += int(np.busday_count(previous + timedelta(days=1), oldest))
    return days


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class Scored:
    """
    One ticker scored on a session: its criteria, the market and session it was scored in, that session's close
    (unrounded, as price_data holds it) and the dates of the rows it was scored from, the session's the last.
    """

    criteria: StockCriteria
    market: str
    session_date: date
    close: float
    dates: pd.Index


@dataclass(frozen=True)
class Skipped:
    """
    A ticker left unscored on its market's session: 'reason' is STALE or FROZEN (app/progress.py), 'last' the date
    its FLAGGED line carries and 'detail' why it was not scored, for the warning a run logs.
    """

    ticker: str
    reason: str
    last: date
    detail: str


@dataclass(frozen=True)
class MarketScore:
    """
    One market's session (its newest price date), the tickers scored on it and those left unscored as stale or
    frozen. A ticker with fewer than MIN_ROWS rows is in neither.
    """

    session_date: date
    scored: list[Scored]
    skipped: list[Skipped]


def with_strength_and_status(scored: list[Scored]) -> list[Scored]:
    """
    'scored' with each ticker's returns compared with its subsector's median over 'scored' (with_sector_strength),
    then its status set (with_status), which counts that comparison among the secondary indicators.
    """
    criteria = with_sector_strength([s.criteria for s in scored])
    return [replace(s, criteria=with_status(sc)) for s, sc in zip(scored, criteria, strict=True)]


class Analyzer:
    def __init__(self, container: Container, now: Callable[[], datetime] = utc_now) -> None:
        self.stock_criteria_repo: StockCriteriaRepo = container.stock_criteria_repo
        self.price_data_repo: PriceDataRepo = container.price_data_repo
        self.benchmark_price_repo: BenchmarkPriceRepo = container.benchmark_price_repo
        self.screened_session_repo: ScreenedSessionRepo = container.screened_session_repo
        self.signal_repo: SignalRepo = container.signal_repo
        self.issuer_repo: IssuerRepo = container.issuer_repo
        # the clock a run's start is read from, timezone aware
        self.now = now
        # each ticker the last analyze left unscored as stale or frozen, with that reason, flagged or not
        self.flagged: dict[str, str] = {}

    def analyze(
        self,
        fetched_by: 'PriceApi | None' = None,
        already_flagged: Collection[str] = (),
        started_at: datetime | None = None,
        run_id: int | None = None,
    ) -> None:
        """
        Scores the tickers of every market, or, after a fetch, only those of the markets 'fetched_by' covers (see
        _analyze), on the sessions complete when the run started: 'started_at', or now when it is None. Records each
        market's session it scored in screened_session, from the stock_analyzer screener_run 'run_id' when given.
        Prints 'PROGRESS stage=analyze status=start' before the work and 'status=done' after it.
        """
        report_progress('analyze', status='start')
        self._analyze(fetched_by, already_flagged, started_at if started_at is not None else self.now(), run_id)
        report_progress('analyze', status='done')

    def _analyze(
        self,
        fetched_by: 'PriceApi | None',
        already_flagged: Collection[str],
        started_at: datetime,
        run_id: int | None,
    ) -> None:
        """
        Only completed daily candles are scored. A run started before a market's close plus the price APIs'
        publishing delay (app/markets.py) fetched that day's row while the session was still in progress, or before
        its final values were published, so each of that market's rows dated after its last complete session is
        dropped first, with one line saying how many: a run during market hours scores the previous session.

        Each ticker is judged against the newest price date of its own market (app/markets.py), since the markets
        close on different holidays: a US ticker is never left unscored because Canada traded on a US holiday.

        Each scored ticker's stock_criteria row is dated (date_created) by its market's session, the newest price date
        of that market's complete candles, never by the run's own day: a run on a weekend, on a holiday in one market,
        or after midnight UTC dates every market's rows by that market's last session. No stored price row is dated on
        a day its market does not trade: every fetch drops those (app/price_apis on_sessions).

        A ticker whose download failed keeps its old rows. Scoring it again would stamp a stale score with the
        session's date_created, so a ticker whose last price is older than its market's newest keeps its previous
        stock_criteria row instead. So does a frozen one (see frozen_since), whose API repeats a dead ticker's last
        close every day: its rows are current, but its price is not. A padded stop (see padded_stop), a ticker that
        traded every day until its API began repeating its last close, counts as stale from its second flat day, dated
        by its last trade, so both price APIs reach the same verdict on a ticker that stopped trading. For the same
        reason, after a fetch only the markets the fetch's API covers are scored: a Tiingo run leaves every Canadian
        ticker's row and date_created as they were. A standalone 'analyze' ('fetched_by' None) scores every market.

        In the markets it scores, the run clears the status of every stock_criteria row it did not write, keeping its
        date and values: a stale or frozen ticker, one with fewer than MIN_ROWS rows, or one with no rows left. No row
        keeps a status its stored prices no longer support.

        Each stale ticker is reported with a 'FLAGGED ticker=<ticker> reason=stale last=<date>' line after its warning
        (see app/progress.py), dated by its last traded day when its rows end in a padded stop, even one that is
        also stale, and each frozen one with 'reason=frozen last=<first flat day>', except one in 'already_flagged':
        the fetch reported it already, as not found or with no prices, and kept the rows judged here. A ticker both
        stale and frozen is reported as stale.

        Each scored ticker's returns are compared with its market's benchmark, as stored by the last fetch that
        covered that market, and then with its subsector's median (with_sector_strength), over the tickers scored in
        this run. Only then is its status set (with_status), since relative strength is one of the secondary
        indicators a high-conviction breakout counts.

        Each market with at least one ticker scored records its session, the market's newest price date, in
        screened_session, with SCREENER_VERSION, rule_settings(), the screener's git commit, 'run_id' and how many
        of its tickers were scored, replacing the row of an earlier run that scored the same session.

        Last, each scored ticker the signal archive keeps (archive_kind), a signal, a near-miss, a watchlist session or
        a continuation, is archived in signal (archive_signals) with the same rules, commit and run as its market's
        session, unless it is a listing that is not its issuer's primary.

        Prints one summary line per market that has prices.
        """
        self.flagged = {}
        df = self.load_price_data()
        if df is None:
            logger.info('No price data found.')
            return

        by_market = self._by_market(df)
        scored: list[Scored] = []
        # each market's session scored, and how many of its tickers were
        sessions: dict[str, tuple[date, int]] = {}
        for market, groups in by_market.items():
            if not groups:
                continue
            if fetched_by is not None and market not in fetched_by.markets:
                print(f'{market}: not scored, not fetched by this run ({fetched_by.name})')
                continue
            groups = self._complete_sessions(market, groups, started_at)
            if not groups:
                continue
            result = self._score_market(market, groups)
            for skipped in result.skipped:
                logger.warning(f'{skipped.ticker} not scored: {skipped.detail}')
                self.flagged[skipped.ticker] = skipped.reason
                if skipped.ticker not in already_flagged:
                    report_flagged(skipped.ticker, skipped.reason, skipped.last)
            scored.extend(result.scored)
            if result.scored:
                sessions[market] = (result.session_date, len(result.scored))
            stale = sum(1 for s in result.skipped if s.reason == STALE)
            frozen = len(result.skipped) - stale
            print(
                f'{market}: skipped {stale} stale ticker{"" if stale == 1 else "s"} '
                f'(stopped trading, or last price before {result.session_date}) '
                f'and {frozen} frozen ticker{"" if frozen == 1 else "s"}'
            )

        scored = with_strength_and_status(scored)
        self.save_to_repository(scored)
        judged = fetched_by.markets if fetched_by is not None else MARKETS
        self._clear_unscored_statuses(judged, {s.criteria.ticker for s in scored})
        settings = json.dumps(rule_settings(), sort_keys=True)
        commit = git_commit()
        self._record_sessions(sessions, run_id, settings, commit)
        self.archive_signals(scored, run_id, settings, commit)

    def score_as_of(
        self,
        session_date: date,
        prices: pd.DataFrame | None = None,
        markets: Collection[str] = MARKETS,
        benchmarks: Mapping[str, Mapping[date, float]] | None = None,
    ) -> list[Scored]:
        """
        Scores every ticker as of 'session_date', as a run on that session would have, using only the price and
        benchmark rows dated on or before it: no indicator, and no subsector median, reads a later row. 'prices' is
        load_price_data()'s frame, loaded here when None, so a caller replaying many sessions loads it once, and
        'benchmarks' each benchmark's closes by date, keyed by its ticker (BENCHMARKS), read from benchmark_price
        when None, so prices from elsewhere (the history database) are scored against their own benchmarks.

        Only the 'markets' that traded on 'session_date' are scored: a market whose newest row on or before it is
        older (a holiday, or a weekend) is left out, rather than scored again on its previous session. A ticker is
        left out as a run leaves it out, stale, frozen or with fewer than MIN_ROWS rows, but without a warning or a
        FLAGGED line, and nothing is stored. The session must be complete: rows of a candle not yet final are scored as
        they stand.

        Returns the tickers scored, with their subsector comparison and status set over the tickers scored on that
        session in every market scored.
        """
        df = self.load_price_data() if prices is None else prices
        if df is None:
            return []
        df = df[df.index.get_level_values('date') <= session_date]
        scored: list[Scored] = []
        for market, groups in self._by_market(df).items():
            if not groups or market not in markets:
                continue
            result = self._score_market(market, groups, benchmarks)
            if result.session_date == session_date:
                scored.extend(result.scored)
        return with_strength_and_status(scored)

    @staticmethod
    def _by_market(df: pd.DataFrame) -> dict[str, list[tuple[str, pd.DataFrame]]]:
        """Each market's tickers with their rows, oldest first, leaving out (with a warning) a ticker of none."""
        by_market: dict[str, list[tuple[str, pd.DataFrame]]] = {market: [] for market in MARKETS}
        for ticker, group_df in df.groupby(level='ticker'):
            try:
                by_market[market_of(str(ticker))].append((str(ticker), group_df))
            except UnsupportedTicker as err:
                logger.warning(f'{ticker} not scored: {err}')
        return by_market

    def _score_market(
        self,
        market: str,
        groups: list[tuple[str, pd.DataFrame]],
        benchmarks: Mapping[str, Mapping[date, float]] | None = None,
    ) -> MarketScore:
        """
        Scores each of 'market's tickers ('groups', each ticker's rows oldest first) on the market's session, its
        newest price date, against the benchmark's closes up to that session, from 'benchmarks' when given and else
        from benchmark_price. A ticker whose last price is older, or whose rows end in a padded stop (padded_stop)
        is skipped as stale, and a frozen one (frozen_since) as frozen. Warns about, flags and stores nothing; the
        subsector comparison and the status are left to the caller (with_strength_and_status), which knows every
        ticker scored on the session.
        """
        newest = max(group_df.index.get_level_values('date').max() for _, group_df in groups)
        closes = (
            self.benchmark_price_repo.closes(BENCHMARKS[market])
            if benchmarks is None
            else benchmarks.get(BENCHMARKS[market], {})
        )
        benchmark = {d: c for d, c in closes.items() if d <= newest}
        if not benchmark:
            logger.info(f'{market}: no closes stored for benchmark {BENCHMARKS[market]}, market strength left empty')
        scored: list[Scored] = []
        skipped: list[Skipped] = []
        for ticker, group_df in groups:
            last = group_df.index.get_level_values('date').max()
            last_trade = padded_stop(group_df)
            if last < newest:
                detail = f'its last price is from {last}, before the newest {market} market date {newest}'
                skipped.append(Skipped(ticker, STALE, last_trade or last, detail))
                continue
            if last_trade is not None:
                detail = f'it last traded on {last_trade}, and its price has been repeated since with no volume'
                skipped.append(Skipped(ticker, STALE, last_trade, detail))
                continue
            since = frozen_since(group_df)
            if since is not None:
                detail = f'its price has been frozen, with no volume, since {since}'
                skipped.append(Skipped(ticker, FROZEN, since, detail))
                continue
            sc = self.analyze_ticker(ticker, group_df, benchmark)
            if sc is not None:
                dates = group_df.index.get_level_values('date')
                scored.append(Scored(sc, market, newest, float(group_df['close'].iloc[-1]), dates))
        return MarketScore(newest, scored, skipped)

    def archive_signals(
        self,
        scored: list[Scored],
        run_id: int | None,
        settings: str,
        commit: str | None,
        source: str = LIVE,
        non_primary: Mapping[str, str] | None = None,
    ) -> Counter[str]:
        """
        Archives each of 'scored' the signal archive keeps (archive_kind) as a record of its kind of the session it
        was scored on from 'source' (LIVE or BACKFILL). A signal is linked to the ticker's previous signal with the
        trading days since it. A continuation is linked to the signal of its run's breakout session, the session
        sessions_since_breakout before it under the same screener version, and left out when that breakout is not
        archived, as when its run began before the archive or under other rules. A record already archived for the
        same ticker, session and version is left as it was, so a rerun of a session adds nothing. A listing that is
        not its issuer's primary (IssuerRepo) is left out, whatever its kind, keeping its status in stock_criteria, so
        a company listed twice is archived once, under its primary listing, by today's issuers: 'non_primary'
        (IssuerRepo.non_primary), read here when not given, so a backfill archiving session after session reads it
        once. Logs and returns how many it archived of each kind.
        """
        if non_primary is None:
            non_primary = self.issuer_repo.non_primary()
        records: list[Signal] = []
        left_out: list[str] = []
        kept: list[tuple[Scored, str, date | None]] = []
        for s in scored:
            sc = s.criteria
            kind = archive_kind(sc)
            if kind is None:
                continue
            if sc.ticker.upper() in non_primary:
                left_out.append(f'{sc.ticker} (under {non_primary[sc.ticker.upper()]})')
                continue
            breakout_session = (
                s.dates[len(s.dates) - 1 - sc.sessions_since_breakout] if kind == CONTINUATION_KIND else None
            )
            kept.append((s, kind, breakout_session))
        breakouts = self.signal_repo.breakouts(
            ((s.criteria.ticker, day) for s, _, day in kept if day is not None), SCREENER_VERSION
        )
        unlinked = 0
        for s, kind, breakout_session in kept:
            sc = s.criteria
            market, session_date, close, dates = s.market, s.session_date, s.close, s.dates
            previous, breakout_id = None, None
            if kind == SIGNAL_KIND:
                previous = self.signal_repo.previous(sc.ticker, session_date)
            elif breakout_session is not None:
                breakout_id = breakouts.get((sc.ticker, breakout_session))
                if breakout_id is None:
                    unlinked += 1
                    continue
            records.append(
                Signal(
                    criteria=sc,
                    market=market,
                    exchange=exchange_of(sc.ticker),
                    time_zone=MARKET_CLOSES[market][0].key,
                    session_date=session_date,
                    close=close,
                    screener_version=SCREENER_VERSION,
                    rule_settings=settings,
                    git_commit=commit,
                    run_id=run_id,
                    source=source,
                    previous_signal_id=previous[0] if previous else None,
                    trading_days_since_previous=(
                        trading_days_between(dates, previous[1], session_date) if previous else None
                    ),
                    kind=kind,
                    breakout_id=breakout_id,
                )
            )
        if left_out:
            logger.info(f'not archived, not the primary listing of their issuer: {", ".join(left_out)}')
        if unlinked:
            logger.info(
                f'not archived, {unlinked} extended session{"" if unlinked == 1 else "s"} of an unarchived breakout'
            )
        if not records:
            return Counter()
        stored = self.signal_repo.archive(records)
        logger.info(f'archived {describe_records(stored)}, of {len(records)} tickers the archive keeps')
        return stored

    def _record_sessions(
        self, sessions: Mapping[str, tuple[date, int]], run_id: int | None, settings: str, commit: str | None
    ) -> None:
        """Records each market's session scored, as 'sessions' maps it, with the rules it was scored by."""
        if not sessions:
            return
        self.screened_session_repo.record(
            ScreenedSession(
                market=market,
                session_date=session_date,
                screener_version=SCREENER_VERSION,
                rule_settings=settings,
                git_commit=commit,
                run_id=run_id,
                tickers_scored=scored,
            )
            for market, (session_date, scored) in sessions.items()
        )
        for market, (session_date, scored) in sessions.items():
            logger.info(
                f'{market}: recorded session {session_date}, {scored} ticker{"" if scored == 1 else "s"} scored'
            )

    def _clear_unscored_statuses(self, markets: Collection[str], scored: Collection[str]) -> None:
        """Clears the status of every stock_criteria row of 'markets' whose ticker is not in 'scored'."""

        def judged(ticker: str) -> bool:
            try:
                return market_of(ticker) in markets
            except UnsupportedTicker:
                return False

        self.stock_criteria_repo.clear_status(
            t for t in self.stock_criteria_repo.tickers_with_status() if t not in scored and judged(t)
        )

    @staticmethod
    def _complete_sessions(
        market: str, groups: list[tuple[str, pd.DataFrame]], started_at: datetime
    ) -> list[tuple[str, pd.DataFrame]]:
        """
        Each ticker's rows of 'market' without those dated after its last complete session for a run started at
        'started_at', leaving out a ticker with no row left. Prints one line when it drops any.
        """
        last = last_complete_session(market, started_at)
        kept: list[tuple[str, pd.DataFrame]] = []
        dropped = 0
        for ticker, group_df in groups:
            complete = group_df.index.get_level_values('date') <= last
            dropped += int((~complete).sum())
            if complete.any():
                kept.append((ticker, group_df[complete].copy()))
        if dropped:
            zone, close = MARKET_CLOSES[market]
            final_at = (datetime.combine(last, close) + PUBLISH_DELAY).time()
            print(
                f'{market}: dropped {dropped} row{"" if dropped == 1 else "s"} dated after {last}, '
                f'whose candles are final only after {final_at:%H:%M} {zone.key} that day'
            )
        return kept

    def load_price_data(self) -> pd.DataFrame | None:
        """Every stored price row, indexed by ticker and date, oldest first, or None when price_data is empty."""
        rows = list(self.price_data_repo.all())  # Iterable[PriceData] -> list
        if not rows:
            return None

        # Build a DataFrame and set a MultiIndex (ticker, date)
        df = pd.DataFrame(
            [
                {
                    'subsector_id': r.subsector_id,
                    'ticker': r.ticker,
                    'date': r.date,
                    'close': r.close,
                    'high': r.high,
                    'low': r.low,
                    'volume': r.volume,
                }
                for r in rows
            ]
        )
        df.set_index(['ticker', 'date'], inplace=True)
        df.sort_index(inplace=True)

        # Ensure dtypes
        df['subsector_id'] = pd.to_numeric(df['subsector_id'], errors='coerce')
        for column in ('close', 'high', 'low'):
            df[column] = pd.to_numeric(df[column], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').astype('Int64')

        # Keep only what you need
        return df[['subsector_id', 'close', 'high', 'low', 'volume']]

    def analyze_ticker(
        self, ticker: str, df: pd.DataFrame, benchmark: Mapping[date, float] | None = None
    ) -> StockCriteria | None:
        """
        The ticker's criteria from its rows, oldest first, or None when it has fewer than MIN_ROWS. 'benchmark' is
        its market's benchmark closes by date; without it the market comparison is left empty. The subsector
        comparison is left empty too: with_sector_strength fills it in once every ticker is scored.
        """
        if len(df) < MIN_ROWS:
            return None

        logger.info(f'generating criteria for {ticker}')

        # A row stored before the screener kept highs (upgrade 3) has none until the next fetch replaces it: its
        # close stands in, the highest price known for that day.
        highs = df['high'].fillna(df['close'])
        sma = df['close'].rolling(window=SMA_DAYS).mean()

        close_series = df['close'].squeeze()
        rsi = RSIIndicator(close=close_series, window=RSI_DAYS).rsi()
        macd_calc = MACD(
            close=close_series, window_slow=MACD_SLOW_DAYS, window_fast=MACD_FAST_DAYS, window_sign=MACD_SIGNAL_DAYS
        )
        macd_line = macd_calc.macd()
        signal_line = macd_calc.macd_signal()
        macd_hist = macd_calc.macd_diff()

        price = df['close'].iloc[-1].item()
        volume = df['volume'].iloc[-1].item()
        # every lookback excludes the scored day: its own high and volume must not raise the bar it is measured by
        high_20d, high_60d = (highs.iloc[-1 - days : -1].max().item() for days in BREAKOUT_WINDOWS)
        longer = tier(price, highs.iloc[-1 - TIER_WINDOW : -1].max().item() if len(df) > TIER_WINDOW else None)
        avg_vol = df['volume'].iloc[-1 - VOLUME_AVG_DAYS : -1].mean().item()
        volume_ratio = volume / avg_vol if avg_vol > 0 else None
        dma_50 = sma.iloc[-1].item()
        dma_50_prior = sma.iloc[-1 - SMA_RISING_DAYS].item()

        rsi_val = rsi.iloc[-1].item()
        macd_val = macd_line.iloc[-1].item()
        signal_val = signal_line.iloc[-1].item()
        hist_val = _at(macd_hist)
        hist_prior = _at(macd_hist, 1)
        days_above = _days_above(macd_line, signal_line)
        # a strengthening positive histogram: above zero and above the day before
        hist_rising = None if hist_val is None or hist_prior is None else int(0 < hist_val and hist_prior < hist_val)

        # moving-average alignment beside the core 50-day average: the 200-day needs 200 rows, and SMA_RISING_DAYS
        # more to say whether it is rising
        sma_short = df['close'].rolling(window=SHORT_SMA_DAYS).mean()
        dma_20, dma_20_prior = _at(sma_short), _at(sma_short, SMA_RISING_DAYS)
        sma_long = df['close'].rolling(window=LONG_SMA_DAYS).mean()
        dma_200, dma_200_prior = _at(sma_long), _at(sma_long, SMA_RISING_DAYS)
        ema_daily = ema(df['close'])
        ema_21, ema_21_prior = _at(ema_daily), _at(ema_daily, EMA_RISING_DAYS)
        ema_weekly = ema(weekly_closes(df))
        ema_21w, ema_21w_prior = _at(ema_weekly), _at(ema_weekly, 1)

        # price action, from the day's own high and low: a row stored before upgrade 3 has neither, and leaves
        # closing strength empty rather than standing its close in for them
        close_position = _close_position(price, _at(df['high']), _at(df['low']))
        consolidation_range_pct, consolidation_vol_ratio = _consolidation(df)
        dist_20d, dist_60d = _distance_pct(price, high_20d), _distance_pct(price, high_60d)

        # relative strength against the market: the benchmark's return between the same two dates
        dates = df.index.get_level_values('date')
        return_1m, return_3m = _return_pct(df['close'], RS_1M_DAYS), _return_pct(df['close'], RS_3M_DAYS)
        benchmark = benchmark or {}
        rs_market_1m = _difference(return_1m, _benchmark_return_pct(dates, RS_1M_DAYS, benchmark))
        rs_market_3m = _difference(return_3m, _benchmark_return_pct(dates, RS_3M_DAYS, benchmark))

        # the core conditions at every row, read on the scored one, and whether that session is a breakout or in the
        # run of an earlier one, from every session before it
        closes = df['close'].to_numpy(dtype=float)
        volumes = df['volume'].to_numpy(dtype=float, na_value=np.nan)
        lows = df['low'].to_numpy(dtype=float)
        conditions = core_conditions(closes, highs.to_numpy(dtype=float), volumes)
        last = len(closes) - 1
        core = conditions.at(last)
        runs = breakout_runs(closes, df['high'].to_numpy(dtype=float), lows, conditions.met())
        position = breakout_position(runs, closes, last)

        return StockCriteria(
            id=0,
            subsector_id=df['subsector_id'].iloc[-1].item(),
            ticker=ticker,
            price=round(price, 2),
            core_criteria_met=all(core),
            core_criteria_score=sum(core),
            high_20d=round(high_20d, 2),
            high_60d=round(high_60d, 2),
            volume=int(volume),
            avg_vol_20d=int(avg_vol),
            # unrounded, so a ratio just under the threshold never reads as on it
            volume_ratio=volume_ratio,
            # unrounded, like volume_ratio, so a value just under the floor never reads as on it
            avg_dollar_volume=average_dollar_volume(closes, volumes, liquidity_row(runs, last)),
            dma_50=round(dma_50, 2),
            dma_50_prev_15d=round(dma_50_prior, 2),
            rsi=round(rsi_val, 2),
            macd=round(macd_val, 2),
            signal=round(signal_val, 2),
            breakout_20d=int(conditions.breakout_20d[last]),
            breakout_60d=int(conditions.breakout_60d[last]),
            breakout_confirmed=int(conditions.confirmed[last]),
            price_gt_50dma=int(conditions.above_sma[last]),
            dma_50_rising=int(conditions.sma_rising[last]),
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
            # unrounded, like volume_ratio, so a value just past its threshold never reads as on it
            close_position=close_position,
            close_upper_quarter=None if close_position is None else int(close_position >= CLOSE_POSITION_MIN),
            consolidation_range_pct=consolidation_range_pct,
            consolidation_tight=_below(consolidation_range_pct, CONSOLIDATION_TIGHT_PCT),
            consolidation_vol_ratio=consolidation_vol_ratio,
            consolidation_vol_declining=_below(consolidation_vol_ratio, 1.0),
            dist_high_20d_pct=dist_20d,
            dist_high_60d_pct=dist_60d,
            near_resistance=_near([dist_20d, dist_60d]),
            high_120d=longer.high,
            # unrounded, like the distances to resistance
            dist_high_120d_pct=longer.distance_pct,
            breakout_120d=longer.broken,
            # unrounded, like volume_ratio; differences in percentage points
            return_1m_pct=return_1m,
            return_3m_pct=return_3m,
            rs_market_1m=rs_market_1m,
            rs_market_3m=rs_market_3m,
            outperforms_market_1m=_positive(rs_market_1m),
            outperforms_market_3m=_positive(rs_market_3m),
            # unrounded, like the distances to resistance
            breakout_margin_pct=position.margin_pct,
            invalidation_distance_pct=position.invalidation_pct,
            sessions_since_breakout=position.sessions,
            breakout_phase=runs.phase[last],
            in_breakout_run=bool(runs.start[last] >= 0),
        )

    def save_to_repository(self, scored: list[Scored]) -> None:
        """Stores each ticker's criteria in stock_criteria, dated by the session its market was scored on."""
        if not scored:
            print('No valid results to save.')
            return
        print(f'saving {len(scored)} criteria rows via StockCriteriaRepo')
        by_session: dict[date, list[StockCriteria]] = {}
        for s in scored:
            by_session.setdefault(s.session_date, []).append(s.criteria)
        for session_date, items in by_session.items():
            self.stock_criteria_repo.bulk_insert(items, session_date)
