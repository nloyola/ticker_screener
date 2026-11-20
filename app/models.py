from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class Ticker:
    ticker: str
    last: float | None
    prev_close: float | None
    volume: int | None
    bid_price: float | None
    ask_price: float | None
    timestamp: str | None


@dataclass(frozen=True)
class Sector:
    id: int
    sector: str
    created_at: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class Subsector:
    id: int
    sector_id: int
    subsector: str
    tickers: str
    created_at: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class PriceData:
    id: int
    subsector_id: int
    ticker: str
    date: date
    close: float
    volume: int
    # adjusted as the close is; None in rows stored before the screener kept them, until the next fetch replaces them
    high: float | None = None
    low: float | None = None
    open: float | None = None


@dataclass(frozen=True)
class StockCriteria:
    id: int
    subsector_id: int
    ticker: str
    price: float | None = None
    core_criteria_met: bool | None = None
    core_criteria_score: int | None = None
    high_20d: float | None = None
    high_60d: float | None = None
    volume: int | None = None
    avg_vol_20d: int | None = None
    # volume / avg_vol_20d, None when the average is 0
    volume_ratio: float | None = None
    dma_50: float | None = None
    dma_50_prev_15d: float | None = None
    rsi: float | None = None
    macd: float | None = None
    signal: float | None = None
    breakout_20d: int | None = None
    breakout_60d: int | None = None
    breakout_confirmed: int | None = None
    price_gt_50dma: int | None = None
    dma_50_rising: int | None = None
    # the secondary indicators: never counted in core_criteria_score, None where the ticker has too few rows
    rsi_50_70: int | None = None
    macd_above_signal: int | None = None
    # MACD minus its signal line, today and the day before
    macd_hist: float | None = None
    macd_hist_prev: float | None = None
    # sessions in a row, ending today, with MACD above its signal line
    macd_days_above: int | None = None
    macd_crossover: int | None = None
    macd_hist_rising: int | None = None
    dma_20: float | None = None
    dma_20_prev_15d: float | None = None
    dma_200: float | None = None
    dma_200_prev_15d: float | None = None
    price_gt_20dma: int | None = None
    dma_20_rising: int | None = None
    price_gt_200dma: int | None = None
    dma_200_rising: int | None = None
    ema_21: float | None = None
    ema_21_prev_5d: float | None = None
    price_gt_21ema: int | None = None
    ema_21_rising: int | None = None
    # the 21-week EMA of the weekly closes, this week's and last week's
    ema_21w: float | None = None
    ema_21w_prev_1w: float | None = None
    price_gt_21wema: int | None = None
    ema_21w_rising: int | None = None
    # where the close sits in the day's high-low range, 0 at the low and 1 at the high
    close_position: float | None = None
    close_upper_quarter: int | None = None
    # the 20 days before today: their high-low range as a percentage of their high, and their later 10 days' average
    # volume over their earlier 10 days'
    consolidation_range_pct: float | None = None
    consolidation_tight: int | None = None
    consolidation_vol_ratio: float | None = None
    consolidation_vol_declining: int | None = None
    # how far the close is below high_20d and high_60d, as a percentage of each: negative above it
    dist_high_20d_pct: float | None = None
    dist_high_60d_pct: float | None = None
    near_resistance: int | None = None
    # the 120-day tier (TIER_WINDOW in app/analyzer.py): the highest high of the 120 days before today, how far the
    # close is below it as a percentage of it (negative above it), and whether the close is above it; None where the
    # ticker has fewer than 120 rows before today
    high_120d: float | None = None
    dist_high_120d_pct: float | None = None
    breakout_120d: int | None = None
    # the close's return over 21 and 63 trading days, in percent, and its differences, in percentage points, from the
    # market benchmark's return over the same dates and from the median return of the subsector's scored tickers
    return_1m_pct: float | None = None
    return_3m_pct: float | None = None
    rs_market_1m: float | None = None
    rs_market_3m: float | None = None
    rs_sector_1m: float | None = None
    rs_sector_3m: float | None = None
    outperforms_market_1m: int | None = None
    outperforms_market_3m: int | None = None
    outperforms_sector_1m: int | None = None
    outperforms_sector_3m: int | None = None
    # how many of the strategy's secondary indicators it passes (SECONDARY_INDICATORS in app/analyzer.py), and its
    # status: 'high_conviction', 'confirmed', 'extended', 'pre_breakout', or None for none
    secondary_passed: int | None = None
    status: str | None = None
    # on a breakout or extended session, the run's position (breakout_position in app/analyzer.py): the close's
    # margin over the level the run broke, in percent of the level, the distance from the close down to that level,
    # the invalidation level, in percent of the close, and the sessions since the run's breakout, 0 on the breakout
    # itself; None on any other session
    breakout_margin_pct: float | None = None
    invalidation_distance_pct: float | None = None
    sessions_since_breakout: int | None = None
    # the liquidity floor's measure (average_dollar_volume in app/analyzer.py): the mean close times volume of the
    # DOLLAR_VOLUME_DAYS sessions before the scored one, in the market's own currency; None when one lacks a volume
    avg_dollar_volume: float | None = None
    # the session's place in a breakout run, which decides its status (breakout_runs in app/analyzer.py): 'breakout',
    # 'extended' or None; not stored
    breakout_phase: str | None = None


@dataclass(frozen=True)
class ScreenedSession:
    """One market's session scored by a run, with the rules it was scored by (SCREENER_VERSION in app/analyzer.py)."""

    market: str
    session_date: date
    screener_version: str
    # rule_settings() as JSON
    rule_settings: str
    # the screener's git commit, with '-dirty' when its checkout had uncommitted changes; None when unknown
    git_commit: str | None
    # the stock_analyzer screener_run the scores came from; None for a run not started by its runner
    run_id: int | None
    tickers_scored: int


@dataclass(frozen=True)
class Signal:
    """
    One archived breakout (table signal): a ticker scoring all four mandatory conditions on a session, with every
    value it was scored with. Its sector and subsector names are read from 'criteria.subsector_id' when it is stored.
    """

    criteria: StockCriteria
    market: str
    # the exchange named by the ticker's suffix, None for a bare US ticker (exchange_of in app/markets.py)
    exchange: str | None
    # the market's time zone, an IANA key
    time_zone: str
    session_date: date
    # the session's close as price_data holds it, unrounded
    close: float
    screener_version: str
    # rule_settings() as JSON
    rule_settings: str
    git_commit: str | None
    run_id: int | None
    # 'live' for a signal archived by the run that scored its session, 'backfill' for a replayed one
    source: str
    # the ticker's previous signal and the trading days since its session, both None for its first
    previous_signal_id: int | None = None
    trading_days_since_previous: int | None = None


@dataclass(frozen=True)
class PathSession:
    """
    One session of a signal's price path (table signal_price): session_index 0 is the signal session and N the Nth
    session after it, with the ticker's prices and its market's benchmark's open and close on that day, as the run
    that wrote the path held them.
    """

    session_index: int
    session_date: date
    open: float | None
    high: float | None
    low: float | None
    close: float
    volume: int | None
    benchmark_open: float | None
    benchmark_close: float | None


# why backfill-history left a listed ticker out (BackfillRun.left_out)
NO_HISTORY = 'no_history'
UNSUPPORTED = 'unsupported'
SHORT_HISTORY = 'short_history'
NEVER_SCORED = 'never_scored'


@dataclass(frozen=True)
class BackfillRun:
    """One finished backfill-history run: what it covered and the biases it carries (backfill_run in schema.sql)."""

    started_at: datetime
    finished_at: datetime
    universe_date: date
    first_session: date | None
    last_session: date | None
    sessions: int
    price_api: str | None
    tickers_listed: int
    tickers_replayed: int
    tickers_scored: int
    # each listed ticker never scored, with why: NO_HISTORY, UNSUPPORTED, SHORT_HISTORY or NEVER_SCORED
    left_out: dict[str, str]
    # each replayed ticker whose history starts late, with its first day
    short_history: dict[str, date]
    signals: int
    relinked: int
    screener_version: str
    git_commit: str | None
    id: int | None = None


@dataclass(frozen=True)
class AnalysisResult:
    """
    One finding of a BreakoutIQ analysis run (analysis_result in schema.sql): its section, the finding within it, the
    horizon it is about (None when it covers every horizon) and its figures, a JSON object whose shape
    payload_version names.
    """

    section: str
    finding: str
    horizon: int | None
    payload_version: int
    payload: dict


@dataclass(frozen=True)
class AnalysisRun:
    """
    One finished BreakoutIQ analysis run (analysis_run in schema.sql): when it ran, the signal sessions its data
    covered, the signals it analysed (those with a resolved horizon), the settings it ran with and the screener that
    ran it.
    """

    started_at: datetime
    finished_at: datetime
    first_session: date | None
    last_session: date | None
    signals: int
    settings: dict
    screener_version: str
    git_commit: str | None
    id: int | None = None
