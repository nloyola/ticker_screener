import logging
import statistics
from collections.abc import Callable, Collection, Mapping
from dataclasses import replace
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD

from app.container import Container
from app.markets import (
    BENCHMARKS,
    MARKET_CLOSES,
    MARKETS,
    PUBLISH_DELAY,
    UnsupportedTicker,
    last_complete_session,
    market_of,
)
from app.models import StockCriteria
from app.progress import FROZEN, STALE, report_flagged, report_progress

if TYPE_CHECKING:
    from app.price_apis import PriceApi
    from app.repositories.benchmark_price_repo import BenchmarkPriceRepo
    from app.repositories.price_data_repo import PriceDataRepo
    from app.repositories.stock_criteria_repo import StockCriteriaRepo

logger = logging.getLogger(__name__)

# The four mandatory conditions of the breakout strategy (GLOSSARY.md, "Core criteria"), scored on the last row.
# The breakout compares the close with the highest high of the BREAKOUT_WINDOWS trading days before it.
BREAKOUT_WINDOWS = (20, 60)
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
# Momentum: the 14-day RSI inside RSI_BAND, ends included.
RSI_BAND = (50.0, 70.0)
# MACD crossover: MACD crossed above its signal line within the last MACD_CROSSOVER_DAYS sessions, the scored day
# included, and has stayed above it since.
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
PROXIMITY_PCT = 5.0
# Relative strength: the close's return over RS_1M_DAYS and RS_3M_DAYS trading days (about 1 and 3 months), each
# compared with the return of the market's benchmark (BENCHMARKS in app/markets.py) between the same two dates, and
# with the median return of the tickers scored in the same run in the ticker's subsector, itself included. The
# subsector median needs SECTOR_MIN_TICKERS returns; with fewer it is left empty. The ticker outperforms when its
# return is above the one it is compared with.
RS_1M_DAYS = 21
RS_3M_DAYS = 63
SECTOR_MIN_TICKERS = 3

# The strategy's statuses (GLOSSARY.md, "Status"), set once a ticker's subsector comparison is known
# (with_status). A confirmed breakout meets the four core conditions, and is high-conviction when it also passes at
# least HIGH_CONVICTION_MIN of the SECONDARY_INDICATORS. A ticker on the pre-breakout watchlist is not a confirmed
# breakout, but passes both trend conditions and closes near its resistance (PROXIMITY_PCT).
HIGH_CONVICTION = 'high_conviction'
CONFIRMED = 'confirmed'
PRE_BREAKOUT = 'pre_breakout'
HIGH_CONVICTION_MIN = 4


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


def with_sector_strength(items: list[StockCriteria]) -> list[StockCriteria]:
    """
    'items' with each one's 1- and 3-month returns compared with the median of those of the items in its subsector
    (subsector_id), itself included; left empty where the subsector has fewer than SECTOR_MIN_TICKERS such returns.
    """
    medians: dict[tuple[int, str], float] = {}
    for field in ('return_1m_pct', 'return_3m_pct'):
        by_subsector: dict[int, list[float]] = {}
        for sc in items:
            value = getattr(sc, field)
            if value is not None:
                by_subsector.setdefault(sc.subsector_id, []).append(value)
        for subsector_id, values in by_subsector.items():
            if len(values) >= SECTOR_MIN_TICKERS:
                medians[subsector_id, field] = statistics.median(values)
    result = []
    for sc in items:
        rs_1m = _difference(sc.return_1m_pct, medians.get((sc.subsector_id, 'return_1m_pct')))
        rs_3m = _difference(sc.return_3m_pct, medians.get((sc.subsector_id, 'return_3m_pct')))
        result.append(
            replace(
                sc,
                rs_sector_1m=rs_1m,
                rs_sector_3m=rs_3m,
                outperforms_sector_1m=_positive(rs_1m),
                outperforms_sector_3m=_positive(rs_3m),
            )
        )
    return result


def with_status(sc: StockCriteria) -> StockCriteria:
    """'sc' with the number of SECONDARY_INDICATORS it passes and its status, None when it has none."""
    passed = sum(passes(sc) for passes in SECONDARY_INDICATORS.values())
    status = None
    if sc.core_criteria_met:
        status = HIGH_CONVICTION if passed >= HIGH_CONVICTION_MIN else CONFIRMED
    elif _all_pass(sc, 'price_gt_50dma', 'dma_50_rising', 'near_resistance'):
        status = PRE_BREAKOUT
    return replace(sc, secondary_passed=passed, status=status)


def _days_above(macd: pd.Series, signal: pd.Series) -> int:
    """How many sessions in a row, ending on the last, MACD has been above its signal line: 0 when it is not."""
    days = 0
    for above in reversed((macd > signal).to_list()):
        if not above:
            break
        days += 1
    return days


def utc_now() -> datetime:
    return datetime.now(UTC)


class Analyzer:
    def __init__(self, container: Container, now: Callable[[], datetime] = utc_now) -> None:
        self.stock_criteria_repo: StockCriteriaRepo = container.stock_criteria_repo
        self.price_data_repo: PriceDataRepo = container.price_data_repo
        self.benchmark_price_repo: BenchmarkPriceRepo = container.benchmark_price_repo
        # the clock a run's start is read from, timezone aware
        self.now = now

    def analyze(
        self,
        fetched_by: 'PriceApi | None' = None,
        already_flagged: Collection[str] = (),
        started_at: datetime | None = None,
    ) -> None:
        """
        Scores the tickers of every market, or, after a fetch, only those of the markets 'fetched_by' covers (see
        _analyze), on the sessions complete when the run started: 'started_at', or now when it is None. Prints
        'PROGRESS stage=analyze status=start' before the work and 'status=done' after it.
        """
        report_progress('analyze', status='start')
        self._analyze(fetched_by, already_flagged, started_at if started_at is not None else self.now())
        report_progress('analyze', status='done')

    def _analyze(self, fetched_by: 'PriceApi | None', already_flagged: Collection[str], started_at: datetime) -> None:
        """
        Only completed daily candles are scored. A run started before a market's close plus the price APIs'
        publishing delay (app/markets.py) fetched that day's row while the session was still in progress, or before
        its final values were published, so each of that market's rows dated after its last complete session is
        dropped first, with one line saying how many: a run during market hours scores the previous session.

        Each ticker is judged against the newest price date of its own market (app/markets.py), since the markets
        close on different holidays: a US ticker is never left unscored because Canada traded on a US holiday.

        A ticker whose download failed keeps its old rows. Scoring it again would stamp a stale score with today's
        date_created, so a ticker whose last price is older than its market's newest keeps its previous stock_criteria
        row instead. So does a frozen one (see frozen_since), whose API repeats a dead ticker's last close every day:
        its rows are current, but its price is not. A padded stop (see padded_stop), a ticker that traded every day
        until its API began repeating its last close, counts as stale from its second flat day, dated by its last
        trade, so both price APIs reach the same verdict on a ticker that stopped trading. For the same reason, after
        a fetch only the markets the fetch's API covers are scored: a Tiingo run leaves every Canadian ticker's row
        and date_created as they were. A standalone 'analyze' ('fetched_by' None) scores every market.

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

        Prints one summary line per market that has prices.
        """
        df = self.load_price_data()
        if df is None:
            logger.info('No price data found.')
            return

        by_market: dict[str, list[tuple[str, pd.DataFrame]]] = {market: [] for market in MARKETS}
        for ticker, group_df in df.groupby(level='ticker'):
            try:
                by_market[market_of(str(ticker))].append((str(ticker), group_df))
            except UnsupportedTicker as err:
                logger.warning(f'{ticker} not scored: {err}')

        results: list[StockCriteria | None] = []
        for market, groups in by_market.items():
            if not groups:
                continue
            if fetched_by is not None and market not in fetched_by.markets:
                print(f'{market}: not scored, not fetched by this run ({fetched_by.name})')
                continue
            groups = self._complete_sessions(market, groups, started_at)
            if not groups:
                continue
            benchmark = self.benchmark_price_repo.closes(BENCHMARKS[market])
            if not benchmark:
                logger.info(
                    f'{market}: no closes stored for benchmark {BENCHMARKS[market]}, market strength left empty'
                )
            newest = max(group_df.index.get_level_values('date').max() for _, group_df in groups)
            stale = 0
            frozen = 0
            for ticker, group_df in groups:
                last = group_df.index.get_level_values('date').max()
                last_trade = padded_stop(group_df)
                if last < newest:
                    logger.warning(
                        f'{ticker} not scored: its last price is from {last}, '
                        f'before the newest {market} market date {newest}'
                    )
                    stale += 1
                    if ticker not in already_flagged:
                        report_flagged(ticker, STALE, last_trade or last)
                    continue
                if last_trade is not None:
                    logger.warning(
                        f'{ticker} not scored: it last traded on {last_trade}, and its price has been repeated '
                        f'since with no volume'
                    )
                    stale += 1
                    if ticker not in already_flagged:
                        report_flagged(ticker, STALE, last_trade)
                    continue
                since = frozen_since(group_df)
                if since is not None:
                    logger.warning(f'{ticker} not scored: its price has been frozen, with no volume, since {since}')
                    frozen += 1
                    if ticker not in already_flagged:
                        report_flagged(ticker, FROZEN, since)
                    continue
                results.append(self.analyze_ticker(ticker, group_df, benchmark))
            print(
                f'{market}: skipped {stale} stale ticker{"" if stale == 1 else "s"} '
                f'(stopped trading, or last price before {newest}) '
                f'and {frozen} frozen ticker{"" if frozen == 1 else "s"}'
            )

        # keep only real StockCriteria objects
        items = [with_status(sc) for sc in with_sector_strength([sc for sc in results if sc is not None])]
        date_created = datetime.now().date()
        self.save_to_repository(items, date_created)
        judged = fetched_by.markets if fetched_by is not None else MARKETS
        self._clear_unscored_statuses(judged, {sc.ticker for sc in items})

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
        # Pull all rows via the repository
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
        rsi = RSIIndicator(close=close_series).rsi()
        macd_calc = MACD(close=close_series)
        macd_line = macd_calc.macd()
        signal_line = macd_calc.macd_signal()
        macd_hist = macd_calc.macd_diff()

        price = df['close'].iloc[-1].item()
        volume = df['volume'].iloc[-1].item()
        # every lookback excludes the scored day: its own high and volume must not raise the bar it is measured by
        high_20d, high_60d = (highs.iloc[-1 - days : -1].max().item() for days in BREAKOUT_WINDOWS)
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

        breakout_20d = price > high_20d
        breakout_60d = price > high_60d
        breakout_confirmed = volume_ratio is not None and volume_ratio >= VOLUME_RATIO_MIN
        price_above_50dma = price > dma_50
        dma_slope_up = dma_50 > dma_50_prior

        core = [breakout_20d or breakout_60d, breakout_confirmed, price_above_50dma, dma_slope_up]

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
            # unrounded, like volume_ratio; differences in percentage points
            return_1m_pct=return_1m,
            return_3m_pct=return_3m,
            rs_market_1m=rs_market_1m,
            rs_market_3m=rs_market_3m,
            outperforms_market_1m=_positive(rs_market_1m),
            outperforms_market_3m=_positive(rs_market_3m),
        )

    def save_to_repository(self, items: list[StockCriteria], date_created: date) -> None:
        if not items:
            print('No valid results to save.')
            return
        print(f'saving {len(items)} criteria rows via StockCriteriaRepo')
        self.stock_criteria_repo.bulk_insert(items, date_created)
