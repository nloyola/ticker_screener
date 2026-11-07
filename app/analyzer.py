import logging
from collections.abc import Collection
from datetime import date, datetime
from typing import TYPE_CHECKING

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD

from app.container import Container
from app.markets import MARKETS, UnsupportedTicker, market_of
from app.models import StockCriteria
from app.progress import STALE, report_flagged, report_progress

if TYPE_CHECKING:
    from app.price_apis import PriceApi
    from app.repositories.price_data_repo import PriceDataRepo
    from app.repositories.stock_criteria_repo import StockCriteriaRepo

logger = logging.getLogger(__name__)

# dma_50_rising compares today's 50-day average with the ones 14 and 24 days ago, so a ticker needs 50 + 24 rows
# and today's to be scored fully. A ticker with fewer rows is not scored at all.
MIN_ROWS = 75


class Analyzer:
    def __init__(self, container: Container) -> None:
        self.stock_criteria_repo: StockCriteriaRepo = container.stock_criteria_repo
        self.price_data_repo: PriceDataRepo = container.price_data_repo

    def analyze(self, fetched_by: 'PriceApi | None' = None, already_flagged: Collection[str] = ()) -> None:
        """
        Scores the tickers of every market, or, after a fetch, only those of the markets 'fetched_by' covers (see
        _analyze). Prints 'PROGRESS stage=analyze status=start' before the work and 'status=done' after it.
        """
        report_progress('analyze', status='start')
        self._analyze(fetched_by, already_flagged)
        report_progress('analyze', status='done')

    def _analyze(self, fetched_by: 'PriceApi | None', already_flagged: Collection[str]) -> None:
        """
        Each ticker is judged against the newest price date of its own market (app/markets.py), since the markets
        close on different holidays: a US ticker is never left unscored because Canada traded on a US holiday.

        A ticker whose download failed keeps its old rows. Scoring it again would stamp a stale score with today's
        date_created, so a ticker whose last price is older than its market's newest keeps its previous stock_criteria
        row instead. For the same reason, after a fetch only the markets the fetch's API covers are scored: a Tiingo
        run leaves every Canadian ticker's row and date_created as they were. A standalone 'analyze' ('fetched_by'
        None) scores every market.

        Each stale ticker is reported with a 'FLAGGED ticker=<ticker> reason=stale last=<date>' line after its warning
        (see app/progress.py), except one in 'already_flagged': the fetch reported it already, as not found or with no
        prices, and kept the rows judged here.

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
            newest = max(group_df.index.get_level_values('date').max() for _, group_df in groups)
            stale = 0
            for ticker, group_df in groups:
                last = group_df.index.get_level_values('date').max()
                if last < newest:
                    logger.warning(
                        f'{ticker} not scored: its last price is from {last}, '
                        f'before the newest {market} market date {newest}'
                    )
                    stale += 1
                    if ticker not in already_flagged:
                        report_flagged(ticker, STALE, last)
                    continue
                results.append(self.analyze_ticker(ticker, group_df))
            print(f'{market}: skipped {stale} stale ticker{"" if stale == 1 else "s"} (last price before {newest})')

        # keep only real StockCriteria objects
        items: list[StockCriteria] = [sc for sc in results if sc is not None]
        date_created = datetime.now().date()
        self.save_to_repository(items, date_created)

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
                    'volume': r.volume,
                }
                for r in rows
            ]
        )
        df.set_index(['ticker', 'date'], inplace=True)
        df.sort_index(inplace=True)

        # Ensure dtypes
        df['subsector_id'] = pd.to_numeric(df['subsector_id'], errors='coerce')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').astype('Int64')

        # Keep only what you need
        return df[['subsector_id', 'close', 'volume']]

    def analyze_ticker(self, ticker: str, df: pd.DataFrame) -> StockCriteria | None:
        if len(df) < MIN_ROWS:
            return None

        logger.info(f'generating criteria for {ticker}')

        # Technical indicators
        df['50dma'] = df['close'].rolling(window=50).mean()
        df['10ema'] = df['close'].ewm(span=10, adjust=False).mean()
        df['volume_avg_20'] = df['volume'].rolling(window=20).mean()
        df['max_20d'] = df['close'].rolling(window=20).max()
        df['max_60d'] = df['close'].rolling(window=60).max()
        df['max_vol_10d'] = df['volume'].rolling(window=10).max()

        close_series = df['close'].squeeze()
        rsi = RSIIndicator(close=close_series).rsi()
        macd_calc = MACD(close=close_series)
        macd_line = macd_calc.macd()
        signal_line = macd_calc.macd_signal()

        price = df['close'].iloc[-1].item()
        volume = df['volume'].iloc[-1].item()
        avg_vol = df['volume_avg_20'].iloc[-1].item()
        dma_50 = df['50dma'].iloc[-1].item()

        dma_50_prev = df['50dma'].iloc[-15].item()
        dma_50_early = df['50dma'].iloc[-25].item()

        ema_10 = df['10ema'].iloc[-1].item()
        high_20d = df['max_20d'].iloc[-2].item()
        high_60d = df['max_60d'].iloc[-2].item()
        high_vol_10d = df['max_vol_10d'].iloc[-2].item()
        rsi_val = rsi.iloc[-1].item()
        macd_val = macd_line.iloc[-1].item()
        signal_val = signal_line.iloc[-1].item()

        breakout_20d = price > high_20d
        breakout_60d = price > high_60d
        breakout_confirmed = volume >= 1.5 * avg_vol and volume >= high_vol_10d
        price_above_50dma = price > dma_50
        price_above_10ema = price > ema_10
        dma_slope_up = dma_50 > dma_50_prev and dma_50_prev > dma_50_early
        rsi_filter = 50 < rsi_val < 75
        macd_bullish = macd_val > signal_val

        core_conditions_met = all(
            [
                breakout_20d or breakout_60d,
                breakout_confirmed,
                price_above_50dma,
                price_above_10ema,
                dma_slope_up,
            ]
        )

        core_criteria_count = sum(
            [
                breakout_20d or breakout_60d,
                breakout_confirmed,
                price_above_50dma,
                price_above_10ema,
                dma_slope_up,
            ]
        )

        return StockCriteria(
            id=0,
            subsector_id=df['subsector_id'].iloc[-1].item(),
            ticker=ticker,
            price=round(price, 2),
            core_criteria_met=core_conditions_met,
            core_criteria_score=core_criteria_count,
            high_20d=round(high_20d, 2),
            high_60d=round(high_60d, 2),
            volume=int(volume),
            avg_vol_20d=int(avg_vol),
            max_vol_10d=int(high_vol_10d),
            dma_50=round(dma_50, 2),
            ema_10=round(ema_10, 2),
            dma_50_prev_15d=round(dma_50_prev, 2),
            dma_50_25d_ago=round(dma_50_early, 2),
            rsi=round(rsi_val, 2),
            macd=round(macd_val, 2),
            signal=round(signal_val, 2),
            breakout_20d=int(breakout_20d),
            breakout_60d=int(breakout_60d),
            breakout_confirmed=int(breakout_confirmed),
            price_gt_50dma=int(price_above_50dma),
            price_gt_10ema=int(price_above_10ema),
            dma_50_rising=int(dma_slope_up),
            rsi_50_75=int(rsi_filter),
            macd_bullish=int(macd_bullish),
        )

    def save_to_repository(self, items: list[StockCriteria], date_created: date) -> None:
        if not items:
            print('No valid results to save.')
            return
        print(f'saving {len(items)} criteria rows via StockCriteriaRepo')
        self.stock_criteria_repo.bulk_insert(items, date_created)
