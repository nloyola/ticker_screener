import argparse

import pandas as pd
from rich.console import Console
from ta.momentum import RSIIndicator
from ta.trend import MACD

from app.container import Container
from app.models import StockCriteria

from .base_command import BaseCommand

console = Console()


class AnalyzerCommand(BaseCommand):
    _NAME = 'analyze'
    _DESCRIPTION = 'analyzes the price data for the stock tickers in the DB'

    def __init__(self, container: Container) -> None:
        super().__init__(self._NAME, self._DESCRIPTION)
        self.stock_criteria_repo = container.stock_criteria_repo
        self.price_data_repo = container.price_data_repo

    def handle(self, args: argparse.Namespace) -> None:
        df = self.load_price_data()
        if df is None:
            print('No price data found.')
            return

        # analyze each ticker -> StockCriteria | None
        results = [self.analyze_ticker(ticker, group_df) for ticker, group_df in df.groupby(level='ticker')]

        # keep only real StockCriteria objects
        items: list[StockCriteria] = [sc for sc in results if sc is not None]
        self.save_to_repository(items)

    def load_price_data(self) -> pd.DataFrame | None:
        # Pull all rows via the repository
        rows = list(self.price_data_repo.all())  # Iterable[PriceData] -> list
        if not rows:
            return None

        # Build a DataFrame and set a MultiIndex (ticker, date)
        df = pd.DataFrame([{'ticker': r.ticker, 'date': r.date, 'close': r.close, 'volume': r.volume} for r in rows])
        df.set_index(['ticker', 'date'], inplace=True)
        df.sort_index(inplace=True)

        # Ensure dtypes
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').astype('Int64')

        # Keep only what you need
        return df[['close', 'volume']]

    def analyze_ticker(self, ticker: str, df: pd.DataFrame) -> StockCriteria | None:
        if df.empty or len(df) < 60:
            return None  # Not enough data

        # Technical indicators
        df['50dma'] = df['close'].rolling(window=50).mean()
        df['10ema'] = df['close'].ewm(span=10).mean()
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

        dma_50_prev = df['50dma'].iloc[-15] if len(df) >= 65 else float('nan')
        dma_50_early = df['50dma'].iloc[-25] if len(df) >= 75 else float('nan')

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
            dma_50_prev_15d=round(dma_50_prev, 2) if pd.notna(dma_50_prev) else None,
            dma_50_25d_ago=round(dma_50_early, 2) if pd.notna(dma_50_early) else None,
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

    def save_to_repository(self, items: list[StockCriteria]) -> None:
        if not items:
            print('No valid results to save.')
            return
        print(f'saving {len(items)} criteria rows via StockCriteriaRepo')
        self.stock_criteria_repo.bulk_insert(items)
