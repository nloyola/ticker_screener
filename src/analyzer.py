import argparse
import io
import json
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from rich.console import Console
from ta.momentum import RSIIndicator
from ta.trend import MACD
from tiingo import TiingoClient

from .base_command import BaseCommand
from .sector import Sector
from .config import Config

console = Console()


class AnalyzerCommand(BaseCommand):
    _NAME = 'analyzer'
    _DESCRIPTION = 'analyzes the price data for the stock tickers in the DB'

    def __init__(self) -> None:
        super().__init__(self._NAME, self._DESCRIPTION)

    def handle(self, args: argparse.Namespace) -> None:
        df = self.load_price_data_from_sqlite()
        if df is None:
            print('No price data found.')
            return

        data = []
        for ticker, group_df in df.groupby(level='ticker'):
            data.append(self.analyze_ticker(ticker, group_df))

        self.save_to_sqlite(data)

    def load_price_data_from_sqlite(self) -> Optional[pd.DataFrame]:
        with sqlite3.connect(Config.get_db_name()) as conn:
            query = """
            SELECT ticker, date, close, volume
            FROM price_data
            ORDER BY ticker, date
            """
            df = pd.read_sql_query(
                query,
                conn,
                parse_dates=['date'],
                index_col=['ticker', 'date'],
            )

            if df.empty:
                return None

            # Keep only what you need, ensure order & dtypes
            df = df[['close', 'volume']].sort_index()
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').astype('Int64')

            return df

    def analyze_ticker(self, ticker: str, df: pd.DataFrame) -> dict:
        if df.empty or len(df) < 60:
            return {'ticker': ticker, 'error': 'Not enough data', 'Available Columns': ', '.join(df.columns)}

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

        return {
            'ticker': ticker,
            'price': round(price, 2),
            'core_criteria_met': core_conditions_met,
            'core_criteria_score': core_criteria_count,
            'high_20d': round(high_20d, 2),
            'high_60d': round(high_60d, 2),
            'volume': int(volume),
            'avg_vol_20d': int(avg_vol),
            'max_vol_10d': int(high_vol_10d),
            'dma_50': round(dma_50, 2),
            'ema_10': round(ema_10, 2),
            'dma_50_prev_15d': round(dma_50_prev, 2) if pd.notna(dma_50_prev) else None,
            'dma_50_25d_ago': round(dma_50_early, 2) if pd.notna(dma_50_early) else None,
            'rsi': round(rsi_val, 2),
            'macd': round(macd_val, 2),
            'signal': round(signal_val, 2),
            'breakout_20d': breakout_20d,
            'breakout_60d': breakout_60d,
            'breakout_confirmed': breakout_confirmed,
            'price_gt_50dma': price_above_50dma,
            'price_gt_10ema': price_above_10ema,
            'dma_50_rising': dma_slope_up,
            'rsi_50_75': rsi_filter,
            'macd_bullish': macd_bullish,
        }

    def save_to_sqlite(self, results: list[dict], db_path: str = Config.get_db_name()):
        if not results:
            print('No results to save.')
            return

        print('saving criteria to sqlite')

        cols = [
            'ticker',
            'price',
            'core_criteria_met',
            'core_criteria_score',
            'high_20d',
            'high_60d',
            'volume',
            'avg_vol_20d',
            'max_vol_10d',
            'dma_50',
            'ema_10',
            'dma_50_prev_15d',
            'dma_50_25d_ago',
            'rsi',
            'macd',
            'signal',
            'breakout_20d',
            'breakout_60d',
            'breakout_confirmed',
            'price_gt_50dma',
            'price_gt_10ema',
            'dma_50_rising',
            'rsi_50_75',
            'macd_bullish',
        ]

        def b(v):  # bool -> 0/1
            return 1 if bool(v) else 0

        def row_to_tuple(r: dict):
            return (
                r.get('ticker'),
                r.get('price'),
                b(r.get('core_criteria_met', 0)),
                int(r.get('core_criteria_score', 0)),
                r.get('high_20d'),
                r.get('high_60d'),
                int(r.get('volume', 0)),
                int(r.get('avg_vol_20d', 0)),
                int(r.get('max_vol_10d', 0)),  # <-- fixed name
                r.get('dma_50'),
                r.get('ema_10'),
                r.get('dma_50_prev_15d'),
                r.get('dma_50_25d_ago'),
                r.get('rsi'),
                r.get('macd'),
                r.get('signal'),
                b(r.get('breakout_20d', 0)),
                b(r.get('breakout_60d', 0)),
                b(r.get('breakout_confirmed', 0)),
                b(r.get('price_gt_50dma', 0)),
                b(r.get('price_gt_10ema', 0)),
                b(r.get('dma_50_rising', 0)),
                b(r.get('rsi_50_75', 0)),
                b(r.get('macd_bullish', 0)),
            )

        # Filter out rows that have an 'error' key
        valid_results = [r for r in results if 'error' not in r]

        if not valid_results:
            print('No valid results to save.')
            return

        values = [row_to_tuple(r) for r in valid_results]

        placeholders = ','.join('?' for _ in cols)
        set_clause = ','.join(f'{c}=excluded.{c}' for c in cols[1:])  # keep ticker as PK

        sql = f"""
            INSERT INTO stock_criteria ({','.join(cols)})
            VALUES ({placeholders})
            ON CONFLICT(ticker) DO UPDATE SET
            {set_clause}
        """

        with sqlite3.connect(db_path) as conn:
            conn.executemany(sql, values)
