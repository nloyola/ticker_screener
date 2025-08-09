import argparse
import os
import sqlite3

from rich.console import Console

from .base_command import BaseCommand
from .config import Config

console = Console()


class DbCreate(BaseCommand):
    _NAME = 'db-create'
    _DESCRIPTION = 'creates the database tables'

    def __init__(self) -> None:
        super().__init__(self._NAME, self._DESCRIPTION)
        os.makedirs(os.path.dirname(Config.get_db_name()), exist_ok=True)

    # def add_arguments(self, parser: argparse.ArgumentParser) -> None:
    # parser.add_argument("--json", help="use the stock tickers from the JSON file")
    # parser.add_argument("--sector", help="filter groups by sector name (case-insensitive)")

    def handle(self, args: argparse.Namespace) -> None:
        conn = sqlite3.connect(Config.get_db_name())
        conn.execute('PRAGMA journal_mode=WAL;')
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS sector (
                sector TEXT,
                subsector TEXT,
                tickers TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS price_data (
                ticker TEXT,
                date TEXT,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (Ticker, Date)
            )
        """)

        c.execute("""
            CREATE TABLE stock_criteria (
                ticker TEXT NOT NULL,
                price REAL,
                core_criteria_met BOOLEAN,
                core_criteria_score INTEGER,
                high_20d REAL,
                high_60d REAL,
                volume INTEGER,
                avg_vol_20d INTEGER,
                max_vol_10d INTEGER,
                dma_50 REAL,
                ema_10 REAL,
                dma_50_prev_15d REAL,
                dma_50_25d_ago REAL,
                rsi REAL,
                macd REAL,
                signal REAL,
                breakout_20d INTEGER,
                breakout_60d INTEGER,
                breakout_confirmed INTEGER,
                price_gt_50dma INTEGER,
                price_gt_10ema INTEGER,
                dma_50_rising INTEGER,
                rsi_50_75 INTEGER,
                macd_bullish INTEGER,
                PRIMARY KEY (ticker)
            )
        """)
