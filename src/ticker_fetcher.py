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


class TickerFetcherCommand(BaseCommand):
    """
    Fetches price data for all the sectors in the 'sector' DB table

    The 'price_data' table holds the last 120 days worth of data.

    If the 'price_data' table already has data for the ticker, then only the latest data is fetched.
    E.g. if the latest date is for 5 days ago, then only the last 5 days worth of data is fetched.

    For debugging purposes, a single sector can be requested. See '--sector' option.

    The Tiingo API is used to fetch price data.
    """

    _NAME = 'ticker-fetcher'
    _DESCRIPTION = 'fetches stock ticker price data'

    def __init__(self) -> None:
        super().__init__(self._NAME, self._DESCRIPTION)

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('--sector', help='filter sectors by name (case-insensitive)')

    def handle(self, args: argparse.Namespace) -> None:
        # print(f"args: {args}")

        sectors = Sector.load_from_sqlite(Config.get_db_name())
        # sectors = [
        #     Sector(sector='Technology', subsector='1', tickers=['XOM', 'CVX', 'SU', 'CVE']),
        #     Sector(sector='Healthcare', subsector='2', tickers=['JNJ', 'PFE', 'MRK']),
        # ]
        sector_filter = args.sector.lower() if args.sector else None

        if sector_filter:
            sectors = [g for g in sectors if g.sector.lower() == sector_filter]
            if not sectors:
                console.print(f"[red]❌ No sectors found for sector:[/red] '{args.sector}'")
                return

        for sector in sectors:
            self.fetch_ticker_data(sector.tickers)

    def fetch_ticker_data(self, tickers: list[str]) -> dict:
        data = {}
        fresh_tickers = []

        for ticker in tickers:
            # fetch from DB or Tiingo
            db_df = self.load_price_data_from_sqlite(ticker)
            if db_df is not None and not db_df.empty:
                data[ticker] = db_df
            else:
                fresh_tickers.append(ticker)

        config = {'session': True, 'api_key': Config.get_tiingo_api_key()}
        client = TiingoClient(config)

        if fresh_tickers:
            print(f'⬇️ Downloading data for tickers: {", ".join(fresh_tickers)}')

            for ticker in fresh_tickers:
                try:
                    price_data = client.get_dataframe(
                        ticker,
                        frequency='daily',
                        startDate=(datetime.now() - timedelta(days=120)).strftime('%Y-%m-%d'),
                        endDate=datetime.now().strftime('%Y-%m-%d'),
                        columns='date,adjClose,adjVolume',
                    )
                    price_data.rename(columns={'adjClose': 'Close', 'adjVolume': 'Volume'}, inplace=True)
                    price_data.index = pd.to_datetime(price_data.index)
                    data[ticker] = price_data

                except Exception as e:
                    print(f'⚠️ Error loading data for {ticker} from Tiingo: {e}')

            # Save all at once
            self.save_multiple_price_data_to_sqlite({t: data[t] for t in fresh_tickers if t in data})
        return data

    def save_multiple_price_data_to_sqlite(self, data: dict):
        print(f'saving multiple tickers to db: {Config.get_db_name()}')
        with sqlite3.connect(Config.get_db_name(), timeout=30) as conn:
            c = conn.cursor()
            all_rows = []
            for ticker, df in data.items():
                rows = [
                    (
                        ticker,
                        date.strftime('%Y-%m-%d'),
                        float(row['Close']),
                        int(row['Volume']),
                    )
                    for date, row in df.iterrows()
                ]
                all_rows.extend(rows)
            c.executemany(
                """
                INSERT OR REPLACE INTO price_data (ticker, date, close, volume)
                VALUES (?, ?, ?, ?)
                """,
                all_rows,
            )
            conn.commit()

    def load_price_data_from_sqlite(self, ticker: str) -> Optional[pd.DataFrame]:
        print(f'load_price_data_from_sqlite: ticker: {ticker}')
        with sqlite3.connect(Config.get_db_name()) as conn:
            query = """
            SELECT date, close, volume
            FROM price_data
            WHERE ticker = ?
            ORDER BY date
            """
            df = pd.read_sql_query(
                query,
                conn,
                params=(ticker,),
                parse_dates=['date'],
                index_col='date',
            )

            if df.empty:
                return None

            # Keep only what you need, ensure order & dtypes
            df = df[['close', 'volume']].sort_index()
            # Optional: enforce types
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').astype('Int64')

            return df
