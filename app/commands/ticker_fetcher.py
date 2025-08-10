import argparse
from datetime import datetime, timedelta

import pandas as pd
from rich.console import Console
from tiingo import TiingoClient

from app.config import Config
from app.container import Container
from app.models import PriceData

from .base_command import BaseCommand

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

    def __init__(self, container: Container) -> None:
        super().__init__(self._NAME, self._DESCRIPTION)
        self.sector_repo = container.sector_repo
        self.price_data_repo = container.price_data_repo

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('--sector', help='filter sectors by name (case-insensitive)')

    def handle(self, args: argparse.Namespace) -> None:
        # print(f"args: {args}")

        sectors = self.sector_repo.all()
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

    from datetime import datetime, timedelta

    def fetch_ticker_data(self, tickers: list[str]) -> dict[str, pd.DataFrame]:
        data: dict[str, pd.DataFrame] = {}
        fresh_tickers: list[str] = []

        # 1) Try DB first
        for ticker in tickers:
            rows = list(self.price_data_repo.all_for_ticker(ticker))  # Iterable[PriceData] -> list
            if rows:
                df = (
                    pd.DataFrame([{'date': r.date, 'close': r.close, 'volume': r.volume} for r in rows])
                    .set_index('date')
                    .sort_index()
                )
                data[ticker] = df
            else:
                fresh_tickers.append(ticker)

        # 2) Download any missing tickers from Tiingo
        if fresh_tickers:
            print(f'⬇️ Downloading data for tickers: {", ".join(fresh_tickers)}')

            config = {'session': True, 'api_key': Config.get_tiingo_api_key()}
            client = TiingoClient(config)

            start = (datetime.now() - timedelta(days=120)).strftime('%Y-%m-%d')
            end = datetime.now().strftime('%Y-%m-%d')

            new_rows: list[PriceData] = []

            for ticker in fresh_tickers:
                try:
                    df = client.get_dataframe(
                        ticker,
                        frequency='daily',
                        startDate=start,
                        endDate=end,
                        columns='date,adjClose,adjVolume',
                    )
                    # Normalize column names to match schema
                    df.rename(columns={'adjClose': 'close', 'adjVolume': 'volume'}, inplace=True)
                    df.index = pd.to_datetime(df.index)  # ensure datetime index

                    # Keep only what we store
                    df = df[['close', 'volume']].copy()
                    data[ticker] = df

                    # Prepare rows for bulk insert
                    for dt, row in df.iterrows():
                        new_rows.append(
                            PriceData(
                                ticker=ticker,
                                date=dt.date(),  # datetime.date
                                close=float(row['close']),
                                volume=int(row['volume']),
                            )
                        )

                except Exception as e:
                    print(f'⚠️ Error loading data for {ticker} from Tiingo: {e}')

            # 3) Save all downloaded rows at once
            if new_rows:
                self.price_data_repo.bulk_insert(new_rows)

        return data
