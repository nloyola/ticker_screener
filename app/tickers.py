from datetime import datetime, timedelta

import pandas as pd
from pandas import DataFrame
from rich.console import Console
from tiingo import TiingoClient

from app.config import Config
from app.container import Container
from app.models import PriceData

console = Console()


class TickerFetcher:
    """
    Fetches price data for all the sectors in the 'sector' DB table

    The 'price_data' table holds the last 120 days worth of data.

    If the 'price_data' table already has data for the ticker, then only the latest data is fetched.
    E.g. if the latest date is for 5 days ago, then only the last 5 days worth of data is fetched.

    For debugging purposes, a single sector can be requested. See '--sector' option.

    The Tiingo API is used to fetch price data.
    """

    def __init__(self, container: Container) -> None:
        self.sector_service = container.sector_service
        self.price_data_repo = container.price_data_repo

    def fetch(self, sector_filter: str | None) -> None:
        sectors = self.sector_service.get_all()

        if sector_filter:
            sectors = [g for g in sectors if g.sector.lower() == sector_filter]
            if not sectors:
                console.print(f"[red]❌ sector not in DB:[/red] '{sector_filter}'")
                return

        for sector in sectors:
            for subsector in sector.subsectors:
                self._fetch_ticker_data(subsector.id, subsector.tickers)

    def _fetch_ticker_data(self, subsector_id, tickers: list[str]) -> dict[str, DataFrame]:
        data: dict[str, DataFrame] = {}
        fresh_tickers: list[str] = []

        # 1) Try DB first
        for ticker in tickers:
            rows = list(self.price_data_repo.all_for_ticker(ticker))
            if rows:
                df = (
                    DataFrame([{'date': r.date, 'close': r.close, 'volume': r.volume} for r in rows])
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
                    df.rename(
                        columns={'adjClose': 'close', 'adjVolume': 'volume'},
                        inplace=True,
                    )
                    df.index = pd.to_datetime(df.index)  # ensure datetime index

                    # Keep only what we store
                    df = df[['close', 'volume']].copy()
                    data[ticker] = df

                    # Prepare rows for bulk insert
                    for dt, row in df.iterrows():
                        new_rows.append(
                            PriceData(
                                id=0,
                                subsector_id=subsector_id,
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
