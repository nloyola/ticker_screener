import logging
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd
from pandas import DataFrame
from tiingo import TiingoClient

from app.container import Container
from app.models import PriceData

if TYPE_CHECKING:
    from app.repositories.price_data_repo import PriceDataRepo
    from app.services.sector_service import SectorService


logger = logging.getLogger(__name__)

# logger.setLevel(logging.DEBUG)


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
        self.tiingo_api_key: str = container.tiingo_api_key
        self.sector_service: SectorService = container.sector_service
        self.price_data_repo: PriceDataRepo = container.price_data_repo

        config = {'session': True, 'api_key': self.tiingo_api_key}
        self.client = TiingoClient(config)

    def fetch(self, **kwargs: str) -> None:
        sector_filter = kwargs.get('sector_filter')
        ticker_filter = kwargs.get('ticker_filter')
        sectors = self.sector_service.get_all()

        if ticker_filter:
            subsector = self.sector_service.subsector_for_ticker(ticker_filter)
            self._fetch_for_tickers(subsector.id, [ticker_filter])
            return

        if sector_filter:
            sectors = [g for g in sectors if g.sector.lower() == sector_filter]
            logger.debug(f'fetching sector: {sector_filter}')
            if not sectors:
                logger.error(f"[red]❌ sector not in DB:[/red] '{sector_filter}'")
                return

        for sector in sectors:
            for subsector in sector.subsectors:
                logger.info(f'fetching sector: {sector.sector}, subsector: {subsector.subsector}')
                self._fetch_for_tickers(subsector.id, subsector.tickers)

    def _fetch_for_tickers(self, subsector_id: int, tickers: list[str]) -> None:
        price_data: list[PriceData] = []
        for ticker in tickers:
            new_data = self._fetch_ticker_data(subsector_id, ticker)
            if new_data is not None:
                price_data = price_data + new_data

        # save all downloaded rows at once
        if price_data:
            self.price_data_repo.bulk_insert(price_data)

    def _fetch_ticker_data(self, subsector_id: int, ticker: str) -> list[PriceData]:
        rows = list(self.price_data_repo.all_for_ticker(ticker))
        if rows:
            df = (
                DataFrame([{'date': r.date, 'close': r.close, 'volume': r.volume} for r in rows])
                .set_index('date')
                .sort_index()
            )
            last_local_date: date = pd.to_datetime(df.index.max()).date()
            fetch_start_date: date = last_local_date + timedelta(days=1)

            data_start_date: date = pd.to_datetime(df.index.min()).date()
            self._remove_stale_data(ticker, data_start_date)
        else:
            data_start_date = None
            fetch_start_date = (datetime.now() - timedelta(days=120)).date()

        fetch_end_date = datetime.now().date()
        logging.debug(f'⬇️ downloading data for ticker: {ticker}, dates: {fetch_start_date} to {fetch_end_date}')

        days_between = (fetch_end_date - fetch_start_date).days
        if days_between < 1:
            logging.debug(f'nothing to downlaod for ticker: {ticker}, dates: {fetch_start_date} to {fetch_end_date}')
            return []

        # Download any missing tickers from Tiingo
        new_rows: list[PriceData] = []

        try:
            df = self.client.get_dataframe(
                ticker,
                frequency='daily',
                startDate=fetch_start_date.strftime('%Y-%m-%d'),
                endDate=fetch_end_date.strftime('%Y-%m-%d'),
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

            for dt, row in df.iterrows():
                new_rows.append(
                    PriceData(
                        id=0,
                        subsector_id=subsector_id,
                        ticker=ticker,
                        date=dt.date(),
                        close=float(row['close']),
                        volume=int(row['volume']),
                    )
                )

            return new_rows

        except Exception as e:
            logging.debug(f'⚠️ Error loading data for {ticker} from Tiingo: {e}')

    def _remove_stale_data(self, ticker: str, data_start_date: date) -> None:
        if data_start_date is None:
            return

        date_120d = (datetime.now() - timedelta(days=120)).date()
        stale_data_num_days = (date_120d - data_start_date).days

        if stale_data_num_days > 0:
            logger.info(f'removing stale data for ticker {ticker}, num_days: {stale_data_num_days}')
            self.price_data_repo.delete_before(date_120d)
