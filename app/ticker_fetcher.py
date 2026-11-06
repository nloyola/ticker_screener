import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING

import pandas as pd
from tiingo import TiingoClient

from app.container import Container
from app.models import PriceData
from app.progress import report_progress

if TYPE_CHECKING:
    from app.repositories.price_data_repo import PriceDataRepo
    from app.services.sector_service import SectorService


logger = logging.getLogger(__name__)

# the days of prices 'price_data' keeps, and each run downloads, per ticker
HISTORY_DAYS = 120


@dataclass
class FetchStats:
    attempted: int = 0
    failed: int = 0

    @property
    def failure_rate(self) -> float:
        return self.failed / self.attempted if self.attempted else 0.0


def window_start(today: date) -> date:
    """The first day of the window 'price_data' keeps, ending today."""
    return today - timedelta(days=HISTORY_DAYS)


class TickerFetcher:
    """
    Fetches price data for all the sectors in the 'sector' DB table

    The 'price_data' table holds the last HISTORY_DAYS days of data. Every run downloads each ticker's whole window
    and replaces the ticker's stored rows with it, because Tiingo restates the adjusted closes of past days after a
    split or dividend: appending only the new days would leave old and new rows on different bases. Tiingo counts one
    request per ticker however long the range is, so the full window costs no more requests.

    A ticker whose download fails keeps its stored rows, and so does one whose download holds no trading day. A
    ticker listed in more than one subsector is downloaded once per run, for the first subsector that lists it.

    'fetch' never deletes the rows of a ticker it could not download. 'prune' deletes every ticker's rows older than
    the window; call it once per run, and only after a fetch that succeeded (run-refresh, 'ticker-fetcher --strict'),
    so a fetch that fails for every ticker leaves 'price_data' as it was.

    For debugging purposes, a single sector can be requested. See '--sector' option.

    Prints a 'PROGRESS stage=fetch subsector=<done>/<total>' line before the first subsector and after each one,
    where the total is the number of subsectors fetched (see app/progress.py).

    A ticker whose download fails is skipped. With 'strict' each failure is logged as a warning;
    either way 'fetch' returns the number of tickers attempted and failed.

    The Tiingo API is used to fetch price data.
    """

    def __init__(self, container: Container) -> None:
        self.tiingo_api_key: str = container.tiingo_api_key
        self.sector_service: SectorService = container.sector_service
        self.price_data_repo: PriceDataRepo = container.price_data_repo

        config = {'session': True, 'api_key': self.tiingo_api_key}
        self.client = TiingoClient(config)
        self.strict = False
        self.stats = FetchStats()
        self._fetched: set[str] = set()

    def fetch(
        self, sector_filter: str | None = None, ticker_filter: str | None = None, strict: bool = False
    ) -> FetchStats:
        self.strict = strict
        self.stats = FetchStats()
        self._fetched = set()
        sectors = self.sector_service.get_all()

        if ticker_filter:
            subsector = self.sector_service.subsector_for_ticker(ticker_filter)
            if subsector is None:
                logger.error(f"ticker not in any subsector: '{ticker_filter}'")
                return self.stats
            report_progress('fetch', subsector='0/1')
            self._fetch_for_tickers(subsector.id, [ticker_filter])
            report_progress('fetch', subsector='1/1')
            return self.stats

        if sector_filter:
            sectors = [g for g in sectors if g.sector.lower() == sector_filter]
            logger.debug(f'fetching sector: {sector_filter}')
            if not sectors:
                logger.error(f"[red]❌ sector not in DB:[/red] '{sector_filter}'")
                return self.stats

        subsectors = [(sector, subsector) for sector in sectors for subsector in sector.subsectors]
        total = len(subsectors)
        report_progress('fetch', subsector=f'0/{total}')
        for done, (sector, subsector) in enumerate(subsectors, start=1):
            logger.info(f'fetching sector: {sector.sector}, subsector: {subsector.subsector}')
            self._fetch_for_tickers(subsector.id, subsector.tickers)
            report_progress('fetch', subsector=f'{done}/{total}')

        return self.stats

    def prune(self) -> int:
        """Deletes every ticker's rows dated before the window and returns how many were deleted."""
        cutoff = window_start(date.today())
        deleted = self.price_data_repo.delete_before(cutoff)
        logger.info(f'removed {deleted} price rows dated before {cutoff}')
        return deleted

    def _fetch_for_tickers(self, subsector_id: int, tickers: list[str]) -> None:
        for ticker in tickers:
            if ticker in self._fetched:
                logger.debug(f'already fetched in this run: {ticker}')
                continue
            self._fetched.add(ticker)
            self.stats.attempted += 1
            rows = self._fetch_ticker_data(subsector_id, ticker)
            if rows is None:
                self.stats.failed += 1
            elif rows:
                self.price_data_repo.replace_for_ticker(ticker, rows)
            else:
                logger.debug(f'no prices in the last {HISTORY_DAYS} days, stored rows kept: {ticker}')

    def _fetch_ticker_data(self, subsector_id: int, ticker: str) -> list[PriceData] | None:
        """Returns the ticker's rows for the whole window, or None when the download fails."""
        fetch_end_date = date.today()
        fetch_start_date = window_start(fetch_end_date)
        logger.debug(f'⬇️ downloading data for ticker: {ticker}, dates: {fetch_start_date} to {fetch_end_date}')

        try:
            # the JSON response is an empty list when the range holds no trading day
            prices = self.client.get_ticker_price(
                ticker,
                frequency='daily',
                startDate=fetch_start_date.strftime('%Y-%m-%d'),
                endDate=fetch_end_date.strftime('%Y-%m-%d'),
                columns='date,adjClose,adjVolume',
            )
            return [
                PriceData(
                    id=0,
                    subsector_id=subsector_id,
                    ticker=ticker,
                    date=pd.to_datetime(price['date']).date(),
                    close=float(price['adjClose']),
                    volume=int(price['adjVolume']),
                )
                for price in prices
            ]

        except Exception as e:
            # the client sends the key in a header, but never let it reach the log
            error = str(e).replace(self.tiingo_api_key, '***') if self.tiingo_api_key else str(e)
            level = logging.WARNING if self.strict else logging.DEBUG
            logger.log(level, f'⚠️ Error loading data for {ticker} from Tiingo: {error}')
            return None
