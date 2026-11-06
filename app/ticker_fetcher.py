import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING

import pandas as pd
import requests
from tiingo import TiingoClient
from tiingo.restclient import RestClientError

from app.container import Container
from app.models import PriceData
from app.progress import report_progress

if TYPE_CHECKING:
    from app.repositories.price_data_repo import PriceDataRepo
    from app.services.sector_service import SectorService


logger = logging.getLogger(__name__)

# the days of prices 'price_data' keeps, and each run downloads, per ticker
HISTORY_DAYS = 120

# a price request that meets a transient error (see 'is_transient') is tried up to RETRY_ATTEMPTS times, waiting
# about RETRY_BASE_DELAY seconds before the first retry and twice as long before each later one
RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY = 2.0
# the longest wait, whatever Tiingo's Retry-After asks for, so a spent hourly limit cannot stall a run
RETRY_MAX_DELAY = 60.0


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


def _response(error: Exception) -> requests.Response | None:
    """The HTTP response behind an error TiingoClient raised, if any: it wraps requests' HTTPError."""
    cause = error.args[0] if isinstance(error, RestClientError) and error.args else error
    return getattr(cause, 'response', None)


def is_transient(error: Exception) -> bool:
    """True for errors worth retrying: a 429, a 5xx, or a connection that failed or timed out."""
    if isinstance(error, requests.ConnectionError | requests.Timeout):
        return True
    response = _response(error)
    return response is not None and (response.status_code == 429 or response.status_code >= 500)


def retry_delay(error: Exception, attempt: int, rand: Callable[[], float] = random.random) -> float:
    """
    Seconds to wait before retrying after 'attempt' (1 for the first) failed with 'error'.

    Honours a Retry-After given in seconds, up to RETRY_MAX_DELAY. Otherwise waits RETRY_BASE_DELAY doubled for each
    earlier attempt, times a random factor between 0.5 and 1.5, so concurrent retries do not line up.
    """
    response = _response(error)
    retry_after = response.headers.get('Retry-After') if response is not None else None
    if retry_after is not None:
        try:
            return min(max(float(retry_after), 0.0), RETRY_MAX_DELAY)
        except ValueError:
            pass  # an HTTP date, which Tiingo does not send: use the backoff
    return min(RETRY_BASE_DELAY * 2 ** (attempt - 1) * (0.5 + rand()), RETRY_MAX_DELAY)


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

    A request that meets a transient error (a 429, a 5xx, a failed connection) is retried up to RETRY_ATTEMPTS times
    in all, with exponential backoff and jitter, honouring Tiingo's Retry-After. Any other error, such as a 404 for an
    unknown ticker, fails at once. A ticker whose download fails after its last attempt is skipped. With 'strict' each
    failure is logged as a warning; either way 'fetch' returns the number of tickers attempted and failed.

    A ticker whose download holds no trading day in the whole window is logged as a warning: it is likely delisted,
    and can be removed at the dashboard's /admin/sectors.

    The Tiingo API is used to fetch price data.
    """

    def __init__(self, container: Container) -> None:
        self.tiingo_api_key: str = container.tiingo_api_key
        self.sector_service: SectorService = container.sector_service
        self.price_data_repo: PriceDataRepo = container.price_data_repo

        config = {'session': True, 'api_key': self.tiingo_api_key}
        self.client = TiingoClient(config)
        self.sleep: Callable[[float], None] = time.sleep
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
                logger.warning(
                    f'no prices in the last {HISTORY_DAYS} days, likely delisted, stored rows kept: {ticker}'
                )

    def _fetch_ticker_data(self, subsector_id: int, ticker: str) -> list[PriceData] | None:
        """Returns the ticker's rows for the whole window, or None when the download fails."""
        fetch_end_date = date.today()
        fetch_start_date = window_start(fetch_end_date)
        logger.debug(f'⬇️ downloading data for ticker: {ticker}, dates: {fetch_start_date} to {fetch_end_date}')

        try:
            return [
                PriceData(
                    id=0,
                    subsector_id=subsector_id,
                    ticker=ticker,
                    date=pd.to_datetime(price['date']).date(),
                    close=float(price['adjClose']),
                    volume=int(price['adjVolume']),
                )
                for price in self._get_prices(ticker, fetch_start_date, fetch_end_date)
            ]
        except Exception as e:
            # a failed request, or a row that is not a price
            level = logging.WARNING if self.strict else logging.DEBUG
            logger.log(level, f'⚠️ Error loading data for {ticker} from Tiingo: {self._redact(e)}')
            return None

    def _get_prices(self, ticker: str, start: date, end: date) -> list[dict]:
        """Requests the ticker's prices, retrying transient errors; raises the last error once the attempts run out."""
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                # the JSON response is an empty list when the range holds no trading day
                return self.client.get_ticker_price(
                    ticker,
                    frequency='daily',
                    startDate=start.strftime('%Y-%m-%d'),
                    endDate=end.strftime('%Y-%m-%d'),
                    columns='date,adjClose,adjVolume',
                )
            except Exception as e:
                if attempt == RETRY_ATTEMPTS or not is_transient(e):
                    raise
                delay = retry_delay(e, attempt)
                logger.info(
                    f'Tiingo request for {ticker} failed (attempt {attempt} of {RETRY_ATTEMPTS}), '
                    f'retrying in {delay:.1f} s: {self._redact(e)}'
                )
                self.sleep(delay)
        raise AssertionError('unreachable')

    def _redact(self, error: Exception) -> str:
        # the client sends the key in a header, but never let it reach the log
        return str(error).replace(self.tiingo_api_key, '***') if self.tiingo_api_key else str(error)
