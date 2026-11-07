import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING

import pandas as pd
import requests

from app.container import Container
from app.models import PriceData
from app.progress import report_progress
from app.services.sector_service import normalize_ticker

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

TIINGO_PRICES_URL = 'https://api.tiingo.com/tiingo/daily/{ticker}/prices'
# seconds to wait for Tiingo to accept the connection, and then for each read of its answer, so a server that stops
# answering fails the attempt (as a requests.Timeout, which is retried) instead of hanging the run
CONNECT_TIMEOUT = 10.0
READ_TIMEOUT = 30.0
# the most characters of an error answer's body quoted in a log line
ERROR_DETAIL_LIMIT = 200


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


class TiingoPrices:
    """Requests a ticker's daily prices from Tiingo's end-of-day endpoint, sending the key in a header."""

    def __init__(self, api_key: str) -> None:
        self.session = requests.Session()
        self.session.headers['Authorization'] = f'Token {api_key}'

    def daily_prices(self, ticker: str, start: date, end: date) -> list[dict]:
        """
        The ticker's rows from 'start' to 'end', each with 'date', 'adjClose' and 'adjVolume'; an empty list when the
        range holds no trading day. Raises requests' HTTPError for an error status and requests.Timeout when Tiingo
        stops answering.
        """
        response = self.session.get(
            TIINGO_PRICES_URL.format(ticker=ticker),
            params={
                'startDate': start.isoformat(),
                'endDate': end.isoformat(),
                'resampleFreq': 'daily',
                'columns': 'date,adjClose,adjVolume',
            },
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        )
        response.raise_for_status()
        return response.json()


def _response(error: Exception) -> requests.Response | None:
    """The HTTP response behind an error, if any: requests' HTTPError carries it."""
    return getattr(error, 'response', None)


def error_detail(error: Exception) -> str:
    """
    The error, followed by what Tiingo said in the body of an error answer (its JSON 'detail', or else the text),
    shortened to ERROR_DETAIL_LIMIT characters.
    """
    response = _response(error)
    if response is None:
        return str(error)
    try:
        body = response.json()
        detail = body.get('detail') if isinstance(body, dict) else None
    except ValueError:
        detail = None
    detail = ' '.join(str(detail if detail is not None else response.text).split())
    if len(detail) > ERROR_DETAIL_LIMIT:
        detail = detail[: ERROR_DETAIL_LIMIT - 1] + '…'
    return f'{error}: {detail}' if detail else str(error)


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
    unknown ticker, fails at once. An attempt that Tiingo stops answering times out (CONNECT_TIMEOUT, READ_TIMEOUT) and
    is retried too. A ticker whose download fails after its last attempt is skipped. With 'strict' each
    failure is logged as a warning; either way 'fetch' returns the number of tickers attempted and failed.

    A ticker whose download holds no trading day in the whole window is logged as a warning: it is likely delisted,
    and can be removed at the dashboard's /admin/sectors.

    Prices come from Tiingo's end-of-day endpoint (see TiingoPrices).
    """

    def __init__(self, container: Container) -> None:
        self.tiingo_api_key: str = container.tiingo_api_key
        self.sector_service: SectorService = container.sector_service
        self.price_data_repo: PriceDataRepo = container.price_data_repo

        self.client = TiingoPrices(self.tiingo_api_key)
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

        if ticker_filter:
            ticker = normalize_ticker(ticker_filter)
            subsector = self.sector_service.subsector_for_ticker(ticker)
            if subsector is None:
                logger.error(f"ticker not in any subsector: '{ticker}'")
                return self.stats
            report_progress('fetch', subsector='0/1')
            self._fetch_for_tickers(subsector.id, [ticker])
            report_progress('fetch', subsector='1/1')
            return self.stats

        sectors = self.sector_service.get_all()

        if sector_filter:
            sectors = [g for g in sectors if g.sector.lower() == sector_filter]
            logger.debug(f'fetching sector: {sector_filter}')
            if not sectors:
                logger.error(f"sector not in the database: '{sector_filter}'")
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
                return self.client.daily_prices(ticker, start, end)
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
        # the key goes in a header, never in the URL, but never let it reach the log, even quoted back in a body
        text = error_detail(error)
        return text.replace(self.tiingo_api_key, '***') if self.tiingo_api_key else text
