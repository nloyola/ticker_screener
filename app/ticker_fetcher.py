import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING

from app.container import Container
from app.markets import UnsupportedTicker, market_of
from app.models import PriceData
from app.price_apis import PriceApi
from app.price_apis.http import RETRY_ATTEMPTS, error_detail, is_transient, response_of, retry_delay
from app.progress import NO_PRICES, NOT_FOUND, report_flagged, report_progress
from app.redaction import redact
from app.services.sector_service import normalize_ticker

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
    # tickers of a market the price API does not cover: neither attempted nor failed, their stored rows kept
    skipped: int = 0
    # the name of the price API the fetch used
    api: str = ''

    @property
    def failure_rate(self) -> float:
        return self.failed / self.attempted if self.attempted else 0.0


def window_start(today: date) -> date:
    """The first day of the window 'price_data' keeps, ending today."""
    return today - timedelta(days=HISTORY_DAYS)


@dataclass
class _Batch:
    """The tickers one subsector contributes to a fetch, after skipping those fetched already or not covered."""

    label: str | None
    subsector_id: int
    tickers: list[str]


class TickerFetcher:
    """
    Fetches price data for all the sectors in the 'sector' DB table, from the price API the PRICE_API setting picks
    (container.price_api_client, see app/price_apis).

    The 'price_data' table holds the last HISTORY_DAYS days of data. Every run downloads each ticker's whole window
    and replaces the ticker's stored rows with it, because the APIs restate the adjusted closes of past days after a
    split or dividend: appending only the new days would leave old and new rows on different bases. It also means a
    ticker never holds a mix of two APIs' rows after a switch. Both APIs count one request per ticker however long the
    range is, so the full window costs no more requests.

    A ticker of a market the API does not cover (a Canadian ticker under Tiingo, see app/markets.py) is skipped: it
    keeps its stored rows, and counts as neither attempted nor failed, so it cannot push a run over its failure limit.
    A ticker with an exchange suffix no market has fails without a request.

    A ticker whose download fails keeps its stored rows, and so does one whose download holds no trading day. A
    ticker listed in more than one subsector is downloaded once per run, for the first subsector that lists it.

    Before the first price request, the API checks it can serve the run (PriceApi.check_quota), and raises
    PriceApiStop when it cannot: then nothing is fetched.

    'fetch' never deletes the rows of a ticker it could not download. 'prune' deletes the rows older than the window of
    every ticker in a market the API covers; call it once per run, and only after a fetch that succeeded (run-refresh,
    'ticker-fetcher --strict'), so a fetch that fails for every ticker leaves 'price_data' as it was. A market the API
    does not cover keeps all its rows, so a Tiingo run never ages out Canadian prices it did not replace.

    For debugging purposes, a single sector can be requested. See '--sector' option.

    Prints a 'PROGRESS stage=fetch subsector=<done>/<total>' line before the first subsector and after each one,
    where the total is the number of subsectors fetched (see app/progress.py).

    A request that meets a transient error (a 429, a 5xx, a failed connection) is retried up to RETRY_ATTEMPTS times
    in all, with exponential backoff and jitter, honouring the API's Retry-After. Any other error, such as a 404 for
    an unknown ticker, fails at once. An attempt that the API stops answering times out (CONNECT_TIMEOUT,
    READ_TIMEOUT in app/price_apis/http.py) and is retried too. A ticker whose download fails after its last attempt
    is given up. With 'strict' each failure is logged as a warning; either way 'fetch' returns the number of tickers
    attempted, failed and skipped.

    A ticker whose download holds no trading day in the whole window is logged as a warning: it is likely delisted,
    and can be removed at the dashboard's /admin/sectors.

    A ticker the API answers with a 404, and one whose download holds no trading day, is also reported with a
    'FLAGGED ticker=<ticker> reason=not_found|no_prices [last=<date>]' line after its warning, where 'last' is the
    newest price date stored for it before this run, read now because a later 'prune' may delete every one of its
    rows (see app/progress.py). 'flagged' holds the tickers reported, so analyze does not report them again as stale.
    """

    def __init__(self, container: Container) -> None:
        self.api: PriceApi = container.price_api_client
        self.sector_service: SectorService = container.sector_service
        self.price_data_repo: PriceDataRepo = container.price_data_repo

        self.sleep: Callable[[float], None] = time.sleep
        self.strict = False
        self.stats = FetchStats()
        self._fetched: set[str] = set()
        self.flagged: set[str] = set()

    def fetch(
        self, sector_filter: str | None = None, ticker_filter: str | None = None, strict: bool = False
    ) -> FetchStats:
        """Fetches every listed ticker's window; raises PriceApiStop, before any price request, if the API cannot."""
        self.strict = strict
        self.stats = FetchStats(api=self.api.name)
        self._fetched = set()
        self.flagged = set()
        logger.info(f'price API: {self.api.name}')

        batches = self._batches(sector_filter, ticker_filter)
        if batches is None:
            return self.stats

        needed = sum(len(batch.tickers) for batch in batches)
        if needed:
            self.api.check_quota(needed)

        total = len(batches)
        report_progress('fetch', subsector=f'0/{total}')
        for done, batch in enumerate(batches, start=1):
            if batch.label:
                logger.info(batch.label)
            self._fetch_for_tickers(batch.subsector_id, batch.tickers)
            report_progress('fetch', subsector=f'{done}/{total}')

        if self.stats.skipped:
            logger.info(f'skipped {self.stats.skipped} tickers not covered by {self.api.name}, stored rows kept')
        return self.stats

    def prune(self) -> int:
        """
        Deletes the rows dated before the window of every ticker in the API's markets, and returns how many were
        deleted.
        """
        cutoff = window_start(date.today())
        markets = sorted(self.api.markets)
        deleted = self.price_data_repo.delete_before(cutoff, markets)
        logger.info(f'removed {deleted} price rows dated before {cutoff} ({", ".join(markets)})')
        return deleted

    def _batches(self, sector_filter: str | None, ticker_filter: str | None) -> list[_Batch] | None:
        """The tickers to fetch, subsector by subsector, or None when the filter matches nothing."""
        if ticker_filter:
            ticker = normalize_ticker(ticker_filter)
            subsector = self.sector_service.subsector_for_ticker(ticker)
            if subsector is None:
                logger.error(f"ticker not in any subsector: '{ticker}'")
                return None
            return [_Batch(None, subsector.id, self._to_fetch([ticker]))]

        sectors = self.sector_service.get_all()

        if sector_filter:
            sectors = [g for g in sectors if g.sector.lower() == sector_filter]
            logger.debug(f'fetching sector: {sector_filter}')
            if not sectors:
                logger.error(f"sector not in the database: '{sector_filter}'")
                return None

        return [
            _Batch(
                f'fetching sector: {sector.sector}, subsector: {subsector.subsector}',
                subsector.id,
                self._to_fetch(subsector.tickers),
            )
            for sector in sectors
            for subsector in sector.subsectors
        ]

    def _to_fetch(self, tickers: list[str]) -> list[str]:
        """
        The tickers to request, leaving out those already in this run and counting those the API does not cover as
        skipped. A ticker with an unsupported suffix counts as attempted and failed, without a request.
        """
        wanted = []
        for ticker in tickers:
            if ticker in self._fetched:
                logger.debug(f'already fetched in this run: {ticker}')
                continue
            self._fetched.add(ticker)
            try:
                market = market_of(ticker)
            except UnsupportedTicker as e:
                self.stats.attempted += 1
                self.stats.failed += 1
                logger.warning(f'⚠️ not fetched from {self.api.name}: {e}')
                continue
            if market not in self.api.markets:
                self.stats.skipped += 1
                logger.debug(f'not covered by {self.api.name} ({market}), stored rows kept: {ticker}')
                continue
            wanted.append(ticker)
        return wanted

    def _fetch_for_tickers(self, subsector_id: int, tickers: list[str]) -> None:
        for ticker in tickers:
            self.stats.attempted += 1
            rows = self._fetch_ticker_data(subsector_id, ticker)
            if rows is None:
                self.stats.failed += 1
            elif rows:
                self.price_data_repo.replace_for_ticker(ticker, rows)
            else:
                logger.warning(
                    f'no prices from {self.api.name} in the last {HISTORY_DAYS} days, likely delisted, '
                    f'stored rows kept: {ticker}'
                )
                self._flag(ticker, NO_PRICES)

    def _flag(self, ticker: str, reason: str) -> None:
        """Prints the ticker's FLAGGED line, with the newest price date stored for it, if any."""
        latest = self.price_data_repo.latest_for_ticker(ticker)
        report_flagged(ticker, reason, latest.date if latest else None)
        self.flagged.add(ticker)

    def _fetch_ticker_data(self, subsector_id: int, ticker: str) -> list[PriceData] | None:
        """Returns the ticker's rows for the whole window, or None when the download fails."""
        fetch_end_date = date.today()
        fetch_start_date = window_start(fetch_end_date)
        logger.debug(
            f'⬇️ downloading data from {self.api.name} for ticker: {ticker}, '
            f'dates: {fetch_start_date} to {fetch_end_date}'
        )

        try:
            return [
                PriceData(
                    id=0,
                    subsector_id=subsector_id,
                    ticker=ticker,
                    date=price.date,
                    close=price.close,
                    volume=price.volume,
                )
                for price in self._get_prices(ticker, fetch_start_date, fetch_end_date)
            ]
        except Exception as e:
            # a failed request, or a row that is not a price
            level = logging.WARNING if self.strict else logging.DEBUG
            logger.log(level, f'⚠️ Error loading data for {ticker} from {self.api.name}: {self._redact(e)}')
            response = response_of(e)
            if response is not None and response.status_code == 404:
                self._flag(ticker, NOT_FOUND)
            return None

    def _get_prices(self, ticker: str, start: date, end: date):
        """Requests the ticker's prices, retrying transient errors; raises the last error once the attempts run out."""
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                return self.api.daily_prices(ticker, start, end)
            except Exception as e:
                if attempt == RETRY_ATTEMPTS or not is_transient(e):
                    raise
                delay = retry_delay(e, attempt)
                logger.info(
                    f'{self.api.name} request for {ticker} failed (attempt {attempt} of {RETRY_ATTEMPTS}), '
                    f'retrying in {delay:.1f} s: {self._redact(e)}'
                )
                self.sleep(delay)
        raise AssertionError('unreachable')

    def _redact(self, error: Exception) -> str:
        # never let the key reach the log: in EODHD's URLs, or quoted back in a body
        return redact(error_detail(error), self.api.api_key)
