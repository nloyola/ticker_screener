import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import TYPE_CHECKING

from app.container import Container
from app.markets import BENCHMARKS, MARKETS, UnsupportedTicker, market_of
from app.price_apis import DailyPrice, PriceApi
from app.price_apis.http import error_detail, response_of, with_retries
from app.redaction import redact
from app.repositories.history_repo import BENCHMARK, TICKER, Coverage

if TYPE_CHECKING:
    from app.repositories.history_repo import HistoryRepo
    from app.services.sector_service import SectorService

logger = logging.getLogger(__name__)

# The years of daily prices the history database holds per ticker and benchmark, ending on the day of the fetch.
HISTORY_YEARS = 10
# A history that starts more than this after the requested start is short: the API has fewer years of the ticker,
# usually because it listed later. The slack covers a start that falls on a weekend or a run of holidays.
SHORT_HISTORY_SLACK = timedelta(days=14)


def history_start(today: date) -> date:
    """The first day the history asks for: HISTORY_YEARS before 'today' (28 February for a 29th)."""
    try:
        return today.replace(year=today.year - HISTORY_YEARS)
    except ValueError:
        return today.replace(year=today.year - HISTORY_YEARS, day=28)


@dataclass
class HistoryReport:
    """What a history fetch requested and what the history holds afterwards for the tickers it was asked for."""

    api: str = ''
    start: date | None = None
    end: date | None = None
    # the requests made, and those skipped because the ticker or benchmark was fetched today already
    requests: int = 0
    done_today: int = 0
    # tickers of a market the API does not cover, and tickers with an exchange suffix no market has: never requested
    not_covered: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    # tickers and benchmarks whose request failed, after retries: fetched again by the next run
    failed: list[str] = field(default_factory=list)
    # tickers and benchmarks with no stored history, and those whose history starts late, with its first day
    no_history: list[str] = field(default_factory=list)
    short_history: dict[str, date] = field(default_factory=dict)
    # how many tickers and benchmarks have stored history
    tickers_with_history: int = 0
    benchmarks_with_history: int = 0


@dataclass
class HistoryPlan:
    """The tickers and benchmarks a fetch covers, those it still has to request today, and its report so far."""

    report: HistoryReport
    tickers: list[str] = field(default_factory=list)
    benchmarks: list[str] = field(default_factory=list)
    # (kind, ticker) pairs, tickers first: history_repo's TICKER or BENCHMARK
    todo: list[tuple[str, str]] = field(default_factory=list)


class HistoryFetch:
    """
    Fetches the full daily history, HISTORY_YEARS long, of every ticker the sectors page lists and of the benchmark of
    every market the price API covers (BENCHMARKS in app/markets.py), from the price API the PRICE_API setting picks,
    into the history database (app/repositories/history_repo.py). It never writes the live database.

    One request per ticker and benchmark on either API, however long the range: the count is printed, and checked
    against the API's quota (PriceApi.check_quota), before the first request. The prices are those the fetcher stores
    (app/price_apis): the adjusted close, with the open, high and low on its basis, so a split leaves no jump.

    Each ticker's prices replace its stored ones, and the ticker is recorded as fetched today from the API, in one
    transaction. Run again the same day, a fetch skips the tickers and benchmarks already fetched today from the same
    API, so an interrupted fetch resumes where it stopped, and a finished one requests nothing.

    A request that meets a transient error is retried as the fetcher's are (app/price_apis/http.py); one that still
    fails is reported, keeps the stored prices and is not recorded, so the next run requests it again. A 404 or an
    answer with no trading day keeps the stored prices and is recorded as fetched: the API has nothing for it today.

    A ticker of a market the API does not cover (a Canadian one under Tiingo) and a ticker with an exchange suffix no
    market has are never requested, and are reported. After the requests, the report lists the tickers and benchmarks
    with no stored history, and those whose history starts more than SHORT_HISTORY_SLACK after the requested start.
    """

    def __init__(self, container: Container) -> None:
        self.api: PriceApi = container.price_api_client
        self.sector_service: SectorService = container.sector_service
        self.history_repo: HistoryRepo = container.history_repo
        self.sleep: Callable[[float], None] = time.sleep
        self.today: Callable[[], date] = date.today

    def plan(self) -> HistoryPlan:
        """What a fetch started now would request, and what it leaves out."""
        today = self.today()
        plan = HistoryPlan(report=HistoryReport(api=self.api.name, start=history_start(today), end=today))
        report = plan.report

        for ticker in self._universe():
            try:
                market = market_of(ticker)
            except UnsupportedTicker:
                report.unsupported.append(ticker)
                continue
            if market not in self.api.markets:
                report.not_covered.append(ticker)
                continue
            plan.tickers.append(ticker)
        plan.benchmarks = [BENCHMARKS[market] for market in MARKETS if market in self.api.markets]

        done = {kind: self.history_repo.fetched_on(kind, today, self.api.setting) for kind in (TICKER, BENCHMARK)}
        plan.todo = [(TICKER, t) for t in plan.tickers if t not in done[TICKER]] + [
            (BENCHMARK, b) for b in plan.benchmarks if b not in done[BENCHMARK]
        ]
        report.requests = len(plan.todo)
        report.done_today = len(plan.tickers) + len(plan.benchmarks) - report.requests
        return plan

    def run(self, announce: Callable[[HistoryPlan], None]) -> HistoryReport:
        """
        Fetches what is left to fetch today and returns the report; 'announce' is called with the plan before the
        quota check and the first request. Raises PriceApiStop, before any request, when the API cannot serve them.
        """
        plan = self.plan()
        report = plan.report
        announce(plan)
        if report.requests:
            self.api.check_quota(report.requests)

        for done, (kind, ticker) in enumerate(plan.todo, start=1):
            if not self._fetch(kind, ticker, report):
                report.failed.append(ticker)
            logger.info(f'history {done}/{report.requests}: {ticker}')

        self._report_coverage(report, {TICKER: plan.tickers, BENCHMARK: plan.benchmarks})
        return report

    def _fetch(self, kind: str, ticker: str, report: HistoryReport) -> bool:
        """Fetches and stores one ticker's history; False when the request failed."""
        assert report.start is not None and report.end is not None
        start, end = report.start, report.end
        try:
            prices: list[DailyPrice] = with_retries(
                lambda: self.api.daily_prices(ticker, start, end),
                ticker=ticker,
                api=self.api,
                sleep=self.sleep,
            )
        except Exception as e:
            response = response_of(e)
            if response is not None and response.status_code == 404:
                logger.warning(f'{self.api.name} does not know {ticker}, stored history kept')
                self.history_repo.mark_fetched(kind, ticker, end, self.api.setting)
                return True
            logger.warning(
                f'⚠️ history of {ticker} not downloaded from {self.api.name}: '
                f'{redact(error_detail(e), self.api.api_key)}'
            )
            return False
        if not prices:
            logger.warning(f'no prices for {ticker} from {self.api.name} since {start}, stored history kept')
            self.history_repo.mark_fetched(kind, ticker, end, self.api.setting)
            return True
        self.history_repo.store(kind, ticker, prices, end, self.api.setting)
        return True

    def _report_coverage(self, report: HistoryReport, names: dict[str, list[str]]) -> None:
        """Fills in the report's history counts, and its tickers and benchmarks with no or short history."""
        assert report.start is not None
        late = report.start + SHORT_HISTORY_SLACK
        for kind, wanted in names.items():
            coverage: dict[str, Coverage] = self.history_repo.coverage(kind)
            for name in wanted:
                stored = coverage.get(name)
                if stored is None:
                    report.no_history.append(name)
                    continue
                if kind == TICKER:
                    report.tickers_with_history += 1
                else:
                    report.benchmarks_with_history += 1
                if stored.first > late:
                    report.short_history[name] = stored.first

    def _universe(self) -> list[str]:
        """Every ticker the sectors page lists, once each, in the order of its first listing."""
        seen: dict[str, None] = {}
        for sector in self.sector_service.get_all():
            for subsector in sector.subsectors:
                for ticker in subsector.tickers:
                    seen.setdefault(ticker, None)
        return list(seen)
