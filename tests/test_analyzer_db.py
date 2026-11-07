"""
Which tickers analyze scores, against a real database: a stale or frozen ticker keeps its previous stock_criteria
row, date_created included, freshness is judged per market, and run-refresh scores only the markets it fetched.
"""

import argparse
import sqlite3
from dataclasses import replace
from datetime import date, timedelta
from types import SimpleNamespace

import pytest
import requests

from app.analyzer import Analyzer
from app.commands.run_refresh_cmd import RunRefreshCmd
from app.db import init_db
from app.markets import US
from app.models import PriceData
from app.price_apis import DailyPrice
from app.repositories.price_data_repo import PriceDataRepo
from app.repositories.stock_criteria_repo import StockCriteriaRepo
from app.services.sector_service import SectorAggregate, SubsectorWithTickers
from app.ticker_fetcher import TickerFetcher, window_start

TODAY = date.today()
LAST_RUN = TODAY - timedelta(days=3)


def price_rows(ticker: str, last_day: date, n: int = 100) -> list[PriceData]:
    """n rising closes ending on last_day, with a volume spike on the last day."""
    return [
        PriceData(
            id=0,
            subsector_id=1,
            ticker=ticker,
            date=last_day - timedelta(days=n - 1 - i),
            close=100.0 + i,
            volume=2000 if i == n - 1 else 1000,
        )
        for i in range(n)
    ]


def criteria(path: str) -> dict[str, tuple[float, str]]:
    with sqlite3.connect(path) as con:
        rows = con.execute('SELECT ticker, price, date_created FROM stock_criteria').fetchall()
    return {ticker: (price, date_created) for ticker, price, date_created in rows}


@pytest.fixture
def path(tmp_path) -> str:
    path = str(tmp_path / 'stock_data.db')
    init_db(path)
    return path


@pytest.fixture
def container(path) -> SimpleNamespace:
    container = SimpleNamespace(db_path=path)
    container.price_data_repo = PriceDataRepo(container)
    container.stock_criteria_repo = StockCriteriaRepo(container)
    container.analyzer = Analyzer(container)
    return container


def score_all(analyzer: Analyzer, date_created: date) -> None:
    """Scores every stored ticker as a run on date_created did."""
    analyzer.save_to_repository(
        [analyzer.analyze_ticker(t, g) for t, g in analyzer.load_price_data().groupby(level='ticker')], date_created
    )


def summary_lines(out: str) -> list[str]:
    return [line for line in out.splitlines() if line.startswith(('US:', 'Canada:'))]


def flagged_lines(out: str) -> list[str]:
    return [line for line in out.splitlines() if line.startswith('FLAGGED')]


def test_stale_ticker_keeps_its_previous_row_and_date(path, container, capsys):
    analyzer = container.analyzer

    # the last run scored both tickers from rows ending yesterday
    container.price_data_repo.bulk_insert(price_rows('AAA', TODAY - timedelta(days=1)))
    container.price_data_repo.bulk_insert(price_rows('STALE', TODAY - timedelta(days=1)))
    score_all(analyzer, LAST_RUN)
    assert criteria(path) == {'AAA': (199.0, LAST_RUN.isoformat()), 'STALE': (199.0, LAST_RUN.isoformat())}

    # today's fetch got a new day for AAA only; STALE's download failed and kept its rows
    container.price_data_repo.replace_for_ticker('AAA', price_rows('AAA', TODAY, n=101))
    analyzer.analyze()

    assert criteria(path) == {'AAA': (200.0, TODAY.isoformat()), 'STALE': (199.0, LAST_RUN.isoformat())}
    out = capsys.readouterr().out
    # US-only data: one line, for the US, and nothing about Canada
    assert summary_lines(out) == [f'US: skipped 1 stale ticker (last price before {TODAY}) and 0 frozen tickers']
    assert flagged_lines(out) == [f'FLAGGED ticker=STALE reason=stale last={TODAY - timedelta(days=1)}']


def test_us_holiday_leaves_us_tickers_scored(path, container, capsys):
    # Canada traded today and the US did not: every US ticker's last price is yesterday's
    yesterday = TODAY - timedelta(days=1)
    container.price_data_repo.bulk_insert(price_rows('AAA', yesterday))
    container.price_data_repo.bulk_insert(price_rows('BBB', yesterday - timedelta(days=1)))
    container.price_data_repo.bulk_insert(price_rows('SHOP.TO', TODAY))

    container.analyzer.analyze()

    assert criteria(path) == {'AAA': (199.0, TODAY.isoformat()), 'SHOP.TO': (199.0, TODAY.isoformat())}
    assert summary_lines(capsys.readouterr().out) == [
        f'US: skipped 1 stale ticker (last price before {yesterday}) and 0 frozen tickers',
        f'Canada: skipped 0 stale tickers (last price before {TODAY}) and 0 frozen tickers',
    ]


def test_standalone_analyze_scores_both_markets(path, container, capsys):
    container.price_data_repo.bulk_insert(price_rows('AAA', TODAY))
    container.price_data_repo.bulk_insert(price_rows('SHOP.TO', TODAY - timedelta(days=3)))
    container.price_data_repo.bulk_insert(price_rows('ABC.V', TODAY - timedelta(days=3)))

    container.analyzer.analyze()

    assert set(criteria(path)) == {'AAA', 'SHOP.TO', 'ABC.V'}
    assert {date_created for _, date_created in criteria(path).values()} == {TODAY.isoformat()}
    assert summary_lines(capsys.readouterr().out) == [
        f'US: skipped 0 stale tickers (last price before {TODAY}) and 0 frozen tickers',
        f'Canada: skipped 0 stale tickers (last price before {TODAY - timedelta(days=3)}) and 0 frozen tickers',
    ]


class TiingoStub:
    """A Tiingo that answers every US ticker with 101 rising closes ending today."""

    name = 'Tiingo'
    markets = frozenset({US})
    api_key = 'test-key'

    def check_quota(self, needed: int) -> None:
        pass

    def daily_prices(self, ticker: str, start: date, end: date) -> list[DailyPrice]:
        return [DailyPrice(date=p.date, close=p.close, volume=p.volume) for p in price_rows(ticker, TODAY, n=101)]


class SectorServiceStub:
    def __init__(self, tickers: list[str]) -> None:
        self.tickers = tickers

    def get_all(self) -> list[SectorAggregate]:
        subsector = SubsectorWithTickers(id=1, subsector='Software', tickers=self.tickers)
        return [SectorAggregate(id=1, sector='Tech', created_at=None, subsectors=[subsector])]


def rows_of(path: str, table: str, ticker: str) -> list[tuple]:
    with sqlite3.connect(path) as con:
        return con.execute(f'SELECT * FROM {table} WHERE ticker = ? ORDER BY id', (ticker,)).fetchall()


def test_tiingo_run_refresh_leaves_canadian_scores_and_prices_as_they_were(path, container, capsys):
    # the last run, an EODHD one, scored both markets; SHOP.TO holds rows older than the window too
    container.price_data_repo.bulk_insert(price_rows('AAA', TODAY - timedelta(days=1)))
    container.price_data_repo.bulk_insert(price_rows('SHOP.TO', TODAY - timedelta(days=1), n=130))
    score_all(container.analyzer, LAST_RUN)
    shop_criteria = rows_of(path, 'stock_criteria', 'SHOP.TO')
    shop_prices = rows_of(path, 'price_data', 'SHOP.TO')
    assert min(row[3] for row in shop_prices) < window_start(TODAY).isoformat()

    container.price_api_client = TiingoStub()
    container.sector_service = SectorServiceStub(['AAA', 'SHOP.TO'])
    container.ticker_fetcher = TickerFetcher(container)
    cmd = RunRefreshCmd()
    parser = argparse.ArgumentParser()
    cmd.add_arguments(parser)
    cmd.handle(parser.parse_args([]), container)

    assert criteria(path)['AAA'] == (200.0, TODAY.isoformat())
    assert rows_of(path, 'stock_criteria', 'SHOP.TO') == shop_criteria
    assert rows_of(path, 'price_data', 'SHOP.TO') == shop_prices
    assert summary_lines(capsys.readouterr().out) == [
        f'US: skipped 0 stale tickers (last price before {TODAY}) and 0 frozen tickers',
        'Canada: not scored, not fetched by this run (Tiingo)',
    ]


def test_stale_ticker_already_flagged_is_counted_but_not_flagged_again(path, container, capsys):
    container.price_data_repo.bulk_insert(price_rows('AAA', TODAY))
    container.price_data_repo.bulk_insert(price_rows('GONE', TODAY - timedelta(days=2)))
    container.price_data_repo.bulk_insert(price_rows('LATE', TODAY - timedelta(days=1)))

    container.analyzer.analyze(already_flagged={'GONE'})

    out = capsys.readouterr().out
    assert summary_lines(out) == [f'US: skipped 2 stale tickers (last price before {TODAY}) and 0 frozen tickers']
    assert flagged_lines(out) == [f'FLAGGED ticker=LATE reason=stale last={TODAY - timedelta(days=1)}']


class TiingoWithDeadTickers(TiingoStub):
    """TiingoStub, except that it does not know GONE (a 404) and has no prices in the window for EMPTY."""

    def daily_prices(self, ticker: str, start: date, end: date) -> list[DailyPrice]:
        if ticker == 'GONE':
            response = requests.Response()
            response.status_code = 404
            raise requests.HTTPError('404 Client Error: Not Found', response=response)
        if ticker == 'EMPTY':
            return []
        return super().daily_prices(ticker, start, end)


def test_run_refresh_flags_each_dead_or_stale_ticker_once(path, container, capsys):
    # every ticker was scored by the last run; GONE and EMPTY keep their rows, which analyze finds stale
    yesterday = TODAY - timedelta(days=1)
    for ticker in ('AAA', 'GONE', 'EMPTY'):
        container.price_data_repo.bulk_insert(price_rows(ticker, yesterday))

    container.price_api_client = TiingoWithDeadTickers()
    container.sector_service = SectorServiceStub(['AAA', 'GONE', 'EMPTY'])
    container.ticker_fetcher = TickerFetcher(container)
    cmd = RunRefreshCmd()
    parser = argparse.ArgumentParser()
    cmd.add_arguments(parser)
    cmd.handle(parser.parse_args([]), container)

    out = capsys.readouterr().out
    assert flagged_lines(out) == [
        f'FLAGGED ticker=GONE reason=not_found last={yesterday}',
        f'FLAGGED ticker=EMPTY reason=no_prices last={yesterday}',
    ]
    assert summary_lines(out) == [f'US: skipped 2 stale tickers (last price before {TODAY}) and 0 frozen tickers']


def frozen_rows(ticker: str, flat: int, n: int = 100, volume_on: int | None = None) -> list[PriceData]:
    """
    n rows ending today whose last 'flat' days repeat one close with zero volume, as Tiingo answers for a ticker that
    stopped trading; 'volume_on' gives that many days before today a volume of 500 instead.
    """
    rows = price_rows(ticker, TODAY, n)
    close = rows[n - flat - 1].close
    rows[n - flat :] = [replace(row, close=close, volume=0) for row in rows[n - flat :]]
    if volume_on is not None:
        rows[n - 1 - volume_on] = replace(rows[n - 1 - volume_on], volume=500)
    return rows


def first_flat_day(flat: int) -> date:
    return TODAY - timedelta(days=flat - 1)


def test_frozen_ticker_keeps_its_previous_row_and_is_flagged(path, container, capsys, caplog):
    analyzer = container.analyzer
    container.price_data_repo.bulk_insert(price_rows('AAA', TODAY - timedelta(days=1)))
    container.price_data_repo.bulk_insert(price_rows('ANSS', TODAY - timedelta(days=1)))
    score_all(analyzer, LAST_RUN)

    # today's fetch got a new day for both, but ANSS's last 10 days are one close with no volume
    container.price_data_repo.replace_for_ticker('AAA', price_rows('AAA', TODAY, n=101))
    container.price_data_repo.replace_for_ticker('ANSS', frozen_rows('ANSS', flat=10))
    with caplog.at_level('WARNING', logger='app.analyzer'):
        analyzer.analyze()

    assert criteria(path) == {'AAA': (200.0, TODAY.isoformat()), 'ANSS': (199.0, LAST_RUN.isoformat())}
    out = capsys.readouterr().out
    assert summary_lines(out) == [f'US: skipped 0 stale tickers (last price before {TODAY}) and 1 frozen ticker']
    assert flagged_lines(out) == [f'FLAGGED ticker=ANSS reason=frozen last={first_flat_day(10)}']
    assert [r.getMessage() for r in caplog.records if r.levelname == 'WARNING'] == [
        f'ANSS not scored: its price has been frozen, with no volume, since {first_flat_day(10)}'
    ]


def test_frozen_ticker_is_flagged_from_the_first_day_of_a_longer_flat_stretch(path, container, capsys):
    container.price_data_repo.bulk_insert(price_rows('AAA', TODAY))
    container.price_data_repo.bulk_insert(frozen_rows('NKLA', flat=40))

    container.analyzer.analyze()

    assert set(criteria(path)) == {'AAA'}
    assert flagged_lines(capsys.readouterr().out) == [f'FLAGGED ticker=NKLA reason=frozen last={first_flat_day(40)}']


@pytest.mark.parametrize(
    'rows',
    [
        pytest.param(frozen_rows('FL', flat=9), id='nine flat days'),
        pytest.param(frozen_rows('FL', flat=10, volume_on=0), id='volume today'),
        pytest.param(frozen_rows('FL', flat=10, volume_on=5), id='volume within the ten days'),
    ],
)
def test_ticker_short_of_frozen_is_scored_as_before(path, container, capsys, rows):
    container.price_data_repo.bulk_insert(price_rows('AAA', TODAY))
    container.price_data_repo.bulk_insert(rows)

    container.analyzer.analyze()

    assert set(criteria(path)) == {'AAA', 'FL'}
    assert {date_created for _, date_created in criteria(path).values()} == {TODAY.isoformat()}
    out = capsys.readouterr().out
    assert summary_lines(out) == [f'US: skipped 0 stale tickers (last price before {TODAY}) and 0 frozen tickers']
    assert flagged_lines(out) == []


def test_frozen_ticker_already_flagged_is_counted_but_not_flagged_again(path, container, capsys):
    container.price_data_repo.bulk_insert(frozen_rows('GONE', flat=10))
    container.price_data_repo.bulk_insert(frozen_rows('FL', flat=10))

    container.analyzer.analyze(already_flagged={'GONE'})

    out = capsys.readouterr().out
    assert summary_lines(out) == [f'US: skipped 0 stale tickers (last price before {TODAY}) and 2 frozen tickers']
    assert flagged_lines(out) == [f'FLAGGED ticker=FL reason=frozen last={first_flat_day(10)}']


def test_ticker_both_stale_and_frozen_is_flagged_once_as_stale(path, container, capsys):
    container.price_data_repo.bulk_insert(price_rows('AAA', TODAY))
    yesterday = TODAY - timedelta(days=1)
    container.price_data_repo.bulk_insert([r for r in frozen_rows('VERO', flat=11) if r.date <= yesterday])

    container.analyzer.analyze()

    out = capsys.readouterr().out
    assert summary_lines(out) == [f'US: skipped 1 stale ticker (last price before {TODAY}) and 0 frozen tickers']
    assert flagged_lines(out) == [f'FLAGGED ticker=VERO reason=stale last={yesterday}']
