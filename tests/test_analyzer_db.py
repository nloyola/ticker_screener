"""
Which tickers analyze scores, against a real database: a stale ticker keeps its previous stock_criteria row,
date_created included, freshness is judged per market, and run-refresh scores only the markets it fetched.
"""

import argparse
import sqlite3
from datetime import date, timedelta
from types import SimpleNamespace

import pytest

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
    # US-only data: one line, for the US, and nothing about Canada
    assert summary_lines(capsys.readouterr().out) == [f'US: skipped 1 stale ticker (last price before {TODAY})']


def test_us_holiday_leaves_us_tickers_scored(path, container, capsys):
    # Canada traded today and the US did not: every US ticker's last price is yesterday's
    yesterday = TODAY - timedelta(days=1)
    container.price_data_repo.bulk_insert(price_rows('AAA', yesterday))
    container.price_data_repo.bulk_insert(price_rows('BBB', yesterday - timedelta(days=1)))
    container.price_data_repo.bulk_insert(price_rows('SHOP.TO', TODAY))

    container.analyzer.analyze()

    assert criteria(path) == {'AAA': (199.0, TODAY.isoformat()), 'SHOP.TO': (199.0, TODAY.isoformat())}
    assert summary_lines(capsys.readouterr().out) == [
        f'US: skipped 1 stale ticker (last price before {yesterday})',
        f'Canada: skipped 0 stale tickers (last price before {TODAY})',
    ]


def test_standalone_analyze_scores_both_markets(path, container, capsys):
    container.price_data_repo.bulk_insert(price_rows('AAA', TODAY))
    container.price_data_repo.bulk_insert(price_rows('SHOP.TO', TODAY - timedelta(days=3)))
    container.price_data_repo.bulk_insert(price_rows('ABC.V', TODAY - timedelta(days=3)))

    container.analyzer.analyze()

    assert set(criteria(path)) == {'AAA', 'SHOP.TO', 'ABC.V'}
    assert {date_created for _, date_created in criteria(path).values()} == {TODAY.isoformat()}
    assert summary_lines(capsys.readouterr().out) == [
        f'US: skipped 0 stale tickers (last price before {TODAY})',
        f'Canada: skipped 0 stale tickers (last price before {TODAY - timedelta(days=3)})',
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
        f'US: skipped 0 stale tickers (last price before {TODAY})',
        'Canada: not scored, not fetched by this run (Tiingo)',
    ]
