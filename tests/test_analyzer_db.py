"""A stale ticker keeps its previous stock_criteria row, date_created included, against a real database."""

import sqlite3
from datetime import date, timedelta
from types import SimpleNamespace

from app.analyzer import Analyzer
from app.db import init_db
from app.models import PriceData
from app.repositories.price_data_repo import PriceDataRepo
from app.repositories.stock_criteria_repo import StockCriteriaRepo

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


def test_stale_ticker_keeps_its_previous_row_and_date(tmp_path):
    path = str(tmp_path / 'stock_data.db')
    init_db(path)
    container = SimpleNamespace(db_path=path)
    container.price_data_repo = PriceDataRepo(container)
    container.stock_criteria_repo = StockCriteriaRepo(container)
    analyzer = Analyzer(container)

    # the last run scored both tickers from rows ending yesterday
    container.price_data_repo.bulk_insert(price_rows('AAA', TODAY - timedelta(days=1)))
    container.price_data_repo.bulk_insert(price_rows('STALE', TODAY - timedelta(days=1)))
    analyzer.save_to_repository(
        [analyzer.analyze_ticker(t, g) for t, g in analyzer.load_price_data().groupby(level='ticker')], LAST_RUN
    )
    assert criteria(path) == {'AAA': (199.0, LAST_RUN.isoformat()), 'STALE': (199.0, LAST_RUN.isoformat())}

    # today's fetch got a new day for AAA only; STALE's download failed and kept its rows
    container.price_data_repo.replace_for_ticker('AAA', price_rows('AAA', TODAY, n=101))
    analyzer.analyze()

    assert criteria(path) == {'AAA': (200.0, TODAY.isoformat()), 'STALE': (199.0, LAST_RUN.isoformat())}
