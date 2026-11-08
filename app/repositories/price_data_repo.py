import sqlite3
from collections.abc import Collection, Iterable, Iterator
from datetime import date

from app.container import Container
from app.db import connect
from app.markets import market_sql
from app.models import PriceData


def _price_data(row: sqlite3.Row) -> PriceData:
    # schema.sql declares price_data.date as TEXT, so no converter applies to it: parse the ISO date here
    return PriceData(**{**dict(row), 'date': date.fromisoformat(row['date'])})


class PriceDataRepo:
    def __init__(self, container: Container):
        self.db_path = container.db_path

    def insert(self, price_data: PriceData) -> None:
        with connect(self.db_path) as con:
            con.execute(
                """
                INSERT OR REPLACE INTO price_data (subsector_id, ticker, date, close, volume, high, low)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    price_data.subsector_id,
                    price_data.ticker,
                    price_data.date,
                    price_data.close,
                    price_data.volume,
                    price_data.high,
                    price_data.low,
                ),
            )

    def bulk_insert(self, items: Iterable[PriceData]) -> None:
        with connect(self.db_path) as con:
            con.executemany(
                """
                INSERT OR REPLACE INTO price_data (subsector_id, ticker, date, close, volume, high, low)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                [(p.subsector_id, p.ticker, p.date, p.close, p.volume, p.high, p.low) for p in items],
            )

    def replace_for_ticker(self, ticker: str, rows: Iterable[PriceData]) -> None:
        """Replaces every stored row of ticker with rows, in one transaction."""
        rows = list(rows)
        others = {p.ticker for p in rows} - {ticker}
        if others:
            raise ValueError(f'rows for {sorted(others)} passed to replace the rows of {ticker}')
        with connect(self.db_path) as con:
            con.execute('DELETE FROM price_data WHERE ticker = ?', (ticker,))
            con.executemany(
                """
                INSERT INTO price_data (subsector_id, ticker, date, close, volume, high, low)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [(p.subsector_id, p.ticker, p.date, p.close, p.volume, p.high, p.low) for p in rows],
            )

    def get(self, ticker: str, d: date) -> PriceData | None:
        with connect(self.db_path) as con:
            row = con.execute(
                """
                SELECT id, subsector_id, ticker, date, close, volume, high, low
                FROM price_data
                WHERE ticker = ? AND date = ?
                """,
                (ticker, d),
            ).fetchone()
            return _price_data(row) if row else None

    def all(self) -> Iterator[PriceData]:
        with connect(self.db_path) as con:
            rows = con.execute("""
                SELECT id, subsector_id, ticker, date, close, volume, high, low
                FROM price_data
                ORDER BY ticker, date
            """).fetchall()
            for r in rows:
                yield _price_data(r)

    def all_for_ticker(self, ticker: str) -> Iterator[PriceData]:
        with connect(self.db_path) as con:
            rows = con.execute(
                """
                SELECT id, subsector_id, ticker, date, close, volume, high, low
                FROM price_data
                WHERE ticker = ?
                ORDER BY date
                """,
                (ticker,),
            )
            for r in rows:
                yield _price_data(r)

    def latest_for_ticker(self, ticker: str) -> PriceData | None:
        with connect(self.db_path) as con:
            row = con.execute(
                """
                SELECT id, subsector_id, ticker, date, close, volume, high, low
                FROM price_data
                WHERE ticker = ?
                ORDER BY date DESC
                LIMIT 1
                """,
                (ticker,),
            ).fetchone()
            return _price_data(row) if row else None

    def delete_before(self, cutoff: date, markets: Collection[str]) -> int:
        """
        Deletes the rows dated before cutoff of the tickers in 'markets' (app/markets.py), and returns how many were
        deleted. Other markets' rows are kept, whatever their date.
        """
        markets = list(markets)
        if not markets:
            return 0
        with connect(self.db_path) as con:
            cur = con.execute(
                f"""
                DELETE FROM price_data
                WHERE date < ? AND {market_sql('ticker')} IN ({','.join('?' * len(markets))})
                """,
                (cutoff, *markets),
            )
            return cur.rowcount
