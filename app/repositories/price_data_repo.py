import sqlite3
from collections.abc import Iterable
from datetime import date, datetime

from app.models import PriceData


class PriceDataRepo:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        sqlite3.register_adapter(date, lambda d: d.isoformat())
        sqlite3.register_converter('DATE', lambda s: datetime.strptime(s.decode(), '%Y-%m-%d').date())
        con = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        con.row_factory = sqlite3.Row
        return con

    def insert(self, price_data: PriceData) -> None:
        with self._connect() as con:
            con.execute(
                """
                INSERT OR REPLACE INTO price_data (ticker, date, close, volume)
                VALUES (?, ?, ?, ?)
                """,
                (price_data.ticker, price_data.date, price_data.close, price_data.volume),
            )

    def bulk_insert(self, items: Iterable[PriceData]) -> None:
        with self._connect() as con:
            con.executemany(
                """
                INSERT OR REPLACE INTO price_data (ticker, date, close, volume)
                VALUES (?, ?, ?, ?)
            """,
                [(p.ticker, p.date, p.close, p.volume) for p in items],
            )

    def get(self, ticker: str, d: date) -> PriceData | None:
        with self._connect() as con:
            row = con.execute(
                """
                SELECT ticker, date, close, volume
                FROM price_data
                WHERE ticker = ? AND date = ?
                """,
                (ticker, d),
            ).fetchone()
            return PriceData(**row) if row else None

    def all(self) -> Iterable[PriceData]:
        with self._connect() as con:
            rows = con.execute("""
                SELECT ticker, date, close, volume
                FROM price_data
                ORDER BY ticker, date
            """).fetchall()
            for r in rows:
                yield PriceData(**r)

    def all_for_ticker(self, ticker: str) -> PriceData | None:
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT ticker, date, close, volume
                FROM price_data
                WHERE ticker = ?
                ORDER BY date
                """,
                (ticker,),
            )
            for r in rows:
                yield PriceData(**r)

    def latest_for_ticker(self, ticker: str) -> PriceData | None:
        with self._connect() as con:
            row = con.execute(
                """
                SELECT ticker, date, close, volume
                FROM price_data
                WHERE ticker = ?
                ORDER BY date DESC
                LIMIT 1
                """,
                (ticker,),
            ).fetchone()
            return PriceData(**row) if row else None
