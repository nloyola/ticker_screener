import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass
class Sector:
    sector: str
    subsector: str
    tickers: list[str]

    @staticmethod
    def load_from_sqlite(db_path: str, table: str = 'sector') -> Iterator['Sector']:
        """
        Loads Sector instances from a SQLite database.

        Args:
            db_path: Path to the SQLite database file.
            table: Name of the table to query.

        Yields:
            Sector instances.
        """
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            query = f'SELECT sector, subsector, tickers FROM {table}'
            for sector, subsector, tickers in cursor.execute(query):
                yield Sector(
                    sector=sector,
                    subsector=subsector,
                    tickers=[ticker.strip() for ticker in tickers.split(',')] if tickers else [],
                )
