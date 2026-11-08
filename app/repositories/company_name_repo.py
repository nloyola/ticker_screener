from __future__ import annotations

from collections.abc import Iterable

from app.container import Container
from app.db import connect


class CompanyNameRepo:
    """Each ticker's company name. Tickers match case-insensitively, as the table's NOCASE key does."""

    def __init__(self, container: Container):
        self.db_path = container.db_path

    def missing(self, tickers: Iterable[str]) -> list[str]:
        """The given tickers that have no stored name, in the order given, each once."""
        with connect(self.db_path) as con:
            # tickers are ASCII, so upper() folds exactly as NOCASE does
            stored = {row['ticker'].upper() for row in con.execute('SELECT ticker FROM company_name')}
        result: list[str] = []
        for ticker in tickers:
            key = ticker.upper()
            if key not in stored:
                stored.add(key)
                result.append(ticker)
        return result

    def get(self, ticker: str) -> str | None:
        with connect(self.db_path) as con:
            row = con.execute('SELECT name FROM company_name WHERE ticker = ?', (ticker,)).fetchone()
            return row['name'] if row else None

    def store(self, ticker: str, name: str, price_api: str) -> None:
        """Stores a ticker's name, looked up from price_api, replacing any name stored before."""
        with connect(self.db_path) as con:
            con.execute(
                """
                INSERT INTO company_name (ticker, name, price_api)
                VALUES (?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    name = excluded.name,
                    price_api = excluded.price_api,
                    stored_at = datetime('now')
                """,
                (ticker, name, price_api),
            )
