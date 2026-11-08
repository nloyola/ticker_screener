from collections.abc import Iterable
from datetime import date

from app.container import Container
from app.db import connect
from app.price_apis import DailyPrice


class BenchmarkPriceRepo:
    """The daily closes of the markets' benchmarks (table benchmark_price, BENCHMARKS in app/markets.py)."""

    def __init__(self, container: Container):
        self.db_path = container.db_path

    def replace(self, ticker: str, prices: Iterable[DailyPrice]) -> None:
        """Replaces every stored close of the benchmark 'ticker' with those of 'prices', in one transaction."""
        with connect(self.db_path) as con:
            con.execute('DELETE FROM benchmark_price WHERE ticker = ?', (ticker,))
            con.executemany(
                'INSERT INTO benchmark_price (ticker, date, close) VALUES (?, ?, ?)',
                [(ticker, p.date, p.close) for p in prices],
            )

    def closes(self, ticker: str) -> dict[date, float]:
        """The benchmark's stored closes by date; empty when it has none."""
        with connect(self.db_path) as con:
            rows = con.execute('SELECT date, close FROM benchmark_price WHERE ticker = ?', (ticker,)).fetchall()
        # date is TEXT, as in price_data, so no converter applies to it
        return {date.fromisoformat(row['date']): row['close'] for row in rows}
