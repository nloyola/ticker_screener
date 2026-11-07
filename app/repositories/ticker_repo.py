from __future__ import annotations

from collections.abc import Sequence

from app.container import Container
from app.db import connect
from app.models import Ticker


class TickerRepo:
    def __init__(self, container: Container):
        self.db_path = container.db_path

    def upsert_many(self, tickers: list[Ticker]) -> None:
        with connect(self.db_path) as con:
            con.executemany(
                """
                INSERT INTO ticker (ticker, last, prev_close, volume, bid_price, ask_price, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    last = excluded.last,
                    prev_close = excluded.prev_close,
                    volume = excluded.volume,
                    bid_price = excluded.bid_price,
                    ask_price = excluded.ask_price,
                    timestamp = excluded.timestamp
                """,
                [
                    (
                        t.ticker,
                        t.last,
                        t.prev_close,
                        t.volume,
                        t.bid_price,
                        t.ask_price,
                        t.timestamp,
                    )
                    for t in tickers
                ],
            )

    def get(self, ticker: str) -> Ticker | None:
        with connect(self.db_path) as con:
            row = con.execute('SELECT * FROM ticker WHERE ticker = ?', (ticker,)).fetchone()
            return Ticker(**row) if row else None

    def all(self) -> Sequence[Ticker]:
        with connect(self.db_path) as con:
            rows = con.execute('SELECT * FROM ticker ORDER BY ticker').fetchall()
            return [Ticker(**row) for row in rows]
