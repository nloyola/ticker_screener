from __future__ import annotations

from datetime import UTC, datetime

from app.container import Container
from app.db import connect


class MetadataRepo:
    def __init__(self, container: Container):
        self.db_path = container.db_path

    def get(self, key: str) -> str | None:
        with connect(self.db_path) as con:
            row = con.execute('SELECT value FROM metadata WHERE key = ?', (key,)).fetchone()
            return row['value'] if row else None

    def set(self, key: str, value: str) -> None:
        with connect(self.db_path) as con:
            con.execute(
                """
                INSERT INTO metadata (key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, value),
            )

    # convenience for your use-case
    def get_tickers_fetched_at(self) -> datetime | None:
        v = self.get('tickers_fetched_at')
        return datetime.fromisoformat(v) if v else None

    def set_tickers_fetched_now(self) -> None:
        now_utc = datetime.now(UTC).isoformat()
        self.set('tickers_fetched_at', now_utc)
