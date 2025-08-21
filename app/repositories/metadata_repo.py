from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

from app.container import Container


class MetadataRepo:
    def __init__(self, container: Container):
        self.db_path = container.db_path

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        con.row_factory = sqlite3.Row
        return con

    def get(self, key: str) -> str | None:
        with self._connect() as con:
            row = con.execute('SELECT value FROM metadata WHERE key = ?', (key,)).fetchone()
            return row['value'] if row else None

    def set(self, key: str, value: str) -> None:
        with self._connect() as con:
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
