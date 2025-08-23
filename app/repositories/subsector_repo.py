from __future__ import annotations

import sqlite3
from collections.abc import Sequence

from app.container import Container
from app.models import Subsector


class SubsectorRepo:
    def __init__(self, container: Container):
        self.db_path = container.db_path

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        con.row_factory = sqlite3.Row
        # Enable FK constraints on every new connection (SQLite is per-connection)
        con.execute('PRAGMA foreign_keys = ON;')
        return con

    def insert(self, subsector: Subsector) -> int | None:
        """
        Inserts a subsector row and returns its id.
        """
        with self._connect() as con:
            cur = con.cursor()
            cur.execute(
                """
                INSERT INTO subsector (sector_id, subsector, tickers)
                VALUES (?, ?, ?)
                """,
                (subsector.sector_id, subsector.subsector, subsector.tickers),
            )
            con.commit()
            return cur.lastrowid

    def get_by_id(self, id_: int) -> Subsector | None:
        with self._connect() as con:
            cur = con.cursor()
            cur.execute(
                'SELECT id, sector_id, subsector, tickers, created_at FROM subsector WHERE id = ?',
                (id_,),
            )
            row = cur.fetchone()
            return Subsector(*row) if row else None

    def list_by_sector(self, sector_id: int) -> Sequence[Subsector]:
        with self._connect() as con:
            cur = con.cursor()
            cur.execute(
                'SELECT id, sector_id, subsector, tickers, created_at FROM subsector WHERE sector_id = ?',
                (sector_id,),
            )
            return [Subsector(*row) for row in cur.fetchall()]

    def delete_by_sector(self, sector_id: int) -> None:
        with self._connect() as con:
            cur = con.cursor()
            cur.execute('DELETE FROM subsector WHERE sector_id = ?', (sector_id,))
            con.commit()

    def clear_all(self) -> None:
        with self._connect() as con:
            con.execute('DELETE FROM subsector')
            con.commit()

    def get_by_ticker(self, ticker: str) -> Subsector | None:
        """
        Returns the Subsector that contains the given ticker, or None if not found.
        Assumes subsector.tickers is a comma-separated string of tickers.
        """
        with self._connect() as con:
            cur = con.cursor()
            cur.execute(
                """
                SELECT id, sector_id, subsector, tickers, created_at
                FROM subsector
                WHERE ',' || tickers || ',' LIKE ?
                """,
                (f'%,{ticker},%',),
            )
            row = cur.fetchone()
            return Subsector(*row) if row else None
