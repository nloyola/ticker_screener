from __future__ import annotations

from collections.abc import Sequence

from app.container import Container
from app.db import connect
from app.models import Subsector


class SubsectorRepo:
    def __init__(self, container: Container):
        self.db_path = container.db_path

    def insert(self, subsector: Subsector) -> int | None:
        """
        Inserts a subsector row and returns its id.
        """
        with connect(self.db_path) as con:
            cur = con.cursor()
            cur.execute(
                """
                INSERT INTO subsector (sector_id, subsector, tickers)
                VALUES (?, ?, ?)
                """,
                (subsector.sector_id, subsector.subsector, subsector.tickers),
            )
            return cur.lastrowid

    def get_by_id(self, id_: int) -> Subsector | None:
        with connect(self.db_path) as con:
            cur = con.cursor()
            cur.execute(
                'SELECT id, sector_id, subsector, tickers, created_at FROM subsector WHERE id = ?',
                (id_,),
            )
            row = cur.fetchone()
            return Subsector(*row) if row else None

    def list_by_sector(self, sector_id: int) -> Sequence[Subsector]:
        with connect(self.db_path) as con:
            cur = con.cursor()
            cur.execute(
                'SELECT id, sector_id, subsector, tickers, created_at FROM subsector WHERE sector_id = ? ORDER BY id',
                (sector_id,),
            )
            return [Subsector(*row) for row in cur.fetchall()]

    def delete_by_sector(self, sector_id: int) -> None:
        with connect(self.db_path) as con:
            cur = con.cursor()
            cur.execute('DELETE FROM subsector WHERE sector_id = ?', (sector_id,))

    def clear_all(self) -> None:
        with connect(self.db_path) as con:
            con.execute('DELETE FROM subsector')
