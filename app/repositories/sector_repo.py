from collections.abc import Iterable
from datetime import datetime

from app.container import Container
from app.db import connect
from app.models import Sector


class SectorRepo:
    def __init__(self, container: Container):
        self.db_path = container.db_path

    def insert(self, sector: Sector) -> int | None:
        with connect(self.db_path) as con:
            cur = con.execute(
                """
                INSERT INTO sector (sector)
                VALUES (?)
                """,
                (sector.sector,),
            )
            return cur.lastrowid

    def all(self) -> Iterable[Sector]:
        with connect(self.db_path) as con:
            rows = con.execute(
                """
                SELECT id, sector, created_at
                FROM sector
                ORDER BY sector
                """
            ).fetchall()
            for r in rows:
                yield Sector(
                    id=r['id'],
                    sector=r['sector'],
                    created_at=r['created_at']
                    if isinstance(r['created_at'], datetime)
                    else datetime.fromisoformat(r['created_at']),
                )

    def find(self, sector_name: str) -> list[Sector]:
        with connect(self.db_path) as con:
            rows = con.execute(
                """
                SELECT id, sector, created_at
                FROM sector
                WHERE sector = ?
                """,
                (sector_name,),
            ).fetchall()
            return [
                Sector(
                    id=r['id'],
                    sector=r['sector'],
                    created_at=r['created_at']
                    if isinstance(r['created_at'], datetime)
                    else datetime.fromisoformat(r['created_at']),
                )
                for r in rows
            ]
