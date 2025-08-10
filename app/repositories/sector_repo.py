import sqlite3
from collections.abc import Iterable
from datetime import datetime

from app.models import Sector


class SectorRepo:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        con.row_factory = sqlite3.Row
        return con

    @staticmethod
    def _tickers_to_str(tickers: list[str] | None) -> str | None:
        return ','.join(tickers) if tickers else None

    @staticmethod
    def _tickers_from_str(s: str | None) -> list[str] | None:
        return s.split(',') if s else None

    def insert(self, sector: Sector) -> None:
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO sector (sector, subsector, tickers)
                VALUES (?, ?, ?)
                """,
                (
                    sector.sector,
                    sector.subsector,
                    self._tickers_to_str(sector.tickers),
                ),
            )

    def all(self) -> Iterable[Sector]:
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT sector, subsector, tickers, created_at
                FROM sector
                ORDER BY sector, subsector
                """
            ).fetchall()
            for r in rows:
                yield Sector(
                    sector=r['sector'],
                    subsector=r['subsector'],
                    tickers=self._tickers_from_str(r['tickers']),
                    created_at=r['created_at']
                    if isinstance(r['created_at'], datetime)
                    else datetime.fromisoformat(r['created_at']),
                )

    def find(self, sector_name: str) -> list[Sector]:
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT sector, subsector, tickers, created_at
                FROM sector
                WHERE sector = ?
                """,
                (sector_name,),
            ).fetchall()
            return [
                Sector(
                    sector=r['sector'],
                    subsector=r['subsector'],
                    tickers=self._tickers_from_str(r['tickers']),
                    created_at=r['created_at']
                    if isinstance(r['created_at'], datetime)
                    else datetime.fromisoformat(r['created_at']),
                )
                for r in rows
            ]
