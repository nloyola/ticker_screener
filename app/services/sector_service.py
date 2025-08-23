from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from app.container import Container
from app.models import Subsector

if TYPE_CHECKING:
    from app.repositories.sector_repo import SectorRepo
    from app.repositories.subsector_repo import SubsectorRepo


@dataclass(frozen=True)
class SubsectorWithTickers:
    id: int
    subsector: str
    tickers: list[str]
    created_at: datetime | None = None


@dataclass(frozen=True)
class SectorAggregate:
    id: int
    sector: str
    created_at: datetime | None
    subsectors: list[SubsectorWithTickers]


class SectorService:
    """
    Aggregate-root service around Sector.

    Responsibilities:
    - Load all sectors.
    - For each sector, load subsectors.
    - Normalize subsector.tickers into a list[str].
    """

    def __init__(self, container: Container):
        self._sectors: SectorRepo = container.sector_repo
        self._subsectors: SubsectorRepo = container.subsector_repo

    def get_all(self) -> list[SectorAggregate]:
        """
        Return all sectors with their subsectors and tickers (as a list[str]).
        """
        aggregates: list[SectorAggregate] = []

        for s in self._sectors.all():  # SectorRepo.all() yields Sector(id, sector, created_at)
            raw_subs = self._subsectors.list_by_sector(s.id)  # list of Subsector rows
            subs: list[SubsectorWithTickers] = []

            for sub in raw_subs:
                tickers_list = self._parse_tickers(getattr(sub, 'tickers', None))
                subs.append(
                    SubsectorWithTickers(
                        id=sub.id,
                        subsector=sub.subsector,
                        tickers=tickers_list,
                        created_at=getattr(sub, 'created_at', None),
                    )
                )

            aggregates.append(
                SectorAggregate(
                    id=s.id,
                    sector=s.sector,
                    created_at=s.created_at,
                    subsectors=subs,
                )
            )

        return aggregates

    def subsector_for_ticker(self, ticker: str) -> Subsector:
        return self._subsectors.get_by_ticker(ticker)

    @staticmethod
    def _parse_tickers(tickers: str | None) -> list[str]:
        """
        Normalize a CSV string like 'XOM, CVX, SU' to ['XOM','CVX','SU'].
        Deduplicates and preserves order of first appearance.
        """
        if not tickers:
            return []
        seen = set()
        out: list[str] = []
        for t in (x.strip().upper() for x in tickers.split(',') if x.strip()):
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out
