from __future__ import annotations

import argparse
import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

from app.commands.base_command import BaseCommand
from app.container import Container
from app.models import Sector, Subsector


class ImporterCommand(BaseCommand):
    _NAME = 'import'
    _DESCRIPTION = 'Imports sectors, subsectors, and tickers from an Excel file.'

    def __init__(self, container: Container) -> None:
        super().__init__(self._NAME, self._DESCRIPTION)
        self.sector_repo = container.sector_repo
        self.subsector_repo = container.subsector_repo

        # cache sector name → id to minimize DB lookups
        self._sector_id_cache: dict[str, int] = {}

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            '--excel', required=True, help='Path to Excel file with columns: sector, subsector, tickers'
        )

    def handle(self, args: argparse.Namespace) -> None:
        self.excel_import(args.excel)

    # ---- internals ---------------------------------------------------------

    @staticmethod
    def _parse_tickers(value: object) -> list[str] | None:
        if not isinstance(value, str) or not value.strip():
            return None
        # split on commas or whitespace; trim empties
        parts = re.split(r'[,\s]+', value.strip())
        parts = [p.upper() for p in parts if p]  # normalize to UPPER
        seen = set()
        uniq = [p for p in parts if not (p in seen or seen.add(p))]  # preserve order, dedupe
        return uniq or None

    @staticmethod
    def _join_tickers(tickers: Iterable[str] | None) -> str | None:
        if not tickers:
            return None
        return ','.join(tickers)

    def _get_or_create_sector_id(self, sector_name: str) -> int:
        """
        Returns sector.id for the given sector name, creating it if needed.
        Caches results in-memory for performance.
        """
        key = sector_name.lower().strip()
        if key in self._sector_id_cache:
            return self._sector_id_cache[key]

        # Try repo lookup (adjust to your API)
        sectors = self.sector_repo.find(sector_name)
        if len(sectors) > 0:
            self._sector_id_cache[key] = sectors[0].id
            return sectors[0].id

        # Create if not found
        created = self.sector_repo.insert(Sector(id=0, sector=sector_name))
        # Some repos return the created model, others return the id — handle both:
        sector_id = created.id if hasattr(created, 'id') else created
        self._sector_id_cache[key] = sector_id
        return sector_id

    def _insert_subsector(self, sector_id: int, name: str | None, tickers_csv: str | None) -> None:
        """
        Inserts a subsector row. If your repo supports upsert or uniqueness
        constraints on (sector_id, subsector), you can switch to upsert here.
        """
        model = Subsector(id=0, sector_id=sector_id, subsector=name, tickers=tickers_csv)
        self.subsector_repo.insert(model)

    def excel_import(self, filename: str) -> None:
        df = pd.read_excel(
            filename,
            dtype={'sector': 'string', 'subsector': 'string', 'tickers': 'string'},
        )

        required = ['sector', 'subsector', 'tickers']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f'Excel file must contain columns: {required} (missing: {missing})')

        # Normalize NA → None
        df = df[required].replace({np.nan: None, pd.NA: None})

        inserted_sectors = 0
        inserted_subsectors = 0

        # Optional: begin a transaction if your repos expose it (pseudo-code):
        # with self.sector_repo.transaction():
        for _, row in df.iterrows():
            sector = (row['sector'] or '').strip()
            if not sector:
                continue  # skip invalid row

            subsector = row['subsector'] or None
            if isinstance(subsector, str):
                subsector = subsector.strip() or None

            tickers = self._parse_tickers(row['tickers'])
            tickers_csv = self._join_tickers(tickers)

            # Ensure sector exists, get id
            pre_cache_len = len(self._sector_id_cache)
            sector_id = self._get_or_create_sector_id(sector)
            if len(self._sector_id_cache) > pre_cache_len:
                inserted_sectors += 1  # newly created

            # Insert subsector row (linked via FK)
            self._insert_subsector(sector_id, subsector, tickers_csv)
            inserted_subsectors += 1

        print(f'Imported {inserted_subsectors} subsector rows and created {inserted_sectors} new sectors (if any).')
