from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

from app.commands.base_command import BaseCommand
from app.container import Container
from app.models import Sector


class ImporterCommand(BaseCommand):
    _NAME = 'import'
    _DESCRIPTION = 'Imports stock tickers from an Excel file.'

    def __init__(self, container: Container) -> None:
        super().__init__(self._NAME, self._DESCRIPTION)
        self.sector_repo = container.sector_repo

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
        return [p for p in parts if p]

    def excel_import(self, filename: str) -> None:
        df = pd.read_excel(
            filename,
            dtype={'sector': 'string', 'subsector': 'string', 'tickers': 'string'},
        )

        required = ['sector', 'subsector', 'tickers']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f'Excel file must contain columns: {required} (missing: {missing})')

        # Normalize NA to None
        df = df[required].replace({np.nan: None, pd.NA: None})

        count = 0
        for _, row in df.iterrows():
            sector = (row['sector'] or '').strip()
            if not sector:
                continue  # skip invalid row

            subsector = row['subsector'] or None
            if isinstance(subsector, str):
                subsector = subsector.strip() or None

            tickers = self._parse_tickers(row['tickers'])

            self.sector_repo.insert(Sector(sector=sector, subsector=subsector, tickers=tickers))
            count += 1

        print(f'Imported {count} sector rows via SectorRepo')
