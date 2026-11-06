from __future__ import annotations

import argparse
import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

from app.commands.base_command import BaseCommand
from app.container import Container
from app.db import SQLite, clear_market_data


class ImporterCommand(BaseCommand):
    _NAME = 'import'
    _DESCRIPTION = 'Replaces all sectors, subsectors, and tickers with those in an Excel file.'

    def __init__(self) -> None:
        super().__init__(self._NAME, self._DESCRIPTION)

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            '--excel', required=True, help='Path to Excel file with columns: sector, subsector, tickers'
        )

    def handle(self, args: argparse.Namespace, container: Container) -> None:
        self.excel_import(container.db, args.excel)

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

    def _read_rows(self, filename: str) -> list[tuple[str, str | None, str | None]]:
        """
        Reads the spreadsheet into (sector, subsector, tickers_csv) rows, skipping
        rows without a sector. Raises before the database is touched if the file
        is missing a required column.
        """
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

        rows: list[tuple[str, str | None, str | None]] = []
        for _, row in df.iterrows():
            sector = (row['sector'] or '').strip()
            if not sector:
                continue  # skip invalid row

            subsector = row['subsector'] or None
            if isinstance(subsector, str):
                subsector = subsector.strip() or None

            tickers_csv = self._join_tickers(self._parse_tickers(row['tickers']))
            rows.append((sector, subsector, tickers_csv))
        return rows

    def excel_import(self, db: SQLite, filename: str) -> None:
        """
        Replaces every sector and subsector with the spreadsheet's contents.

        New subsector rows get new ids, so the market data tables, whose rows
        point at subsector ids, are emptied too; the next screener run refills
        them. Everything happens in one transaction: a failure leaves the
        database as it was. Tables not owned by the screener are never touched.
        """
        rows = self._read_rows(filename)

        with db.connect() as con:
            con.execute('DELETE FROM subsector')
            con.execute('DELETE FROM sector')
            clear_market_data(con)

            sector_ids: dict[str, int] = {}
            for sector, subsector, tickers_csv in rows:
                sector_id = sector_ids.get(sector)
                if sector_id is None:
                    cur = con.execute('INSERT INTO sector (sector) VALUES (?)', (sector,))
                    sector_id = cur.lastrowid
                    sector_ids[sector] = sector_id
                con.execute(
                    'INSERT INTO subsector (sector_id, subsector, tickers) VALUES (?, ?, ?)',
                    (sector_id, subsector, tickers_csv),
                )

        print(
            f'Replaced all sectors: imported {len(sector_ids)} sectors and {len(rows)} subsectors; '
            'emptied the market data tables.'
        )
