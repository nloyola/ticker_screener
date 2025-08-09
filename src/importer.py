import argparse
import sqlite3

import pandas as pd
from rich.console import Console

from src.db_create import DbCreate

from .base_command import BaseCommand
from .config import Config

console = Console()


class ImporterCommand(BaseCommand):
    _NAME = 'import'
    _DESCRIPTION = 'Imports stock tickers from an Excel file.'

    def __init__(self) -> None:
        super().__init__(self._NAME, self._DESCRIPTION)

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('--excel', help='import ticker symbols from an Excel file')

    def handle(self, args: argparse.Namespace) -> None:
        excel_filename = args.excel
        if excel_filename is not None:
            self.excel_import(excel_filename)

    def excel_import(self, filename: str) -> None:
        excel_file = filename
        df = pd.read_excel(excel_file)

        # Ensure DataFrame has the required columns
        required_columns = ['sector', 'subsector', 'tickers']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f'Excel file must contain columns: {required_columns}')

        # Save to SQLite database
        table_name = 'sector'
        with sqlite3.connect(Config.get_db_name()) as conn:
            df[required_columns].to_sql(table_name, conn, if_exists='replace', index=False)

        console.print(f'Imported {len(df)} records into {Config.get_db_name()}:{table_name}')
