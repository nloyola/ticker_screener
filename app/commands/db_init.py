import argparse
from pathlib import Path

from rich.console import Console

from app.container import Container
from app.db import MARKET_DATA_TABLES, clear_market_data, init_db

from .base_command import BaseCommand

console = Console()


class DbInit(BaseCommand):
    _NAME = 'db-init'
    _DESCRIPTION = 'creates missing tables and empties the market data tables, keeping sectors'

    def __init__(self) -> None:
        super().__init__(self._NAME, self._DESCRIPTION)

    def handle(self, args: argparse.Namespace, container: Container) -> None:
        db_file = Path(container.db_path)
        tables = ', '.join(MARKET_DATA_TABLES)

        if db_file.exists():
            try:
                confirm = (
                    input(
                        f"Database '{db_file}' already exists. Empty {tables}? "
                        'Sectors, subsectors and all other tables are kept. [y/N]: '
                    )
                    .strip()
                    .lower()
                )
            except EOFError:  # no answer, e.g. stdin was a piped .env already read as config
                print()
                confirm = ''
            if confirm != 'y':
                print('Keeping existing database unchanged.')
                return

        init_db(container.db_path)
        with container.db.connect() as con:
            clear_market_data(con)
        print(f"DB ready at: '{db_file}'. Emptied {tables}; kept sectors and subsectors.")
