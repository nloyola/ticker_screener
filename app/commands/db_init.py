import argparse
import os
from pathlib import Path

from rich.console import Console

from app.container import Container
from app.db import init_db

from .base_command import BaseCommand

console = Console()


class DbInit(BaseCommand):
    _NAME = 'db-init'
    _DESCRIPTION = 'initializes the database'

    def __init__(self, container: Container) -> None:
        super().__init__(self._NAME, self._DESCRIPTION)
        self.db_path = container.db_path

    def handle(self, args: argparse.Namespace) -> None:
        db_file = Path(self.db_path)

        if db_file.exists():
            confirm = input(f"Database '{db_file}' already exists. Delete and recreate? [y/N]: ").strip().lower()
            if confirm != 'y':
                print('Keeping existing database.')
                return
            else:
                os.remove(db_file)
                print(f"Deleted '{db_file}'.")

        init_db(self.db_path)
        print(f"DB created at: '{db_file}'.")
