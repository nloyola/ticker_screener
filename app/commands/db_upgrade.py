import argparse
import sys
from pathlib import Path

from app.container import Container
from app.db import SCHEMA_VERSION

from .base_command import BaseCommand


class DbUpgrade(BaseCommand):
    """
    Brings the database's schema up to date and does nothing else. Every command upgrades the database before it
    runs (main.py), so this one only reports: it is what a release runs before the dashboard serves, since the
    dashboard reads the columns the upgrades add and never upgrades the database itself.
    """

    _NAME = 'db-upgrade'
    _DESCRIPTION = 'upgrades the database schema to the current version, changing nothing else'

    def __init__(self) -> None:
        super().__init__(self._NAME, self._DESCRIPTION)

    def handle(self, args: argparse.Namespace, container: Container) -> None:
        db_file = Path(container.db_path)
        if not db_file.exists():  # upgrade_db leaves a missing file alone; a release needs the database to exist
            print(f"No database at '{db_file}': nothing to upgrade.", file=sys.stderr)
            sys.exit(1)
        print(f"DB at '{db_file}' is at schema version {SCHEMA_VERSION}.")
