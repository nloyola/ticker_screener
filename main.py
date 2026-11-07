import argparse
import json
import logging.config
import pathlib
import sys

from app.commands.analyzer import AnalyzerCommand
from app.commands.base_command import BaseCommand, CLI_Interface
from app.commands.db_init import DbInit
from app.commands.importer import ImporterCommand
from app.commands.run_refresh_cmd import RunRefreshCmd
from app.commands.ticker_fetcher_cmd import TickerFetcherCmd
from app.commands.ticker_sync_cmd import TickerSyncCmd
from app.config import DB_NAME, ConfigError, load_config
from app.db import upgrade_db
from app.wiring import build_container

logger = logging.getLogger(__name__)

# logging borrowed from here:
# - https://github.com/mCodingLLC/VideosSampleCode


# Define some ANSI color codes
RESET = '\033[0m'
RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: GREEN,
        logging.INFO: RESET,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: RED + '\033[1m',  # bold red
    }

    def format(self, record):
        color = self.COLORS.get(record.levelno, RESET)
        message = super().format(record)
        return f'{color}{message}{RESET}'


# config/ sits beside this file: found from here, so the commands run from any directory
LOGGING_CONFIG = pathlib.Path(__file__).resolve().parent / 'config' / '0-stdout.json'


def setup_logging():
    config = json.loads(LOGGING_CONFIG.read_text(encoding='utf-8'))

    logging.config.dictConfig(config)

    # Replace stdout formatter with color formatter
    console_handler = logging.getHandlerByName('stdout')
    if console_handler:
        console_handler.setFormatter(
            ColorFormatter('%(levelname)s | %(module)s(%(lineno)d) | %(asctime)s | %(message)s')
        )


class CommandLineInterface(CLI_Interface):
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(
            description='A command line interface to the stocks analyzer',
        )
        self.subparsers = self.parser.add_subparsers(dest='command', help='Available commands')

        self.commands = {}

    def add_command(self, name: str, handler: 'BaseCommand', help_text: str) -> None:
        """Registers a new command to the CLI."""
        self.commands[name] = handler
        subparser = self.subparsers.add_parser(name, help=help_text)
        handler.add_arguments(subparser)

    def execute(self) -> None:
        """
        Parses the command line, checks the config keys the named command needs, upgrades the database's schema,
        builds the container and runs the command. A missing key stops here with one line naming it and exit 2,
        before any work.
        """
        args = self.parser.parse_args()
        command = self.commands.get(args.command)
        if command is None:
            self.parser.print_help()
            return
        try:
            config = load_config(require=command.REQUIRED_CONFIG)
        except ConfigError as err:
            print(f'{self.parser.prog} {args.command}: {err}', file=sys.stderr)
            sys.exit(2)
        if DB_NAME in command.REQUIRED_CONFIG:
            upgrade_db(config.db_name)  # before the command opens the database, so it sees the current schema
        command.handle(args, build_container(config))


def main() -> None:
    setup_logging()

    cli = CommandLineInterface()
    DbInit().add_to_cli(cli)
    ImporterCommand().add_to_cli(cli)
    TickerFetcherCmd().add_to_cli(cli)
    TickerSyncCmd().add_to_cli(cli)
    AnalyzerCommand().add_to_cli(cli)
    RunRefreshCmd().add_to_cli(cli)
    cli.execute()


# Main entry point
if __name__ == '__main__':
    main()
