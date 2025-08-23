import argparse
import atexit
import json
import logging.config
import logging.handlers
import pathlib

from app.commands.analyzer import AnalyzerCommand
from app.commands.base_command import BaseCommand, CLI_Interface
from app.commands.db_init import DbInit
from app.commands.importer import ImporterCommand
from app.commands.market_value import MarketValueCommand
from app.commands.ticker_fetcher_cmd import TickerFetcherCmd
from app.commands.ticker_sync_cmd import TickerSyncCmd
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


def setup_logging():
    config_file = pathlib.Path('config/0-stdout.json')
    with open(config_file) as f_in:
        config = json.load(f_in)

    logging.config.dictConfig(config)

    # Replace stdout formatter with color formatter
    console_handler = logging.getHandlerByName('stdout')
    if console_handler:
        console_handler.setFormatter(
            ColorFormatter('%(levelname)s | %(module)s(%(lineno)d) | %(asctime)s | %(message)s')
        )

    queue_handler = logging.getHandlerByName('queue_handler')
    if queue_handler is not None:
        queue_handler.listener.start()
        atexit.register(queue_handler.listener.stop)


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
        """Parses and executes the command provided in the command line arguments."""
        args = self.parser.parse_args()
        if args.command in self.commands:
            self.commands[args.command].handle(args)
        else:
            self.parser.print_help()


def main() -> None:
    setup_logging()
    logging.basicConfig(level=logging.INFO)

    container = build_container()

    cli = CommandLineInterface()
    DbInit(container).add_to_cli(cli)
    ImporterCommand(container).add_to_cli(cli)
    TickerFetcherCmd(container).add_to_cli(cli)
    TickerSyncCmd(container).add_to_cli(cli)
    AnalyzerCommand(container).add_to_cli(cli)
    MarketValueCommand().add_to_cli(cli)
    cli.execute()


# Main entry point
if __name__ == '__main__':
    main()
