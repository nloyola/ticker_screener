import argparse

from app.commands.analyzer import AnalyzerCommand
from app.commands.base_command import BaseCommand, CLI_Interface
from app.commands.db_init import DbInit
from app.commands.importer import ImporterCommand
from app.commands.market_value import MarketValueCommand
from app.commands.ticker_fetcher import TickerFetcherCommand
from app.wiring import build_container


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
    container = build_container()

    cli = CommandLineInterface()
    DbInit().add_to_cli(cli)
    ImporterCommand(container).add_to_cli(cli)
    TickerFetcherCommand(container).add_to_cli(cli)
    AnalyzerCommand(container).add_to_cli(cli)
    MarketValueCommand().add_to_cli(cli)
    cli.execute()


# Main entry point
if __name__ == '__main__':
    main()
