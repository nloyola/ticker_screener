import argparse

from rich.console import Console

console = Console()


class CLI_Interface:
    def add_command(self, name: str, handler: 'BaseCommand', help_text: str) -> None:
        pass


class BaseCommand:
    """Base class for creating commands."""

    def __init__(self, name: str, description: str) -> None:
        self._name = name
        self._description = description

    def name(self):
        return self._name

    def description(self):
        return self._description

    def add_to_cli(self, cli: CLI_Interface) -> None:
        cli.add_command(self._name, self, self._description)

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Hook to add arguments to the subparser."""
        pass

    def handle(self, args: argparse.Namespace) -> None:
        """The logic for handling the command."""
        raise NotImplementedError('Subclasses must implement the handle() method.')
