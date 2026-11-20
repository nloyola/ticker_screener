import argparse
import logging
import sys
from typing import TYPE_CHECKING

from app.breakoutiq.dataset import MissingHistory, UnknownVersion
from app.breakoutiq.run import summarize
from app.breakoutiq.thresholds import DEFAULTS, DESCRIPTIONS, Thresholds, names
from app.config import DB_NAME, HISTORY_DATABASE_FILE
from app.container import Container

from .base_command import BaseCommand

if TYPE_CHECKING:
    from app.breakoutiq.run import Analysis

logger = logging.getLogger(__name__)

# the threshold set per horizon, with HORIZON=PERCENT; every other threshold takes one value of its default's type
BAR = 'min_net_excess_return'


class BreakoutIQCmd(BaseCommand):
    _NAME = 'breakoutiq-analysis'
    _DESCRIPTION = (
        "runs every section of BreakoutIQ's analysis of the signal archive and stores the run in analysis_run, with "
        'its results in analysis_result; reads the regime of backfilled signals from HISTORY_DATABASE_FILE. Each '
        "threshold option replaces that threshold's default for this run, and the run records every threshold it used"
    )

    REQUIRED_CONFIG = (DB_NAME, HISTORY_DATABASE_FILE)

    def __init__(self) -> None:
        super().__init__(self._NAME, self._DESCRIPTION)

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            '--screener-version',
            metavar='VERSION',
            help='the screener version whose archived records to analyse (default the newest in the archive)',
        )
        for name in names():
            option = '--' + name.replace('_', '-')
            default = getattr(DEFAULTS, name)
            if name == BAR:
                bars = ', '.join(f'{h}={bar:g}' for h, bar in default.items())
                parser.add_argument(
                    option,
                    type=_bar,
                    action='append',
                    default=[],
                    metavar='HORIZON=PERCENT',
                    help=f'{DESCRIPTIONS[name]}: one horizon, repeated for several (default {bars})',
                )
            else:
                parser.add_argument(
                    option,
                    type=type(default),
                    metavar=type(default).__name__.upper(),
                    help=f'{DESCRIPTIONS[name]} (default {default:g})',
                )

    def handle(self, args: argparse.Namespace, container: Container) -> None:
        analysis: Analysis = container.breakoutiq_analysis
        try:
            thresholds = _thresholds(args)
        except ValueError as err:
            print(f'breakoutiq-analysis: {err}', file=sys.stderr)
            sys.exit(2)
        changed = [
            f'{name}={value}' for name, value in thresholds.settings().items() if value != DEFAULTS.settings()[name]
        ]
        if changed:
            print(f'breakoutiq-analysis: thresholds changed from the defaults: {", ".join(changed)}')
        try:
            summary = analysis.run(thresholds, args.screener_version)
        except UnknownVersion as err:
            print(f'breakoutiq-analysis: {err}', file=sys.stderr)
            sys.exit(2)
        except MissingHistory as err:
            logger.error(f'breakoutiq-analysis: {err}; set HISTORY_DATABASE_FILE to the file history-fetch wrote')
            sys.exit(1)
        summarize(summary)


def _bar(text: str) -> tuple[int, float]:
    """A HORIZON=PERCENT option's horizon and bar."""
    horizon, sep, bar = text.partition('=')
    try:
        if not sep:
            raise ValueError
        return int(horizon), float(bar)
    except ValueError:
        raise argparse.ArgumentTypeError(f'expected HORIZON=PERCENT, such as 20=0.5, got {text!r}') from None


def _thresholds(args: argparse.Namespace) -> Thresholds:
    """The defaults with every threshold option given replaced; ValueError when one is out of range."""
    given = {name: getattr(args, name) for name in names() if name != BAR and getattr(args, name) is not None}
    return Thresholds(**given).with_bars(dict(getattr(args, BAR)))
