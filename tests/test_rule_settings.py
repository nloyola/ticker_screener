"""The rule settings snapshot (app.analyzer.rule_settings) that travels with SCREENER_VERSION.

The snapshot is written out by hand, so these tests read the analyzer's source for its module-level constants and
fail when one is neither in the snapshot nor named below as not being a rule setting: a threshold added to the
analyzer without being listed would otherwise leave the snapshot silently short of the rules it claims to describe.
"""

import ast
import json
import re
from pathlib import Path

import app.analyzer as analyzer
from app.analyzer import SCREENER_VERSION, rule_settings

ANALYZER_SOURCE = Path(analyzer.__file__).read_text()

# The analyzer's module-level constants that are not rule settings: the version the snapshot travels with, the
# names of the statuses, which label a result rather than decide it, and the statuses archived as signals with the
# sources a signal is archived from, which say where a result is kept.
NOT_RULE_SETTINGS = {
    'SCREENER_VERSION',
    'HIGH_CONVICTION',
    'CONFIRMED',
    'PRE_BREAKOUT',
    'SIGNAL_STATUSES',
    'LIVE',
    'BACKFILL',
}


def module_constants(source: str) -> set[str]:
    """The upper-case names assigned at the top level of 'source'."""
    names: set[str] = set()
    for node in ast.parse(source).body:
        targets = (
            node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        names.update(t.id for t in targets if isinstance(t, ast.Name) and t.id.isupper())
    return names


def unlisted(source: str) -> set[str]:
    """The constants of 'source' missing from the snapshot and not named in NOT_RULE_SETTINGS."""
    return module_constants(source) - NOT_RULE_SETTINGS - set(rule_settings())


def test_snapshot_lists_every_threshold_constant():
    assert unlisted(ANALYZER_SOURCE) == set()


def test_snapshot_lists_only_analyzer_constants():
    assert set(rule_settings()) == module_constants(ANALYZER_SOURCE) - NOT_RULE_SETTINGS


def test_a_new_constant_left_out_of_the_snapshot_is_caught():
    source = ANALYZER_SOURCE + '\nGAP_DOWN_MAX_PCT = 3.0\n'
    assert unlisted(source) == {'GAP_DOWN_MAX_PCT'}


def test_snapshot_values_are_the_analyzers():
    for name, value in rule_settings().items():
        current = getattr(analyzer, name)
        expected = list(current) if isinstance(current, tuple | dict) else current
        assert value == expected, name


def test_snapshot_round_trips_through_json():
    settings = rule_settings()
    assert json.loads(json.dumps(settings)) == settings


def test_version_is_semver():
    assert re.fullmatch(r'\d+\.\d+\.\d+', SCREENER_VERSION)
