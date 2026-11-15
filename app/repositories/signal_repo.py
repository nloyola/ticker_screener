from collections.abc import Iterable
from datetime import date

from app.container import Container
from app.db import connect
from app.models import Signal
from app.repositories.stock_criteria_repo import COLUMNS as CRITERIA_COLUMNS
from app.repositories.stock_criteria_repo import row_values

# the stock_criteria columns a signal does not copy: it stores the sector and subsector names instead of the
# subsector id, and its session date instead of the day the row was written
_NOT_ARCHIVED = ('subsector_id', 'date_created')
_ARCHIVED = tuple(c for c in CRITERIA_COLUMNS if c not in _NOT_ARCHIVED)
_ARCHIVED_AT = tuple(CRITERIA_COLUMNS.index(c) for c in _ARCHIVED)

_OWN = (
    'market',
    'exchange',
    'time_zone',
    'session_date',
    'close',
    'screener_version',
    'rule_settings',
    'git_commit',
    'run_id',
    'source',
    'previous_signal_id',
    'trading_days_since_previous',
)

# the names of the subsector with the given id and of its sector, when the signal is stored
_SECTOR = 'SELECT se.sector FROM subsector su JOIN sector se ON se.id = su.sector_id WHERE su.id = ?'
_SUBSECTOR = 'SELECT subsector FROM subsector WHERE id = ?'

_INSERT = f"""
INSERT OR IGNORE INTO signal ({','.join((*_OWN, 'sector', 'subsector', *_ARCHIVED))})
VALUES ({','.join(['?'] * len(_OWN))}, ({_SECTOR}), ({_SUBSECTOR}), {','.join(['?'] * len(_ARCHIVED))})
"""


def _values(s: Signal) -> tuple:
    criteria = row_values(s.criteria, s.session_date)
    subsector_id = criteria[CRITERIA_COLUMNS.index('subsector_id')]
    return (
        s.market,
        s.exchange,
        s.time_zone,
        s.session_date.isoformat(),
        float(s.close),
        s.screener_version,
        s.rule_settings,
        s.git_commit,
        s.run_id,
        s.source,
        s.previous_signal_id,
        s.trading_days_since_previous,
        subsector_id,
        subsector_id,
        *(criteria[i] for i in _ARCHIVED_AT),
    )


class SignalRepo:
    """The append-only archive of confirmed breakouts, one row per ticker, session and screener version."""

    def __init__(self, container: Container):
        self.db_path = container.db_path

    def previous(self, ticker: str, before: date) -> tuple[int, date] | None:
        """The id and session date of the ticker's latest signal on a session before 'before', of any version."""
        with connect(self.db_path) as con:
            row = con.execute(
                'SELECT id, session_date FROM signal WHERE ticker = ? AND session_date < ? '
                'ORDER BY session_date DESC, id DESC LIMIT 1',
                (ticker, before.isoformat()),
            ).fetchone()
        return (row['id'], date.fromisoformat(row['session_date'])) if row else None

    def archive(self, signals: Iterable[Signal]) -> int:
        """
        Stores each signal, leaving one already archived for its ticker, session and version as it was, and returns
        how many it stored.
        """
        stored = 0
        with connect(self.db_path) as con:
            for s in signals:
                stored += con.execute(_INSERT, _values(s)).rowcount
        return stored
