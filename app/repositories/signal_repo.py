from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

from app.container import Container
from app.db import connect
from app.models import SIGNAL_KIND, Signal
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
    'kind',
    'breakout_id',
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
        s.kind,
        s.breakout_id,
        subsector_id,
        subsector_id,
        *(criteria[i] for i in _ARCHIVED_AT),
    )


@dataclass(frozen=True)
class SignalLink:
    """A signal's link to the ticker's previous signal, both None for a ticker's first."""

    id: int
    ticker: str
    session_date: date
    previous_signal_id: int | None
    trading_days_since_previous: int | None


class SignalRepo:
    """
    The append-only signal archive, one record per ticker, session and screener version, each of one kind (KINDS in
    app/models.py). The links between signals, previous and links, read signals of one screener version alone, so
    records scored under different rules never link to each other.
    """

    def __init__(self, container: Container):
        self.db_path = container.db_path

    def previous(self, ticker: str, before: date, screener_version: str) -> tuple[int, date] | None:
        """
        The id and session date of the ticker's latest signal archived under 'screener_version' on a session before
        'before'.
        """
        with connect(self.db_path) as con:
            row = con.execute(
                'SELECT id, session_date FROM signal '
                'WHERE ticker = ? AND session_date < ? AND screener_version = ? AND kind = ? '
                'ORDER BY session_date DESC, id DESC LIMIT 1',
                (ticker, before.isoformat(), screener_version, SIGNAL_KIND),
            ).fetchone()
        return (row['id'], date.fromisoformat(row['session_date'])) if row else None

    def breakouts(self, wanted: Iterable[tuple[str, date]], screener_version: str) -> dict[tuple[str, date], int]:
        """
        The id of each (ticker, session date) of 'wanted' that has a signal archived under 'screener_version', read
        over one connection; one with none is left out.
        """
        found: dict[tuple[str, date], int] = {}
        wanted = set(wanted)
        if not wanted:
            return found
        with connect(self.db_path) as con:
            for ticker, session_date in wanted:
                row = con.execute(
                    'SELECT id FROM signal WHERE ticker = ? AND session_date = ? AND screener_version = ? AND kind = ?',
                    (ticker, session_date.isoformat(), screener_version, SIGNAL_KIND),
                ).fetchone()
                if row:
                    found[(ticker, session_date)] = row['id']
        return found

    def archive(self, records: Iterable[Signal]) -> Counter[str]:
        """
        Stores each record, leaving one already archived for its ticker, session and version as it was, whatever its
        kind, and returns how many it stored of each kind.
        """
        stored: Counter[str] = Counter()
        with connect(self.db_path) as con:
            for s in records:
                stored[s.kind] += con.execute(_INSERT, _values(s)).rowcount
        return stored

    def links(self, tickers: Iterable[str], screener_version: str) -> list[SignalLink]:
        """
        Each signal of 'tickers' archived under 'screener_version', with its previous signal's id and the trading days
        since it, oldest first.
        """
        tickers = list(tickers)
        if not tickers:
            return []
        with connect(self.db_path) as con:
            rows = con.execute(
                'SELECT id, ticker, session_date, previous_signal_id, trading_days_since_previous FROM signal '
                f'WHERE ticker IN ({",".join(["?"] * len(tickers))}) AND screener_version = ? AND kind = ? '
                'ORDER BY ticker, session_date, id',
                [*tickers, screener_version, SIGNAL_KIND],
            ).fetchall()
        return [
            SignalLink(
                id=row['id'],
                ticker=row['ticker'],
                session_date=date.fromisoformat(row['session_date']),
                previous_signal_id=row['previous_signal_id'],
                trading_days_since_previous=row['trading_days_since_previous'],
            )
            for row in rows
        ]

    def relink(self, links: Iterable[SignalLink]) -> int:
        """Stores each link's previous signal and trading days since it, and returns how many signals it changed."""
        changed = 0
        with connect(self.db_path) as con:
            for link in links:
                changed += con.execute(
                    'UPDATE signal SET previous_signal_id = ?, trading_days_since_previous = ? WHERE id = ?',
                    (link.previous_signal_id, link.trading_days_since_previous, link.id),
                ).rowcount
        return changed
