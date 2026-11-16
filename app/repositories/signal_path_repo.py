from collections.abc import Collection, Iterable
from dataclasses import dataclass
from datetime import date

from app.container import Container
from app.db import connect
from app.models import PathSession

_COLUMNS = (
    'session_index',
    'session_date',
    'open',
    'high',
    'low',
    'close',
    'volume',
    'benchmark_open',
    'benchmark_close',
)


@dataclass(frozen=True)
class TrackedSignal:
    """A signal whose path is not frozen, with the last session of the path it has, None when it has none yet."""

    id: int
    ticker: str
    market: str
    session_date: date
    last_index: int | None
    last_date: date | None


class SignalPathRepo:
    """
    Each signal's own price path (signal_price) and whether it is frozen (signal_track.path_frozen_at), and the
    horizons of a signal whose ticker stopped getting prices, marked ended in signal_outcome.
    """

    def __init__(self, container: Container):
        self.db_path = container.db_path

    def unfrozen(self, markets: Collection[str]) -> list[TrackedSignal]:
        """Every signal of 'markets' whose path is not frozen, oldest session first."""
        markets = list(markets)
        if not markets:
            return []
        with connect(self.db_path) as con:
            rows = con.execute(
                f"""
                SELECT s.id, s.ticker, s.market, s.session_date,
                       (SELECT MAX(p.session_index) FROM signal_price p WHERE p.signal_id = s.id) AS last_index,
                       (SELECT MAX(p.session_date) FROM signal_price p WHERE p.signal_id = s.id) AS last_date
                FROM signal s
                LEFT JOIN signal_track t ON t.signal_id = s.id
                WHERE t.path_frozen_at IS NULL AND s.market IN ({','.join('?' * len(markets))})
                ORDER BY s.session_date, s.id
                """,
                markets,
            ).fetchall()
        return [
            TrackedSignal(
                id=row['id'],
                ticker=row['ticker'],
                market=row['market'],
                session_date=date.fromisoformat(row['session_date']),
                last_index=row['last_index'],
                last_date=date.fromisoformat(row['last_date']) if row['last_date'] else None,
            )
            for row in rows
        ]

    def path(self, signal_id: int) -> list[PathSession]:
        """The signal's stored path, signal session first."""
        with connect(self.db_path) as con:
            rows = con.execute(
                f'SELECT {",".join(_COLUMNS)} FROM signal_price WHERE signal_id = ? ORDER BY session_index',
                (signal_id,),
            ).fetchall()
        return [PathSession(**{**dict(row), 'session_date': date.fromisoformat(row['session_date'])}) for row in rows]

    def write(self, signal_id: int, path: Iterable[PathSession], frozen: bool) -> None:
        """
        In one transaction: replaces the signal's whole path with 'path', freezes it when 'frozen', and deletes the
        signal's ended horizons, since its ticker has prices again; a complete horizon is never touched.
        """
        with connect(self.db_path) as con:
            con.execute('DELETE FROM signal_price WHERE signal_id = ?', (signal_id,))
            con.executemany(
                f'INSERT INTO signal_price (signal_id, {",".join(_COLUMNS)}) VALUES (?{", ?" * len(_COLUMNS)})',
                [
                    (
                        signal_id,
                        p.session_index,
                        p.session_date.isoformat(),
                        p.open,
                        p.high,
                        p.low,
                        p.close,
                        p.volume,
                        p.benchmark_open,
                        p.benchmark_close,
                    )
                    for p in path
                ],
            )
            con.execute('INSERT OR IGNORE INTO signal_track (signal_id) VALUES (?)', (signal_id,))
            if frozen:
                con.execute(
                    "UPDATE signal_track SET path_frozen_at = datetime('now') WHERE signal_id = ?", (signal_id,)
                )
            con.execute("DELETE FROM signal_outcome WHERE signal_id = ? AND status = 'ended'", (signal_id,))

    def end(self, signal_id: int, horizons: Iterable[int], reason: str, last_session_date: date) -> None:
        """
        Marks each of the signal's 'horizons' ended, with the reason and the last session its path has, leaving a
        complete horizon as it was.
        """
        with connect(self.db_path) as con:
            con.executemany(
                """
                INSERT INTO signal_outcome (signal_id, horizon, status, ended_reason, last_session_date)
                VALUES (?, ?, 'ended', ?, ?)
                ON CONFLICT (signal_id, horizon) DO UPDATE SET
                    status = 'ended', ended_reason = excluded.ended_reason,
                    last_session_date = excluded.last_session_date, updated_at = datetime('now')
                WHERE signal_outcome.status <> 'complete'
                """,
                [(signal_id, h, reason, last_session_date.isoformat()) for h in horizons],
            )
