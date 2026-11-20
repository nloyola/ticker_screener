from collections.abc import Collection, Iterable
from dataclasses import dataclass
from datetime import date

from app.container import Container
from app.db import connect
from app.models import CONTINUATION_KIND, PathSession
from app.signal_paths import HORIZONS, PATH_SESSIONS

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
    """
    A record whose path is not frozen, of a kind with a path of its own (not a continuation), with the last session
    of the path it has, None when it has none yet.
    """

    id: int
    ticker: str
    market: str
    session_date: date
    last_index: int | None
    last_date: date | None
    kind: str


class SignalPathRepo:
    """
    Each record's own price path (signal_price) and whether it is frozen (signal_track.path_frozen_at), and the
    horizons of a record whose ticker stopped getting prices, marked ended in signal_outcome, its continuations' with
    them. A continuation has no path of its own.
    """

    def __init__(self, container: Container):
        self.db_path = container.db_path

    def unfrozen(self, markets: Collection[str], source: str | None = None) -> list[TrackedSignal]:
        """
        Every record of 'markets' with a path of its own (not a continuation) whose path is not frozen, of 'source'
        alone when given, oldest session first.
        """
        markets = list(markets)
        if not markets:
            return []
        with connect(self.db_path) as con:
            rows = con.execute(
                f"""
                SELECT s.id, s.ticker, s.market, s.session_date, s.kind,
                       (SELECT MAX(p.session_index) FROM signal_price p WHERE p.signal_id = s.id) AS last_index,
                       (SELECT MAX(p.session_date) FROM signal_price p WHERE p.signal_id = s.id) AS last_date
                FROM signal s
                LEFT JOIN signal_track t ON t.signal_id = s.id
                WHERE t.path_frozen_at IS NULL AND s.market IN ({','.join('?' * len(markets))})
                  AND (? IS NULL OR s.source = ?) AND s.kind <> ?
                ORDER BY s.session_date, s.id
                """,
                [*markets, source, source, CONTINUATION_KIND],
            ).fetchall()
        return [
            TrackedSignal(
                id=row['id'],
                ticker=row['ticker'],
                market=row['market'],
                session_date=date.fromisoformat(row['session_date']),
                last_index=row['last_index'],
                last_date=date.fromisoformat(row['last_date']) if row['last_date'] else None,
                kind=row['kind'],
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
        ended horizons of the signal and of its continuations, since its ticker has prices again; a complete horizon is
        never touched.
        """
        self.write_all([(signal_id, path, frozen)])

    def write_all(self, paths: Iterable[tuple[int, Iterable[PathSession], bool]]) -> None:
        """As write, for each (signal id, path, frozen) of 'paths', all in one transaction."""
        with connect(self.db_path) as con:
            for signal_id, path, frozen in paths:
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
                con.execute(
                    "DELETE FROM signal_outcome WHERE status = 'ended' "
                    'AND (signal_id = ? OR signal_id IN (SELECT id FROM signal WHERE breakout_id = ?))',
                    (signal_id, signal_id),
                )

    def end(self, signal_id: int, horizons: Iterable[int], reason: str, last_session_date: date) -> None:
        """
        Marks each of the signal's 'horizons' ended, with the reason and the last session its path has, leaving a
        complete horizon as it was.
        """
        self.end_all([(signal_id, horizons, reason, last_session_date)])

    def end_all(self, ends: Iterable[tuple[int, Iterable[int], str, date]]) -> None:
        """As end, for each (signal id, horizons, reason, last session date) of 'ends', all in one transaction."""
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
                [
                    (signal_id, h, reason, last_session_date.isoformat())
                    for signal_id, horizons, reason, last_session_date in ends
                    for h in horizons
                ],
            )

    def end_continuations(self, ends: Iterable[tuple[int, int, str, date]]) -> None:
        """
        For each (breakout id, last session index, reason, last session date) of 'ends', a signal whose path stops at
        that index, marks ended each horizon h of each of its continuations, k sessions after it, that the path does
        not reach (k + h beyond the last index) and that the continuation is measured at (k + h at most
        PATH_SESSIONS), all in one transaction, leaving a complete horizon as it was.
        """
        ends = list(ends)
        if not ends:
            return
        rows = []
        with connect(self.db_path) as con:
            for breakout_id, last_index, reason, last_date in ends:
                for row in con.execute(
                    'SELECT id, sessions_since_breakout FROM signal WHERE breakout_id = ?', (breakout_id,)
                ):
                    offset = row['sessions_since_breakout']
                    horizons = [h for h in HORIZONS if last_index < offset + h <= PATH_SESSIONS]
                    rows.append((row['id'], horizons, reason, last_date))
        self.end_all(rows)
