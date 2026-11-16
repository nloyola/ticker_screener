import json
from datetime import UTC, date, datetime

from app.container import Container
from app.db import connect
from app.models import BackfillRun

_COLUMNS = (
    'started_at, finished_at, universe_date, first_session, last_session, sessions, price_api, tickers_listed, '
    'tickers_replayed, tickers_scored, left_out, short_history, signals, relinked, screener_version, git_commit'
)


class BackfillRunRepo:
    """
    Each finished backfill-history run, with what it covered and the biases it carries. Its times are stored in UTC,
    as SQLite's datetime('now') writes them, and read back timezone aware.
    """

    def __init__(self, container: Container):
        self.db_path = container.db_path

    def add(self, run: BackfillRun) -> int:
        """Stores 'run' and returns its id."""
        with connect(self.db_path) as con:
            cursor = con.execute(
                f'INSERT INTO backfill_run ({_COLUMNS}) VALUES ({", ".join("?" * 16)})',
                (
                    _timestamp(run.started_at),
                    _timestamp(run.finished_at),
                    run.universe_date.isoformat(),
                    _day(run.first_session),
                    _day(run.last_session),
                    run.sessions,
                    run.price_api,
                    run.tickers_listed,
                    run.tickers_replayed,
                    run.tickers_scored,
                    json.dumps(run.left_out, sort_keys=True),
                    json.dumps({t: d.isoformat() for t, d in run.short_history.items()}, sort_keys=True),
                    run.signals,
                    run.relinked,
                    run.screener_version,
                    run.git_commit,
                ),
            )
            assert cursor.lastrowid is not None
            return cursor.lastrowid

    def all(self) -> list[BackfillRun]:
        """Every stored run, oldest first."""
        with connect(self.db_path) as con:
            rows = con.execute(f'SELECT id, {_COLUMNS} FROM backfill_run ORDER BY id').fetchall()
        return [
            BackfillRun(
                id=row['id'],
                started_at=row['started_at'].replace(tzinfo=UTC),
                finished_at=row['finished_at'].replace(tzinfo=UTC),
                universe_date=date.fromisoformat(row['universe_date']),
                first_session=_parse_day(row['first_session']),
                last_session=_parse_day(row['last_session']),
                sessions=row['sessions'],
                price_api=row['price_api'],
                tickers_listed=row['tickers_listed'],
                tickers_replayed=row['tickers_replayed'],
                tickers_scored=row['tickers_scored'],
                left_out=json.loads(row['left_out']),
                short_history={t: date.fromisoformat(d) for t, d in json.loads(row['short_history']).items()},
                signals=row['signals'],
                relinked=row['relinked'],
                screener_version=row['screener_version'],
                git_commit=row['git_commit'],
            )
            for row in rows
        ]


def _timestamp(moment: datetime) -> str:
    """'moment' in UTC, as SQLite's datetime('now') writes it."""
    return moment.astimezone(UTC).strftime('%Y-%m-%d %H:%M:%S')


def _day(day: date | None) -> str | None:
    return None if day is None else day.isoformat()


def _parse_day(text: str | None) -> date | None:
    return None if text is None else date.fromisoformat(text)
