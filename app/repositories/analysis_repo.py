import json
from collections.abc import Iterable
from datetime import UTC, date, datetime

from app.container import Container
from app.db import connect
from app.models import AnalysisResult, AnalysisRun

_RUN_COLUMNS = 'started_at, finished_at, first_session, last_session, signals, settings, screener_version, git_commit'


class AnalysisRepo:
    """
    BreakoutIQ's analysis runs (analysis_run) and their findings (analysis_result). A run is stored with its results
    in one transaction, so a reader never sees a run with only some of them. Its times are stored in UTC, as SQLite's
    datetime('now') writes them, and read back timezone aware.
    """

    def __init__(self, container: Container):
        self.db_path = container.db_path

    def add(self, run: AnalysisRun, results: Iterable[AnalysisResult]) -> int:
        """Stores 'run' with its 'results' and returns the run's id."""
        with connect(self.db_path) as con:
            cursor = con.execute(
                f'INSERT INTO analysis_run ({_RUN_COLUMNS}) VALUES ({", ".join("?" * 8)})',
                (
                    _timestamp(run.started_at),
                    _timestamp(run.finished_at),
                    _day(run.first_session),
                    _day(run.last_session),
                    run.signals,
                    json.dumps(run.settings, sort_keys=True),
                    run.screener_version,
                    run.git_commit,
                ),
            )
            run_id = cursor.lastrowid
            assert run_id is not None
            con.executemany(
                'INSERT INTO analysis_result (run_id, section, finding, horizon, payload_version, payload) '
                'VALUES (?, ?, ?, ?, ?, ?)',
                [
                    (run_id, r.section, r.finding, r.horizon, r.payload_version, json.dumps(r.payload, sort_keys=True))
                    for r in results
                ],
            )
            return run_id

    def runs(self) -> list[AnalysisRun]:
        """Every stored run, oldest first."""
        with connect(self.db_path) as con:
            rows = con.execute(f'SELECT id, {_RUN_COLUMNS} FROM analysis_run ORDER BY id').fetchall()
        return [
            AnalysisRun(
                id=row['id'],
                started_at=row['started_at'].replace(tzinfo=UTC),
                finished_at=row['finished_at'].replace(tzinfo=UTC),
                first_session=_parse_day(row['first_session']),
                last_session=_parse_day(row['last_session']),
                signals=row['signals'],
                settings=json.loads(row['settings']),
                screener_version=row['screener_version'],
                git_commit=row['git_commit'],
            )
            for row in rows
        ]

    def results(self, run_id: int) -> list[AnalysisResult]:
        """The results of run 'run_id', in the order they were stored."""
        with connect(self.db_path) as con:
            rows = con.execute(
                'SELECT section, finding, horizon, payload_version, payload FROM analysis_result '
                'WHERE run_id = ? ORDER BY id',
                (run_id,),
            ).fetchall()
        return [
            AnalysisResult(
                section=row['section'],
                finding=row['finding'],
                horizon=row['horizon'],
                payload_version=row['payload_version'],
                payload=json.loads(row['payload']),
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
