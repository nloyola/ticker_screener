"""
backfill_run, each backfill-history run's coverage and biases: every database gains it, a market data reset keeps its
rows, and it refuses rows that break its rules.
"""

import sqlite3
from datetime import UTC, date, datetime

import pytest

from app.db import MARKET_DATA_TABLES, SCHEMA_VERSION, SQLite, clear_market_data, init_db, upgrade_db
from app.models import BackfillRun
from app.repositories.backfill_run_repo import BackfillRunRepo


def _query(path: str, sql: str) -> list[tuple]:
    with sqlite3.connect(path) as con:
        return con.execute(sql).fetchall()


def _schema_sql(path: str) -> str:
    sql = _query(path, "SELECT sql FROM sqlite_master WHERE name = 'backfill_run'")[0][0]
    return ' '.join(sql.split())


@pytest.fixture
def db_path(tmp_path) -> str:
    path = str(tmp_path / 'stock_data.db')
    init_db(path)
    return path


@pytest.fixture
def version_12_db(tmp_path) -> str:
    """A database at version 12: today's schema without backfill_run."""
    path = str(tmp_path / 'v12.db')
    init_db(path)
    with sqlite3.connect(path) as con:
        con.execute('DROP TABLE backfill_run')
        con.execute('PRAGMA user_version = 12')
    return path


def _run(**changes) -> BackfillRun:
    values = {
        'started_at': datetime(2026, 11, 17, 1, 0, tzinfo=UTC),
        'finished_at': datetime(2026, 11, 17, 1, 40, 5, tzinfo=UTC),
        'universe_date': date(2026, 11, 16),
        'first_session': date(2017, 9, 1),
        'last_session': date(2026, 11, 16),
        'sessions': 2310,
        'price_api': 'eodhd',
        'tickers_listed': 3,
        'tickers_replayed': 2,
        'tickers_scored': 2,
        'left_out': {'NOHIST': 'no_history'},
        'short_history': {'NEWCO': date(2021, 3, 4)},
        'signals': 812,
        'relinked': 4,
        'screener_version': '1.0.0',
        'git_commit': 'abc1234',
    }
    return BackfillRun(**{**values, **changes})


def test_upgrade_adds_the_table_as_schema_sql_creates_it(version_12_db, db_path):
    upgrade_db(version_12_db)

    assert _query(version_12_db, 'PRAGMA user_version') == [(SCHEMA_VERSION,)]
    assert _schema_sql(version_12_db) == _schema_sql(db_path)


def test_the_table_is_not_market_data(db_path):
    assert 'backfill_run' not in MARKET_DATA_TABLES
    BackfillRunRepo(SimpleContainer(db_path)).add(_run())
    with SQLite(db_path).connect() as con:
        clear_market_data(con)
    assert _query(db_path, 'SELECT COUNT(*) FROM backfill_run') == [(1,)]


def test_a_run_reads_back_as_stored(db_path):
    repo = BackfillRunRepo(SimpleContainer(db_path))
    first = repo.add(_run())
    second = repo.add(_run(first_session=None, last_session=None, sessions=0, price_api=None, short_history={}))

    assert repo.all() == [
        _run(id=first),
        _run(id=second, first_session=None, last_session=None, sessions=0, price_api=None, short_history={}),
    ]


@pytest.mark.parametrize(
    'changes',
    [
        {'first_session': None},
        {'sessions': 0},
        {'first_session': None, 'last_session': None},
        {'signals': -1},
    ],
)
def test_the_table_refuses_an_inconsistent_range_or_count(db_path, changes):
    with pytest.raises(sqlite3.IntegrityError):
        BackfillRunRepo(SimpleContainer(db_path)).add(_run(**changes))


class SimpleContainer:
    def __init__(self, db_path: str):
        self.db_path = db_path
