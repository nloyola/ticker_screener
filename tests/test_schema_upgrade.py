"""price_data.date is a real YYYY-MM-DD date, guaranteed by the schema, and older databases are upgraded to it."""

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from app import db
from app.db import SCHEMA_VERSION, init_db, open_connection, upgrade_db

MAIN = Path(__file__).resolve().parent.parent / 'main.py'

# price_data as schema.sql created it before the guard
OLD_PRICE_DATA = """
CREATE TABLE price_data (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       subsector_id INTEGER,
       ticker TEXT,
       date TEXT,
       close REAL,
       volume INTEGER,
       UNIQUE (ticker, date)
)"""

OLD_ROWS = [
    (1, 10, 'XOM', '2026-11-04', 110.5, 1000),
    (2, 10, 'XOM', '2026-11-05', 111.0, 1200),
    (3, None, 'CVX', '2026-11-05', None, None),
]


def _execute(db_path: str, *statements: str) -> None:
    con = open_connection(db_path)
    try:
        with con:
            for statement in statements:
                con.execute(statement)
    finally:
        con.close()


def _query(db_path: str, sql: str) -> list[tuple]:
    con = open_connection(db_path)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def _dump(db_path: str) -> list[str]:
    con = sqlite3.connect(db_path)
    try:
        return list(con.iterdump())
    finally:
        con.close()


def _insert_price(db_path: str, ticker: str, day) -> None:
    con = open_connection(db_path)
    try:
        with con:
            con.execute('INSERT INTO price_data (ticker, date, close) VALUES (?, ?, 1.0)', (ticker, day))
    finally:
        con.close()


def _table_sql(db_path: str) -> str:
    # a renamed table's stored CREATE statement quotes its new name; the upgrade's indentation differs too
    sql = _query(db_path, "SELECT sql FROM sqlite_master WHERE name = 'price_data'")[0][0]
    return ' '.join(sql.replace('"', '').replace('(', ' ( ').replace(')', ' ) ').split())


@pytest.fixture
def current_db(tmp_path) -> str:
    path = str(tmp_path / 'current.db')
    init_db(path)
    return path


@pytest.fixture
def old_db(tmp_path) -> str:
    """A database from before the guard: today's schema with the old price_data, at user_version 0, whose highest
    id (4) was deleted, so the AUTOINCREMENT sequence is above every id left."""
    path = str(tmp_path / 'old.db')
    init_db(path)
    _execute(path, 'DROP TABLE price_data', OLD_PRICE_DATA, 'PRAGMA user_version = 0')
    con = open_connection(path)
    try:
        with con:
            con.executemany('INSERT INTO price_data VALUES (?, ?, ?, ?, ?, ?)', OLD_ROWS)
            con.execute("INSERT INTO price_data VALUES (4, 10, 'XOM', '2026-11-06', 112.0, 900)")
            con.execute('DELETE FROM price_data WHERE id = 4')
    finally:
        con.close()
    return path


def test_new_database_is_at_the_current_version(current_db):
    assert _query(current_db, 'PRAGMA user_version') == [(SCHEMA_VERSION,)]


@pytest.mark.parametrize(
    'bad',
    [None, '2026-11-05 00:00:00', '2026-11-05T00:00:00', '20261105', '2026-1-05', '2026-02-30', '2026-13-01', ''],
)
def test_insert_refuses_a_date_that_is_not_yyyy_mm_dd(current_db, bad):
    with pytest.raises(sqlite3.IntegrityError):
        _insert_price(current_db, 'XOM', bad)
    assert _query(current_db, 'SELECT count(*) FROM price_data') == [(0,)]


def test_insert_accepts_a_yyyy_mm_dd_date(current_db):
    _insert_price(current_db, 'XOM', '2026-11-05')
    assert _query(current_db, 'SELECT date FROM price_data') == [('2026-11-05',)]


def test_upgrade_rebuilds_an_old_table_keeping_rows_ids_and_sequence(old_db, current_db):
    upgrade_db(old_db)

    assert _query(old_db, 'PRAGMA user_version') == [(SCHEMA_VERSION,)]
    assert _query(old_db, 'SELECT * FROM price_data ORDER BY id') == OLD_ROWS
    assert _query(old_db, "SELECT seq FROM sqlite_sequence WHERE name = 'price_data'") == [(4,)]
    # exactly the table schema.sql creates, UNIQUE (ticker, date) included
    assert _table_sql(old_db) == _table_sql(current_db)
    assert _query(old_db, "PRAGMA index_list('price_data')") == _query(current_db, "PRAGMA index_list('price_data')")
    assert _query(old_db, "SELECT count(*) FROM sqlite_master WHERE name = 'price_data_new'") == [(0,)]

    with pytest.raises(sqlite3.IntegrityError):
        _insert_price(old_db, 'XOM', '2026-11-05 00:00:00')
    _insert_price(old_db, 'XOM', '2026-11-07')
    assert _query(old_db, "SELECT id FROM price_data WHERE date = '2026-11-07'") == [(5,)]


def test_upgrade_keeps_the_sequence_of_an_emptied_table(old_db):
    _execute(old_db, 'DELETE FROM price_data')
    upgrade_db(old_db)
    _insert_price(old_db, 'XOM', '2026-11-07')
    assert _query(old_db, 'SELECT id FROM price_data') == [(5,)]


def test_second_upgrade_changes_nothing(old_db):
    upgrade_db(old_db)
    before = _dump(old_db)
    mtime = os.stat(old_db).st_mtime_ns

    upgrade_db(old_db)

    assert _dump(old_db) == before
    assert os.stat(old_db).st_mtime_ns == mtime


def test_upgrade_with_a_bad_date_fails_and_changes_nothing(old_db):
    _execute(old_db, "INSERT INTO price_data (ticker, date) VALUES ('BAD', '2026-11-05 00:00:00')")
    before = _dump(old_db)

    with pytest.raises(sqlite3.IntegrityError):
        upgrade_db(old_db)

    assert _dump(old_db) == before
    assert _query(old_db, 'PRAGMA user_version') == [(0,)]


def test_upgrade_leaves_a_missing_file_alone(tmp_path):
    path = tmp_path / 'missing.db'
    upgrade_db(str(path))
    assert not path.exists()


def test_upgrade_refuses_a_newer_database(current_db):
    _execute(current_db, f'PRAGMA user_version = {SCHEMA_VERSION + 1}')
    with pytest.raises(RuntimeError, match='newer'):
        upgrade_db(current_db)


def test_init_db_upgrades_an_old_table(old_db):
    init_db(old_db)
    assert _query(old_db, 'PRAGMA user_version') == [(SCHEMA_VERSION,)]
    assert _query(old_db, 'SELECT * FROM price_data ORDER BY id') == OLD_ROWS
    with pytest.raises(sqlite3.IntegrityError):
        _insert_price(old_db, 'XOM', None)


def test_every_command_upgrades_the_database_first(old_db):
    """analyze on an old database with no prices: the upgrade runs before the command opens it."""
    env = {**os.environ, 'DB_NAME': old_db}
    env.pop('TIINGO_API_KEY', None)
    result = subprocess.run(
        [sys.executable, str(MAIN), 'analyze'],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        stdin=subprocess.DEVNULL,
    )
    assert result.returncode == 0, result.stderr
    assert _query(old_db, 'PRAGMA user_version') == [(SCHEMA_VERSION,)]
    assert _query(old_db, 'SELECT * FROM price_data ORDER BY id') == OLD_ROWS


def test_upgrade_reports_its_own_error_when_sqlite_already_rolled_back(old_db, monkeypatch):
    """SQLite ends the transaction itself on some errors; the ROLLBACK after it must not replace the real error."""

    def failing_upgrade(con):
        con.execute('ROLLBACK')  # as SQLite does on a full disk or an I/O error
        raise ValueError('the real error')

    monkeypatch.setattr(db, 'UPGRADES', (failing_upgrade,))
    with pytest.raises(ValueError, match='the real error'):
        upgrade_db(old_db)
    assert _query(old_db, 'PRAGMA user_version') == [(0,)]
