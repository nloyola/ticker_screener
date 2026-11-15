"""
The signal table, the breakout archive: every database gains it, a market data reset keeps its rows, it refuses rows
that break its rules, and it keeps every value and flag column stock_criteria has, so a new indicator cannot miss
the archive.
"""

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from app.db import (
    MARKET_DATA_TABLES,
    SCHEMA_VERSION,
    SQLite,
    clear_market_data,
    connect,
    init_db,
    open_connection,
    upgrade_db,
)

MAIN = Path(__file__).resolve().parent.parent / 'main.py'

# the stock_criteria columns a signal does not copy: its own row id, the subsector id (a signal stores the sector
# and subsector names instead, since ids are deleted and reused) and the day the row was written (a signal has the
# session date and archived_at)
NOT_ARCHIVED = {'id', 'subsector_id', 'date_created'}

INSERT = (
    'INSERT INTO signal (ticker, market, time_zone, session_date, close, status, screener_version, rule_settings, '
    'source, previous_signal_id, trading_days_since_previous) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'
)
ROW = ('XOM', 'US', 'America/New_York', '2026-11-13', 110.5, 'confirmed', '1.0.0', '{}', 'live', None, None)


def _execute(db_path: str, *statements: tuple) -> None:
    con = open_connection(db_path)
    try:
        with con:
            for sql, *params in statements:
                con.execute(sql, params)
    finally:
        con.close()


def _query(db_path: str, sql: str) -> list[tuple]:
    con = open_connection(db_path)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def _schema_sql(db_path: str, name: str) -> str:
    sql = _query(db_path, f"SELECT sql FROM sqlite_master WHERE name = '{name}'")[0][0]
    return ' '.join(sql.replace('IF NOT EXISTS ', '').split())


def _columns(db_path: str, table: str) -> dict[str, str]:
    return {row[1]: row[2] for row in _query(db_path, f"PRAGMA table_info('{table}')")}


def _missing_from_signal(db_path: str) -> list[str]:
    """The stock_criteria value and flag columns signal lacks, or has with another type, as 'name TYPE'."""
    signal = _columns(db_path, 'signal')
    return [
        f'{name} {kind}'
        for name, kind in _columns(db_path, 'stock_criteria').items()
        if name not in NOT_ARCHIVED and signal.get(name) != kind
    ]


@pytest.fixture
def db_path(tmp_path) -> str:
    path = str(tmp_path / 'stock_data.db')
    init_db(path)
    return path


@pytest.fixture
def version_10_db(tmp_path) -> str:
    """A database at version 10: today's schema without signal."""
    path = str(tmp_path / 'v10.db')
    init_db(path)
    _execute(path, ('DROP TABLE signal',), ('PRAGMA user_version = 10',))
    return path


def test_new_database_has_the_table_and_its_session_date_index(db_path):
    assert _columns(db_path, 'signal')
    assert _query(db_path, "PRAGMA index_list('signal')")  # the unique index, which also serves lookups by ticker
    assert _query(db_path, "PRAGMA index_info('idx_signal_session_date')") == [(0, 5, 'session_date')]


def test_upgrade_adds_the_table_and_index(version_10_db, db_path):
    upgrade_db(version_10_db)

    assert _query(version_10_db, 'PRAGMA user_version') == [(SCHEMA_VERSION,)]
    assert _schema_sql(version_10_db, 'signal') == _schema_sql(db_path, 'signal')
    assert _schema_sql(version_10_db, 'idx_signal_session_date') == _schema_sql(db_path, 'idx_signal_session_date')


def test_signal_keeps_every_value_and_flag_column_of_stock_criteria(db_path):
    assert _missing_from_signal(db_path) == []
    assert {'secondary_passed', 'status'} <= _columns(db_path, 'signal').keys()


def test_a_column_added_to_stock_criteria_alone_is_reported(db_path):
    _execute(db_path, ('ALTER TABLE stock_criteria ADD COLUMN new_indicator REAL',))
    assert _missing_from_signal(db_path) == ['new_indicator REAL']


def test_signal_is_not_market_data():
    assert 'signal' not in MARKET_DATA_TABLES


def test_clearing_market_data_keeps_the_signals(db_path):
    _execute(db_path, (INSERT, *ROW))
    with SQLite(db_path).connect() as con:
        clear_market_data(con)
    assert _query(db_path, 'SELECT ticker, session_date FROM signal') == [('XOM', '2026-11-13')]


def test_db_init_keeps_the_signals(db_path):
    _execute(db_path, (INSERT, *ROW))
    env = {**os.environ, 'DB_NAME': db_path}
    for key in ('TIINGO_API_KEY', 'EODHD_API_KEY', 'PRICE_API'):
        env.pop(key, None)
    result = subprocess.run(
        [sys.executable, str(MAIN), 'db-init'], env=env, input='y\n', capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    emptied = result.stdout.split('Emptied ')[1].split(';')[0]
    assert 'signal' not in emptied
    assert _query(db_path, 'SELECT ticker, session_date FROM signal') == [('XOM', '2026-11-13')]


def test_a_rerun_of_a_session_cannot_add_a_second_signal(db_path):
    _execute(db_path, (INSERT, *ROW))
    with pytest.raises(sqlite3.IntegrityError):
        _execute(db_path, (INSERT, *ROW[:-3], 'backfill', None, None))
    # the same ticker and session under another screener version is another signal
    _execute(db_path, (INSERT, *ROW[:6], '2.0.0', *ROW[7:]))


def test_a_repeated_signal_links_to_the_previous_one(db_path):
    _execute(db_path, (INSERT, *ROW))
    _execute(db_path, (INSERT, *ROW[:3], '2026-11-20', *ROW[4:-2], 1, 5))
    assert _query(db_path, 'SELECT previous_signal_id, trading_days_since_previous FROM signal ORDER BY id') == [
        (None, None),
        (1, 5),
    ]


@pytest.mark.parametrize(
    'changes',
    [
        {1: 'Mars'},
        {3: '2026-11-31'},
        {4: None},
        {5: 'pre_breakout'},
        {5: None},
        {7: 'not json'},
        {8: 'replay'},
        {9: 1},  # a previous signal without the trading days since it
        {10: 5},  # trading days without a previous signal
    ],
)
def test_signal_refuses_a_bad_row(db_path, changes):
    _execute(db_path, (INSERT, *ROW))
    row = [*ROW[:3], '2026-11-20', *ROW[4:]]
    for index, value in changes.items():
        row[index] = value
    with pytest.raises(sqlite3.IntegrityError):
        _execute(db_path, (INSERT, *row))


def test_a_repeated_signal_must_point_at_a_signal_that_exists(db_path):
    # through connect(), as the repositories write: SQLite checks foreign keys only on connections that turn them on
    with pytest.raises(sqlite3.IntegrityError), connect(db_path) as con:
        con.execute(INSERT, (*ROW[:-2], 99, 5))
