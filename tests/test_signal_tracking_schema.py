"""
The signal tracking tables: signal_price, each signal's price path, signal_outcome, its outcome per horizon, and
signal_track, what is decided once for it. Every database gains them, a market data reset keeps their rows, and
they refuse rows that break their rules.
"""

import sqlite3

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

TABLES = ('signal_price', 'signal_outcome', 'signal_track')

SIGNAL = (
    'INSERT INTO signal (id, ticker, market, time_zone, session_date, close, status, screener_version, '
    "rule_settings, source) VALUES (1, 'XOM', 'US', 'America/New_York', '2026-11-13', 110.5, 'confirmed', '1.0.0', "
    "'{}', 'live')",
)
PRICE = (
    'INSERT INTO signal_price (signal_id, session_index, session_date, open, high, low, close, volume, '
    'benchmark_open, benchmark_close) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'
)
PRICE_ROW = (1, 0, '2026-11-13', 109.0, 111.0, 108.5, 110.5, 1000, 600.0, 602.0)
OUTCOME = (
    'INSERT INTO signal_outcome (signal_id, horizon, status, entry_date, entry_price, exit_date, exit_price, '
    'cost_bps, gross_return_pct, net_return_pct, benchmark_return_pct, excess_return_pct, mfe_pct, mae_pct, '
    'ended_reason, last_session_date, completed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'
)
COMPLETE_ROW = (
    1, 5, 'complete', '2026-11-16', 111.0, '2026-11-20', 115.0, 20, 3.6, 3.4, 1.0, 2.4, 4.5, -1.2, None, None,
    '2026-11-20 23:00:00',
)  # fmt: skip
ENDED_ROW = (1, 20, 'ended', *[None] * 11, 'no prices', '2026-11-18', None)
TRACK = (
    'INSERT INTO signal_track (signal_id, path_frozen_at, breakout_level, breakout_failed, failure_date, '
    'failure_decided_at) VALUES (?, ?, ?, ?, ?, ?)'
)
TRACK_ROW = (1, None, 109.8, 1, '2026-11-17', '2026-11-17 23:00:00')


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


def _primary_key(db_path: str, table: str) -> list[str]:
    columns = sorted((row[5], row[1]) for row in _query(db_path, f"PRAGMA table_info('{table}')") if row[5])
    return [name for _, name in columns]


def _with_rows(db_path: str) -> str:
    _execute(db_path, SIGNAL, (PRICE, *PRICE_ROW), (OUTCOME, *COMPLETE_ROW), (OUTCOME, *ENDED_ROW), (TRACK, *TRACK_ROW))
    return db_path


def _counts(db_path: str) -> list[tuple]:
    return [_query(db_path, f'SELECT count(*) FROM {table}')[0] for table in TABLES]


@pytest.fixture
def db_path(tmp_path) -> str:
    path = str(tmp_path / 'stock_data.db')
    init_db(path)
    return path


@pytest.fixture
def version_11_db(tmp_path) -> str:
    """A database at version 11: today's schema without the tracking tables."""
    path = str(tmp_path / 'v11.db')
    init_db(path)
    _execute(path, *((f'DROP TABLE {table}',) for table in TABLES), ('PRAGMA user_version = 11',))
    return path


def test_new_database_has_the_tables_keyed_by_signal_and_session_index_or_horizon(db_path):
    assert _primary_key(db_path, 'signal_price') == ['signal_id', 'session_index']
    assert _primary_key(db_path, 'signal_outcome') == ['signal_id', 'horizon']
    assert _primary_key(db_path, 'signal_track') == ['signal_id']


def test_upgrade_adds_the_tables_as_schema_sql_creates_them(version_11_db, db_path):
    upgrade_db(version_11_db)

    assert _query(version_11_db, 'PRAGMA user_version') == [(SCHEMA_VERSION,)]
    for table in TABLES:
        assert _schema_sql(version_11_db, table) == _schema_sql(db_path, table)


def test_the_tables_are_not_market_data():
    assert not set(TABLES) & set(MARKET_DATA_TABLES)


def test_clearing_market_data_keeps_their_rows(db_path):
    _with_rows(db_path)
    with SQLite(db_path).connect() as con:
        clear_market_data(con)
    assert _counts(db_path) == [(1,), (2,), (1,)]


def test_a_session_or_horizon_cannot_be_stored_twice(db_path):
    _with_rows(db_path)
    for statement in (
        (PRICE, *PRICE_ROW),
        (PRICE, *PRICE_ROW[:1], 1, *PRICE_ROW[2:]),  # another index on the same date
        (OUTCOME, *COMPLETE_ROW),
        (TRACK, *TRACK_ROW),
    ):
        with pytest.raises(sqlite3.IntegrityError):
            _execute(db_path, statement)


@pytest.mark.parametrize(
    'changes',
    [
        {1: -1},
        {1: 121},
        {2: '2026-11-31'},
        {6: None},  # no close
    ],
)
def test_signal_price_refuses_a_bad_row(db_path, changes):
    _execute(db_path, SIGNAL)
    row = list(PRICE_ROW)
    for index, value in changes.items():
        row[index] = value
    with pytest.raises(sqlite3.IntegrityError):
        _execute(db_path, (PRICE, *row))


@pytest.mark.parametrize(
    ('row', 'changes'),
    [
        (COMPLETE_ROW, {1: 15}),  # not a horizon
        (COMPLETE_ROW, {2: 'done'}),
        (COMPLETE_ROW, {16: None}),  # complete without when
        (COMPLETE_ROW, {6: None}),  # complete without an exit price
        (COMPLETE_ROW, {13: None}),  # complete without the MAE
        (COMPLETE_ROW, {7: -5}),
        (COMPLETE_ROW, {5: '2026-02-30'}),
        (COMPLETE_ROW, {2: 'pending'}),  # pending with a completion time
        (ENDED_ROW, {14: None}),  # ended without a reason
        (ENDED_ROW, {15: None}),  # ended without its last session
        (ENDED_ROW, {2: 'pending'}),  # pending with an end reason
    ],
)
def test_signal_outcome_refuses_a_bad_row(db_path, row, changes):
    _execute(db_path, SIGNAL)
    row = list(row)
    for index, value in changes.items():
        row[index] = value
    with pytest.raises(sqlite3.IntegrityError):
        _execute(db_path, (OUTCOME, *row))


def test_a_pending_outcome_needs_no_measures(db_path):
    _execute(db_path, SIGNAL, (OUTCOME, 1, 40, 'pending', *[None] * 14))
    assert _query(db_path, 'SELECT horizon, status FROM signal_outcome') == [(40, 'pending')]


@pytest.mark.parametrize(
    'changes',
    [
        {3: 2},
        {4: None},  # failed without the session
        {3: 0},  # held, yet with a failure session
        {5: None},  # decided without when
        {3: None, 4: None},  # undecided, yet with a decision time
    ],
)
def test_signal_track_refuses_a_bad_row(db_path, changes):
    _execute(db_path, SIGNAL)
    row = list(TRACK_ROW)
    for index, value in changes.items():
        row[index] = value
    with pytest.raises(sqlite3.IntegrityError):
        _execute(db_path, (TRACK, *row))


def test_signal_track_accepts_an_undecided_and_a_held_breakout(db_path):
    _execute(
        db_path,
        SIGNAL,
        (TRACK, 1, None, 109.8, None, None, None),
    )
    _execute(db_path, ('UPDATE signal_track SET breakout_failed = 0, failure_decided_at = ? WHERE signal_id = 1', 'x'))
    assert _query(db_path, 'SELECT breakout_failed, failure_date FROM signal_track') == [(0, None)]


@pytest.mark.parametrize('table', TABLES)
def test_rows_must_point_at_a_signal_that_exists(db_path, table):
    # through connect(), as the repositories write: SQLite checks foreign keys only on connections that turn them on
    statement = {'signal_price': (PRICE, *PRICE_ROW), 'signal_outcome': (OUTCOME, *COMPLETE_ROW)}.get(
        table, (TRACK, *TRACK_ROW)
    )
    with pytest.raises(sqlite3.IntegrityError), connect(db_path) as con:
        con.execute(statement[0], statement[1:])
