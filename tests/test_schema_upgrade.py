"""
price_data.date is a real YYYY-MM-DD date, guaranteed by the schema, price_data has the day's high and low,
stock_criteria has the breakout strategy's columns, core and secondary (trend, momentum and price action), and older
databases are upgraded to all three.
"""

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
# the same rows once upgraded: no high and no low until the next fetch replaces them
UPGRADED_ROWS = [(*row, None, None) for row in OLD_ROWS]

# price_data as schema.sql created it from the date guard (version 1) until the high and low (version 3)
GUARDED_PRICE_DATA = """
CREATE TABLE price_data (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       subsector_id INTEGER,
       ticker TEXT,
       date TEXT NOT NULL CHECK (date(date) IS date),
       close REAL,
       volume INTEGER,
       UNIQUE (ticker, date)
)"""


def _execute(db_path: str, *statements: str) -> None:
    con = open_connection(db_path)
    try:
        with con:
            for statement in statements:
                con.execute(statement)
    finally:
        con.close()


def _query(db_path: str, sql: str, *params) -> list[tuple]:
    con = open_connection(db_path)
    try:
        return con.execute(sql, params).fetchall()
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


def _table_sql(db_path: str, table: str = 'price_data') -> str:
    # a renamed table's stored CREATE statement quotes its new name; the upgrade's indentation differs too
    sql = _query(db_path, 'SELECT sql FROM sqlite_master WHERE name = ?', table)[0][0]
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
    assert _query(old_db, 'SELECT * FROM price_data ORDER BY id') == UPGRADED_ROWS
    assert _query(old_db, "SELECT seq FROM sqlite_sequence WHERE name = 'price_data'") == [(4,)]
    # exactly the table schema.sql creates, UNIQUE (ticker, date) included
    assert _table_sql(old_db) == _table_sql(current_db)
    assert _query(old_db, "PRAGMA index_list('price_data')") == _query(current_db, "PRAGMA index_list('price_data')")
    assert _query(old_db, "SELECT count(*) FROM sqlite_master WHERE name = 'price_data_new'") == [(0,)]

    with pytest.raises(sqlite3.IntegrityError):
        _insert_price(old_db, 'XOM', '2026-11-05 00:00:00')
    _insert_price(old_db, 'XOM', '2026-11-07')
    assert _query(old_db, "SELECT id FROM price_data WHERE date = '2026-11-07'") == [(5,)]


@pytest.fixture
def version_2_db(tmp_path) -> str:
    """A database at version 2: today's schema with price_data guarded but without its high and low."""
    path = str(tmp_path / 'v2.db')
    init_db(path)
    _execute(path, 'DROP TABLE price_data', GUARDED_PRICE_DATA, 'PRAGMA user_version = 2')
    con = open_connection(path)
    try:
        with con:
            con.executemany('INSERT INTO price_data VALUES (?, ?, ?, ?, ?, ?)', OLD_ROWS)
    finally:
        con.close()
    return path


def test_upgrade_adds_high_and_low_keeping_every_row(version_2_db, current_db):
    upgrade_db(version_2_db)

    assert _query(version_2_db, 'PRAGMA user_version') == [(SCHEMA_VERSION,)]
    assert _query(version_2_db, 'SELECT * FROM price_data ORDER BY id') == UPGRADED_ROWS
    # exactly the table schema.sql creates: ALTER TABLE puts the new columns before UNIQUE (ticker, date)
    assert _table_sql(version_2_db) == _table_sql(current_db)
    _execute(
        version_2_db,
        "INSERT INTO price_data (ticker, date, close, volume, high, low) VALUES ('XOM', '2026-11-06', 2, 3, 2.5, 1.5)",
    )
    assert _query(version_2_db, "SELECT high, low FROM price_data WHERE date = '2026-11-06'") == [(2.5, 1.5)]


# stock_criteria as schema.sql created it before the breakout strategy's core criteria (version 4)
CORE_5_STOCK_CRITERIA = """
CREATE TABLE stock_criteria (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       subsector_id INTEGER,
       ticker TEXT NOT NULL,
       price REAL,
       core_criteria_met BOOLEAN,
       core_criteria_score INTEGER,
       high_20d REAL,
       high_60d REAL,
       volume INTEGER,
       avg_vol_20d INTEGER,
       max_vol_10d INTEGER,
       dma_50 REAL,
       ema_10 REAL,
       dma_50_prev_15d REAL,
       dma_50_25d_ago REAL,
       rsi REAL,
       macd REAL,
       signal REAL,
       breakout_20d INTEGER,
       breakout_60d INTEGER,
       breakout_confirmed INTEGER,
       price_gt_50dma INTEGER,
       price_gt_10ema INTEGER,
       dma_50_rising INTEGER,
       rsi_50_75 INTEGER,
       macd_bullish INTEGER,
       date_created DATE NOT NULL DEFAULT (date('now')),
       UNIQUE (ticker)
)"""


@pytest.fixture
def version_3_db(tmp_path) -> str:
    """A database at version 3: today's schema with stock_criteria scoring the five old core criteria, two rows."""
    path = str(tmp_path / 'v3.db')
    init_db(path)
    _execute(
        path,
        'DROP TABLE stock_criteria',
        CORE_5_STOCK_CRITERIA,
        'PRAGMA user_version = 3',
        'INSERT INTO stock_criteria (id, subsector_id, ticker, price, core_criteria_score, max_vol_10d, ema_10, '
        'dma_50_prev_15d, dma_50_25d_ago, price_gt_10ema, date_created) '
        "VALUES (1, 10, 'XOM', 110.5, 5, 900, 108.2, 101.0, 99.5, 1, '2026-11-05')",
        "INSERT INTO stock_criteria (id, subsector_id, ticker, price, date_created) VALUES (2, 10, 'CVX', 150.0, "
        "'2026-11-05')",
    )
    return path


def test_upgrade_reworks_stock_criteria_keeping_every_row(version_3_db, current_db):
    upgrade_db(version_3_db)

    assert _query(version_3_db, 'PRAGMA user_version') == [(SCHEMA_VERSION,)]
    assert _table_sql(version_3_db, 'stock_criteria') == _table_sql(current_db, 'stock_criteria')
    assert _query(
        version_3_db, 'SELECT id, ticker, price, core_criteria_score, dma_50_prev_15d, volume_ratio FROM stock_criteria'
    ) == [(1, 'XOM', 110.5, 5, 101.0, None), (2, 'CVX', 150.0, None, None, None)]


# stock_criteria as schema.sql created it with the four core criteria (version 4), before the secondary indicators
CORE_4_STOCK_CRITERIA = """
CREATE TABLE stock_criteria (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       subsector_id INTEGER,
       ticker TEXT NOT NULL,
       price REAL,
       core_criteria_met BOOLEAN,
       core_criteria_score INTEGER,
       high_20d REAL,
       high_60d REAL,
       volume INTEGER,
       avg_vol_20d INTEGER,
       dma_50 REAL,
       dma_50_prev_15d REAL,
       rsi REAL,
       macd REAL,
       signal REAL,
       breakout_20d INTEGER,
       breakout_60d INTEGER,
       breakout_confirmed INTEGER,
       price_gt_50dma INTEGER,
       dma_50_rising INTEGER,
       rsi_50_75 INTEGER,
       macd_bullish INTEGER,
       date_created DATE NOT NULL DEFAULT (date('now')),
       volume_ratio REAL,
       UNIQUE (ticker)
)"""


@pytest.fixture
def version_4_db(tmp_path) -> str:
    """A database at version 4: today's schema with stock_criteria holding no secondary indicators, two rows."""
    path = str(tmp_path / 'v4.db')
    init_db(path)
    _execute(
        path,
        'DROP TABLE stock_criteria',
        CORE_4_STOCK_CRITERIA,
        'PRAGMA user_version = 4',
        'INSERT INTO stock_criteria (id, subsector_id, ticker, price, core_criteria_score, rsi, rsi_50_75, '
        "macd_bullish, volume_ratio, date_created) VALUES (1, 10, 'XOM', 110.5, 4, 72.5, 1, 0, 1.8, '2026-11-06')",
        "INSERT INTO stock_criteria (id, subsector_id, ticker, price, date_created) VALUES (2, 10, 'CVX', 150.0, "
        "'2026-11-06')",
    )
    return path


def test_upgrade_adds_the_secondary_indicators_keeping_every_row(version_4_db, current_db):
    upgrade_db(version_4_db)

    assert _query(version_4_db, 'PRAGMA user_version') == [(SCHEMA_VERSION,)]
    assert _table_sql(version_4_db, 'stock_criteria') == _table_sql(current_db, 'stock_criteria')
    # the renamed flags keep their stored values, judged on the old RSI band until the next analyze replaces them
    assert _query(
        version_4_db,
        'SELECT id, ticker, core_criteria_score, rsi_50_70, macd_above_signal, volume_ratio, macd_hist, dma_200, '
        'ema_21w_rising FROM stock_criteria',
    ) == [(1, 'XOM', 4, 1, 0, 1.8, None, None, None), (2, 'CVX', None, None, None, None, None, None, None)]


@pytest.fixture
def version_5_db(version_4_db) -> str:
    """A database at version 5: the version 4 one with the trend and momentum indicators, without the price action."""
    con = open_connection(version_4_db)
    try:
        with con:
            db.UPGRADES[4](con)
            con.execute('PRAGMA user_version = 5')
            con.execute("UPDATE stock_criteria SET macd_hist = 0.25, dma_200 = 98.5 WHERE ticker = 'XOM'")
    finally:
        con.close()
    return version_4_db


def test_upgrade_adds_the_price_action_indicators_keeping_every_row(version_5_db, current_db):
    upgrade_db(version_5_db)

    assert _query(version_5_db, 'PRAGMA user_version') == [(SCHEMA_VERSION,)]
    assert _table_sql(version_5_db, 'stock_criteria') == _table_sql(current_db, 'stock_criteria')
    assert _query(
        version_5_db,
        'SELECT id, ticker, core_criteria_score, macd_hist, dma_200, close_position, consolidation_tight, '
        'near_resistance FROM stock_criteria',
    ) == [(1, 'XOM', 4, 0.25, 98.5, None, None, None), (2, 'CVX', None, None, None, None, None, None)]


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
    assert _query(old_db, 'SELECT * FROM price_data ORDER BY id') == UPGRADED_ROWS
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
    assert _query(old_db, 'SELECT * FROM price_data ORDER BY id') == UPGRADED_ROWS


def test_upgrade_reports_its_own_error_when_sqlite_already_rolled_back(old_db, monkeypatch):
    """SQLite ends the transaction itself on some errors; the ROLLBACK after it must not replace the real error."""

    def failing_upgrade(con):
        con.execute('ROLLBACK')  # as SQLite does on a full disk or an I/O error
        raise ValueError('the real error')

    monkeypatch.setattr(db, 'UPGRADES', (failing_upgrade,))
    with pytest.raises(ValueError, match='the real error'):
        upgrade_db(old_db)
    assert _query(old_db, 'PRAGMA user_version') == [(0,)]
