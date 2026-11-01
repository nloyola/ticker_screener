"""The stock_analyzer dashboard reads the database while the screener writes it."""

import threading
import time

import pytest

from app.db import BUSY_TIMEOUT_MS, SQLite, init_db, open_connection


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / 'stock_data.db')
    init_db(path)
    return path


def test_connections_wait_for_locks_and_keep_wal(db_path):
    with SQLite(db_path).connect() as con:
        assert con.execute('PRAGMA busy_timeout').fetchone()[0] == BUSY_TIMEOUT_MS
        con.execute("INSERT INTO metadata (key, value) VALUES ('k', 'v')")
    con = open_connection(db_path)
    try:
        assert con.execute('PRAGMA busy_timeout').fetchone()[0] == BUSY_TIMEOUT_MS
        assert con.execute('PRAGMA journal_mode').fetchone()[0] == 'wal'
    finally:
        con.close()


def test_write_succeeds_while_another_connection_holds_a_read_transaction(db_path):
    reader = open_connection(db_path, isolation_level=None)
    try:
        reader.execute('BEGIN')
        assert reader.execute('SELECT count(*) FROM metadata').fetchone()[0] == 0

        started = time.monotonic()
        with SQLite(db_path).connect() as con:
            con.execute("INSERT INTO metadata (key, value) VALUES ('k', 'v')")
        assert time.monotonic() - started < 1, 'the write waited on the reader'

        # the reader keeps its snapshot until its transaction ends
        assert reader.execute('SELECT count(*) FROM metadata').fetchone()[0] == 0
        reader.execute('COMMIT')
        assert reader.execute('SELECT count(*) FROM metadata').fetchone()[0] == 1
    finally:
        reader.close()


def test_write_waits_for_another_writer_instead_of_failing(db_path):
    holder = open_connection(db_path, isolation_level=None, check_same_thread=False)
    holder.execute('BEGIN IMMEDIATE')
    release = threading.Timer(0.5, lambda: (holder.execute('COMMIT'), holder.close()))
    release.start()
    try:
        with SQLite(db_path).connect() as con:
            con.execute("INSERT INTO metadata (key, value) VALUES ('k', 'v')")
    finally:
        release.join()
