"""The stock_analyzer dashboard reads the database while the screener writes it."""

import sqlite3
import threading
import time
from datetime import date
from types import SimpleNamespace

import pytest

from app.db import BUSY_TIMEOUT_MS, SQLite, init_db, open_connection
from app.models import PriceData, StockCriteria
from app.repositories.metadata_repo import MetadataRepo
from app.repositories.price_data_repo import PriceDataRepo
from app.repositories.sector_repo import SectorRepo
from app.repositories.stock_criteria_repo import StockCriteriaRepo
from app.repositories.subsector_repo import SubsectorRepo
from app.repositories.ticker_repo import TickerRepo


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


@pytest.mark.parametrize(
    'repo_class', [MetadataRepo, PriceDataRepo, SectorRepo, StockCriteriaRepo, SubsectorRepo, TickerRepo]
)
def test_repositories_close_their_connections(db_path, repo_class):
    repo = repo_class(SimpleNamespace(db_path=db_path))
    with repo._connect() as con:
        con.execute('SELECT 1')
    with pytest.raises(sqlite3.ProgrammingError, match='closed'):
        con.execute('SELECT 1')


def test_repositories_commit_on_success_and_roll_back_on_error(db_path):
    repo = MetadataRepo(SimpleNamespace(db_path=db_path))
    repo.set('kept', 'v')
    with pytest.raises(RuntimeError):
        with repo._connect() as con:
            con.execute("INSERT INTO metadata (key, value) VALUES ('dropped', 'v')")
            raise RuntimeError
    assert repo.get('kept') == 'v'
    assert repo.get('dropped') is None


def test_dates_round_trip_on_every_connection(db_path):
    """
    Dates are stored as ISO text and read back as dates: the DATE adapter and converter are registered once, so a
    DATE column reads back as a date on any connection, and PriceDataRepo parses price_data.date, declared TEXT.
    """
    container = SimpleNamespace(db_path=db_path)
    PriceDataRepo(container).replace_for_ticker(
        'XOM', [PriceData(id=0, subsector_id=1, ticker='XOM', date=date(2026, 11, 5), close=110.5, volume=1000)]
    )
    StockCriteriaRepo(container).bulk_insert([StockCriteria(id=0, subsector_id=1, ticker='XOM')], date(2026, 11, 6))

    assert [p.date for p in PriceDataRepo(container).all_for_ticker('XOM')] == [date(2026, 11, 5)]
    with SQLite(db_path).connect() as con:
        assert con.execute('SELECT date_created FROM stock_criteria').fetchone()[0] == date(2026, 11, 6)
    raw = open_connection(db_path)
    try:
        assert raw.execute('SELECT date, typeof(date) FROM price_data').fetchone() == ('2026-11-05', 'text')
    finally:
        raw.close()
