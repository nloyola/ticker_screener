"""The company_name table: every database gains it, db-init keeps its rows, and its repository."""

import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.db import SCHEMA_VERSION, init_db, open_connection, upgrade_db
from app.repositories.company_name_repo import CompanyNameRepo

MAIN = Path(__file__).resolve().parent.parent / 'main.py'


def _query(db_path: str, sql: str) -> list[tuple]:
    con = open_connection(db_path)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def _table_sql(db_path: str) -> str:
    sql = _query(db_path, "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'company_name'")[0][0]
    return ' '.join(sql.replace('IF NOT EXISTS ', '').split())


def _run(db_path: str, *args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, 'DB_NAME': db_path}
    for key in ('TIINGO_API_KEY', 'EODHD_API_KEY', 'PRICE_API'):
        env.pop(key, None)
    return subprocess.run(
        [sys.executable, str(MAIN), *args],
        env=env,
        input=stdin,
        capture_output=True,
        text=True,
        timeout=120,
        stdin=None if stdin is not None else subprocess.DEVNULL,
    )


@pytest.fixture
def db_path(tmp_path) -> str:
    path = str(tmp_path / 'stock_data.db')
    init_db(path)
    return path


@pytest.fixture
def repo(db_path) -> CompanyNameRepo:
    return CompanyNameRepo(SimpleNamespace(db_path=db_path))


@pytest.fixture
def version_1_db(tmp_path) -> str:
    """A database from before the table: today's schema without company_name, at user_version 1."""
    path = str(tmp_path / 'v1.db')
    init_db(path)
    con = open_connection(path)
    try:
        con.execute('DROP TABLE company_name')
        con.execute('PRAGMA user_version = 1')
    finally:
        con.close()
    return path


def test_a_new_database_has_the_empty_table(db_path):
    assert _query(db_path, 'SELECT count(*) FROM company_name') == [(0,)]


def test_the_upgrade_creates_the_table_schema_sql_creates(version_1_db, db_path):
    upgrade_db(version_1_db)
    assert _query(version_1_db, 'PRAGMA user_version') == [(SCHEMA_VERSION,)]
    assert _query(version_1_db, 'SELECT count(*) FROM company_name') == [(0,)]
    assert _table_sql(version_1_db) == _table_sql(db_path)


def test_every_command_gains_the_table_first(version_1_db):
    result = _run(version_1_db, 'analyze')
    assert result.returncode == 0, result.stderr
    assert _query(version_1_db, 'SELECT count(*) FROM company_name') == [(0,)]


def test_db_init_keeps_the_names(db_path, repo):
    repo.store('XOM', 'Exxon Mobil Corp', 'tiingo')
    result = _run(db_path, 'db-init', stdin='y\n')
    assert result.returncode == 0, result.stderr
    assert 'company_name' not in result.stdout  # not among the tables it says it emptied
    assert repo.get('XOM') == 'Exxon Mobil Corp'


def test_missing_lists_the_tickers_with_no_name_in_order_once(repo):
    repo.store('XOM', 'Exxon Mobil Corp', 'tiingo')
    assert repo.missing(['SU.TO', 'XOM', 'CVX', 'su.to', 'xom']) == ['SU.TO', 'CVX']
    assert repo.missing([]) == []


def test_store_matches_tickers_case_insensitively_and_replaces(repo, db_path):
    repo.store('su.to', 'Suncor', 'eodhd')
    repo.store('SU.TO', 'Suncor Energy Inc', 'eodhd')
    assert repo.get('Su.To') == 'Suncor Energy Inc'
    assert _query(db_path, 'SELECT count(*) FROM company_name') == [(1,)]
    assert repo.missing(['SU.TO']) == []


def test_get_without_a_name_is_none(repo):
    assert repo.get('XOM') is None


def test_store_refuses_an_unknown_price_api(repo):
    with pytest.raises(sqlite3.IntegrityError):
        repo.store('XOM', 'Exxon Mobil Corp', 'yahoo')
