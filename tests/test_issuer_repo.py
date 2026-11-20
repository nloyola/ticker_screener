"""
IssuerRepo, which reads the dashboard's issuers: each non-primary listing mapped to its primary, matched without
regard to case, and none at all when the dashboard has not yet created the tables.
"""

import sqlite3
from types import SimpleNamespace

import pytest

from app.db import init_db
from app.repositories.issuer_repo import IssuerRepo

# the dashboard's issuer tables as its migration 0014 creates them; the screener's schema.sql never holds them
ISSUER_TABLES = """
CREATE TABLE IF NOT EXISTS issuer (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    name           TEXT NOT NULL,
    primary_ticker TEXT NOT NULL COLLATE NOCASE
);
CREATE TABLE IF NOT EXISTS issuer_listing (
    ticker    TEXT PRIMARY KEY COLLATE NOCASE,
    issuer_id INTEGER NOT NULL REFERENCES issuer(id) ON DELETE CASCADE
);
"""


def add_issuer(path: str, name: str, primary: str, *listings: str) -> None:
    """Groups 'listings' under a new issuer called 'name' whose primary listing is 'primary', as the dashboard does."""
    with sqlite3.connect(path) as con:
        con.executescript(ISSUER_TABLES)
        issuer_id = con.execute('INSERT INTO issuer (name, primary_ticker) VALUES (?, ?)', (name, primary)).lastrowid
        con.executemany(
            'INSERT INTO issuer_listing (ticker, issuer_id) VALUES (?, ?)', [(t, issuer_id) for t in listings]
        )


@pytest.fixture
def path(tmp_path) -> str:
    path = str(tmp_path / 'stock_data.db')
    init_db(path)
    return path


def test_a_database_without_the_issuer_tables_has_no_issuers(path):
    assert IssuerRepo(SimpleNamespace(db_path=path)).non_primary() == {}


def test_empty_issuer_tables_give_no_issuers(path):
    with sqlite3.connect(path) as con:
        con.executescript(ISSUER_TABLES)

    assert IssuerRepo(SimpleNamespace(db_path=path)).non_primary() == {}


def test_each_non_primary_listing_maps_to_its_primary_upper_case(path):
    add_issuer(path, 'Stantec', 'STN', 'STN', 'stn.to')
    add_issuer(path, 'Alphabet', 'googl', 'GOOG', 'GOOGL')

    assert IssuerRepo(SimpleNamespace(db_path=path)).non_primary() == {'STN.TO': 'STN', 'GOOG': 'GOOGL'}
