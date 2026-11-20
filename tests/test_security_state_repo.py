"""
SecurityStateRepo, which reads the dashboard's security master: each ticker flagged for review or inactive, upper case,
mapped to its state, and none at all when the dashboard has not yet created the table.
"""

import sqlite3
from types import SimpleNamespace

import pytest

from app.db import init_db
from app.repositories.security_state_repo import SecurityStateRepo

# the columns of the dashboard's security_master (its migration 0015) the screener reads; schema.sql never holds it
SECURITY_TABLE = """
CREATE TABLE IF NOT EXISTS security_master (
    ticker TEXT PRIMARY KEY COLLATE NOCASE,
    state  TEXT NOT NULL DEFAULT 'active' CHECK (state IN ('active', 'flagged', 'inactive'))
);
"""


def set_state(path: str, ticker: str, state: str) -> None:
    """Gives 'ticker' a security master row in 'state', as the dashboard's re-verification or an admin does."""
    with sqlite3.connect(path) as con:
        con.executescript(SECURITY_TABLE)
        con.execute(
            'INSERT INTO security_master (ticker, state) VALUES (?, ?) ON CONFLICT (ticker) DO UPDATE SET state = ?',
            (ticker, state, state),
        )


@pytest.fixture
def path(tmp_path) -> str:
    path = str(tmp_path / 'stock_data.db')
    init_db(path)
    return path


def test_a_database_without_the_security_master_holds_no_ticker(path):
    assert SecurityStateRepo(SimpleNamespace(db_path=path)).held() == {}


def test_only_flagged_and_inactive_tickers_are_held_upper_case(path):
    set_state(path, 'XOM', 'active')
    set_state(path, 'key', 'flagged')
    set_state(path, 'PARA', 'inactive')

    assert SecurityStateRepo(SimpleNamespace(db_path=path)).held() == {'KEY': 'flagged', 'PARA': 'inactive'}
