"""
The price history database (HISTORY_DATABASE_FILE): years of adjusted daily prices of the tickers and the markets'
benchmarks, for replaying the screener over past sessions. It is a file of its own, never DB_NAME, so years of prices
stay out of price_data's window and out of the live database's backups: it can always be fetched again.

Only this module touches it, and it creates its own tables: the dashboard never reads it, and schema.sql does not
describe it.
"""

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

from app.container import Container
from app.db import connect
from app.price_apis import DailyPrice

# what 'history_fetch.kind' says a row is about, and the table each kind's prices are in
TICKER = 'ticker'
BENCHMARK = 'benchmark'
PRICE_TABLES = {TICKER: 'history_price', BENCHMARK: 'history_benchmark_price'}

# CREATE ... IF NOT EXISTS only, so it runs on every open. A later change to one of these tables must upgrade the
# tables an earlier version created, as app/db.py's UPGRADES does for the live database.
HISTORY_SCHEMA = """
-- each ticker's adjusted daily prices: the close, and the open, high and low on the close's basis
CREATE TABLE IF NOT EXISTS history_price (
    ticker TEXT NOT NULL,
    date   TEXT NOT NULL CHECK (date IS date(date)),
    open   REAL NOT NULL,
    high   REAL NOT NULL,
    low    REAL NOT NULL,
    close  REAL NOT NULL,
    volume INTEGER NOT NULL,
    PRIMARY KEY (ticker, date)
) WITHOUT ROWID;

-- the same, for each market's benchmark (BENCHMARKS in app/markets.py)
CREATE TABLE IF NOT EXISTS history_benchmark_price (
    ticker TEXT NOT NULL,
    date   TEXT NOT NULL CHECK (date IS date(date)),
    open   REAL NOT NULL,
    high   REAL NOT NULL,
    low    REAL NOT NULL,
    close  REAL NOT NULL,
    volume INTEGER NOT NULL,
    PRIMARY KEY (ticker, date)
) WITHOUT ROWID;

-- the day each ticker or benchmark was last fetched, and from which price API (the PRICE_API setting), so a fetch
-- run again the same day, after an interruption, skips those already done
CREATE TABLE IF NOT EXISTS history_fetch (
    kind       TEXT NOT NULL CHECK (kind IN ('ticker', 'benchmark')),
    ticker     TEXT NOT NULL,
    fetched_on TEXT NOT NULL CHECK (fetched_on IS date(fetched_on)),
    api        TEXT NOT NULL,
    PRIMARY KEY (kind, ticker)
) WITHOUT ROWID;
"""


@dataclass(frozen=True)
class Coverage:
    """The stored history of one ticker or benchmark: its first and last day and how many days it holds."""

    first: date
    last: date
    days: int


def init_history_db(db_path: str) -> None:
    """Creates the history database's tables that are missing; the file too, when there is none."""
    con = sqlite3.connect(db_path)
    try:
        con.executescript(HISTORY_SCHEMA)
    finally:
        con.close()


class HistoryRepo:
    """The history database's tables. Creates the ones missing when it is built."""

    def __init__(self, container: Container):
        self.db_path = container.history_db_path
        init_history_db(self.db_path)

    def fetched_on(self, kind: str, day: date, api: str) -> set[str]:
        """The tickers of 'kind' fetched on 'day' from the price API 'api'."""
        with connect(self.db_path) as con:
            rows = con.execute(
                'SELECT ticker FROM history_fetch WHERE kind = ? AND fetched_on = ? AND api = ?',
                (kind, day.isoformat(), api),
            ).fetchall()
        return {row['ticker'] for row in rows}

    def store(self, kind: str, ticker: str, prices: Iterable[DailyPrice], day: date, api: str) -> None:
        """
        Replaces every stored day of the ticker with 'prices', and records it as fetched on 'day' from 'api', in one
        transaction: an interrupted fetch leaves each ticker either as it was or fully replaced and recorded.
        """
        table = PRICE_TABLES[kind]
        with connect(self.db_path) as con:
            con.execute(f'DELETE FROM {table} WHERE ticker = ?', (ticker,))
            con.executemany(
                f'INSERT INTO {table} (ticker, date, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)',
                [(ticker, p.date.isoformat(), p.open, p.high, p.low, p.close, p.volume) for p in prices],
            )
            self._record(con, kind, ticker, day, api)

    def mark_fetched(self, kind: str, ticker: str, day: date, api: str) -> None:
        """Records the ticker as fetched on 'day' from 'api' without changing its stored prices."""
        with connect(self.db_path) as con:
            self._record(con, kind, ticker, day, api)

    def coverage(self, kind: str) -> dict[str, Coverage]:
        """Each stored ticker of 'kind' with its stored history."""
        with connect(self.db_path) as con:
            rows = con.execute(
                f'SELECT ticker, MIN(date) AS first, MAX(date) AS last, COUNT(*) AS days '
                f'FROM {PRICE_TABLES[kind]} GROUP BY ticker'
            ).fetchall()
        return {
            row['ticker']: Coverage(date.fromisoformat(row['first']), date.fromisoformat(row['last']), row['days'])
            for row in rows
        }

    def prices(self, kind: str, ticker: str) -> list[DailyPrice]:
        """The ticker's stored days, oldest first."""
        with connect(self.db_path) as con:
            rows = con.execute(
                f'SELECT date, open, high, low, close, volume FROM {PRICE_TABLES[kind]} WHERE ticker = ? ORDER BY date',
                (ticker,),
            ).fetchall()
        return [
            DailyPrice(
                date=date.fromisoformat(row['date']),
                close=row['close'],
                volume=row['volume'],
                high=row['high'],
                low=row['low'],
                open=row['open'],
            )
            for row in rows
        ]

    @staticmethod
    def _record(con: sqlite3.Connection, kind: str, ticker: str, day: date, api: str) -> None:
        con.execute(
            'INSERT INTO history_fetch (kind, ticker, fetched_on, api) VALUES (?, ?, ?, ?) '
            'ON CONFLICT (kind, ticker) DO UPDATE SET fetched_on = excluded.fetched_on, api = excluded.api',
            (kind, ticker, day.isoformat(), api),
        )
