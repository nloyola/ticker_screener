from app.container import Container
from app.db import connect

# the dashboard's security master, read here and never written: each ticker's identity and its state
_TABLE = 'security_master'

# the states that take a ticker off the screen: its identity changed and waits for an admin's review, or it no longer
# trades
HELD_STATES = ('flagged', 'inactive')


class SecurityStateRepo:
    """
    The state of each ticker in the dashboard's security master (table security_master, which the dashboard owns and
    its weekly re-verification keeps): active, flagged for an admin's review, or inactive, and the exchange it lists
    on. The screener only reads it, to leave the held tickers out of a run and to name a US ticker's exchange, which
    its bare symbol does not, and a database the dashboard has not yet given the table holds none. The parent's
    GLOSSARY.md ("Security master") explains the states.
    """

    def __init__(self, container: Container):
        self.db_path = container.db_path

    def exchanges(self) -> dict[str, str]:
        """
        Each ticker's exchange as the security master records it from EODHD's fundamentals ('NYSE', 'NASDAQ'), keyed
        by the ticker upper case, leaving out a ticker whose exchange is unknown. Empty when the database has no
        security master.
        """
        with connect(self.db_path) as con:
            found = con.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (_TABLE,)).fetchone()
            if found is None:
                return {}
            rows = con.execute(f"SELECT upper(ticker) AS ticker, exchange FROM {_TABLE} WHERE exchange <> ''")
            return {row['ticker']: row['exchange'] for row in rows}

    def held(self) -> dict[str, str]:
        """
        Each ticker flagged for review or inactive, upper case, mapped to its state. Empty when every ticker is
        active, and when the database has no security master.
        """
        with connect(self.db_path) as con:
            found = con.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (_TABLE,)).fetchone()
            if found is None:
                return {}
            states = ','.join('?' * len(HELD_STATES))
            rows = con.execute(
                f'SELECT upper(ticker) AS ticker, state FROM {_TABLE} WHERE state IN ({states})', HELD_STATES
            )
            return {row['ticker']: row['state'] for row in rows}
