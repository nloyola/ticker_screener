from app.container import Container
from app.db import connect

# the dashboard's issuer tables, read here and never written: each groups the listings of one company
_TABLES = ('issuer', 'issuer_listing')


class IssuerRepo:
    """
    The issuers the dashboard's admin pages keep (tables issuer and issuer_listing, which the dashboard owns): each
    groups the listings of one company, one of them its primary. The screener only reads them, and a database the
    dashboard has not yet given them has no issuers. The parent's GLOSSARY.md ("Issuer") explains them.
    """

    def __init__(self, container: Container):
        self.db_path = container.db_path

    def non_primary(self) -> dict[str, str]:
        """
        Each listing that is not its issuer's primary, upper case, mapped to that primary, upper case. A ticker no
        issuer holds is its own primary and is left out, so the result is empty when there are no issuers.
        """
        return {ticker: primary for ticker, primary in self.listings().items() if ticker != primary}

    def listings(self) -> dict[str, str]:
        """
        Each listing an issuer holds, its primary included, upper case, mapped to that issuer's primary, upper case.
        A ticker no issuer holds is left out, so the result is empty when there are no issuers.
        """
        with connect(self.db_path) as con:
            found = {
                row['name']
                for row in con.execute(
                    f"SELECT name FROM sqlite_master WHERE type = 'table' AND name IN ({','.join('?' * len(_TABLES))})",
                    _TABLES,
                )
            }
            if found != set(_TABLES):
                return {}
            rows = con.execute(
                """
                SELECT upper(l.ticker) AS ticker, upper(i.primary_ticker) AS primary_ticker
                  FROM issuer_listing l JOIN issuer i ON i.id = l.issuer_id
                """
            )
            return {row['ticker']: row['primary_ticker'] for row in rows}
