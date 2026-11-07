"""
The price APIs the screener can fetch daily prices from, behind one interface (PriceApi).

Each API lives in its own module (tiingo.py, eodhd.py), and only app/wiring.py picks one, from the PRICE_API config
setting, so either can be deleted without touching the fetcher.
"""

from collections.abc import Collection
from dataclasses import dataclass
from datetime import date
from typing import Protocol


@dataclass(frozen=True)
class DailyPrice:
    """One trading day of a ticker, as stored: the close and the volume, both adjusted for splits."""

    date: date
    close: float
    volume: int


class PriceApiStop(Exception):
    """The price API cannot serve this run, found before any price request: the run stops, saying why."""


class PriceApi(Protocol):
    # the API's name, as log lines and messages show it
    name: str
    # the markets (app.markets) whose tickers it can fetch
    markets: Collection[str]
    # the key it sends, so error messages that quote it back can be redacted
    api_key: str

    def daily_prices(self, ticker: str, start: date, end: date) -> list[DailyPrice]:
        """
        The ticker's rows from 'start' to 'end'; an empty list when the range holds no trading day. Raises requests'
        HTTPError for an error status, requests.Timeout when the API stops answering, and another error for a row
        that is not a price.
        """
        ...

    def check_quota(self, needed: int) -> None:
        """Raises PriceApiStop when the API cannot serve 'needed' price requests today; never costs a request."""
        ...
