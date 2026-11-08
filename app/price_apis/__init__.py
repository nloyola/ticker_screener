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
    """
    One trading day of a ticker, as stored: the close, the day's high and low and the volume, all adjusted for splits,
    and the prices on one basis, so the high is never below the close nor the low above it (see daily_price).
    """

    date: date
    close: float
    volume: int
    high: float
    low: float


def daily_price(day: date, close: float, volume: int, high: float, low: float) -> DailyPrice:
    """
    The day's price, with the high raised to the close and the low lowered to it where they fall short: rounding in an
    API's adjustment, or EODHD's scaling of the raw high and low, can leave them a hair on the wrong side of the
    adjusted close, and a day's range always holds its close.
    """
    return DailyPrice(date=day, close=close, volume=volume, high=max(high, close), low=min(low, close))


class PriceApiStop(Exception):
    """The price API cannot serve this run, found before any price request: the run stops, saying why."""


class PriceApi(Protocol):
    # the API's name, as log lines and messages show it
    name: str
    # the PRICE_API setting that picks it, as the company_name table records where a name came from
    setting: str
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

    def company_name(self, ticker: str) -> str | None:
        """
        The ticker's company name, or None when the API does not know the ticker or gives it no name. One request
        (EODHD counts each search as one call). Raises requests' HTTPError for an error status, requests.Timeout when
        the API stops answering, and ValueError for an answer of the wrong shape.
        """
        ...

    def check_quota(self, needed: int) -> None:
        """Raises PriceApiStop when the API cannot serve 'needed' price requests today; never costs a request."""
        ...
