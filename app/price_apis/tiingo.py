"""Daily prices and company names from Tiingo, which covers US tickers only."""

from datetime import date

import pandas as pd
import requests

from app.config import TIINGO
from app.markets import US
from app.price_apis import DailyPrice, daily_price
from app.price_apis.http import CONNECT_TIMEOUT, READ_TIMEOUT, company_name_of

TIINGO_PRICES_URL = 'https://api.tiingo.com/tiingo/daily/{ticker}/prices'
TIINGO_META_URL = 'https://api.tiingo.com/tiingo/daily/{ticker}'


class TiingoPrices:
    """
    Requests a ticker's daily prices from Tiingo's end-of-day endpoint, sending the key in a header. Uses 'adjClose'
    as the close, 'adjHigh' and 'adjLow' as the high and low, and 'adjVolume' as the volume. One request per ticker,
    however long the range.
    """

    name = 'Tiingo'
    setting = TIINGO
    markets = frozenset({US})

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers['Authorization'] = f'Token {api_key}'

    def daily_prices(self, ticker: str, start: date, end: date) -> list[DailyPrice]:
        response = self.session.get(
            TIINGO_PRICES_URL.format(ticker=ticker),
            params={
                'startDate': start.isoformat(),
                'endDate': end.isoformat(),
                'resampleFreq': 'daily',
                'columns': 'date,adjClose,adjHigh,adjLow,adjVolume',
            },
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        )
        response.raise_for_status()
        return self.to_prices(response.json())

    @staticmethod
    def to_prices(rows: list[dict]) -> list[DailyPrice]:
        """
        Tiingo's rows, each with 'date', 'adjClose', 'adjHigh', 'adjLow' and 'adjVolume', as prices; raises for a row
        that is not one.
        """
        return [
            daily_price(
                pd.to_datetime(row['date']).date(),
                close=float(row['adjClose']),
                volume=int(row['adjVolume']),
                high=float(row['adjHigh']),
                low=float(row['adjLow']),
            )
            for row in rows
        ]

    def company_name(self, ticker: str) -> str | None:
        """The name in the ticker's metadata (Tiingo's /tiingo/daily/{ticker}); None for a 404."""
        response = self.session.get(TIINGO_META_URL.format(ticker=ticker), timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return self.to_name(response.json())

    @staticmethod
    def to_name(meta: object) -> str | None:
        """The name in Tiingo's metadata; None when the metadata names no ticker, as Tiingo says it does not know it."""
        if not isinstance(meta, dict):
            raise ValueError(f'unexpected answer from Tiingo metadata: {type(meta).__name__}, not an object')
        if not str(meta.get('ticker') or '').strip():
            return None
        return company_name_of(meta.get('name'))

    def check_quota(self, needed: int) -> None:
        """Tiingo has no usage endpoint that costs nothing: its hourly and daily limits show as failed requests."""
