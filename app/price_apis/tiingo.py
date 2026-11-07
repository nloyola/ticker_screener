"""Daily prices from Tiingo, which covers US tickers only."""

from datetime import date

import pandas as pd
import requests

from app.markets import US
from app.price_apis import DailyPrice
from app.price_apis.http import CONNECT_TIMEOUT, READ_TIMEOUT

TIINGO_PRICES_URL = 'https://api.tiingo.com/tiingo/daily/{ticker}/prices'


class TiingoPrices:
    """
    Requests a ticker's daily prices from Tiingo's end-of-day endpoint, sending the key in a header. Uses 'adjClose'
    as the close and 'adjVolume' as the volume. One request per ticker, however long the range.
    """

    name = 'Tiingo'
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
                'columns': 'date,adjClose,adjVolume',
            },
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        )
        response.raise_for_status()
        return self.to_prices(response.json())

    @staticmethod
    def to_prices(rows: list[dict]) -> list[DailyPrice]:
        """Tiingo's rows, each with 'date', 'adjClose' and 'adjVolume', as prices; raises for a row that is not one."""
        return [
            DailyPrice(
                date=pd.to_datetime(row['date']).date(), close=float(row['adjClose']), volume=int(row['adjVolume'])
            )
            for row in rows
        ]

    def check_quota(self, needed: int) -> None:
        """Tiingo has no usage endpoint that costs nothing: its hourly and daily limits show as failed requests."""
