from datetime import UTC, datetime

import requests
from rich.console import Console

from app.container import Container
from app.models import Ticker

TIINGO_IEX_URL = 'https://api.tiingo.com/iex'

console = Console()


class TickerSync:
    """
    Fetches ticker symbols from Tiingo
    """

    def __init__(self, container: Container) -> None:
        self.metadata_repo = container.metadata_repo
        self.ticker_repo = container.ticker_repo
        self.tiingo_api_key = container.tiingo_api_key

    def fetch_all_tiingo_iex_tickers(self) -> list[dict]:
        headers = {'Content-Type': 'application/json', 'Authorization': f'Token {self.tiingo_api_key}'}
        resp = requests.get(TIINGO_IEX_URL, headers=headers, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def to_ticker(self, obj: dict) -> Ticker:
        return Ticker(
            ticker=obj.get('ticker'),
            last=obj.get('last'),
            prev_close=obj.get('prevClose'),
            volume=obj.get('volume'),
            bid_price=obj.get('bidPrice'),
            ask_price=obj.get('askPrice'),
            timestamp=obj.get('timestamp'),
        )

    def sync(self) -> None:
        self.metadata_repo.set_tickers_fetched_now()
        tickers = self.fetch_all_tiingo_iex_tickers()
        now_iso = datetime.now(UTC).isoformat(timespec='seconds')
        ticker_objs = [self.to_ticker(t) for t in tickers if t.get('ticker')]
        self.ticker_repo.upsert_many(ticker_objs)
        print(f'Upserted {len(ticker_objs)} tickers into DB at {now_iso}')
