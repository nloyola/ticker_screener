"""Daily prices from EODHD, which covers US and Canadian tickers."""

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime

import requests

from app.markets import CANADA, US, market_of
from app.price_apis import DailyPrice, PriceApiStop
from app.price_apis.http import CONNECT_TIMEOUT, READ_TIMEOUT, error_detail, response_of
from app.redaction import redact

logger = logging.getLogger(__name__)

EODHD_EOD_URL = 'https://eodhd.com/api/eod/{symbol}'
EODHD_USER_URL = 'https://eodhd.com/api/user'
# the usage check is a courtesy before the run: it must not hold the run up for long
USAGE_TIMEOUT = (5.0, 10.0)


def eodhd_symbol(ticker: str) -> str:
    """The symbol EODHD knows a ticker by: a bare (US) ticker gets '.US', a Canadian one is sent as written."""
    return ticker if market_of(ticker) == CANADA else f'{ticker}.US'


@dataclass(frozen=True)
class EodhdUsage:
    """The account's usage, as EODHD's /api/user reports it."""

    spent: int  # calls spent on 'day'
    day: str  # the day 'spent' belongs to (GMT); an older day means nothing was spent since midnight GMT
    daily_limit: int
    extra: int  # extra calls bought, used once the daily limit is spent

    def calls_left(self, today: date) -> int:
        """The calls the account can still spend today: what is left of the daily limit, plus the extra calls."""
        spent = self.spent if self.day == today.isoformat() else 0
        return max(self.daily_limit - spent, 0) + self.extra


class EodhdPrices:
    """
    Requests a ticker's daily prices from EODHD's end-of-day endpoint, sending the key as the 'api_token' query
    parameter: every URL it builds holds the key, so any error quoting one must go through error_detail() or redact().

    Uses 'adjusted_close' (adjusted for splits and dividends) as the close and 'volume' (adjusted for splits) as the
    volume; the raw OHLC fields are not used. One call per ticker, whatever the range; an unknown ticker answers 404
    and still costs a call.
    """

    name = 'EODHD'
    markets = frozenset({US, CANADA})

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.session = requests.Session()

    def daily_prices(self, ticker: str, start: date, end: date) -> list[DailyPrice]:
        response = self.session.get(
            EODHD_EOD_URL.format(symbol=eodhd_symbol(ticker)),
            params={'from': start.isoformat(), 'to': end.isoformat(), 'fmt': 'json', 'api_token': self.api_key},
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        )
        response.raise_for_status()
        return self.to_prices(response.json())

    @staticmethod
    def to_prices(rows: list[dict]) -> list[DailyPrice]:
        """EODHD's rows as prices; raises for a row that is not one."""
        return [
            DailyPrice(
                date=date.fromisoformat(row['date']), close=float(row['adjusted_close']), volume=int(row['volume'])
            )
            for row in rows
        ]

    def usage(self) -> EodhdUsage:
        """The account's usage from /api/user, which costs no call. Raises requests' errors, or ValueError."""
        response = self.session.get(
            EODHD_USER_URL, params={'api_token': self.api_key, 'fmt': 'json'}, timeout=USAGE_TIMEOUT
        )
        response.raise_for_status()
        body = response.json()
        try:
            return EodhdUsage(
                spent=int(body['apiRequests']),
                day=str(body.get('apiRequestsDate') or ''),
                daily_limit=int(body['dailyRateLimit']),
                extra=int(body.get('extraLimit') or 0),
            )
        except (KeyError, TypeError) as e:
            raise ValueError(f'unexpected answer from EODHD /api/user: missing or bad field {e}') from e

    def check_quota(self, needed: int) -> None:
        """
        Stops the run (PriceApiStop) when EODHD has fewer calls left today than 'needed', or refuses the key. Any
        other failure of the usage check is logged as a warning, and the run carries on.
        """
        try:
            usage = self.usage()
        except Exception as e:
            response = response_of(e)
            detail = redact(error_detail(e), self.api_key)
            if response is not None and response.status_code in (401, 403):
                raise PriceApiStop(f'EODHD refused the API key, no price was requested: {detail}') from None
            logger.warning(f'EODHD usage check failed, carrying on without it: {detail}')
            return

        left = usage.calls_left(datetime.now(UTC).date())
        if left < needed:
            raise PriceApiStop(
                f'EODHD has {left} calls left today ({usage.spent} of {usage.daily_limit} spent on {usage.day}, '
                f'{usage.extra} extra), fewer than the {needed} tickers to fetch: no price was requested'
            )
        logger.info(f'EODHD has {left} calls left today, {needed} needed')
