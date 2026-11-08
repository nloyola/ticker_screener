"""
The market a ticker trades on, read from the ticker itself: one rule for the whole screener.

Canadian tickers are stored as EODHD writes them, with their exchange's suffix ('SHOP.TO' on the Toronto Stock
Exchange, 'ABC.V' on the TSX Venture). US tickers are bare ('AAPL'). Share classes use a dash ('BBD-B.TO'), never a
dot, so any other dot is not a ticker the screener can fetch.
"""

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

US = 'US'
CANADA = 'Canada'

# every market, in the order summaries list them
MARKETS = (US, CANADA)

# the exchange suffixes the screener accepts, and their market
SUFFIX_MARKETS = {'TO': CANADA, 'V': CANADA}

# each market's regular close, in its exchanges' time zone: NYSE and NASDAQ, and the TSX and TSX Venture
MARKET_CLOSES = {
    US: (ZoneInfo('America/New_York'), time(16)),
    CANADA: (ZoneInfo('America/Toronto'), time(16)),
}

# each market's broad-market benchmark, an index fund every fetch downloads beside the tickers so the analyzer can
# compare their returns with the market's (relative strength): SPDR S&P 500 for the US, iShares S&P/TSX 60 for Canada
BENCHMARKS = {US: 'SPY', CANADA: 'XIU.TO'}

# how long after a market's close the price APIs have published that day's final candle: EODHD updates NYSE and
# NASDAQ about 15 minutes after the close and other exchanges, the TSX among them, 2 to 3 hours after it; Tiingo
# states no time, its end-of-day jobs running after the close. Before then a row dated that day is not final.
PUBLISH_DELAY = timedelta(hours=3)


class UnsupportedTicker(ValueError):
    """A ticker whose suffix names no exchange the screener covers."""


def market_of(ticker: str) -> str:
    """The ticker's market: US when it is bare, Canada for '.TO' or '.V'. Raises UnsupportedTicker otherwise."""
    if '.' not in ticker:
        return US
    base, _, suffix = ticker.rpartition('.')
    market = SUFFIX_MARKETS.get(suffix)
    if market is None or not base or '.' in base:
        allowed = ', '.join(f'.{s}' for s in SUFFIX_MARKETS)
        raise UnsupportedTicker(f'unsupported exchange suffix: {ticker} (only {allowed})')
    return market


def market_sql(column: str) -> str:
    """
    The same rule as market_of, as an SQL expression over a ticker column: 'Canada' for a ticker ending in one of
    SUFFIX_MARKETS' suffixes, 'US' otherwise. GLOB, unlike LIKE, is case-sensitive, as the suffixes are. A ticker
    market_of refuses counts as US here, but none is ever stored: the fetch fails it without a request.
    """
    whens = ' '.join(f"WHEN {column} GLOB '*.{suffix}' THEN '{market}'" for suffix, market in SUFFIX_MARKETS.items())
    return f"CASE {whens} ELSE '{US}' END"


def last_complete_session(market: str, started_at: datetime) -> date:
    """
    The newest day whose daily candle is final in 'market' for a run started at 'started_at' (timezone aware): the
    run's own day in that market's time zone once its close plus PUBLISH_DELAY has passed, and the day before until
    then. A weekend or holiday has no rows, so the day returned need not be a trading day.
    """
    zone, close = MARKET_CLOSES[market]
    local = started_at.astimezone(zone)
    complete_at = datetime.combine(local.date(), close, tzinfo=zone) + PUBLISH_DELAY
    return local.date() if local >= complete_at else local.date() - timedelta(days=1)
