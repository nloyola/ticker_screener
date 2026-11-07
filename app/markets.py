"""
The market a ticker trades on, read from the ticker itself: one rule for the whole screener.

Canadian tickers are stored as EODHD writes them, with their exchange's suffix ('SHOP.TO' on the Toronto Stock
Exchange, 'ABC.V' on the TSX Venture). US tickers are bare ('AAPL'). Share classes use a dash ('BBD-B.TO'), never a
dot, so any other dot is not a ticker the screener can fetch.
"""

US = 'US'
CANADA = 'Canada'

# the exchange suffixes the screener accepts, and their market
SUFFIX_MARKETS = {'TO': CANADA, 'V': CANADA}


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
