"""
The market a ticker trades on, read from the ticker itself: one rule for the whole screener.

Canadian tickers are stored as EODHD writes them, with their exchange's suffix ('SHOP.TO' on the Toronto Stock
Exchange, 'ABC.V' on the TSX Venture). US tickers are bare ('AAPL'). Share classes use a dash ('BBD-B.TO'), never a
dot, so any other dot is not a ticker the screener can fetch.
"""

US = 'US'
CANADA = 'Canada'

# every market, in the order summaries list them
MARKETS = (US, CANADA)

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


def market_sql(column: str) -> str:
    """
    The same rule as market_of, as an SQL expression over a ticker column: 'Canada' for a ticker ending in one of
    SUFFIX_MARKETS' suffixes, 'US' otherwise. GLOB, unlike LIKE, is case-sensitive, as the suffixes are. A ticker
    market_of refuses counts as US here, but none is ever stored: the fetch fails it without a request.
    """
    whens = ' '.join(f"WHEN {column} GLOB '*.{suffix}' THEN '{market}'" for suffix, market in SUFFIX_MARKETS.items())
    return f"CASE {whens} ELSE '{US}' END"
