"""
The market a ticker trades on, read from the ticker itself: one rule for the whole screener.

Canadian tickers are stored as EODHD writes them, with their exchange's suffix ('SHOP.TO' on the Toronto Stock
Exchange, 'ABC.V' on the TSX Venture). US tickers are bare ('AAPL'). Share classes use a dash ('BBD-B.TO'), never a
dot, so any other dot is not a ticker the screener can fetch.
"""

from datetime import date, datetime, time, timedelta
from functools import cache
from zoneinfo import ZoneInfo

US = 'US'
CANADA = 'Canada'

# every market, in the order summaries list them
MARKETS = (US, CANADA)

# the exchange suffixes the screener accepts, and their market
SUFFIX_MARKETS = {'TO': CANADA, 'V': CANADA}

# the exchange each suffix names: the Toronto Stock Exchange and the TSX Venture. A bare US ticker names none, since
# no price API the screener uses says whether it lists on the NYSE or NASDAQ
SUFFIX_EXCHANGES = {'TO': 'TSX', 'V': 'TSXV'}

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


def exchange_of(ticker: str) -> str | None:
    """The exchange the ticker's suffix names ('TSX' for '.TO', 'TSXV' for '.V'), or None for a bare ticker."""
    if '.' not in ticker:
        return None
    market_of(ticker)  # refuses an unsupported suffix
    return SUFFIX_EXCHANGES[ticker.rpartition('.')[2]]


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


# The exchange holiday calendars, one per market: the NYSE's for the US and the TSX's for Canada, computed from their
# published rules, so they need no yearly update, except for an unscheduled closure (a national day of mourning, say),
# which goes in CLOSURES when the exchange announces it. The same rules as the dashboard's internal/domain/holidays.go,
# which must change with them.

# the weekdays an exchange closed outside its holiday rules
CLOSURES: dict[str, frozenset[date]] = {
    US: frozenset(
        {
            date(2018, 12, 5),  # National day of mourning for George H. W. Bush
            date(2025, 1, 9),  # National day of mourning for Jimmy Carter
        }
    ),
    CANADA: frozenset(),
}

SATURDAY, SUNDAY = 5, 6
MONDAY, THURSDAY = 0, 3


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The nth 'weekday' (Monday 0) of 'month' in 'year'."""
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """The last 'weekday' (Monday 0) of 'month' in 'year'."""
    last = date(year + month // 12, month % 12 + 1, 1) - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _easter(year: int) -> date:
    """Easter Sunday in 'year', in the Gregorian calendar (the anonymous Gregorian algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7  # noqa: E741
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _nyse_holidays(year: int) -> list[date]:
    """
    NYSE Rule 7.2: a holiday on a Saturday is observed the Friday before, and one on a Sunday the Monday after, except
    New Year's Day, which is not observed when it falls on a Saturday.
    """

    def observed(d: date) -> date:
        if d.weekday() == SATURDAY:
            return d - timedelta(days=1)
        if d.weekday() == SUNDAY:
            return d + timedelta(days=1)
        return d

    out: list[date] = []
    new_year = date(year, 1, 1)
    if new_year.weekday() != SATURDAY:
        out.append(observed(new_year))
    out += [
        _nth_weekday(year, 1, MONDAY, 3),  # Martin Luther King Jr. Day
        _nth_weekday(year, 2, MONDAY, 3),  # Washington's Birthday
        _easter(year) - timedelta(days=2),  # Good Friday
        _last_weekday(year, 5, MONDAY),  # Memorial Day
    ]
    if year >= 2022:
        out.append(observed(date(year, 6, 19)))  # Juneteenth
    out += [
        observed(date(year, 7, 4)),  # Independence Day
        _nth_weekday(year, 9, MONDAY, 1),  # Labor Day
        _nth_weekday(year, 11, THURSDAY, 4),  # Thanksgiving Day
        observed(date(year, 12, 25)),  # Christmas Day
    ]
    return out


def _tsx_holidays(year: int) -> list[date]:
    """
    The TSX's calendar: a holiday on a weekend is observed on the next weekday that is not already a holiday, so a
    weekend Christmas and Boxing Day close the Monday and Tuesday after.
    """
    out: list[date] = []

    def add(d: date) -> None:
        while d in out or d.weekday() in (SATURDAY, SUNDAY):
            d += timedelta(days=1)
        out.append(d)

    may_24 = date(year, 5, 24)
    for d in (
        date(year, 1, 1),  # New Year's Day
        _nth_weekday(year, 2, MONDAY, 3),  # Family Day
        _easter(year) - timedelta(days=2),  # Good Friday
        may_24 - timedelta(days=(may_24.weekday() - MONDAY) % 7),  # Victoria Day, the Monday before 25 May
        date(year, 7, 1),  # Canada Day
        _nth_weekday(year, 8, MONDAY, 1),  # Civic Holiday
        _nth_weekday(year, 9, MONDAY, 1),  # Labour Day
        _nth_weekday(year, 10, MONDAY, 2),  # Thanksgiving Day
        date(year, 12, 25),  # Christmas Day
        date(year, 12, 26),  # Boxing Day
    ):
        add(d)
    return out


def is_holiday(market: str, day: date) -> bool:
    """
    Whether 'market's exchange is closed on 'day', a weekday, for a holiday: an exchange holiday, or the weekday it is
    observed on when it falls on a weekend, or an unscheduled closure. Neither exchange observes a holiday in the year
    before it: the NYSE skips a Saturday New Year's Day, and the TSX observes it after.
    """
    return day in _closed_days(market, day.year)


@cache
def _closed_days(market: str, year: int) -> frozenset[date]:
    """The weekdays 'market's exchange is closed in 'year': its holidays as observed, and its unscheduled closures."""
    holidays = _tsx_holidays(year) if market == CANADA else _nyse_holidays(year)
    return frozenset(holidays) | {d for d in CLOSURES[market] if d.year == year}


def is_session(market: str, day: date) -> bool:
    """Whether 'market' trades on 'day': a weekday that is not one of its exchange's holidays."""
    return day.weekday() not in (SATURDAY, SUNDAY) and not is_holiday(market, day)
