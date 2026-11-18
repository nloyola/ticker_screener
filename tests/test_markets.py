"""
When a market's daily candle is complete: last_complete_session, for runs at either side of the cutoff; and which
days each market trades, against the exchanges' published holiday calendars.
"""

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

from app.markets import CANADA, US, UnsupportedTicker, exchange_of, is_holiday, is_session, last_complete_session

NEW_YORK = ZoneInfo('America/New_York')
TORONTO = ZoneInfo('America/Toronto')
MONDAY = date(2026, 11, 9)
SUNDAY = date(2026, 11, 8)


@pytest.mark.parametrize(
    ('market', 'started_at', 'expected'),
    [
        pytest.param(US, datetime.combine(MONDAY, time(9, 29), tzinfo=NEW_YORK), SUNDAY, id='US before the open'),
        pytest.param(US, datetime.combine(MONDAY, time(11), tzinfo=NEW_YORK), SUNDAY, id='US market open'),
        pytest.param(US, datetime.combine(MONDAY, time(16, 30), tzinfo=NEW_YORK), SUNDAY, id='US just after close'),
        pytest.param(US, datetime.combine(MONDAY, time(18, 59), tzinfo=NEW_YORK), SUNDAY, id='US before publishing'),
        pytest.param(US, datetime.combine(MONDAY, time(19), tzinfo=NEW_YORK), MONDAY, id='US published'),
        pytest.param(US, datetime.combine(MONDAY, time(23, 59), tzinfo=NEW_YORK), MONDAY, id='US late evening'),
        # 01:30 UTC on Tuesday is 20:30 on Monday in New York
        pytest.param(US, datetime(2026, 11, 10, 1, 30, tzinfo=UTC), MONDAY, id='US evening, next day in UTC'),
        # 14:00 UTC on Monday is 09:00 in New York
        pytest.param(US, datetime(2026, 11, 9, 14, tzinfo=UTC), SUNDAY, id='US morning, given in UTC'),
        pytest.param(CANADA, datetime.combine(MONDAY, time(15), tzinfo=TORONTO), SUNDAY, id='Canada market open'),
        pytest.param(CANADA, datetime.combine(MONDAY, time(19, 30), tzinfo=TORONTO), MONDAY, id='Canada published'),
        # summer time: 19:00 in New York is 23:00 UTC, not 00:00
        pytest.param(US, datetime(2026, 7, 6, 22, 59, tzinfo=UTC), date(2026, 7, 5), id='US summer, before'),
        pytest.param(US, datetime(2026, 7, 6, 23, tzinfo=UTC), date(2026, 7, 6), id='US summer, published'),
    ],
)
def test_last_complete_session(market, started_at, expected):
    assert last_complete_session(market, started_at) == expected


@pytest.mark.parametrize(
    ('ticker', 'expected'),
    [('AAPL', None), ('BRK-B', None), ('SHOP.TO', 'TSX'), ('BBD-B.TO', 'TSX'), ('ABC.V', 'TSXV')],
)
def test_exchange_of(ticker, expected):
    assert exchange_of(ticker) == expected


def test_exchange_of_refuses_an_unsupported_suffix():
    with pytest.raises(UnsupportedTicker):
        exchange_of('VOD.L')


# the exchanges' published calendars, every weekday each closed or will close for a holiday in the year, the same as
# the dashboard's internal/domain/holidays_test.go
# fmt: off
PUBLISHED_HOLIDAYS = {
    US: {
        2021: ['2021-01-01', '2021-01-18', '2021-02-15', '2021-04-02', '2021-05-31', '2021-07-05',
               '2021-09-06', '2021-11-25', '2021-12-24'],
        2022: ['2022-01-17', '2022-02-21', '2022-04-15', '2022-05-30', '2022-06-20', '2022-07-04',
               '2022-09-05', '2022-11-24', '2022-12-26'],
        2025: ['2025-01-01', '2025-01-09', '2025-01-20', '2025-02-17', '2025-04-18', '2025-05-26',
               '2025-06-19', '2025-07-04', '2025-09-01', '2025-11-27', '2025-12-25'],
        2026: ['2026-01-01', '2026-01-19', '2026-02-16', '2026-04-03', '2026-05-25', '2026-06-19',
               '2026-07-03', '2026-09-07', '2026-11-26', '2026-12-25'],
        2027: ['2027-01-01', '2027-01-18', '2027-02-15', '2027-03-26', '2027-05-31', '2027-06-18',
               '2027-07-05', '2027-09-06', '2027-11-25', '2027-12-24'],
    },
    CANADA: {
        2020: ['2020-01-01', '2020-02-17', '2020-04-10', '2020-05-18', '2020-07-01', '2020-08-03',
               '2020-09-07', '2020-10-12', '2020-12-25', '2020-12-28'],
        2021: ['2021-01-01', '2021-02-15', '2021-04-02', '2021-05-24', '2021-07-01', '2021-08-02',
               '2021-09-06', '2021-10-11', '2021-12-27', '2021-12-28'],
        2022: ['2022-01-03', '2022-02-21', '2022-04-15', '2022-05-23', '2022-07-01', '2022-08-01',
               '2022-09-05', '2022-10-10', '2022-12-26', '2022-12-27'],
        2023: ['2023-01-02', '2023-02-20', '2023-04-07', '2023-05-22', '2023-07-03', '2023-08-07',
               '2023-09-04', '2023-10-09', '2023-12-25', '2023-12-26'],
        2025: ['2025-01-01', '2025-02-17', '2025-04-18', '2025-05-19', '2025-07-01', '2025-08-04',
               '2025-09-01', '2025-10-13', '2025-12-25', '2025-12-26'],
        2026: ['2026-01-01', '2026-02-16', '2026-04-03', '2026-05-18', '2026-07-01', '2026-08-03',
               '2026-09-07', '2026-10-12', '2026-12-25', '2026-12-28'],
        2027: ['2027-01-01', '2027-02-15', '2027-03-26', '2027-05-24', '2027-07-01', '2027-08-02',
               '2027-09-06', '2027-10-11', '2027-12-27', '2027-12-28'],
    },
}
# fmt: on


@pytest.mark.parametrize(
    ('market', 'year'), [(market, year) for market, years in PUBLISHED_HOLIDAYS.items() for year in years]
)
def test_is_holiday_matches_the_published_calendar(market, year):
    published = {date.fromisoformat(d) for d in PUBLISHED_HOLIDAYS[market][year]}
    weekdays = [d for d in (date(year, 1, 1) + timedelta(days=i) for i in range(366)) if d.year == year]
    weekdays = [d for d in weekdays if d.weekday() < 5]
    assert {d for d in weekdays if is_holiday(market, d)} == published


@pytest.mark.parametrize(
    ('market', 'day', 'expected'),
    [
        pytest.param(US, date(2026, 11, 13), True, id='US Friday'),
        pytest.param(US, date(2026, 11, 14), False, id='US Saturday'),
        pytest.param(CANADA, date(2026, 11, 15), False, id='Canada Sunday'),
        pytest.param(US, date(2026, 11, 26), False, id='US Thanksgiving'),
        pytest.param(CANADA, date(2026, 11, 26), True, id='Canada on US Thanksgiving'),
        pytest.param(CANADA, date(2026, 10, 12), False, id='Canadian Thanksgiving'),
        pytest.param(US, date(2026, 10, 12), True, id='US on Canadian Thanksgiving'),
    ],
)
def test_is_session(market, day, expected):
    assert is_session(market, day) is expected
