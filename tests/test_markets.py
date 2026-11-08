"""When a market's daily candle is complete: last_complete_session, for runs at either side of the cutoff."""

from datetime import UTC, date, datetime, time
from zoneinfo import ZoneInfo

import pytest

from app.markets import CANADA, US, last_complete_session

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
