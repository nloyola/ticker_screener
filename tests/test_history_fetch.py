"""history-fetch fills the history database with each listed ticker's and benchmark's years of prices, resumably."""

import argparse
import hashlib
from datetime import date, datetime, time, timedelta
from types import SimpleNamespace

import pytest

from app.commands.history_fetch_cmd import HistoryFetchCmd
from app.db import init_db
from app.history_fetch import HISTORY_YEARS, SHORT_HISTORY_SLACK, HistoryFetch, history_start
from app.markets import CANADA, MARKET_CLOSES, US, is_session
from app.price_apis import DailyPrice, PriceApiStop
from app.price_apis.eodhd import EodhdPrices
from app.price_apis.http import RETRY_ATTEMPTS
from app.repositories.history_repo import BENCHMARK, TICKER, HistoryRepo
from tests.test_ticker_fetcher import FakeSectorService, http_error

TODAY = date(2026, 11, 16)
NEW_YORK = MARKET_CLOSES[US][0]
START = history_start(TODAY)


def sessions(first: date, days: int) -> list[date]:
    """The first 'days' US sessions on or after 'first', since a fetch drops a row dated on no session."""
    out: list[date] = []
    day = first
    while len(out) < days:
        if is_session(US, day):
            out.append(day)
        day += timedelta(days=1)
    return out


def prices(first: date, days: int, close: float = 10.0) -> list[DailyPrice]:
    return [
        DailyPrice(date=d, close=close + i, volume=100, high=close + i, low=close + i, open=close + i)
        for i, d in enumerate(sessions(first, days))
    ]


class FakeApi:
    """A price API answering from 'answers': rows, or an exception to raise; a name with no answer raises."""

    name = 'Tiingo'
    setting = 'tiingo'
    markets = frozenset({US})
    api_key = 'test-key'

    def __init__(self, answers: dict[str, list]) -> None:
        self.answers = answers
        self.calls: list[tuple[str, date, date]] = []
        self.quota_checks: list[int] = []

    def daily_prices(self, ticker: str, start: date, end: date) -> list[DailyPrice]:
        self.calls.append((ticker, start, end))
        script = self.answers[ticker]
        answer = script.pop(0) if len(script) > 1 else script[0]
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def check_quota(self, needed: int) -> None:
        self.quota_checks.append(needed)


@pytest.fixture
def history(tmp_path) -> HistoryRepo:
    return HistoryRepo(SimpleNamespace(history_db_path=str(tmp_path / 'history.db')))


def make_fetch(
    history: HistoryRepo, api, subsectors: dict[int, list[str]], today: date = TODAY, at: time = time(21)
) -> HistoryFetch:
    """A fetch on 'today' at 'at' New York time: by default after the day's candles are final."""
    fetch = HistoryFetch(
        SimpleNamespace(price_api_client=api, sector_service=FakeSectorService(subsectors), history_repo=history)
    )
    fetch.sleep = lambda seconds: None
    fetch.now = lambda: datetime.combine(today, at, tzinfo=NEW_YORK)
    return fetch


def run(fetch: HistoryFetch):
    return fetch.run(lambda plan: None)


def full(close: float = 10.0) -> list:
    return [prices(START, 5, close)]


def test_history_start_is_the_named_number_of_years_back():
    assert history_start(date(2026, 11, 16)) == date(2026 - HISTORY_YEARS, 11, 16)
    assert history_start(date(2028, 2, 29)) == date(2028 - HISTORY_YEARS, 2, 28)


def test_stores_every_listed_ticker_once_and_the_benchmark(history):
    api = FakeApi({'CVX': full(), 'XOM': full(20.0), 'SPY': full(400.0)})

    report = run(make_fetch(history, api, {1: ['CVX', 'XOM'], 2: ['XOM']}))

    assert [call[0] for call in api.calls] == ['CVX', 'XOM', 'SPY']
    assert {call[1:] for call in api.calls} == {(START, TODAY)}
    assert api.quota_checks == [3]
    assert history.prices(TICKER, 'XOM') == prices(START, 5, 20.0)
    assert history.prices(BENCHMARK, 'SPY') == prices(START, 5, 400.0)
    assert history.prices(TICKER, 'SPY') == []
    assert (report.requests, report.tickers_with_history, report.benchmarks_with_history) == (3, 2, 1)
    assert (report.failed, report.no_history, report.short_history) == ([], [], {})


def test_a_second_fetch_the_same_day_requests_nothing(history):
    api = FakeApi({'CVX': full(), 'SPY': full()})
    run(make_fetch(history, api, {1: ['CVX']}))
    api.calls.clear()

    report = run(make_fetch(history, api, {1: ['CVX']}))

    assert api.calls == []
    assert (report.requests, report.done_today, report.tickers_with_history) == (0, 2, 1)


def test_a_fetch_on_a_later_day_or_another_api_requests_again(history):
    api = FakeApi({'CVX': full(), 'SPY': full()})
    run(make_fetch(history, api, {1: ['CVX']}))
    api.calls.clear()

    run(make_fetch(history, api, {1: ['CVX']}, today=TODAY + timedelta(days=1)))
    assert [call[0] for call in api.calls] == ['CVX', 'SPY']

    eodhd = FakeApi({'CVX': full(), 'SPY': full(), 'XIU.TO': full()})
    eodhd.name, eodhd.setting, eodhd.markets = 'EODHD', 'eodhd', frozenset({US, CANADA})
    run(make_fetch(history, eodhd, {1: ['CVX']}, today=TODAY + timedelta(days=1)))
    assert [call[0] for call in eodhd.calls] == ['CVX', 'SPY', 'XIU.TO']


def test_an_interrupted_fetch_resumes_without_refetching_finished_tickers(history):
    api = FakeApi({'AAA': full(), 'BBB': [KeyboardInterrupt()], 'CCC': full(), 'SPY': full()})
    with pytest.raises(KeyboardInterrupt):
        run(make_fetch(history, api, {1: ['AAA', 'BBB', 'CCC']}))
    assert history.prices(TICKER, 'AAA') == prices(START, 5)
    assert history.prices(TICKER, 'BBB') == []

    api.answers['BBB'] = full(30.0)
    api.calls.clear()
    report = run(make_fetch(history, api, {1: ['AAA', 'BBB', 'CCC']}))

    assert [call[0] for call in api.calls] == ['BBB', 'CCC', 'SPY']
    assert (report.requests, report.done_today, report.tickers_with_history) == (3, 1, 3)


def test_a_failed_request_keeps_the_stored_history_and_is_retried_next_run(history):
    api = FakeApi({'CVX': full(), 'SPY': full()})
    run(make_fetch(history, api, {1: ['CVX']}))

    api.answers['CVX'] = [http_error(503)]
    tomorrow = TODAY + timedelta(days=1)
    report = run(make_fetch(history, api, {1: ['CVX']}, today=tomorrow))

    assert report.failed == ['CVX']
    assert [call[0] for call in api.calls].count('CVX') == 1 + RETRY_ATTEMPTS
    assert history.prices(TICKER, 'CVX') == prices(START, 5)
    assert history.fetched_on(TICKER, tomorrow, 'tiingo') == set()


@pytest.mark.parametrize('answer', [http_error(404), []])
def test_unknown_or_empty_is_reported_as_no_history_and_not_requested_again_today(history, answer):
    api = FakeApi({'GONE': [answer], 'SPY': full()})

    report = run(make_fetch(history, api, {1: ['GONE']}))

    assert (report.failed, report.no_history) == ([], ['GONE'])
    api.calls.clear()
    run(make_fetch(history, api, {1: ['GONE']}))
    assert api.calls == []


def test_a_candle_not_yet_final_is_not_stored(history):
    # Tuesday's candle is in the answer from the open on, but final only 3 hours after the close
    tuesday = date(2026, 11, 17)
    answer = [prices(tuesday - timedelta(days=4), 5)]
    api = FakeApi({'CVX': answer, 'SPY': answer})

    report = run(make_fetch(history, api, {1: ['CVX']}, today=tuesday, at=time(12)))

    assert [p.date for p in history.prices(TICKER, 'CVX')][-1] == tuesday - timedelta(days=1)
    assert [p.date for p in history.prices(BENCHMARK, 'SPY')][-1] == tuesday - timedelta(days=1)
    assert report.tickers_with_history == 1

    run(make_fetch(history, api, {1: ['CVX']}, today=tuesday + timedelta(days=1), at=time(12)))
    assert [p.date for p in history.prices(TICKER, 'CVX')][-1] == tuesday


def test_an_answer_holding_only_a_candle_not_yet_final_keeps_the_stored_history(history):
    api = FakeApi({'CVX': full(), 'SPY': full()})
    run(make_fetch(history, api, {1: ['CVX']}))

    tomorrow = TODAY + timedelta(days=1)
    api.answers['CVX'] = [prices(tomorrow, 1)]
    report = run(make_fetch(history, api, {1: ['CVX']}, today=tomorrow, at=time(12)))

    assert history.prices(TICKER, 'CVX') == prices(START, 5)
    assert (report.failed, report.no_history) == ([], [])


def test_a_row_dated_on_no_session_is_not_stored(history, caplog):
    # an answer holding a Saturday row and one on Thanksgiving, a US holiday, for the ticker and the benchmark
    saturday, thanksgiving = date(2016, 11, 19), date(2016, 11, 24)
    answer = sorted(
        prices(START, 5)
        + [DailyPrice(date=d, close=1.0, volume=0, high=1.0, low=1.0, open=1.0) for d in (saturday, thanksgiving)],
        key=lambda p: p.date,
    )
    api = FakeApi({'CVX': [answer], 'SPY': [answer]})

    with caplog.at_level('WARNING', logger='app.price_apis'):
        run(make_fetch(history, api, {1: ['CVX']}))

    assert history.prices(TICKER, 'CVX') == prices(START, 5)
    assert history.prices(BENCHMARK, 'SPY') == prices(START, 5)
    assert [r.getMessage() for r in caplog.records] == [
        f'{name}: dropped 2 price rows dated on no session ({saturday}, {thanksgiving})' for name in ('CVX', 'SPY')
    ]


def test_a_late_start_is_reported_as_short_history(history):
    late = START + SHORT_HISTORY_SLACK + timedelta(days=1)
    api = FakeApi({'NEW': [prices(late, 3)], 'OLD': [prices(START + SHORT_HISTORY_SLACK, 3)], 'SPY': full()})

    report = run(make_fetch(history, api, {1: ['NEW', 'OLD']}))

    assert report.short_history == {'NEW': late}


def test_tickers_the_api_does_not_cover_are_reported_and_never_requested(history):
    api = FakeApi({'CVX': full(), 'SPY': full()})

    report = run(make_fetch(history, api, {1: ['CVX', 'SHOP.TO', 'BAD.XX']}))

    assert [call[0] for call in api.calls] == ['CVX', 'SPY']
    assert (report.not_covered, report.unsupported) == (['SHOP.TO'], ['BAD.XX'])
    assert report.no_history == []


def test_the_quota_stop_comes_before_any_request(history):
    class Spent(FakeApi):
        def check_quota(self, needed: int) -> None:
            raise PriceApiStop('no calls left')

    api = Spent({'CVX': full(), 'SPY': full()})
    with pytest.raises(PriceApiStop):
        run(make_fetch(history, api, {1: ['CVX']}))
    assert api.calls == []


def test_a_split_inside_the_history_leaves_no_jump():
    # a 2-for-1 split on the third day: EODHD's raw prices halve, its adjusted closes do not
    rows = [
        {'date': '2020-01-02', 'open': 100, 'high': 102, 'low': 98, 'close': 100, 'adjusted_close': 50, 'volume': 10},
        {'date': '2020-01-03', 'open': 100, 'high': 104, 'low': 99, 'close': 102, 'adjusted_close': 51, 'volume': 10},
        {'date': '2020-01-06', 'open': 51, 'high': 53, 'low': 50, 'close': 52, 'adjusted_close': 52, 'volume': 20},
    ]

    stored = EodhdPrices.to_prices(rows)

    assert [(p.open, p.high, p.low, p.close) for p in stored] == [
        (50.0, 51.0, 49.0, 50.0),
        (50.0, 52.0, 49.5, 51.0),
        (51.0, 53.0, 50.0, 52.0),
    ]


def digest(path: str) -> str:
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def test_the_command_leaves_the_live_database_unchanged(tmp_path, capsys):
    live = str(tmp_path / 'stock_data.db')
    init_db(live)
    before = digest(live)
    history = HistoryRepo(SimpleNamespace(history_db_path=str(tmp_path / 'history.db')))
    api = FakeApi({'CVX': full(), 'SPY': full()})
    container = SimpleNamespace(history_fetch=make_fetch(history, api, {1: ['CVX']}), db_path=live)

    HistoryFetchCmd().handle(argparse.Namespace(), container)

    assert digest(live) == before
    out = capsys.readouterr().out
    assert 'history-fetch: 2 requests on Tiingo' in out
    assert 'history-fetch: 1 tickers and 1 benchmarks have history on Tiingo' in out


def test_the_command_exits_1_when_a_request_failed(history, capsys):
    api = FakeApi({'CVX': [ValueError('bad JSON')], 'SPY': full()})
    container = SimpleNamespace(history_fetch=make_fetch(history, api, {1: ['CVX']}))

    with pytest.raises(SystemExit) as exit_info:
        HistoryFetchCmd().handle(argparse.Namespace(), container)

    assert exit_info.value.code == 1
    assert 'history-fetch: failed, run again to retry: CVX' in capsys.readouterr().out


def test_the_command_reports_an_interruption_in_one_line(history, capsys):
    api = FakeApi({'CVX': [KeyboardInterrupt()], 'SPY': full()})
    container = SimpleNamespace(history_fetch=make_fetch(history, api, {1: ['CVX']}))

    with pytest.raises(SystemExit) as exit_info:
        HistoryFetchCmd().handle(argparse.Namespace(), container)

    assert exit_info.value.code == 130
    assert capsys.readouterr().err == 'history-fetch: interrupted, run it again to resume\n'
