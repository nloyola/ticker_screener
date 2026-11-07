"""TickerFetcher downloads each ticker's whole window and replaces its stored rows, against a real database."""

import argparse
import logging
import sys
from datetime import date, timedelta
from types import SimpleNamespace

import pytest
import requests
from requests.adapters import HTTPAdapter

from app.commands.ticker_fetcher_cmd import TickerFetcherCmd
from app.db import init_db
from app.markets import CANADA, US
from app.models import PriceData
from app.price_apis import DailyPrice
from app.price_apis.http import (
    CONNECT_TIMEOUT,
    ERROR_DETAIL_LIMIT,
    READ_TIMEOUT,
    RETRY_ATTEMPTS,
    RETRY_MAX_DELAY,
    error_detail,
    retry_delay,
)
from app.price_apis.tiingo import TiingoPrices
from app.repositories.price_data_repo import PriceDataRepo
from app.services.sector_service import SectorAggregate, SubsectorWithTickers
from app.ticker_fetcher import HISTORY_DAYS, TickerFetcher, window_start

TODAY = date.today()
OLD_DAY = window_start(TODAY) - timedelta(days=1)


class FakeApi:
    """The parts of a price API (app.price_apis.PriceApi) every fake shares: Tiingo's name and markets, no quota."""

    name = 'Tiingo'
    markets = frozenset({US})
    api_key = 'test-key'

    def check_quota(self, needed: int) -> None:
        pass


def tiingo_rows(rows: list[tuple[date, float]]) -> list[DailyPrice]:
    """Rows as Tiingo sends them, parsed as TiingoPrices parses them, so a malformed row fails the same way."""
    return TiingoPrices.to_prices(
        [{'date': f'{d.isoformat()}T00:00:00.000Z', 'adjClose': close, 'adjVolume': 100} for d, close in rows]
    )


class FakeTiingo(FakeApi):
    """Answers daily_prices from canned responses; a ticker with no response raises, as a failed request does."""

    def __init__(self, prices: dict[str, list[tuple[date, float]]]) -> None:
        self.prices = prices
        self.requests: list[tuple[str, str, str]] = []

    def daily_prices(self, ticker: str, start: date, end: date) -> list[DailyPrice]:
        self.requests.append((ticker, start.isoformat(), end.isoformat()))
        if ticker not in self.prices:
            raise RuntimeError(f'500 Server Error for ticker {ticker}, token test-key')
        return tiingo_rows(self.prices[ticker])


def tiingo_response(status: int, body: bytes = b'[]', retry_after: str | None = None) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response.reason = 'Not Found' if status == 404 else 'Error'
    response._content = body
    response.url = 'https://api.tiingo.com/tiingo/daily/CVX/prices'
    if retry_after is not None:
        response.headers['Retry-After'] = retry_after
    return response


def http_error(status: int, retry_after: str | None = None, body: bytes = b'') -> requests.HTTPError:
    """The error raise_for_status raises for an HTTP error status, carrying the response."""
    response = tiingo_response(status, body, retry_after)
    return requests.HTTPError(f'{status} Error for url: {response.url}', response=response)


class ScriptedTiingo(FakeApi):
    """Answers each daily_prices call with the next item of the ticker's script: an exception to raise, or rows."""

    def __init__(self, scripts: dict[str, list]) -> None:
        self.scripts = scripts
        self.calls: list[str] = []

    def daily_prices(self, ticker: str, start: date, end: date) -> list[DailyPrice]:
        self.calls.append(ticker)
        answer = self.scripts[ticker].pop(0)
        if isinstance(answer, Exception):
            raise answer
        return tiingo_rows(answer)


class FakeSectorService:
    def __init__(self, subsectors: dict[int, list[str]]) -> None:
        self.sectors = [
            SectorAggregate(
                id=1,
                sector='Energy',
                created_at=None,
                subsectors=[
                    SubsectorWithTickers(id=sub_id, subsector=f'sub{sub_id}', tickers=tickers)
                    for sub_id, tickers in subsectors.items()
                ],
            )
        ]

    def get_all(self) -> list[SectorAggregate]:
        return self.sectors

    def subsector_for_ticker(self, ticker: str):
        for subsector in self.sectors[0].subsectors:
            if ticker in subsector.tickers:
                return subsector
        return None


@pytest.fixture
def repo(tmp_path) -> PriceDataRepo:
    path = str(tmp_path / 'stock_data.db')
    init_db(path)
    return PriceDataRepo(SimpleNamespace(db_path=path))


def make_fetcher(repo: PriceDataRepo, api, subsectors: dict[int, list[str]]) -> TickerFetcher:
    container = SimpleNamespace(
        price_api_client=api, sector_service=FakeSectorService(subsectors), price_data_repo=repo
    )
    fetcher = TickerFetcher(container)
    fetcher.sleep = lambda seconds: None
    return fetcher


def store(repo: PriceDataRepo, ticker: str, rows: list[tuple[date, float]], subsector_id: int = 1) -> None:
    repo.bulk_insert(
        PriceData(id=0, subsector_id=subsector_id, ticker=ticker, date=d, close=close, volume=100) for d, close in rows
    )


def stored(repo: PriceDataRepo, ticker: str) -> list[tuple[int, date, float]]:
    return [(p.subsector_id, p.date, p.close) for p in repo.all_for_ticker(ticker)]


def days_ago(n: int) -> date:
    return TODAY - timedelta(days=n)


def test_asks_for_the_whole_window_once_per_ticker(repo):
    tiingo = FakeTiingo({'CVX': [(days_ago(1), 1.0)], 'XOM': [(days_ago(1), 2.0)]})
    store(repo, 'CVX', [(days_ago(2), 9.0)])  # stored rows no longer shorten the request

    stats = make_fetcher(repo, tiingo, {1: ['CVX', 'XOM'], 2: ['XOM']}).fetch()

    start, end = (TODAY - timedelta(days=HISTORY_DAYS)).isoformat(), TODAY.isoformat()
    assert tiingo.requests == [('CVX', start, end), ('XOM', start, end)]
    assert (stats.attempted, stats.failed) == (2, 0)
    assert stored(repo, 'XOM') == [(1, days_ago(1), 2.0)]  # for the first subsector that lists it


def test_restated_history_replaces_every_stored_row(repo):
    # a 2-for-1 split: Tiingo halves every past adjusted close, and the oldest stored day left the window
    store(repo, 'CVX', [(OLD_DAY, 50.0), (days_ago(3), 100.0), (days_ago(2), 102.0)])
    tiingo = FakeTiingo({'CVX': [(days_ago(3), 50.0), (days_ago(2), 51.0), (days_ago(1), 52.0)]})

    make_fetcher(repo, tiingo, {1: ['CVX']}).fetch()

    assert stored(repo, 'CVX') == [(1, days_ago(3), 50.0), (1, days_ago(2), 51.0), (1, days_ago(1), 52.0)]


def test_failed_download_keeps_its_rows_and_others_are_replaced(repo):
    store(repo, 'CVX', [(OLD_DAY, 1.0), (days_ago(2), 2.0)])
    store(repo, 'XOM', [(days_ago(2), 3.0)])
    tiingo = FakeTiingo({'XOM': [(days_ago(1), 4.0)]})

    stats = make_fetcher(repo, tiingo, {1: ['CVX', 'XOM']}).fetch()

    assert (stats.attempted, stats.failed) == (2, 1)
    assert stored(repo, 'CVX') == [(1, OLD_DAY, 1.0), (1, days_ago(2), 2.0)]
    assert stored(repo, 'XOM') == [(1, days_ago(1), 4.0)]


def test_empty_download_keeps_the_stored_rows(repo):
    store(repo, 'CVX', [(days_ago(2), 2.0)])

    stats = make_fetcher(repo, FakeTiingo({'CVX': []}), {1: ['CVX']}).fetch()

    assert (stats.attempted, stats.failed) == (1, 0)
    assert stored(repo, 'CVX') == [(1, days_ago(2), 2.0)]


def test_error_message_never_shows_the_key(repo, caplog):
    make_fetcher(repo, FakeTiingo({}), {1: ['CVX']}).fetch(strict=True)

    assert 'CVX' in caplog.text
    assert 'test-key' not in caplog.text


def test_error_detail_quoted_back_never_shows_the_key(repo, caplog):
    tiingo = ScriptedTiingo({'CVX': [http_error(401, body=b'{"detail": "Invalid token test-key"}')]})

    make_fetcher(repo, tiingo, {1: ['CVX']}).fetch(strict=True)

    assert 'Invalid token ***' in caplog.text
    assert 'test-key' not in caplog.text


def test_a_single_ticker_is_fetched_and_stored_normalized(repo):
    tiingo = FakeTiingo({'XOM': [(days_ago(1), 2.0)]})

    make_fetcher(repo, tiingo, {1: ['CVX'], 2: ['XOM']}).fetch(ticker_filter=' xom ')

    assert [ticker for ticker, _, _ in tiingo.requests] == ['XOM']
    assert stored(repo, 'XOM') == [(2, days_ago(1), 2.0)]


def test_prune_deletes_only_rows_before_the_window(repo):
    store(repo, 'CVX', [(OLD_DAY, 1.0), (window_start(TODAY), 2.0)])

    deleted = make_fetcher(repo, FakeTiingo({}), {}).prune()

    assert deleted == 1
    assert stored(repo, 'CVX') == [(1, window_start(TODAY), 2.0)]


def test_prune_under_tiingo_keeps_canadian_rows_of_any_age(repo):
    store(repo, 'CVX', [(OLD_DAY, 1.0), (window_start(TODAY), 2.0)])
    store(repo, 'SHOP.TO', [(OLD_DAY, 3.0), (window_start(TODAY), 4.0)])
    store(repo, 'ABC.V', [(OLD_DAY, 5.0)])

    deleted = make_fetcher(repo, FakeTiingo({}), {}).prune()

    assert deleted == 1
    assert stored(repo, 'CVX') == [(1, window_start(TODAY), 2.0)]
    assert stored(repo, 'SHOP.TO') == [(1, OLD_DAY, 3.0), (1, window_start(TODAY), 4.0)]
    assert stored(repo, 'ABC.V') == [(1, OLD_DAY, 5.0)]


def test_prune_under_eodhd_ages_out_both_markets(repo):
    store(repo, 'CVX', [(OLD_DAY, 1.0), (window_start(TODAY), 2.0)])
    store(repo, 'SHOP.TO', [(OLD_DAY, 3.0), (window_start(TODAY), 4.0)])
    store(repo, 'ABC.V', [(OLD_DAY, 5.0)])
    eodhd = FakeTiingo({})
    eodhd.name, eodhd.markets = 'EODHD', frozenset({US, CANADA})

    deleted = make_fetcher(repo, eodhd, {}).prune()

    assert deleted == 3
    assert stored(repo, 'CVX') == [(1, window_start(TODAY), 2.0)]
    assert stored(repo, 'SHOP.TO') == [(1, window_start(TODAY), 4.0)]
    assert stored(repo, 'ABC.V') == []


def test_replace_for_ticker_rejects_another_tickers_rows(repo):
    store(repo, 'CVX', [(days_ago(2), 2.0)])
    xom = PriceData(id=0, subsector_id=1, ticker='XOM', date=days_ago(1), close=1.0, volume=1)

    with pytest.raises(ValueError):
        repo.replace_for_ticker('CVX', [xom])

    assert stored(repo, 'CVX') == [(1, days_ago(2), 2.0)]


def run_cmd(fetcher: TickerFetcher, *argv: str) -> None:
    cmd = TickerFetcherCmd()
    parser = argparse.ArgumentParser()
    cmd.add_arguments(parser)
    cmd.handle(parser.parse_args(argv), SimpleNamespace(ticker_fetcher=fetcher))


def test_strict_fetch_where_every_download_fails_leaves_price_data_unchanged(repo):
    # issue #6: every stored row is older than the window, and the Tiingo key is wrong
    store(repo, 'CVX', [(OLD_DAY, 1.0)])
    store(repo, 'XOM', [(OLD_DAY, 2.0)])
    fetcher = make_fetcher(repo, FakeTiingo({}), {1: ['CVX', 'XOM']})

    with pytest.raises(SystemExit) as exit_info:
        run_cmd(fetcher, '--strict')

    assert exit_info.value.code == 1
    assert stored(repo, 'CVX') == [(1, OLD_DAY, 1.0)]
    assert stored(repo, 'XOM') == [(1, OLD_DAY, 2.0)]


def test_successful_strict_fetch_prunes_once(repo):
    # CVX fails, so only the prune can remove its old row; XOM's download replaces its own
    store(repo, 'CVX', [(OLD_DAY, 1.0), (days_ago(2), 2.0)])
    store(repo, 'XOM', [(OLD_DAY, 3.0)])
    fetcher = make_fetcher(repo, FakeTiingo({'XOM': [(days_ago(1), 4.0)], 'SHEL': [(days_ago(1), 5.0)]}), {})
    fetcher.sector_service = FakeSectorService({1: ['CVX', 'XOM', 'SHEL']})
    prunes = []
    prune = fetcher.prune
    fetcher.prune = lambda: prunes.append(prune())

    run_cmd(fetcher, '--strict', '--max-failure-rate', '0.5')

    assert prunes == [1]
    assert stored(repo, 'CVX') == [(1, days_ago(2), 2.0)]
    assert stored(repo, 'XOM') == [(1, days_ago(1), 4.0)]


def test_fetch_without_strict_never_prunes(repo):
    store(repo, 'CVX', [(OLD_DAY, 1.0)])

    run_cmd(make_fetcher(repo, FakeTiingo({}), {1: ['CVX']}))

    assert stored(repo, 'CVX') == [(1, OLD_DAY, 1.0)]


def test_retries_a_429_and_stores_the_next_answer(repo):
    tiingo = ScriptedTiingo({'CVX': [http_error(429), [(days_ago(1), 1.0)]]})

    stats = make_fetcher(repo, tiingo, {1: ['CVX']}).fetch()

    assert tiingo.calls == ['CVX', 'CVX']
    assert (stats.attempted, stats.failed) == (1, 0)
    assert stored(repo, 'CVX') == [(1, days_ago(1), 1.0)]


def test_three_server_errors_fail_the_ticker_once(repo):
    store(repo, 'CVX', [(days_ago(2), 2.0)])
    tiingo = ScriptedTiingo({'CVX': [http_error(503)] * 3})

    stats = make_fetcher(repo, tiingo, {1: ['CVX']}).fetch()

    assert tiingo.calls == ['CVX'] * RETRY_ATTEMPTS
    assert (stats.attempted, stats.failed) == (1, 1)
    assert stored(repo, 'CVX') == [(1, days_ago(2), 2.0)]


def test_connection_errors_are_retried(repo):
    tiingo = ScriptedTiingo(
        {'CVX': [requests.ConnectionError('reset'), requests.Timeout('slow'), [(days_ago(1), 1.0)]]}
    )

    stats = make_fetcher(repo, tiingo, {1: ['CVX']}).fetch()

    assert len(tiingo.calls) == 3
    assert stats.failed == 0


@pytest.mark.parametrize('error', [http_error(404), http_error(401), ValueError('bad JSON')])
def test_other_errors_fail_at_once(repo, error):
    tiingo = ScriptedTiingo({'CVX': [error]})

    stats = make_fetcher(repo, tiingo, {1: ['CVX']}).fetch()

    assert tiingo.calls == ['CVX']
    assert stats.failed == 1


def test_waits_grow_and_honour_retry_after(repo):
    tiingo = ScriptedTiingo({'CVX': [http_error(500), http_error(429, retry_after='7')] + [[(days_ago(1), 1.0)]]})
    fetcher = make_fetcher(repo, tiingo, {1: ['CVX']})
    waits: list[float] = []
    fetcher.sleep = waits.append

    fetcher.fetch()

    assert len(waits) == 2
    assert 1.0 <= waits[0] <= 3.0  # about 2 s, with jitter
    assert waits[1] == 7.0


def test_retry_delay():
    assert retry_delay(http_error(429, retry_after='7'), 1, rand=lambda: 0.5) == 7.0
    assert retry_delay(http_error(429, retry_after='3600'), 1, rand=lambda: 0.5) == RETRY_MAX_DELAY
    # a date or a garbled Retry-After falls back to the backoff
    assert retry_delay(http_error(429, retry_after='Wed, 21 Oct 2026 07:28:00 GMT'), 2, rand=lambda: 0.5) == 4.0
    assert retry_delay(http_error(503), 1, rand=lambda: 0.0) == 1.0
    assert retry_delay(http_error(503), 1, rand=lambda: 1.0) == 3.0
    assert retry_delay(requests.ConnectionError(), 2, rand=lambda: 0.5) == 4.0


def test_retry_logs_never_show_the_key(repo, caplog):
    caplog.set_level(logging.DEBUG)
    tiingo = ScriptedTiingo({'CVX': [http_error(503)] * 3})

    make_fetcher(repo, tiingo, {1: ['CVX']}).fetch(strict=True)

    assert 'CVX' in caplog.text
    assert '503' in caplog.text
    assert 'test-key' not in caplog.text


def test_ticker_with_no_rows_in_the_window_is_logged_as_a_warning(repo, caplog):
    make_fetcher(repo, FakeTiingo({'CVX': []}), {1: ['CVX']}).fetch()

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert 'CVX' in warnings[0].getMessage()
    assert 'delisted' in warnings[0].getMessage()


def test_malformed_row_fails_only_that_ticker(repo):
    tiingo = ScriptedTiingo({'CVX': [[(days_ago(1), None)]], 'XOM': [[(days_ago(1), 2.0)]]})

    stats = make_fetcher(repo, tiingo, {1: ['CVX', 'XOM']}).fetch()

    assert tiingo.calls == ['CVX', 'XOM']  # not retried
    assert (stats.attempted, stats.failed) == (2, 1)
    assert stored(repo, 'XOM') == [(1, days_ago(1), 2.0)]


def test_not_found_warning_carries_tiingos_detail(repo, caplog):
    body = b'{"detail": "Error: Ticker \'NOPE\' not found"}'
    tiingo = ScriptedTiingo({'NOPE': [http_error(404, body=body)]})

    stats = make_fetcher(repo, tiingo, {1: ['NOPE']}).fetch(strict=True)

    assert stats.failed == 1
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "Ticker 'NOPE' not found" in warnings[0]
    assert '404' in warnings[0]


def test_retried_ticker_writes_no_error_record(repo, caplog):
    caplog.set_level(logging.DEBUG)
    body = b'{"detail": "busy"}'
    tiingo = ScriptedTiingo({'CVX': [http_error(503, body=body)] * 3, 'XOM': [http_error(503, body=body), []]})

    make_fetcher(repo, tiingo, {1: ['CVX', 'XOM']}).fetch(strict=True)

    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []
    retries = [r.getMessage() for r in caplog.records if 'retrying' in r.getMessage()]
    assert len(retries) == 3 and all('busy' in m for m in retries)


def test_error_detail():
    assert error_detail(ValueError('bad JSON')) == 'bad JSON'
    assert error_detail(http_error(404, body=b'{"detail": "gone"}')).endswith(': gone')
    assert error_detail(http_error(502, body=b'<html>\n  Bad   gateway</html>')).endswith(': <html> Bad gateway</html>')
    assert error_detail(http_error(503)) == '503 Error for url: https://api.tiingo.com/tiingo/daily/CVX/prices'
    long = error_detail(http_error(500, body=b'x' * 1000))
    assert long.endswith('x…') and len(long.rsplit(': ', 1)[1]) == ERROR_DETAIL_LIMIT


class ScriptedAdapter(HTTPAdapter):
    """Stands in for the network under a real requests.Session: answers each request with the next scripted item."""

    def __init__(self, answers: list) -> None:
        super().__init__()
        self.answers = answers
        self.sent: list[tuple[requests.PreparedRequest, object]] = []

    def send(self, request, **kwargs):
        self.sent.append((request, kwargs['timeout']))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        answer.request = request
        return answer


def tiingo_prices(answers: list) -> tuple[TiingoPrices, ScriptedAdapter]:
    client = TiingoPrices('test-key')
    adapter = ScriptedAdapter(answers)
    client.session.mount('https://', adapter)
    return client, adapter


def test_tiingo_prices_sends_the_key_in_a_header_with_a_timeout():
    body = b'[{"date": "2026-11-05T00:00:00.000Z", "adjClose": 1.5, "adjVolume": 10}]'
    client, adapter = tiingo_prices([tiingo_response(200, body)])

    rows = client.daily_prices('CVX', date(2026, 7, 9), date(2026, 11, 6))

    assert rows == [DailyPrice(date=date(2026, 11, 5), close=1.5, volume=10)]
    ((request, timeout),) = adapter.sent
    assert request.method == 'GET'
    assert request.url == (
        'https://api.tiingo.com/tiingo/daily/CVX/prices?startDate=2026-07-09&endDate=2026-11-06'
        '&resampleFreq=daily&columns=date%2CadjClose%2CadjVolume'
    )
    assert request.headers['Authorization'] == 'Token test-key'
    assert 'test-key' not in request.url
    assert timeout == (CONNECT_TIMEOUT, READ_TIMEOUT)


def test_tiingo_prices_raises_http_error_with_the_response():
    client, _ = tiingo_prices([tiingo_response(404, b'{"detail": "not found"}')])

    with pytest.raises(requests.HTTPError) as error:
        client.daily_prices('CVX', date(2026, 7, 9), date(2026, 11, 6))

    assert error.value.response.status_code == 404


def test_timed_out_request_is_retried_and_then_succeeds(repo):
    body = f'[{{"date": "{days_ago(1).isoformat()}T00:00:00.000Z", "adjClose": 1.0, "adjVolume": 100}}]'.encode()
    client, adapter = tiingo_prices([requests.ReadTimeout('read timed out'), tiingo_response(200, body)])
    fetcher = make_fetcher(repo, client, {1: ['CVX']})

    stats = fetcher.fetch()

    assert len(adapter.sent) == 2
    assert (stats.attempted, stats.failed) == (1, 0)
    assert stored(repo, 'CVX') == [(1, days_ago(1), 1.0)]


def flagged_lines(out: str) -> list[str]:
    return [line for line in out.splitlines() if line.startswith('FLAGGED')]


def test_not_found_ticker_is_flagged_with_its_last_stored_date(repo, capsys):
    store(repo, 'GONE', [(days_ago(40), 1.0), (days_ago(30), 2.0)])
    tiingo = ScriptedTiingo({'GONE': [http_error(404)], 'NEW': [http_error(404)], 'CVX': [[(days_ago(1), 3.0)]]})

    fetcher = make_fetcher(repo, tiingo, {1: ['GONE', 'NEW', 'CVX']})
    fetcher.fetch(strict=True)

    assert flagged_lines(capsys.readouterr().out) == [
        f'FLAGGED ticker=GONE reason=not_found last={days_ago(30)}',
        'FLAGGED ticker=NEW reason=not_found',
    ]
    assert fetcher.flagged == {'GONE', 'NEW'}


def test_not_found_is_flagged_without_strict_too(repo, capsys):
    make_fetcher(repo, ScriptedTiingo({'GONE': [http_error(404)]}), {1: ['GONE']}).fetch()

    assert flagged_lines(capsys.readouterr().out) == ['FLAGGED ticker=GONE reason=not_found']


@pytest.mark.parametrize('error', [http_error(401), http_error(503), ValueError('bad JSON')])
def test_other_failures_are_not_flagged(repo, capsys, error):
    fetcher = make_fetcher(repo, ScriptedTiingo({'CVX': [error] * RETRY_ATTEMPTS}), {1: ['CVX']})
    fetcher.fetch(strict=True)

    assert flagged_lines(capsys.readouterr().out) == []
    assert fetcher.flagged == set()


def test_empty_download_is_flagged_with_the_date_stored_before_the_run(repo, capsys):
    # BLD's newest stored row is older than the window, so the prune after this fetch deletes every one of its rows
    store(repo, 'BLD', [(OLD_DAY - timedelta(days=5), 1.0), (OLD_DAY, 2.0)])
    fetcher = make_fetcher(repo, FakeTiingo({'BLD': [], 'NONE': [], 'CVX': [(days_ago(1), 3.0)]}), {})
    fetcher.sector_service = FakeSectorService({1: ['BLD', 'NONE', 'CVX']})

    run_cmd(fetcher, '--strict')

    assert flagged_lines(capsys.readouterr().out) == [
        f'FLAGGED ticker=BLD reason=no_prices last={OLD_DAY}',
        'FLAGGED ticker=NONE reason=no_prices',
    ]
    assert stored(repo, 'BLD') == []  # pruned: only the line still knows its last date
    assert fetcher.flagged == {'BLD', 'NONE'}


def test_flagged_line_follows_its_warning_and_is_printed_once(repo, capsys):
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.WARNING)
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        fetcher = make_fetcher(repo, FakeTiingo({'CVX': [], 'XOM': []}), {1: ['CVX', 'XOM'], 2: ['CVX']})
        fetcher.fetch(strict=True)
    finally:
        root.removeHandler(handler)

    lines = [line for line in capsys.readouterr().out.splitlines() if not line.startswith('PROGRESS')]
    assert len(lines) == 4
    assert 'delisted' in lines[0] and 'CVX' in lines[0]
    assert lines[1] == 'FLAGGED ticker=CVX reason=no_prices'
    assert 'delisted' in lines[2] and 'XOM' in lines[2]
    assert lines[3] == 'FLAGGED ticker=XOM reason=no_prices'


def test_fetched_tickers_are_not_flagged(repo, capsys):
    fetcher = make_fetcher(repo, FakeTiingo({'CVX': [(days_ago(1), 1.0)]}), {1: ['CVX']})
    fetcher.fetch(strict=True)

    assert flagged_lines(capsys.readouterr().out) == []
