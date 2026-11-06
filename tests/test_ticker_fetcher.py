"""TickerFetcher downloads each ticker's whole window and replaces its stored rows, against a real database."""

import argparse
import logging
from datetime import date, timedelta
from types import SimpleNamespace

import pytest
import requests
from tiingo.restclient import RestClientError

from app.commands.ticker_fetcher_cmd import TickerFetcherCmd
from app.db import init_db
from app.models import PriceData
from app.repositories.price_data_repo import PriceDataRepo
from app.services.sector_service import SectorAggregate, SubsectorWithTickers
from app.ticker_fetcher import (
    HISTORY_DAYS,
    RETRY_ATTEMPTS,
    RETRY_MAX_DELAY,
    TickerFetcher,
    retry_delay,
    window_start,
)

TODAY = date.today()
OLD_DAY = window_start(TODAY) - timedelta(days=1)


class FakeTiingo:
    """Answers get_ticker_price from canned responses; a ticker with no response raises, as a failed request does."""

    def __init__(self, prices: dict[str, list[tuple[date, float]]]) -> None:
        self.prices = prices
        self.requests: list[tuple[str, str, str]] = []

    def get_ticker_price(self, ticker, frequency, startDate, endDate, columns):
        self.requests.append((ticker, startDate, endDate))
        if ticker not in self.prices:
            raise RuntimeError(f'500 Server Error for ticker {ticker}, token test-key')
        return [
            {'date': f'{d.isoformat()}T00:00:00.000Z', 'adjClose': close, 'adjVolume': 100}
            for d, close in self.prices[ticker]
        ]


def http_error(status: int, retry_after: str | None = None) -> RestClientError:
    """The error TiingoClient raises for an HTTP error status: a RestClientError wrapping requests' HTTPError."""
    response = requests.Response()
    response.status_code = status
    if retry_after is not None:
        response.headers['Retry-After'] = retry_after
    return RestClientError(
        requests.HTTPError(f'{status} Error for url: https://api.tiingo.com?token=test-key', response=response)
    )


class ScriptedTiingo:
    """Answers each get_ticker_price call with the next item of the ticker's script: an exception to raise, or rows."""

    def __init__(self, scripts: dict[str, list]) -> None:
        self.scripts = scripts
        self.calls: list[str] = []

    def get_ticker_price(self, ticker, frequency, startDate, endDate, columns):
        self.calls.append(ticker)
        answer = self.scripts[ticker].pop(0)
        if isinstance(answer, Exception):
            raise answer
        return [{'date': f'{d.isoformat()}T00:00:00.000Z', 'adjClose': close, 'adjVolume': 100} for d, close in answer]


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


def make_fetcher(repo: PriceDataRepo, tiingo: FakeTiingo, subsectors: dict[int, list[str]]) -> TickerFetcher:
    container = SimpleNamespace(
        tiingo_api_key='test-key', sector_service=FakeSectorService(subsectors), price_data_repo=repo
    )
    fetcher = TickerFetcher(container)
    fetcher.client = tiingo
    fetcher.sleep = lambda seconds: None
    return fetcher


def store(repo: PriceDataRepo, ticker: str, rows: list[tuple[date, float]], subsector_id: int = 1) -> None:
    repo.bulk_insert(
        PriceData(id=0, subsector_id=subsector_id, ticker=ticker, date=d, close=close, volume=100) for d, close in rows
    )


def stored(repo: PriceDataRepo, ticker: str) -> list[tuple[int, date, float]]:
    # the date column is declared TEXT, so rows come back with ISO date strings
    return [(p.subsector_id, date.fromisoformat(str(p.date)), p.close) for p in repo.all_for_ticker(ticker)]


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


def test_prune_deletes_only_rows_before_the_window(repo):
    store(repo, 'CVX', [(OLD_DAY, 1.0), (window_start(TODAY), 2.0)])

    deleted = make_fetcher(repo, FakeTiingo({}), {}).prune()

    assert deleted == 1
    assert stored(repo, 'CVX') == [(1, window_start(TODAY), 2.0)]


def test_replace_for_ticker_rejects_another_tickers_rows(repo):
    store(repo, 'CVX', [(days_ago(2), 2.0)])
    xom = PriceData(id=0, subsector_id=1, ticker='XOM', date=days_ago(1), close=1.0, volume=1)

    with pytest.raises(ValueError):
        repo.replace_for_ticker('CVX', [xom])

    assert stored(repo, 'CVX') == [(1, days_ago(2), 2.0)]


def run_cmd(fetcher: TickerFetcher, *argv: str) -> None:
    cmd = TickerFetcherCmd(SimpleNamespace(ticker_fetcher=fetcher))
    parser = argparse.ArgumentParser()
    cmd.add_arguments(parser)
    cmd.handle(parser.parse_args(argv))


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
