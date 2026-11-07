"""The two price APIs behind PRICE_API: EODHD's prices and usage check, Tiingo's skipped markets, the market rule."""

import argparse
import json
import logging
import sqlite3
from datetime import UTC, date, datetime
from types import SimpleNamespace

import pytest
import requests

from app.commands.run_refresh_cmd import RunRefreshCmd
from app.commands.ticker_fetcher_cmd import TickerFetcherCmd
from app.markets import CANADA, US, UnsupportedTicker, market_of, market_sql
from app.price_apis import DailyPrice, PriceApiStop
from app.price_apis.eodhd import USAGE_TIMEOUT, EodhdPrices, EodhdUsage, eodhd_symbol
from tests.test_ticker_fetcher import (  # noqa: F401
    FakeTiingo,
    ScriptedAdapter,
    ScriptedTiingo,
    days_ago,
    http_error,
    make_fetcher,
    repo,
    store,
    stored,
)

EODHD_KEY = 'eodhd-secret-5678'
TODAY_GMT = datetime.now(UTC).date().isoformat()


def answer(status: int, body, url: str = 'https://eodhd.com/api/eod/X') -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response.reason = 'OK' if status == 200 else 'Error'
    response._content = json.dumps(body).encode()
    response.url = url
    return response


def usage_answer(spent: int = 0, limit: int = 100_000, extra: int = 0, day: str = TODAY_GMT) -> requests.Response:
    body = {'apiRequests': spent, 'apiRequestsDate': day, 'dailyRateLimit': limit, 'extraLimit': str(extra)}
    return answer(200, body, 'https://eodhd.com/api/user')


def eod_answer(rows: list[tuple[date, float, int]]) -> requests.Response:
    body = [
        {'date': d.isoformat(), 'open': 1, 'high': 1, 'low': 1, 'close': 999.0, 'adjusted_close': close, 'volume': vol}
        for d, close, vol in rows
    ]
    return answer(200, body)


def eodhd(answers: list) -> tuple[EodhdPrices, ScriptedAdapter]:
    client = EodhdPrices(EODHD_KEY)
    adapter = ScriptedAdapter(answers)
    client.session.mount('https://', adapter)
    return client, adapter


def stored_rows(repo, ticker: str) -> list[tuple[date, float, int]]:  # noqa: F811
    return [(p.date, p.close, p.volume) for p in repo.all_for_ticker(ticker)]


def run_fetch_cmd(fetcher, *argv: str) -> None:
    cmd = TickerFetcherCmd()
    parser = argparse.ArgumentParser()
    cmd.add_arguments(parser)
    cmd.handle(parser.parse_args(argv), SimpleNamespace(ticker_fetcher=fetcher))


# the market rule


@pytest.mark.parametrize(
    'ticker, market', [('AAPL', US), ('BRK-B', US), ('SHOP.TO', CANADA), ('BBD-B.TO', CANADA), ('ABC.V', CANADA)]
)
def test_market_is_read_from_the_suffix(ticker, market):
    assert market_of(ticker) == market


@pytest.mark.parametrize('ticker', ['AAPL', 'BRK-B', 'SHOP.TO', 'BBD-B.TO', 'ABC.V', 'TOTO', 'VEEV'])
def test_market_sql_agrees_with_market_of(ticker):
    with sqlite3.connect(':memory:') as con:
        (market,) = con.execute(f'SELECT {market_sql("t")} FROM (SELECT ? AS t)', (ticker,)).fetchone()
    assert market == market_of(ticker)


@pytest.mark.parametrize('ticker', ['VOD.L', 'BRK.B', 'SHOP.TO.TO', '.TO', 'SHOP.'])
def test_any_other_suffix_is_refused(ticker):
    with pytest.raises(UnsupportedTicker):
        market_of(ticker)


def test_eodhd_symbol_adds_us_to_bare_tickers_only():
    assert eodhd_symbol('AAPL') == 'AAPL.US'
    assert eodhd_symbol('BRK-B') == 'BRK-B.US'
    assert eodhd_symbol('SHOP.TO') == 'SHOP.TO'
    assert eodhd_symbol('ABC.V') == 'ABC.V'


# EODHD runs


def test_eodhd_run_stores_adjusted_close_and_volume_for_bare_and_suffixed_tickers(repo):  # noqa: F811
    client, adapter = eodhd(
        [
            usage_answer(),
            eod_answer([(days_ago(2), 150.5, 1000), (days_ago(1), 151.25, 1200)]),
            eod_answer([(days_ago(1), 98.75, 3400)]),
        ]
    )

    stats = make_fetcher(repo, client, {1: ['AAPL', 'SHOP.TO']}).fetch(strict=True)

    assert (stats.attempted, stats.failed, stats.skipped, stats.api) == (2, 0, 0, 'EODHD')
    assert stored_rows(repo, 'AAPL') == [(days_ago(2), 150.5, 1000), (days_ago(1), 151.25, 1200)]
    assert stored_rows(repo, 'SHOP.TO') == [(days_ago(1), 98.75, 3400)]
    urls = [request.url for request, _ in adapter.sent]
    assert urls[0].startswith('https://eodhd.com/api/user?')
    assert urls[1].startswith(f'https://eodhd.com/api/eod/AAPL.US?from={days_ago(120).isoformat()}&to=')
    assert urls[2].startswith('https://eodhd.com/api/eod/SHOP.TO?')
    assert all('fmt=json' in url and f'api_token={EODHD_KEY}' in url for url in urls)
    assert adapter.sent[0][1] == USAGE_TIMEOUT


def test_eodhd_404_fails_the_ticker_and_its_log_line_hides_the_key(repo, caplog):  # noqa: F811
    url = f'https://eodhd.com/api/eod/NOPE.US?from=x&api_token={EODHD_KEY}&fmt=json'
    not_found = answer(404, 'Ticker Not Found.', url)
    client, _ = eodhd([usage_answer(), not_found])

    stats = make_fetcher(repo, client, {1: ['NOPE']}).fetch(strict=True)

    assert (stats.attempted, stats.failed) == (1, 1)
    assert 'Error loading data for NOPE from EODHD' in caplog.text
    assert EODHD_KEY not in caplog.text


def test_eodhd_run_with_too_few_calls_left_stops_before_any_price_request(repo, caplog):  # noqa: F811
    client, adapter = eodhd([usage_answer(spent=99_999, limit=100_000, extra=0)])
    fetcher = make_fetcher(repo, client, {1: ['AAPL', 'SHOP.TO']})

    with pytest.raises(SystemExit) as exit_info:
        run_fetch_cmd(fetcher, '--strict')

    assert exit_info.value.code == 1
    assert len(adapter.sent) == 1  # the usage check only
    assert 'EODHD has 1 calls left today' in caplog.text
    assert 'fewer than the 2 tickers to fetch' in caplog.text


def test_extra_calls_count_once_the_daily_limit_is_spent():
    usage = EodhdUsage(spent=100_500, day='2026-11-07', daily_limit=100_000, extra=2_000)

    assert usage.calls_left(date(2026, 11, 7)) == 2_000
    # a counter from an earlier day has not been reset because nothing was spent since midnight GMT
    assert usage.calls_left(date(2026, 11, 8)) == 102_000


@pytest.mark.parametrize('status', [401, 403])
def test_eodhd_run_whose_key_is_refused_stops_before_any_price_request(repo, caplog, status):  # noqa: F811
    refused = answer(status, {'error': 'Unauthenticated'}, f'https://eodhd.com/api/user?api_token={EODHD_KEY}')
    client, adapter = eodhd([refused])
    fetcher = make_fetcher(repo, client, {1: ['AAPL']})

    with pytest.raises(PriceApiStop, match='EODHD refused the API key'):
        fetcher.fetch(strict=True)

    assert len(adapter.sent) == 1


def test_run_refresh_stopped_by_eodhd_exits_1_without_pruning_or_analyzing(repo, tmp_path):  # noqa: F811
    client, adapter = eodhd([answer(401, {}, 'https://eodhd.com/api/user')])
    fetcher = make_fetcher(repo, client, {1: ['AAPL']})
    analyzed = []
    container = SimpleNamespace(
        db_path=str(tmp_path / 'stock_data.db'),
        ticker_fetcher=fetcher,
        analyzer=SimpleNamespace(analyze=lambda: analyzed.append(True)),
    )
    cmd = RunRefreshCmd()
    parser = argparse.ArgumentParser()
    cmd.add_arguments(parser)

    with pytest.raises(SystemExit) as exit_info:
        cmd.handle(parser.parse_args([]), container)

    assert exit_info.value.code == 1
    assert analyzed == []
    assert len(adapter.sent) == 1


@pytest.mark.parametrize(
    'failure',
    [requests.ReadTimeout(f'read timed out: https://eodhd.com/api/user?api_token={EODHD_KEY}'), 'server error'],
)
def test_eodhd_usage_check_that_fails_otherwise_warns_and_carries_on(repo, caplog, failure):  # noqa: F811
    usage_failure = failure if isinstance(failure, Exception) else answer(500, {}, 'https://eodhd.com/api/user')
    client, adapter = eodhd([usage_failure, eod_answer([(days_ago(1), 10.0, 5)])])

    stats = make_fetcher(repo, client, {1: ['AAPL']}).fetch(strict=True)

    assert (stats.attempted, stats.failed) == (1, 0)
    assert len(adapter.sent) == 2
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and 'EODHD usage check failed, carrying on' in warnings[0]
    assert EODHD_KEY not in caplog.text


def test_eodhd_rows_parse_as_prices():
    rows = [{'date': '2026-11-06', 'close': 100.0, 'adjusted_close': 99.5, 'volume': 42}]

    assert EodhdPrices.to_prices(rows) == [DailyPrice(date=date(2026, 11, 6), close=99.5, volume=42)]


# Tiingo runs


def test_tiingo_run_skips_suffixed_tickers_without_attempting_them(repo, caplog, capsys):  # noqa: F811
    caplog.set_level(logging.INFO)
    store(repo, 'SHOP.TO', [(days_ago(5), 80.0)])
    tiingo = FakeTiingo({'AAPL': [(days_ago(1), 1.0)]})
    fetcher = make_fetcher(repo, tiingo, {1: ['AAPL', 'SHOP.TO', 'ABC.V'], 2: ['SHOP.TO']})

    run_fetch_cmd(fetcher, '--strict')

    assert [ticker for ticker, _, _ in tiingo.requests] == ['AAPL']
    assert (fetcher.stats.attempted, fetcher.stats.failed, fetcher.stats.skipped) == (1, 0, 2)
    assert capsys.readouterr().out.splitlines()[-1] == (
        'ticker-fetcher: 1 attempted, 0 failed (0%, threshold 50%), 2 skipped (not covered by Tiingo)'
    )
    assert 'skipped 2 tickers not covered by Tiingo' in caplog.text
    assert stored(repo, 'SHOP.TO') == [(1, days_ago(5), 80.0)]


def test_skipped_tickers_cannot_push_a_run_over_its_failure_limit(repo):  # noqa: F811
    tiingo = FakeTiingo({'AAPL': [(days_ago(1), 1.0)]})

    run_fetch_cmd(make_fetcher(repo, tiingo, {1: ['AAPL', 'A.TO', 'B.TO', 'C.TO']}), '--strict')  # no exit


def test_summary_prints_the_skipped_count_even_at_zero(repo, capsys):  # noqa: F811
    run_fetch_cmd(make_fetcher(repo, FakeTiingo({'AAPL': [(days_ago(1), 1.0)]}), {1: ['AAPL']}), '--strict')

    assert capsys.readouterr().out.splitlines()[-1].endswith(', 0 skipped (not covered by Tiingo)')


def test_ticker_with_an_unsupported_suffix_fails_without_a_request(repo, caplog):  # noqa: F811
    tiingo = FakeTiingo({'AAPL': [(days_ago(1), 1.0)]})

    stats = make_fetcher(repo, tiingo, {1: ['AAPL', 'VOD.L']}).fetch(strict=True)

    assert [ticker for ticker, _, _ in tiingo.requests] == ['AAPL']
    assert (stats.attempted, stats.failed, stats.skipped) == (2, 1, 0)
    assert 'unsupported exchange suffix: VOD.L' in caplog.text


# the log names the API


@pytest.mark.parametrize('api_name', ['Tiingo', 'EODHD'])
def test_the_log_names_the_api_in_use(repo, caplog, api_name):  # noqa: F811
    caplog.set_level(logging.INFO)
    api = ScriptedTiingo({'AAPL': [http_error(503)] * 3})
    api.name = api_name
    api.markets = frozenset({US, CANADA})

    make_fetcher(repo, api, {1: ['AAPL']}).fetch(strict=True)

    assert f'price API: {api_name}' in caplog.text
    assert f'{api_name} request for AAPL failed (attempt 1' in caplog.text
    assert f'Error loading data for AAPL from {api_name}' in caplog.text
