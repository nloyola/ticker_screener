"""
Company names: each price API's lookup, on recorded answers, and ticker-fetcher looking up the missing names after
the prices, against a real database.
"""

import argparse
import logging
from types import SimpleNamespace

import pytest
import requests

from app.commands.ticker_fetcher_cmd import TickerFetcherCmd
from app.db import open_connection
from app.price_apis.http import CONNECT_TIMEOUT, READ_TIMEOUT, RETRY_ATTEMPTS
from app.price_apis.tiingo import TiingoPrices
from app.repositories.company_name_repo import CompanyNameRepo
from tests.test_price_apis import EODHD_KEY, answer, eodhd
from tests.test_ticker_fetcher import (  # noqa: F401
    FakeTiingo,
    ScriptedAdapter,
    days_ago,
    http_error,
    make_fetcher,
    repo,
    stored,
    tiingo_response,
)

# answers as Tiingo's metadata request and EODHD's search give them, cut to the fields the lookup reads and a few
# beside them
TIINGO_CVX = (
    b'{"ticker": "CVX", "name": "Chevron  Corp", "exchangeCode": "NYSE", '
    b'"startDate": "1962-01-02", "endDate": "2026-11-06", "description": "..."}'
)
TIINGO_UNKNOWN = b'{"detail": "Error: Ticker \'NOPE\' not found"}'
EODHD_SHOP_SEARCH = [
    # SHOP's popular US listing comes first, and a prefix match on TO: neither is the .TO listing
    {'Code': 'SHOP', 'Exchange': 'US', 'Name': 'Shopify Inc', 'Type': 'Common Stock', 'Country': 'USA'},
    {'Code': 'SHOPX', 'Exchange': 'TO', 'Name': 'Shop Something Ltd', 'Type': 'Common Stock', 'Country': 'Canada'},
    {'Code': 'shop', 'Exchange': 'TO', 'Name': 'Shopify Inc.', 'Type': 'Common Stock', 'Country': 'Canada'},
]


def tiingo(answers: list) -> tuple[TiingoPrices, ScriptedAdapter]:
    client = TiingoPrices('test-key')
    adapter = ScriptedAdapter(answers)
    client.session.mount('https://', adapter)
    return client, adapter


def names(repo) -> dict[str, tuple[str, str]]:  # noqa: F811
    """Every stored name, by ticker, with the price API it came from."""
    con = open_connection(repo.db_path)
    try:
        return {row[0]: (row[1], row[2]) for row in con.execute('SELECT ticker, name, price_api FROM company_name')}
    finally:
        con.close()


def name_repo(repo) -> CompanyNameRepo:  # noqa: F811
    return CompanyNameRepo(SimpleNamespace(db_path=repo.db_path))


def progress_names(out: str) -> list[str]:
    return [line for line in out.splitlines() if line.startswith('PROGRESS stage=names')]


# Tiingo's lookup


def test_tiingo_name_comes_from_the_ticker_metadata_with_the_key_in_a_header():
    client, adapter = tiingo([tiingo_response(200, TIINGO_CVX)])

    assert client.company_name('CVX') == 'Chevron Corp'
    ((request, timeout),) = adapter.sent
    assert request.url == 'https://api.tiingo.com/tiingo/daily/CVX'
    assert request.headers['Authorization'] == 'Token test-key'
    assert timeout == (CONNECT_TIMEOUT, READ_TIMEOUT)


def test_tiingo_unknown_ticker_has_no_name():
    client, _ = tiingo([tiingo_response(404, TIINGO_UNKNOWN)])

    assert client.company_name('NOPE') is None


@pytest.mark.parametrize('body', [b'{"ticker": "", "name": "Ghost"}', b'{"ticker": "CVX", "name": "  "}'])
def test_tiingo_metadata_naming_no_ticker_or_no_name_has_no_name(body):
    client, _ = tiingo([tiingo_response(200, body)])

    assert client.company_name('CVX') is None


def test_tiingo_error_status_raises_with_the_response():
    client, _ = tiingo([tiingo_response(500, b'{"detail": "boom"}')])

    with pytest.raises(requests.HTTPError) as error:
        client.company_name('CVX')

    assert error.value.response.status_code == 500


def test_tiingo_answer_that_is_not_an_object_raises():
    client, _ = tiingo([tiingo_response(200, b'[]')])

    with pytest.raises(ValueError, match='not an object'):
        client.company_name('CVX')


# EODHD's lookup


def test_eodhd_name_of_a_toronto_ticker_is_its_listing_on_to():
    client, adapter = eodhd([answer(200, EODHD_SHOP_SEARCH, 'https://eodhd.com/api/search/SHOP')])

    assert client.company_name('SHOP.TO') == 'Shopify Inc.'
    ((request, timeout),) = adapter.sent
    assert request.url.startswith('https://eodhd.com/api/search/SHOP?')
    assert 'exchange=TO' in request.url and 'limit=500' in request.url
    assert timeout == (CONNECT_TIMEOUT, READ_TIMEOUT)


@pytest.mark.parametrize('ticker, code, exchange', [('BRK-B', 'BRK-B', 'US'), ('ABC.V', 'ABC', 'V')])
def test_eodhd_searches_the_ticker_s_own_exchange(ticker, code, exchange):
    match = {'Code': code, 'Exchange': exchange, 'Name': 'The Company'}
    client, adapter = eodhd([answer(200, [match])])

    assert client.company_name(ticker) == 'The Company'
    ((request, _),) = adapter.sent
    assert request.url.startswith(f'https://eodhd.com/api/search/{code}?') and f'exchange={exchange}' in request.url


def test_eodhd_ticker_with_no_exact_match_on_its_exchange_has_no_name():
    # only SHOP's US listing and a prefix match: the bare ticker SHOPX on US has neither
    client, _ = eodhd([answer(200, EODHD_SHOP_SEARCH)])

    assert client.company_name('SHOPX') is None


def test_eodhd_404_has_no_name():
    client, _ = eodhd([answer(404, {'error': 'not found'})])

    assert client.company_name('NOPE') is None


def test_eodhd_error_status_raises_with_the_response():
    client, _ = eodhd([answer(402, {'error': 'limit reached'})])

    with pytest.raises(requests.HTTPError) as error:
        client.company_name('AAPL')

    assert error.value.response.status_code == 402


def test_eodhd_answer_that_is_not_a_list_raises():
    client, _ = eodhd([answer(200, {'Code': 'AAPL'})])

    with pytest.raises(ValueError, match='not a list'):
        client.company_name('AAPL')


# ticker-fetcher


def test_fetch_stores_the_names_of_the_tickers_it_stored(repo, capsys):  # noqa: F811
    api = FakeTiingo({'CVX': [(days_ago(1), 1.0)], 'XOM': [(days_ago(1), 2.0)]})
    api.names = {'CVX': 'Chevron Corp', 'XOM': 'Exxon Mobil Corp'}

    make_fetcher(repo, api, {1: ['CVX', 'XOM'], 2: ['XOM']}).fetch()

    assert api.name_lookups == ['CVX', 'XOM']
    assert names(repo) == {'CVX': ('Chevron Corp', 'tiingo'), 'XOM': ('Exxon Mobil Corp', 'tiingo')}
    assert progress_names(capsys.readouterr().out) == [
        'PROGRESS stage=names ticker=0/2',
        'PROGRESS stage=names ticker=1/2',
        'PROGRESS stage=names ticker=2/2',
    ]


def test_second_fetch_makes_no_name_lookups(repo, capsys):  # noqa: F811
    prices = {'CVX': [(days_ago(1), 1.0)]}
    first = FakeTiingo(prices)
    first.names = {'CVX': 'Chevron Corp'}
    make_fetcher(repo, first, {1: ['CVX']}).fetch()
    capsys.readouterr()

    second = FakeTiingo(prices)
    second.names = {'CVX': 'Renamed Corp'}
    make_fetcher(repo, second, {1: ['CVX']}).fetch()

    assert second.__dict__.get('name_lookups', []) == []
    assert names(repo) == {'CVX': ('Chevron Corp', 'tiingo')}
    assert progress_names(capsys.readouterr().out) == ['PROGRESS stage=names ticker=0/0']


def test_a_name_stored_under_another_case_is_not_looked_up(repo):  # noqa: F811
    name_repo(repo).store('cvx', 'Chevron Corp', 'eodhd')
    api = FakeTiingo({'CVX': [(days_ago(1), 1.0)]})

    make_fetcher(repo, api, {1: ['CVX']}).fetch()

    assert api.__dict__.get('name_lookups', []) == []


def test_tickers_not_stored_are_not_looked_up(repo):  # noqa: F811
    # CVX fails, XOM holds no trading day, SHOP.TO is not covered by Tiingo, VOD.L has no market
    api = FakeTiingo({'XOM': [], 'AAPL': [(days_ago(1), 1.0)]})
    api.names = {'AAPL': 'Apple Inc'}

    make_fetcher(repo, api, {1: ['CVX', 'XOM', 'SHOP.TO', 'VOD.L', 'AAPL']}).fetch()

    assert api.name_lookups == ['AAPL']


def test_unknown_name_is_not_stored_and_is_logged(repo, caplog):  # noqa: F811
    caplog.set_level(logging.INFO)
    api = FakeTiingo({'CVX': [(days_ago(1), 1.0)]})

    make_fetcher(repo, api, {1: ['CVX']}).fetch()

    assert api.name_lookups == ['CVX']
    assert names(repo) == {}
    assert 'no company name on Tiingo for CVX' in caplog.text


def test_failed_lookup_leaves_the_outcome_and_failure_count_unchanged(repo, caplog, capsys):  # noqa: F811
    prices = {'CVX': [(days_ago(1), 1.0)], 'XOM': [(days_ago(1), 2.0)]}
    api = FakeTiingo(prices)
    api.names = {'CVX': http_error(401, body=b'{"detail": "Invalid token test-key"}'), 'XOM': 'Exxon Mobil Corp'}
    fetcher = make_fetcher(repo, api, {1: ['CVX', 'XOM']})

    cmd = TickerFetcherCmd()
    parser = argparse.ArgumentParser()
    cmd.add_arguments(parser)
    cmd.handle(parser.parse_args(['--strict', '--max-failure-rate', '0']), SimpleNamespace(ticker_fetcher=fetcher))

    assert (fetcher.stats.attempted, fetcher.stats.failed) == (2, 0)
    assert 'ticker-fetcher: 2 attempted, 0 failed' in capsys.readouterr().out
    assert names(repo) == {'XOM': ('Exxon Mobil Corp', 'tiingo')}
    assert stored(repo, 'CVX') == [(1, days_ago(1), 1.0)]
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and 'company name of CVX not looked up on Tiingo' in warnings[0]
    assert 'Invalid token ***' in warnings[0]
    assert 'test-key' not in caplog.text


def test_transient_lookup_error_is_retried(repo):  # noqa: F811
    client, adapter = tiingo(
        [
            tiingo_response(200, f'[{{"date": "{days_ago(1).isoformat()}", "adjClose": 1, "adjVolume": 1}}]'.encode()),
            tiingo_response(503),
            tiingo_response(200, TIINGO_CVX),
        ]
    )

    make_fetcher(repo, client, {1: ['CVX']}).fetch()

    assert len(adapter.sent) == 3
    assert names(repo) == {'CVX': ('Chevron Corp', 'tiingo')}


def test_lookup_that_keeps_failing_gives_up_after_the_retries(repo):  # noqa: F811
    api = FakeTiingo({'CVX': [(days_ago(1), 1.0)]})
    attempts = []

    def company_name(ticker):
        attempts.append(ticker)
        raise http_error(503)

    api.company_name = company_name

    stats = make_fetcher(repo, api, {1: ['CVX']}).fetch()

    assert attempts == ['CVX'] * RETRY_ATTEMPTS
    assert (stats.attempted, stats.failed) == (1, 0)
    assert names(repo) == {}


def test_eodhd_run_stores_its_names_as_eodhd_and_hides_the_key(repo, caplog):  # noqa: F811
    eod = [{'date': days_ago(1).isoformat(), 'adjusted_close': 1.0, 'volume': 5}]
    client, _ = eodhd(
        [
            answer(200, {'apiRequests': 0, 'dailyRateLimit': 100, 'extraLimit': 0}, 'https://eodhd.com/api/user'),
            answer(200, eod),
            answer(200, eod),
            answer(200, EODHD_SHOP_SEARCH),
            requests.ConnectionError(f'refused: https://eodhd.com/api/search/AAPL?api_token={EODHD_KEY}'),
            requests.ConnectionError(f'refused: https://eodhd.com/api/search/AAPL?api_token={EODHD_KEY}'),
            requests.ConnectionError(f'refused: https://eodhd.com/api/search/AAPL?api_token={EODHD_KEY}'),
        ]
    )

    stats = make_fetcher(repo, client, {1: ['SHOP.TO', 'AAPL']}).fetch(strict=True)

    assert (stats.attempted, stats.failed) == (2, 0)
    assert names(repo) == {'SHOP.TO': ('Shopify Inc.', 'eodhd')}
    assert 'company name of AAPL not looked up on EODHD' in caplog.text
    assert EODHD_KEY not in caplog.text
