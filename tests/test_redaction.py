"""No API key, and no api_token= value, gets into the screener's output, whichever way it tries to escape."""

import io
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest
import requests

import main
from app import redaction
from app.redaction import RedactFilter, redact, redact_api_tokens, redacting_excepthook, register_secrets
from tests.test_ticker_fetcher import ScriptedTiingo, http_error, make_fetcher, repo  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent
TIINGO_KEY = 'tiingo-secret-1234'
EODHD_KEY = 'eodhd-secret-5678'
URL = f'https://eodhd.com/api/eod/SHOP.TO?api_token={EODHD_KEY}&fmt=json'


@pytest.fixture(autouse=True)
def secrets(monkeypatch):
    """Both keys in the environment, and the registered secrets restored afterwards."""
    monkeypatch.setenv('TIINGO_API_KEY', TIINGO_KEY)
    monkeypatch.setenv('EODHD_API_KEY', EODHD_KEY)
    monkeypatch.setattr(redaction, '_registered', set())


@pytest.fixture
def output():
    """
    The screener's own logging setup (config/0-stdout.json) at DEBUG, its stdout handler writing to a buffer.
    Returns a function giving what was written; the logging state is restored afterwards.
    """
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    main.setup_logging()
    handler = logging.getHandlerByName('stdout')
    buffer = io.StringIO()
    handler.setStream(buffer)
    root.setLevel(logging.DEBUG)
    yield buffer.getvalue
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)


def assert_clean(text: str) -> None:
    assert TIINGO_KEY not in text
    assert EODHD_KEY not in text
    assert 'secret' not in text


def test_redact_masks_both_keys_and_any_api_token_value(monkeypatch):
    monkeypatch.delenv('EODHD_API_KEY')
    register_secrets([EODHD_KEY, None, ''])

    text = redact(f'{TIINGO_KEY} {EODHD_KEY} {URL} "api_token=other-value" api_token=x y', 'extra-key')

    assert text == ('*** *** https://eodhd.com/api/eod/SHOP.TO?api_token=***&fmt=json "api_token=***" api_token=*** y')
    assert redact('extra-key here', 'extra-key') == '*** here'


def test_redact_api_tokens_stops_at_ampersand_whitespace_and_quotes():
    assert redact_api_tokens('a?api_token=k1&b=2') == 'a?api_token=***&b=2'
    assert redact_api_tokens("'api_token=k2' api_token=k3\tz") == "'api_token=***' api_token=***\tz"
    assert redact_api_tokens('api_token=&x') == 'api_token=***&x'


def test_the_stdout_handler_carries_the_filter():
    root = logging.getLogger()
    saved = root.handlers[:]
    try:
        main.setup_logging()
        assert any(isinstance(f, RedactFilter) for f in logging.getHandlerByName('stdout').filters)
    finally:
        root.handlers[:] = saved


@pytest.mark.parametrize(
    'error',
    [
        http_error(401, body=f'{{"detail": "bad token {EODHD_KEY}"}}'.encode()),
        requests.ConnectionError(f"HTTPSConnectionPool(host='eodhd.com'): Max retries exceeded with url: {URL}"),
        requests.Timeout(f"HTTPSConnectionPool(host='eodhd.com'): Read timed out. (read timeout=30) url: {URL}"),
    ],
    ids=['http-error', 'connection-error', 'timeout'],
)
def test_request_errors_holding_the_url_never_show_the_key(repo, output, error):  # noqa: F811
    attempts = [error] * 5  # enough for every retry of a transient error
    make_fetcher(repo, ScriptedTiingo({'CVX': attempts}), {1: ['CVX']}).fetch(strict=True)

    text = output()
    assert 'CVX' in text
    assert_clean(text)


def test_error_detail_masks_api_token_in_the_url():
    from app.price_apis.http import error_detail

    response = requests.Response()
    response.status_code = 404
    response._content = b''
    response.url = URL
    error = requests.HTTPError(f'404 Error for url: {URL}', response=response)

    assert error_detail(error) == '404 Error for url: https://eodhd.com/api/eod/SHOP.TO?api_token=***&fmt=json'
    assert error_detail(requests.ConnectionError(URL)).endswith('api_token=***&fmt=json')


def test_a_logged_traceback_never_shows_the_key(output):
    try:
        raise RuntimeError(f'request failed: {URL} with {TIINGO_KEY}')
    except RuntimeError:
        logging.getLogger('app.test').exception('fetch failed for %s', EODHD_KEY)

    text = output()
    assert 'Traceback' in text
    assert 'RuntimeError: request failed' in text
    assert_clean(text)


def test_a_debug_urllib3_line_never_shows_the_key(output):
    path = f'/api/eod/SHOP.TO?api_token={EODHD_KEY}&fmt=json'
    logging.getLogger('urllib3.connectionpool').debug(
        '%s://%s:%s "%s %s %s" %s %s', 'https', 'eodhd.com', 443, 'GET', path, 'HTTP/1.1', 200, None
    )

    text = output()
    assert '"GET /api/eod/SHOP.TO?api_token=***&fmt=json HTTP/1.1" 200' in text
    assert_clean(text)


def test_registered_secrets_are_masked_when_the_environment_has_none(monkeypatch, output):
    monkeypatch.delenv('TIINGO_API_KEY')
    monkeypatch.delenv('EODHD_API_KEY')
    register_secrets([TIINGO_KEY])

    logging.getLogger('app.test').warning('key is %s', TIINGO_KEY)

    assert 'key is ***' in output()


def test_the_excepthook_prints_a_redacted_traceback(capsys):
    try:
        raise ValueError(f'crash at {URL} using {TIINGO_KEY}')
    except ValueError as e:
        redacting_excepthook(type(e), e, e.__traceback__)

    err = capsys.readouterr().err
    assert err.startswith('Traceback (most recent call last):')
    assert 'ValueError: crash at' in err
    assert_clean(err)


def test_main_installs_the_excepthook_for_a_crash():
    script = (
        'import main\n'
        'def crash(self):\n'
        f'    raise RuntimeError("boom {URL} {TIINGO_KEY}")\n'
        'main.CommandLineInterface.execute = crash\n'
        'main.main()\n'
    )
    env = {**os.environ, 'TIINGO_API_KEY': TIINGO_KEY, 'EODHD_API_KEY': EODHD_KEY}

    result = subprocess.run([sys.executable, '-c', script], cwd=ROOT, env=env, capture_output=True, text=True)

    assert result.returncode == 1
    assert 'RuntimeError: boom' in result.stderr
    assert 'api_token=***' in result.stderr
    assert_clean(result.stderr + result.stdout)
