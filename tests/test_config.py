"""Config is read once the command is known, and a missing key stops the command before any work."""

import argparse
import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app import config
from app.commands.db_init import DbInit
from app.config import Config, ConfigError, load_config
from app.wiring import build_container

MAIN = str(Path(__file__).resolve().parent.parent / 'main.py')


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """No config keys in the environment, no .env in the working directory, and a terminal-less empty stdin."""
    for key in config.KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'stdin', io.StringIO(''))
    return tmp_path


def test_a_complete_environment_wins_and_stdin_is_left_alone(clean_env, monkeypatch):
    monkeypatch.setenv('DB_NAME', '/data/a.db')
    monkeypatch.setenv('TIINGO_API_KEY', 'env-key')
    (clean_env / '.env').write_text('DB_NAME=/file.db\nTIINGO_API_KEY=file-key\n')
    stdin = io.StringIO('DB_NAME=/other.db\n')
    monkeypatch.setattr(sys, 'stdin', stdin)

    cfg = load_config(require=('DB_NAME', 'TIINGO_API_KEY'))

    assert cfg.db_name == '/data/a.db'
    assert cfg.tiingo_api_key == 'env-key'
    assert stdin.tell() == 0


def test_reads_a_dotenv_piped_on_stdin(clean_env, monkeypatch):
    monkeypatch.setattr(sys, 'stdin', io.StringIO('DB_NAME=/piped.db\nTIINGO_API_KEY=piped-key\n'))

    cfg = load_config(require=('DB_NAME', 'TIINGO_API_KEY'))

    assert cfg.db_name == '/piped.db'
    assert cfg.tiingo_api_key == 'piped-key'


def test_missing_keys_are_all_named(clean_env):
    with pytest.raises(ConfigError) as err:
        load_config(require=('DB_NAME', 'TIINGO_API_KEY'))

    assert 'DB_NAME, TIINGO_API_KEY' in str(err.value)


def test_only_the_required_keys_must_be_set(clean_env, monkeypatch):
    monkeypatch.setenv('DB_NAME', '/data/a.db')

    cfg = load_config(require=('DB_NAME',))

    assert cfg.db_name == '/data/a.db'
    assert cfg.tiingo_api_key is None


def test_an_unknown_key_is_a_programming_error(clean_env):
    with pytest.raises(ValueError):
        load_config(require=('NOPE',))


def run_main(tmp_path, *argv: str, **env: str) -> subprocess.CompletedProcess:
    """Runs main.py with only 'env' as config, and an empty stdin.

    It runs from tmp_path, where no .env is: from the repo root, a developer's own .env would supply the keys.
    """
    base = {k: v for k, v in os.environ.items() if k not in config.KEYS}
    return subprocess.run(
        [sys.executable, MAIN, *argv],
        env={**base, **env},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
    )


def test_db_init_runs_without_a_tiingo_key(tmp_path):
    db = tmp_path / 'a.db'

    result = run_main(tmp_path, 'db-init', DB_NAME=str(db))

    assert result.returncode == 0, result.stderr
    assert db.exists()


@pytest.mark.parametrize('command', ['run-refresh', 'ticker-fetcher', 'ticker-sync'])
def test_tiingo_commands_stop_naming_the_missing_key(tmp_path, command):
    result = run_main(tmp_path, command, DB_NAME=str(tmp_path / 'a.db'))

    assert result.returncode == 2
    assert result.stderr.strip() == (
        f'main.py {command}: missing required config: TIINGO_API_KEY (set in the environment or in .env)'
    )
    assert 'PROGRESS' not in result.stdout
    assert not (tmp_path / 'a.db').exists()


@pytest.mark.parametrize('command', ['db-init', 'analyze', 'run-refresh'])
def test_a_missing_db_name_is_named_not_a_traceback(tmp_path, command):
    result = run_main(tmp_path, command, TIINGO_API_KEY='test-key')

    assert result.returncode == 2
    assert 'missing required config: DB_NAME' in result.stderr
    assert 'Traceback' not in result.stderr


class RecordingContainer:
    """Wraps the real container and records each name a command reads from it."""

    def __init__(self, container) -> None:
        self._container = container
        self.read: list[str] = []

    def __getattr__(self, name: str):
        self.read.append(name)
        return getattr(self._container, name)


def test_db_init_builds_nothing_it_does_not_use(tmp_path):
    container = RecordingContainer(
        build_container(Config(db_name=str(tmp_path / 'a.db'), tiingo_api_key=None, eodhd_api_key=None))
    )

    DbInit().handle(argparse.Namespace(), container)

    assert container.read == ['db_path', 'db_path', 'db']


def test_reading_a_key_the_command_did_not_require_names_it():
    container = build_container(Config(db_name='/data/a.db', tiingo_api_key=None, eodhd_api_key=None))

    with pytest.raises(ConfigError, match='PRICE_API'):
        container.ticker_fetcher  # noqa: B018


def test_the_dotenv_file_is_read_when_stdin_is_not_a_terminal(clean_env):
    (clean_env / '.env').write_text('DB_NAME=/file.db\nTIINGO_API_KEY=file-key\n')

    cfg = load_config(require=('DB_NAME', 'TIINGO_API_KEY'))

    assert cfg.db_name == '/file.db'
    assert cfg.tiingo_api_key == 'file-key'


def test_stdin_wins_over_the_dotenv_file(clean_env, monkeypatch):
    (clean_env / '.env').write_text('DB_NAME=/file.db\nTIINGO_API_KEY=file-key\n')
    monkeypatch.setattr(sys, 'stdin', io.StringIO('DB_NAME=/piped.db\n'))

    cfg = load_config(require=('DB_NAME', 'TIINGO_API_KEY'))

    assert cfg.db_name == '/piped.db'
    assert cfg.tiingo_api_key == 'file-key'


def test_an_empty_environment_value_is_filled_from_the_dotenv_file(clean_env, monkeypatch):
    monkeypatch.setenv('TIINGO_API_KEY', '')
    monkeypatch.setenv('DB_NAME', '/data/a.db')
    (clean_env / '.env').write_text('TIINGO_API_KEY=file-key\n')

    cfg = load_config(require=('DB_NAME', 'TIINGO_API_KEY'))

    assert cfg.tiingo_api_key == 'file-key'
    assert os.environ['TIINGO_API_KEY'] == ''


def test_db_init_keeps_the_database_when_stdin_has_no_answer(tmp_path):
    db = tmp_path / 'a.db'
    assert run_main(tmp_path, 'db-init', DB_NAME=str(db)).returncode == 0

    result = run_main(tmp_path, 'db-init', DB_NAME=str(db))

    assert result.returncode == 0, result.stderr
    assert 'Keeping existing database unchanged.' in result.stdout


def test_eodhd_key_is_loaded_without_being_required(clean_env, monkeypatch):
    monkeypatch.setenv('DB_NAME', '/data/a.db')
    monkeypatch.setenv('EODHD_API_KEY', 'eodhd-key')

    cfg = load_config(require=('DB_NAME',))

    assert cfg.eodhd_api_key == 'eodhd-key'
    assert cfg.tiingo_api_key is None


def test_eodhd_key_is_loaded_from_the_env_file(clean_env):
    (clean_env / '.env').write_text('DB_NAME=/file.db\nEODHD_API_KEY=file-eodhd\n')

    cfg = load_config(require=('DB_NAME',))

    assert cfg.eodhd_api_key == 'file-eodhd'


@pytest.mark.parametrize('command', ['run-refresh', 'ticker-fetcher'])
def test_eodhd_commands_stop_naming_the_missing_eodhd_key(tmp_path, command):
    result = run_main(tmp_path, command, DB_NAME=str(tmp_path / 'a.db'), PRICE_API='eodhd', TIINGO_API_KEY='tiingo-key')

    assert result.returncode == 2
    assert result.stderr.strip() == (
        f'main.py {command}: missing required config: EODHD_API_KEY (set in the environment or in .env)'
    )
    assert 'PROGRESS' not in result.stdout
    assert not (tmp_path / 'a.db').exists()


@pytest.mark.parametrize('value', ['EODHD', 'polygon', ' tiingo'])
def test_an_unknown_price_api_stops_naming_it_and_the_allowed_ones(tmp_path, value):
    result = run_main(
        tmp_path,
        'run-refresh',
        DB_NAME=str(tmp_path / 'a.db'),
        PRICE_API=value,
        TIINGO_API_KEY='tiingo-key',
        EODHD_API_KEY='eodhd-key',
    )

    assert result.returncode == 2
    assert result.stderr.strip() == f"main.py run-refresh: PRICE_API is '{value}', not one of: tiingo, eodhd"
    assert not (tmp_path / 'a.db').exists()


def test_price_api_defaults_to_tiingo_and_needs_only_its_key(clean_env, monkeypatch):
    monkeypatch.setenv('DB_NAME', '/data/a.db')
    monkeypatch.setenv('TIINGO_API_KEY', 'tiingo-key')

    cfg = load_config(require=('DB_NAME', 'PRICE_API'))

    assert cfg.price_api == 'tiingo'
    assert cfg.eodhd_api_key is None


def test_eodhd_needs_only_its_own_key(clean_env, monkeypatch):
    monkeypatch.setenv('DB_NAME', '/data/a.db')
    monkeypatch.setenv('PRICE_API', 'eodhd')
    monkeypatch.setenv('EODHD_API_KEY', 'eodhd-key')

    cfg = load_config(require=('DB_NAME', 'PRICE_API'))

    assert cfg.price_api == 'eodhd'
    assert cfg.tiingo_api_key is None


def test_price_api_in_the_dotenv_file_applies_when_the_environment_holds_the_keys(clean_env, monkeypatch):
    monkeypatch.setenv('DB_NAME', '/data/a.db')
    monkeypatch.setenv('TIINGO_API_KEY', 'tiingo-key')
    monkeypatch.setenv('EODHD_API_KEY', 'eodhd-key')
    (clean_env / '.env').write_text('PRICE_API=eodhd\nEODHD_API_KEY=file-eodhd\n')
    stdin = io.StringIO('PRICE_API=tiingo\n')
    monkeypatch.setattr(sys, 'stdin', stdin)

    cfg = load_config(require=('DB_NAME', 'PRICE_API'))

    assert (cfg.price_api, cfg.eodhd_api_key) == ('eodhd', 'eodhd-key')
    assert stdin.tell() == 0


def test_price_api_in_the_environment_wins_over_the_dotenv_file(clean_env, monkeypatch):
    monkeypatch.setenv('DB_NAME', '/data/a.db')
    monkeypatch.setenv('PRICE_API', 'tiingo')
    monkeypatch.setenv('TIINGO_API_KEY', 'tiingo-key')
    (clean_env / '.env').write_text('PRICE_API=eodhd\nEODHD_API_KEY=file-eodhd\n')

    cfg = load_config(require=('DB_NAME', 'PRICE_API'))

    assert cfg.price_api == 'tiingo'


def test_price_api_is_read_from_the_dotenv_file(clean_env):
    (clean_env / '.env').write_text('DB_NAME=/file.db\nPRICE_API=eodhd\nEODHD_API_KEY=file-eodhd\n')

    cfg = load_config(require=('DB_NAME', 'PRICE_API'))

    assert (cfg.price_api, cfg.eodhd_api_key) == ('eodhd', 'file-eodhd')


@pytest.mark.parametrize('api, client', [('tiingo', 'TiingoPrices'), ('eodhd', 'EodhdPrices')])
def test_the_container_builds_the_chosen_apis_client(api, client):
    container = build_container(
        Config(db_name='/data/a.db', tiingo_api_key='t-key', eodhd_api_key='e-key', price_api=api)
    )

    assert type(container.price_api_client).__name__ == client
