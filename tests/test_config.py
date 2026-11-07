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


def test_environment_is_the_whole_config_and_stdin_is_left_alone(clean_env, monkeypatch):
    monkeypatch.setenv('DB_NAME', '/data/a.db')
    monkeypatch.setenv('TIINGO_API_KEY', 'env-key')
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
    container = RecordingContainer(build_container(Config(db_name=str(tmp_path / 'a.db'), tiingo_api_key=None)))

    DbInit().handle(argparse.Namespace(), container)

    assert container.read == ['db_path', 'db_path', 'db']


def test_reading_a_key_the_command_did_not_require_names_it():
    container = build_container(Config(db_name='/data/a.db', tiingo_api_key=None))

    with pytest.raises(ConfigError, match='TIINGO_API_KEY'):
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
