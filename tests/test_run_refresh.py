"""run-refresh and the screener lock it shares with the stock_analyzer admin pages."""

import argparse
import os
import signal
import subprocess
import sys
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from app.commands.run_refresh_cmd import EXIT_LOCK_HELD, RUN_ID_ENV, RunRefreshCmd
from app.screener_lock import ScreenerLockHeld, lock_path, screener_lock
from app.ticker_fetcher import FetchStats

STARTED_AT = datetime(2026, 11, 9, 15, tzinfo=UTC)


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / 'stock_data.db')


def test_lock_file_is_beside_the_database(db_path, tmp_path):
    assert lock_path(db_path) == tmp_path / 'screener.lock'


def test_lock_is_exclusive_and_released_on_exit(db_path):
    with screener_lock(db_path):
        with pytest.raises(ScreenerLockHeld):
            with screener_lock(db_path):
                pass
    with screener_lock(db_path):
        pass


def test_lock_is_released_when_its_holder_is_killed(db_path):
    holder = subprocess.Popen(
        [
            sys.executable,
            '-c',
            'import sys, time\n'
            'from app.screener_lock import screener_lock\n'
            'with screener_lock(sys.argv[1]):\n'
            '    print("locked", flush=True)\n'
            '    time.sleep(60)\n',
            db_path,
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == 'locked'
        with pytest.raises(ScreenerLockHeld):
            with screener_lock(db_path):
                pass
    finally:
        holder.send_signal(signal.SIGKILL)
        holder.wait()
        holder.stdout.close()
    with screener_lock(db_path):
        pass


class FakeFetcher:
    api = SimpleNamespace(name='Tiingo', setting='tiingo', markets=frozenset({'US'}))

    def __init__(self, stats: FetchStats) -> None:
        self.stats = stats
        self.calls = []
        self.prunes = 0
        self.flagged = {'GONE': 'not_found', 'BOTH': 'no_prices'}

    def fetch(self, **kwargs) -> FetchStats:
        self.calls.append(kwargs)
        return self.stats

    def prune(self) -> int:
        self.prunes += 1
        return 0


def make_cmd(db_path: str, stats: FetchStats) -> tuple[RunRefreshCmd, SimpleNamespace, FakeFetcher, list[object]]:
    fetcher = FakeFetcher(stats)
    # analyze appends its call, and path and outcome updates theirs, to the same list, so a test sees their order
    analyzed = []
    container = SimpleNamespace(
        db_path=db_path,
        ticker_fetcher=fetcher,
        analyzer=SimpleNamespace(
            now=lambda: STARTED_AT,
            analyze=lambda fetched_by=None, already_flagged=(), started_at=None, run_id=None: analyzed.append(
                (fetched_by, already_flagged, started_at, run_id)
            ),
            flagged={'OLD': 'stale', 'BOTH': 'stale'},
            sessions={'US': ('2026-11-09', 3)},
        ),
        signal_paths=SimpleNamespace(
            update=lambda markets, flagged, started_at: analyzed.append(('paths', markets, flagged, started_at))
        ),
        signal_outcome_updates=SimpleNamespace(update=lambda: analyzed.append('outcomes')),
        screen_export=SimpleNamespace(
            write=lambda sessions, run_id, data_source: analyzed.append(('export', sessions, run_id, data_source))
        ),
    )
    return RunRefreshCmd(), container, fetcher, analyzed


def run(cmd: RunRefreshCmd, container: SimpleNamespace, *argv: str) -> None:
    parser = argparse.ArgumentParser()
    cmd.add_arguments(parser)
    cmd.handle(parser.parse_args(argv), container)


def test_fetches_strictly_then_analyzes(db_path, monkeypatch):
    monkeypatch.setenv(RUN_ID_ENV, '5')
    cmd, container, fetcher, analyzed = make_cmd(db_path, FetchStats(attempted=10, failed=1))

    run(cmd, container)

    assert fetcher.calls == [{'strict': True}]
    assert fetcher.prunes == 1
    # analyze scores only the markets this run's API fetched, does not flag again what the fetch flagged, and
    # judges which sessions are complete by when the run started; it records the sessions with the runner's run id
    # then the signal paths of the markets fetched, ending the signals of each ticker flagged, by the fetch's reason
    # where both flagged it, then the outcomes from those paths, and last the screen export of the sessions analyze
    # scored, with the run id and the API the fetch used
    assert analyzed == [
        (fetcher.api, fetcher.flagged, STARTED_AT, 5),
        ('paths', fetcher.api.markets, {'GONE': 'not_found', 'BOTH': 'no_prices', 'OLD': 'stale'}, STARTED_AT),
        'outcomes',
        ('export', {'US': ('2026-11-09', 3)}, 5, 'tiingo'),
    ]


@pytest.mark.parametrize(
    'stats, argv',
    [
        (FetchStats(attempted=0, failed=0), ()),
        (FetchStats(attempted=4, failed=4), ()),
        (FetchStats(attempted=10, failed=3), ('--max-failure-rate', '0.2')),
    ],
)
def test_failed_fetch_exits_1_without_pruning_or_analyzing(db_path, stats, argv):
    cmd, container, fetcher, analyzed = make_cmd(db_path, stats)

    with pytest.raises(SystemExit) as exit_info:
        run(cmd, container, *argv)

    assert exit_info.value.code == 1
    assert fetcher.prunes == 0
    assert analyzed == []
    with screener_lock(db_path):  # released on the way out
        pass


def test_held_lock_exits_75_without_fetching(db_path, capsys):
    cmd, container, fetcher, analyzed = make_cmd(db_path, FetchStats(attempted=10, failed=0))

    with screener_lock(db_path):
        with pytest.raises(SystemExit) as exit_info:
            run(cmd, container)

    assert EXIT_LOCK_HELD == 75
    assert exit_info.value.code == 75
    assert fetcher.calls == []
    assert analyzed == []
    assert 'another refresh holds the lock' in capsys.readouterr().err


def test_main_exits_75_when_the_lock_is_held(db_path):
    env = {**os.environ, 'DB_NAME': db_path, 'TIINGO_API_KEY': 'test-key', 'SCREEN_EXPORT_DIR': '/nonexistent'}
    with screener_lock(db_path):
        result = subprocess.run(
            [sys.executable, 'main.py', 'run-refresh'], env=env, capture_output=True, text=True, timeout=120
        )
    assert result.returncode == 75, result.stderr
