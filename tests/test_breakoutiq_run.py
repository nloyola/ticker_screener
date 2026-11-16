"""
The whole BreakoutIQ analysis as one command: breakoutiq-analysis builds the dataset from the archive, runs every
section and stores one analysis_run with all their results, which read back with their versioned payloads.
"""

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.breakoutiq.dataset import load_dataset
from app.breakoutiq.indicators import FLAGS
from app.breakoutiq.run import SECTIONS, Analysis, analyse, settings
from app.db import init_db
from app.repositories.analysis_data_repo import AnalysisDataRepo
from app.repositories.analysis_repo import AnalysisRepo
from app.repositories.benchmark_price_repo import BenchmarkPriceRepo
from app.repositories.history_repo import HistoryRepo, init_history_db

MAIN = Path(__file__).resolve().parent.parent / 'main.py'

SESSIONS = pd.bdate_range('2019-01-02', periods=160)
SIGNALS_PER_DATE = 3
HORIZONS = (5, 20)


@pytest.fixture
def db(tmp_path) -> SimpleNamespace:
    path = str(tmp_path / 'stock_data.db')
    history = str(tmp_path / 'history.db')
    init_db(path)
    init_history_db(history)
    c = SimpleNamespace(db_path=path, history_db_path=history)
    c.analysis_data_repo = AnalysisDataRepo(c)
    c.analysis_repo = AnalysisRepo(c)
    c.benchmark_price_repo = BenchmarkPriceRepo(c)
    c.history_repo = HistoryRepo(c)
    return c


def _archive(path: str, seed: int = 7) -> int:
    """
    Stores a random archive: SIGNALS_PER_DATE signals on each of SESSIONS, every one with complete outcomes at
    HORIZONS but for every tenth, whose 20-session horizon ended. Returns the number of signals.
    """
    rng = np.random.default_rng(seed)
    signals, outcomes = [], []
    signal_id = 0
    for day in SESSIONS:
        for ticker in rng.choice(40, size=SIGNALS_PER_DATE, replace=False):
            signal_id += 1
            close = 100.0
            values = {
                'id': signal_id,
                'ticker': f'T{ticker}',
                'market': 'US',
                'time_zone': 'America/New_York',
                'session_date': day.date().isoformat(),
                'close': close,
                'status': 'confirmed',
                'screener_version': '1.0.0',
                'rule_settings': '{}',
                'source': 'backfill',
                'sector': f'S{rng.integers(0, 3)}',
                'high_20d': close / (1 + rng.exponential(0.02)),
                'high_60d': close * 1.1,
                'breakout_60d': 0,
                'dma_50': close / (1 + rng.exponential(0.06)),
                'dma_50_prev_15d': 80.0,
                'volume_ratio': 1.5 + rng.exponential(1),
                'rsi': rng.normal(62, 6),
                'macd_hist': rng.normal(0.5, 0.3),
                'dma_20': close / (1 + rng.normal(0.03, 0.02)),
                'dma_200': close / (1 + rng.normal(0.15, 0.1)),
                'close_position': rng.uniform(0, 1),
                **{flag: int(rng.integers(0, 2)) for flag in FLAGS[:6]},
            }
            signals.append(values)
            for horizon in HORIZONS:
                if horizon == 20 and signal_id % 10 == 0:
                    outcomes.append((signal_id, horizon, 'ended'))
                else:
                    outcomes.append((signal_id, horizon, rng.normal(0.5, 5)))
    with sqlite3.connect(path) as con:
        for values in signals:
            con.execute(
                f'INSERT INTO signal ({", ".join(values)}) VALUES ({", ".join("?" * len(values))})',
                tuple(values.values()),
            )
        for signal_id, horizon, excess in outcomes:
            if excess == 'ended':
                con.execute(
                    'INSERT INTO signal_outcome (signal_id, horizon, status, ended_reason, last_session_date) '
                    "VALUES (?, ?, 'ended', 'history_ended', '2020-01-02')",
                    (signal_id, horizon),
                )
                continue
            con.execute(
                'INSERT INTO signal_outcome (signal_id, horizon, status, entry_date, entry_price, exit_date, '
                'exit_price, cost_bps, gross_return_pct, net_return_pct, benchmark_return_pct, excess_return_pct, '
                "mfe_pct, mae_pct, completed_at) VALUES (?, ?, 'complete', '2019-01-03', 100, '2019-02-01', 105, 20, "
                "?, ?, 1.0, ?, 7.0, -2.0, datetime('now'))",
                (signal_id, horizon, excess + 1.2, excess + 1.0, excess),
            )
    return len(signals)


def _run(db: SimpleNamespace) -> subprocess.CompletedProcess:
    env = {**os.environ, 'DB_NAME': db.db_path, 'HISTORY_DATABASE_FILE': db.history_db_path}
    for key in ('TIINGO_API_KEY', 'EODHD_API_KEY', 'PRICE_API'):
        env.pop(key, None)
    return subprocess.run(
        [sys.executable, str(MAIN), 'breakoutiq-analysis'],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        stdin=subprocess.DEVNULL,
    )


def test_the_command_stores_one_complete_run_that_reloads_with_its_payloads(db):
    count = _archive(db.db_path)

    done = _run(db)

    assert done.returncode == 0, done.stderr
    runs = db.analysis_repo.runs()
    assert len(runs) == 1
    run = runs[0]
    assert run.signals == count
    assert (run.first_session, run.last_session) == (SESSIONS[0].date(), SESSIONS[-1].date())
    assert run.settings == json.loads(json.dumps(settings(with_history=True)))
    assert run.settings['payload_versions'] == {s.name: s.payload_version for s in SECTIONS}
    assert run.git_commit is None or run.git_commit

    stored = db.analysis_repo.results(run.id)
    # every section stored its results, each with its section's payload version
    versions = {s.name: s.payload_version for s in SECTIONS}
    assert {r.section for r in stored} == set(versions)
    assert all(r.payload_version == versions[r.section] for r in stored)
    # the stored results are the sections' own, in order, their payloads reloading as written
    expected = analyse(load_dataset(db, with_history=True))
    assert [(r.section, r.finding, r.horizon) for r in stored] == [(r.section, r.finding, r.horizon) for r in expected]
    assert [r.payload for r in stored] == [json.loads(json.dumps(r.payload)) for r in expected]
    # the baseline's counts are the archive's, censored horizons included
    overall = {r.horizon: r.payload for r in stored if r.section == 'baseline' and r.finding == 'all'}
    assert overall[5]['signals'] == count and overall[5]['censored'] == 0
    assert overall[20]['signals'] == count and overall[20]['censored'] == count // 10
    assert overall[20]['shown'] and overall[20]['measures']

    summary = [line for line in done.stdout.splitlines() if line.startswith('breakoutiq-analysis:')]
    assert summary[0] == (
        f'breakoutiq-analysis: read {count} signals ({SESSIONS[0].date()} to {SESSIONS[-1].date()}), '
        f'{count * len(HORIZONS)} resolved horizons, {count // 10} of them censored'
    )
    assert summary[-1] == f'breakoutiq-analysis: stored run {run.id}'
    for section in SECTIONS:
        n = sum(1 for r in stored if r.section == section.name)
        assert any(line.startswith(f'breakoutiq-analysis: {section.name}: {n} result') for line in summary)


def test_an_archive_with_no_resolved_outcome_stores_an_empty_run(db):
    with sqlite3.connect(db.db_path) as con:
        con.execute(
            'INSERT INTO signal (ticker, market, time_zone, session_date, close, status, screener_version, '
            "rule_settings, source) VALUES ('T', 'US', 'America/New_York', '2026-11-13', 10, 'confirmed', '1.0.0', "
            "'{}', 'live')"
        )

    summary = Analysis(db).run(with_history=True)

    run = db.analysis_repo.runs()[0]
    assert (summary.run_id, summary.signals, summary.rows) == (run.id, 0, 0)
    assert (run.signals, run.first_session, run.last_session) == (0, None, None)
    assert db.analysis_repo.results(run.id) == []
