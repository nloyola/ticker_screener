"""
analysis_run and analysis_result, BreakoutIQ's analysis runs and findings: every database gains them, a market data
reset keeps their rows, a run reads back with its results and versioned payloads, the tables refuse rows that break
their rules, a finding with no horizon is unique within its run, and a payload with no finite value stores it as
null.
"""

import sqlite3
from datetime import UTC, date, datetime
from types import SimpleNamespace

import pytest

from app.db import MARKET_DATA_TABLES, SCHEMA_VERSION, SQLite, clear_market_data, init_db, upgrade_db
from app.models import AnalysisResult, AnalysisRun
from app.repositories.analysis_repo import AnalysisRepo

TABLES = ('analysis_run', 'analysis_result', 'analysis_result_no_horizon')


def _query(path: str, sql: str, params=()) -> list[tuple]:
    with sqlite3.connect(path) as con:
        return con.execute(sql, params).fetchall()


def _schema_sql(path: str, table: str) -> str:
    sql = _query(path, 'SELECT sql FROM sqlite_master WHERE name = ?', (table,))[0][0]
    return ' '.join(sql.split())


@pytest.fixture
def db_path(tmp_path) -> str:
    path = str(tmp_path / 'stock_data.db')
    init_db(path)
    return path


@pytest.fixture
def version_13_db(tmp_path) -> str:
    """A database at version 13: today's schema without the analysis tables."""
    path = str(tmp_path / 'v13.db')
    init_db(path)
    with sqlite3.connect(path) as con:
        con.execute('DROP TABLE analysis_result')
        con.execute('DROP TABLE analysis_run')
        con.execute('PRAGMA user_version = 13')
    return path


@pytest.fixture
def version_14_db(tmp_path) -> str:
    """A database at version 14: the analysis tables without the index of findings with no horizon."""
    path = str(tmp_path / 'v14.db')
    init_db(path)
    with sqlite3.connect(path) as con:
        con.execute('DROP INDEX analysis_result_no_horizon')
        con.execute('PRAGMA user_version = 14')
    return path


def _run(**changes) -> AnalysisRun:
    values = {
        'started_at': datetime(2026, 11, 16, 23, 0, tzinfo=UTC),
        'finished_at': datetime(2026, 11, 16, 23, 4, 30, tzinfo=UTC),
        'first_session': date(2016, 11, 18),
        'last_session': date(2026, 5, 1),
        'signals': 5120,
        'settings': {'GAP_SESSIONS': 120, 'DEVELOPMENT_SHARE': 0.6},
        'screener_version': '1.0.0',
        'git_commit': 'abc1234',
    }
    return AnalysisRun(**{**values, **changes})


RESULTS = [
    AnalysisResult('baseline', 'all', 20, 1, {'signals': 5120, 'median_net_return_pct': 1.25}),
    AnalysisResult('baseline', 'all', None, 1, {'horizons': [5, 10, 20]}),
    AnalysisResult('secondary', 'rsi', 20, 2, {'bands': [[50, 60], [60, 70]], 'label': 'preliminary'}),
]


def test_upgrade_adds_the_tables_as_schema_sql_creates_them(version_13_db, db_path):
    upgrade_db(version_13_db)

    assert _query(version_13_db, 'PRAGMA user_version') == [(SCHEMA_VERSION,)]
    for table in TABLES:
        assert _schema_sql(version_13_db, table) == _schema_sql(db_path, table)


def test_upgrade_adds_the_index_of_findings_with_no_horizon(version_14_db, db_path):
    upgrade_db(version_14_db)

    assert _query(version_14_db, 'PRAGMA user_version') == [(SCHEMA_VERSION,)]
    assert _schema_sql(version_14_db, 'analysis_result_no_horizon') == _schema_sql(
        db_path, 'analysis_result_no_horizon'
    )


def test_a_finding_with_no_horizon_is_unique_within_its_run(db_path):
    repo = AnalysisRepo(SimpleNamespace(db_path=db_path))
    with pytest.raises(sqlite3.IntegrityError):
        repo.add(_run(), [RESULTS[1], RESULTS[1]])

    assert repo.runs() == []


def test_a_payload_value_that_is_not_finite_is_stored_as_null(db_path):
    repo = AnalysisRepo(SimpleNamespace(db_path=db_path))
    payload = {'estimate': float('nan'), 'bounds': [float('inf'), 1.5], 'nested': {'vif': float('-inf')}}

    run_id = repo.add(_run(settings={'bar': float('nan')}), [AnalysisResult('baseline', 'all', 20, 1, payload)])

    (result,) = repo.results(run_id)
    assert result.payload == {'estimate': None, 'bounds': [None, 1.5], 'nested': {'vif': None}}
    assert repo.runs()[0].settings == {'bar': None}


def test_the_tables_are_not_market_data(db_path):
    assert not set(TABLES) & set(MARKET_DATA_TABLES)
    AnalysisRepo(SimpleNamespace(db_path=db_path)).add(_run(), RESULTS)
    with SQLite(db_path).connect() as con:
        clear_market_data(con)
    assert _query(db_path, 'SELECT COUNT(*) FROM analysis_run') == [(1,)]
    assert _query(db_path, 'SELECT COUNT(*) FROM analysis_result') == [(len(RESULTS),)]


def test_a_run_reads_back_with_its_results_and_payload_versions(db_path):
    repo = AnalysisRepo(SimpleNamespace(db_path=db_path))
    first = repo.add(_run(), RESULTS)
    second = repo.add(_run(first_session=None, last_session=None, signals=0), [])

    assert repo.runs() == [_run(id=first), _run(id=second, first_session=None, last_session=None, signals=0)]
    assert repo.results(first) == RESULTS
    assert repo.results(second) == []


def test_a_run_whose_results_fail_is_not_stored(db_path):
    repo = AnalysisRepo(SimpleNamespace(db_path=db_path))
    with pytest.raises(sqlite3.IntegrityError):
        repo.add(_run(), [RESULTS[0], RESULTS[0]])

    assert repo.runs() == []


@pytest.mark.parametrize(
    'changes',
    [{'first_session': None}, {'signals': 0}, {'first_session': None, 'last_session': None}, {'signals': -1}],
)
def test_the_run_table_refuses_an_inconsistent_range_or_count(db_path, changes):
    with pytest.raises(sqlite3.IntegrityError):
        AnalysisRepo(SimpleNamespace(db_path=db_path)).add(_run(**changes), [])


@pytest.mark.parametrize(
    'result',
    [AnalysisResult('baseline', 'all', 15, 1, {}), AnalysisResult('baseline', 'all', 20, 0, {})],
)
def test_the_result_table_refuses_an_unknown_horizon_or_version(db_path, result):
    with pytest.raises(sqlite3.IntegrityError):
        AnalysisRepo(SimpleNamespace(db_path=db_path)).add(_run(), [result])
