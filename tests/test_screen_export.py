"""
The screen export (app/screen_export.py), against a real database: a session with a breakout, a Canadian breakout,
a watchlist ticker and a breakout that fails the next session, scored and tracked as run-refresh does, writes one file
per session in the schema the parent's issue #70 settles, every number equal to the stored value, unrounded, and every
price in its listing's currency; a failure is reported in each file whose FOLLOW_UP_SESSIONS window holds its session;
a rerun of the session replaces the file; an issuer's non-primary listing is unranked
and listed last; and a run that scored no session writes none.
"""

import json
import sqlite3
from dataclasses import replace
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.analyzer import SCREENER_VERSION, SECONDARY_INDICATORS, Analyzer, rule_settings
from app.db import init_db
from app.export_settings import EXPORT_VERSION, FOLLOW_UP_SESSIONS, export_settings
from app.markets import CANADA, US, is_session
from app.repositories.benchmark_price_repo import BenchmarkPriceRepo
from app.repositories.issuer_repo import IssuerRepo
from app.repositories.price_data_repo import PriceDataRepo
from app.repositories.screen_export_repo import DayRange, ExportRow, ScreenExportRepo
from app.repositories.screened_session_repo import ScreenedSessionRepo
from app.repositories.sector_repo import SectorRepo
from app.repositories.security_state_repo import SecurityStateRepo
from app.repositories.signal_outcome_repo import SignalOutcomeRepo
from app.repositories.signal_path_repo import SignalPathRepo
from app.repositories.signal_repo import SignalRepo
from app.repositories.stock_criteria_repo import StockCriteriaRepo
from app.repositories.subsector_repo import SubsectorRepo
from app.screen_export import ScreenExport, export_file_name, export_row
from app.services.sector_service import SectorService
from app.signal_outcome_updates import SignalOutcomeUpdates
from app.signal_paths import SignalPaths
from tests.test_analyzer import flags
from tests.test_analyzer_db import EVENING, NEW_YORK, TODAY, breakout_rows, price_rows, shift, table_rows
from tests.test_issuer_repo import add_issuer
from tests.test_security_state_repo import set_exchange

RUN_TIME = datetime(2026, 11, 17, 23, 45, 12, tzinfo=UTC)

# the export's keys, exactly, at every level ("Export design" in the parent's issue #70)
RUN_KEYS = [
    'export_version',
    'run_time',
    'as_of',
    'sessions',
    'data_source',
    'screener_version',
    'git_commit',
    'run_id',
    'rule_settings',
    'export_settings',
    'universe',
    'status_counts',
]
ROW_KEYS = [
    'signal_id',
    'ticker',
    'screener_ticker',
    'archive_id',
    'market',
    'exchange',
    'currency',
    'sector',
    'subsector',
    'issuer',
    'primary_listing',
    'status',
    'session_date',
    'mandatory',
    'secondary',
    'context',
    'risk',
    'technical_rank',
    'breakout_date',
    'days_since_breakout',
]
MANDATORY_KEYS = {
    'breakout': [
        'pass',
        'close',
        'high_20d',
        'high_60d',
        'breakout_20d',
        'breakout_60d',
        'lookback_cleared',
        'breakout_level',
        'breakout_margin_pct',
        'required_margin_pct',
        'average_daily_range_pct',
        'margin_met',
    ],
    'volume': ['pass', 'volume', 'avg_volume_20d', 'volume_ratio'],
    'above_sma_50': ['pass', 'sma_50'],
    'sma_50_rising': ['pass', 'sma_50', 'sma_50_15d_ago'],
}
SECONDARY_KEYS = {
    'rsi': ['rsi'],
    'macd': [
        'macd',
        'signal',
        'macd_hist',
        'macd_hist_prev',
        'macd_days_above',
        'macd_above_signal',
        'macd_crossover',
        'macd_hist_rising',
    ],
    'relative_strength': [
        'return_1m_pct',
        'return_3m_pct',
        'rs_market_1m',
        'rs_market_3m',
        'rs_sector_1m',
        'rs_sector_3m',
        'outperforms_market_1m',
        'outperforms_market_3m',
        'outperforms_sector_1m',
        'outperforms_sector_3m',
    ],
    'moving_average_alignment': [
        'sma_20',
        'sma_20_15d_ago',
        'sma_200',
        'sma_200_15d_ago',
        'price_gt_20dma',
        'price_gt_50dma',
        'price_gt_200dma',
        'dma_20_rising',
        'dma_50_rising',
        'dma_200_rising',
    ],
    'consolidation_quality': [
        'consolidation_range_pct',
        'consolidation_tight',
        'consolidation_vol_ratio',
        'consolidation_vol_declining',
    ],
    'closing_strength': ['high', 'low', 'close_position', 'close_upper_quarter'],
    'ema_21_alignment': [
        'ema_21',
        'ema_21_5d_ago',
        'ema_21w',
        'ema_21w_1w_ago',
        'price_gt_21ema',
        'ema_21_rising',
        'price_gt_21wema',
        'ema_21w_rising',
    ],
}
CONTEXT_KEYS = [
    'dist_high_20d_pct',
    'dist_high_60d_pct',
    'near_resistance',
    'high_120d',
    'dist_high_120d_pct',
    'in_120d_tier',
    'avg_dollar_volume',
]
RISK_KEYS = [
    'invalidation_level',
    'invalidation_basis',
    'pct_to_invalidation',
    'swing_low',
    'swing_low_date',
    'pct_to_swing_low',
    'wide_risk',
]
FOLLOW_UP_KEYS = [
    'signal_id',
    'archive_id',
    'ticker',
    'breakout_date',
    'breakout_level',
    'event',
    'event_date',
    'sessions_after_breakout',
    'close',
    'low',
    'volume',
    'avg_volume_20d',
]

# each number of a row that stock_criteria stores, by its place in the row, with the column it must equal exactly
STORED_NUMBERS = {
    ('mandatory', 'breakout', 'close'): 'price',
    ('mandatory', 'breakout', 'high_20d'): 'high_20d',
    ('mandatory', 'breakout', 'high_60d'): 'high_60d',
    ('mandatory', 'breakout', 'breakout_level'): 'breakout_level',
    ('mandatory', 'breakout', 'breakout_margin_pct'): 'breakout_margin_pct',
    ('mandatory', 'breakout', 'required_margin_pct'): 'required_margin_pct',
    ('mandatory', 'breakout', 'average_daily_range_pct'): 'average_daily_range_pct',
    ('mandatory', 'volume', 'volume'): 'volume',
    ('mandatory', 'volume', 'avg_volume_20d'): 'avg_vol_20d',
    ('mandatory', 'volume', 'volume_ratio'): 'volume_ratio',
    ('mandatory', 'above_sma_50', 'sma_50'): 'dma_50',
    ('mandatory', 'sma_50_rising', 'sma_50_15d_ago'): 'dma_50_prev_15d',
    ('mandatory', 'core_score'): 'core_criteria_score',
    ('context', 'dist_high_20d_pct'): 'dist_high_20d_pct',
    ('context', 'dist_high_60d_pct'): 'dist_high_60d_pct',
    ('context', 'high_120d'): 'high_120d',
    ('context', 'dist_high_120d_pct'): 'dist_high_120d_pct',
    ('context', 'avg_dollar_volume'): 'avg_dollar_volume',
    ('risk', 'invalidation_level'): 'breakout_level',
    ('risk', 'pct_to_invalidation'): 'invalidation_distance_pct',
    ('risk', 'swing_low'): 'swing_low',
    ('days_since_breakout',): 'sessions_since_breakout',
}
STORED_SECONDARY = {
    'rsi': 'rsi',
    'macd': 'macd',
    'signal': 'signal',
    'macd_hist': 'macd_hist',
    'macd_hist_prev': 'macd_hist_prev',
    'macd_days_above': 'macd_days_above',
    'return_1m_pct': 'return_1m_pct',
    'return_3m_pct': 'return_3m_pct',
    'rs_market_1m': 'rs_market_1m',
    'rs_market_3m': 'rs_market_3m',
    'rs_sector_1m': 'rs_sector_1m',
    'rs_sector_3m': 'rs_sector_3m',
    'sma_20': 'dma_20',
    'sma_20_15d_ago': 'dma_20_prev_15d',
    'sma_200': 'dma_200',
    'sma_200_15d_ago': 'dma_200_prev_15d',
    'consolidation_range_pct': 'consolidation_range_pct',
    'consolidation_vol_ratio': 'consolidation_vol_ratio',
    'close_position': 'close_position',
    'ema_21': 'ema_21',
    'ema_21_5d_ago': 'ema_21_prev_5d',
    'ema_21w': 'ema_21w',
    'ema_21w_1w_ago': 'ema_21w_prev_1w',
}


@pytest.fixture
def path(tmp_path) -> str:
    path = str(tmp_path / 'stock_data.db')
    init_db(path)
    return path


@pytest.fixture
def export_dir(tmp_path) -> Path:
    return tmp_path / 'screens'


@pytest.fixture
def container(path, export_dir) -> SimpleNamespace:
    container = SimpleNamespace(db_path=path, screen_export_dir=str(export_dir))
    container.price_data_repo = PriceDataRepo(container)
    container.benchmark_price_repo = BenchmarkPriceRepo(container)
    container.stock_criteria_repo = StockCriteriaRepo(container)
    container.screened_session_repo = ScreenedSessionRepo(container)
    container.signal_repo = SignalRepo(container)
    container.issuer_repo = IssuerRepo(container)
    container.security_state_repo = SecurityStateRepo(container)
    container.signal_path_repo = SignalPathRepo(container)
    container.signal_paths = SignalPaths(container)
    container.signal_outcome_repo = SignalOutcomeRepo(container)
    container.signal_outcome_updates = SignalOutcomeUpdates(container)
    container.sector_repo = SectorRepo(container)
    container.subsector_repo = SubsectorRepo(container)
    container.sector_service = SectorService(container)
    container.screen_export_repo = ScreenExportRepo(container)
    container.analyzer = Analyzer(container, now=lambda: EVENING)
    container.screen_export = ScreenExport(container, now=lambda: RUN_TIME)
    return container


def run(container, started_at: datetime = EVENING) -> Path | None:
    """One live run on the stored prices, as run-refresh runs it after its fetch: analyze, paths, outcomes, export."""
    container.analyzer.analyze(run_id=7, started_at=started_at)
    container.signal_paths.update(markets=(US, CANADA), flagged={}, started_at=started_at)
    container.signal_outcome_updates.update()
    return container.screen_export.write(container.analyzer.sessions, run_id=7, data_source='eodhd')


def read(file: Path) -> dict:
    return json.loads(file.read_text(encoding='utf-8'))


def at(row: dict, place: tuple[str, ...]) -> object:
    for key in place:
        row = row[key]
    return row


def test_a_session_run_twice_writes_one_file_in_the_schema_with_the_stored_values(path, container, export_dir):
    # on TODAY, BRK (on the NYSE, as only the security master says) and SHOP.TO break out over their 198 highs and
    # NEAR is on the watchlist; FALL has no status
    breakout_rows(path, container)
    set_exchange(path, 'BRK', 'NYSE')

    first = run(container)
    second = run(container)

    # the rerun of the session replaced its file, leaving no temporary file behind
    assert first == second == export_dir / export_file_name(TODAY)
    assert [p.name for p in export_dir.iterdir()] == [f'screen_{TODAY.isoformat()}.json']
    export = read(second)
    assert list(export) == ['run', 'rows', 'follow_up']

    header = export['run']
    assert list(header) == RUN_KEYS
    assert header == {
        'export_version': EXPORT_VERSION,
        'run_time': '2026-11-17T23:45:12Z',
        'as_of': TODAY.isoformat(),
        'sessions': {US: TODAY.isoformat(), CANADA: TODAY.isoformat()},
        'data_source': 'eodhd',
        'screener_version': SCREENER_VERSION,
        'git_commit': header['git_commit'],
        'run_id': 7,
        'rule_settings': rule_settings(),
        'export_settings': export_settings(),
        # the Oil subsector lists BRK, NEAR and FALL; SHOP.TO has prices under it, but it does not list it
        'universe': {US: {'listed': 3, 'scored': 3}, CANADA: {'listed': 0, 'scored': 1}},
        'status_counts': {'high_conviction': 0, 'confirmed': 2, 'extended': 0, 'pre_breakout': 1},
    }

    rows = {r['screener_ticker']: r for r in export['rows']}
    assert list(rows) == ['BRK', 'SHOP.TO', 'NEAR']
    for row in rows.values():
        assert list(row) == ROW_KEYS
        for check, keys in MANDATORY_KEYS.items():
            assert list(row['mandatory'][check]) == keys
        assert list(row['mandatory']) == [*MANDATORY_KEYS, 'core_score']
        assert list(row['secondary']) == ['checks', 'score']
        checks = row['secondary']['checks']
        assert [(c['id'], c['name']) for c in checks] == [(f'S{i}', n) for i, n in enumerate(SECONDARY_INDICATORS, 1)]
        assert [list(c) for c in checks] == [['id', 'name', 'pass', 'values']] * 7
        assert {c['name']: list(c['values']) for c in checks} == SECONDARY_KEYS
        assert sum(c['pass'] for c in checks) == row['secondary']['score']
        assert list(row['context']) == CONTEXT_KEYS
        assert list(row['risk']) == RISK_KEYS

    # every number equals the value stock_criteria stores, unrounded, None where it stores none
    stored = table_rows(path, 'stock_criteria')
    for ticker, row in rows.items():
        for place, column in STORED_NUMBERS.items():
            assert at(row, place) == stored[ticker][column], (ticker, place)
        values = {k: v for c in row['secondary']['checks'] for k, v in c['values'].items()}
        for key, column in STORED_SECONDARY.items():
            assert values[key] == stored[ticker][column], (ticker, key)
    assert rows['BRK']['mandatory']['breakout']['close'] == 202.456

    archive = {t: r['id'] for t, r in table_rows(path, 'signal').items()}
    signal_date = TODAY.isoformat()
    # BRK: a US breakout in dollars, in EODHD's format, at its exact 198 level, 4.456 / 198 above it
    assert {k: v for k, v in rows['BRK'].items() if k not in ('mandatory', 'secondary', 'context', 'risk')} == {
        'signal_id': f'{signal_date}:BRK.US',
        'ticker': 'BRK.US',
        'screener_ticker': 'BRK',
        'archive_id': archive['BRK'],
        'market': US,
        'exchange': 'NYSE',
        'currency': 'USD',
        'sector': 'Energy',
        'subsector': 'Oil',
        'issuer': None,
        'primary_listing': True,
        'status': 'confirmed',
        'session_date': signal_date,
        'technical_rank': 1,
        'breakout_date': signal_date,
        'days_since_breakout': 0,
    }
    breakout = rows['BRK']['mandatory']['breakout']
    assert breakout['breakout_level'] == 198.0
    assert breakout['breakout_margin_pct'] == pytest.approx(4.456 / 198 * 100)
    assert (breakout['pass'], breakout['lookback_cleared'], breakout['margin_met']) == (True, 60, True)
    assert rows['BRK']['risk'] == {
        'invalidation_level': 198.0,
        'invalidation_basis': 'breakout_level',
        'pct_to_invalidation': stored['BRK']['invalidation_distance_pct'],
        'swing_low': None,
        'swing_low_date': None,
        'pct_to_swing_low': None,
        'wide_risk': False,
    }
    # its rows store no high or low for the day
    closing = rows['BRK']['secondary']['checks'][5]['values']
    assert (closing['high'], closing['low']) == (None, None)

    # SHOP.TO: a Canadian breakout in Canadian dollars, as stored, its exchange from its suffix
    shop = rows['SHOP.TO']
    assert (shop['ticker'], shop['currency'], shop['exchange'], shop['market']) == ('SHOP.TO', 'CAD', 'TSX', CANADA)
    assert shop['technical_rank'] == 2
    assert shop['signal_id'] == f'{signal_date}:SHOP.TO'

    # NEAR, on the watchlist: archived as a watchlist record, outside every run, so every value of a run is null
    near = rows['NEAR']
    assert (near['status'], near['archive_id'], near['technical_rank']) == ('pre_breakout', archive['NEAR'], 3)
    assert (near['breakout_date'], near['days_since_breakout']) == (None, None)
    assert near['mandatory']['breakout']['breakout_level'] is None
    assert near['mandatory']['breakout']['margin_met'] is None
    assert near['mandatory']['breakout']['lookback_cleared'] is None
    assert {k: near['risk'][k] for k in ('invalidation_level', 'invalidation_basis', 'wide_risk')} == {
        'invalidation_level': None,
        'invalidation_basis': None,
        'wide_risk': None,
    }
    assert near['context']['in_120d_tier'] is False
    # its day's high and low are at its close
    closing = near['secondary']['checks'][5]['values']
    assert (closing['high'], closing['low']) == (197.0, 197.0)
    # no breakout failed or was retested on the session
    assert export['follow_up'] == []


def us_sessions_after(day: date, sessions: int) -> date:
    """The US session 'sessions' US sessions after 'day'."""
    for _ in range(sessions):
        day += timedelta(days=1)
        while not is_session(US, day):
            day += timedelta(days=1)
    return day


def test_the_next_sessions_failed_breakout_is_reported_in_each_file_whose_window_holds_it(path, container, export_dir):
    breakout_rows(path, container)
    run(container)
    brk_id = table_rows(path, 'signal')['BRK']['id']

    # the next session BRK closes at 197.9, below its exact 198 level, on its usual volume
    next_day = shift(TODAY, 1)
    container.price_data_repo.bulk_insert([replace(price_rows('BRK', next_day, n=1)[0], close=197.9)])
    started = datetime.combine(next_day, time(20), tzinfo=NEW_YORK)
    container.signal_paths.update(markets=(US, CANADA), flagged={}, started_at=started)
    container.signal_outcome_updates.update()
    file = container.screen_export.write({US: (next_day, 1)}, run_id=8, data_source='eodhd')

    entry = {
        'signal_id': f'{TODAY.isoformat()}:BRK.US',
        'archive_id': brk_id,
        'ticker': 'BRK.US',
        'breakout_date': TODAY.isoformat(),
        'breakout_level': 198.0,
        'event': 'failed_breakout',
        'event_date': next_day.isoformat(),
        'sessions_after_breakout': 1,
        'close': 197.9,
        'low': None,
        'volume': 2_000_000,
        # the 20 sessions before it: 19 on 1,000,000 shares and the breakout's 2,000,000
        'avg_volume_20d': 1_050_000.0,
    }
    follow_up = read(file)['follow_up']
    assert [list(e) for e in follow_up] == [FOLLOW_UP_KEYS]
    assert follow_up == [entry]
    with sqlite3.connect(path) as con:
        close, volume = con.execute(
            'SELECT close, volume FROM signal_price WHERE signal_id = ? AND session_index = 1', (brk_id,)
        ).fetchone()
    assert (close, volume) == (entry['close'], entry['volume'])
    # the breakout's own session's file reports nothing: the failure happened on the next one
    assert read(export_dir / export_file_name(TODAY))['follow_up'] == []

    # a later file reports it while its window holds the failure's session, as when a later run decided it late
    last_in_window = us_sessions_after(next_day, FOLLOW_UP_SESSIONS - 1)
    file = container.screen_export.write({US: (last_in_window, 1)}, run_id=9, data_source='eodhd')
    assert read(file)['follow_up'] == [entry]
    file = container.screen_export.write(
        {US: (us_sessions_after(last_in_window, 1), 1)}, run_id=10, data_source='eodhd'
    )
    assert read(file)['follow_up'] == []


def test_an_issuers_other_listing_is_unranked_and_listed_last(path, container):
    breakout_rows(path, container)
    add_issuer(path, 'Breaker', 'SHOP.TO', 'BRK', 'SHOP.TO')

    export = read(run(container))

    rows = [(r['screener_ticker'], r['issuer'], r['primary_listing'], r['technical_rank']) for r in export['rows']]
    assert rows == [('SHOP.TO', 'SHOP.TO', True, 1), ('NEAR', None, True, 2), ('BRK', 'SHOP.TO', False, None)]
    # BRK is not archived, as its issuer's other listing, but keeps its status, counted with the others
    assert export['rows'][2]['archive_id'] is None
    assert export['run']['status_counts']['confirmed'] == 2


def test_a_run_that_scored_no_session_writes_no_file(container, export_dir):
    assert container.screen_export.write({}, run_id=7, data_source='tiingo') is None
    assert not export_dir.exists()


def test_the_file_is_named_by_the_latest_session_across_the_markets(path, container, export_dir):
    breakout_rows(path, container)
    run(container)
    earlier = shift(TODAY, -1)

    file = container.screen_export.write({US: (TODAY, 3), CANADA: (earlier, 1)}, run_id=7, data_source='eodhd')

    header = read(file)['run']
    assert file.name == export_file_name(TODAY)
    assert header['sessions'] == {US: TODAY.isoformat(), CANADA: earlier.isoformat()}
    # each market's rows are those of its own session: SHOP.TO's row is TODAY's, not the earlier session's
    assert [r['screener_ticker'] for r in read(file)['rows']] == ['BRK', 'NEAR']


def test_a_market_the_run_did_not_score_has_no_session(path, container):
    breakout_rows(path, container)
    run(container)

    header = read(container.screen_export.write({US: (TODAY, 3)}, run_id=7, data_source='tiingo'))['run']

    assert header['sessions'] == {US: TODAY.isoformat(), CANADA: None}
    assert header['universe'][CANADA] == {'listed': 0, 'scored': 0}
    assert isinstance(date.fromisoformat(header['as_of']), date)


def stored_row(**values: float | int | str | None) -> dict:
    """The export row of a US breakout session's stored criteria, but for 'values', with no day's range."""
    sc = flags(**({'breakout_level': 100.0, 'breakout_margin_pct': 2.0, 'required_margin_pct': 1.5} | values))
    row = ExportRow(sc, TODAY, sector=None, subsector=None, archive_id=None, day=DayRange(None, None))
    return export_row(row, rank=1, issuer=None, primary=True, exchange=None)


@pytest.mark.parametrize(
    ('sessions', 'required', 'met'),
    [(0, 1.5, True), (0, 2.5, False), (3, 2.5, None)],
    ids=['breakout cleared its margin', 'breakout short of it', 'extended session'],
)
def test_the_margin_is_judged_met_on_the_breakout_session_alone(sessions, required, met):
    row = stored_row(sessions_since_breakout=sessions, required_margin_pct=required)

    assert row['mandatory']['breakout']['margin_met'] is met


def test_a_value_that_is_not_finite_is_exported_as_null():
    # a session whose range window holds a close of 0 has an infinite average daily range
    row = stored_row(sessions_since_breakout=0, average_daily_range_pct=float('inf'), volume_ratio=float('nan'))

    assert row['mandatory']['breakout']['average_daily_range_pct'] is None
    assert row['mandatory']['volume']['volume_ratio'] is None
    json.dumps(row, allow_nan=False)
