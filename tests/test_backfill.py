"""
Seeding the signal archive (app/backfill.py): every session the stored prices can fully score is replayed and its
signals archived as backfill, once, with values that match the prices on the session.
"""

import argparse
import os
import sqlite3
from collections import Counter
from datetime import date, datetime, time, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from app.analyzer import Analyzer
from app.backfill import REPLAY_MIN_SESSIONS, REPLAY_MIN_WEEKS, Backfill, replayable_sessions
from app.commands.backfill_cmd import BackfillCmd
from app.db import init_db
from app.markets import CANADA, US
from app.models import PriceData
from app.repositories.benchmark_price_repo import BenchmarkPriceRepo
from app.repositories.issuer_repo import IssuerRepo
from app.repositories.price_data_repo import PriceDataRepo
from app.repositories.screened_session_repo import ScreenedSessionRepo
from app.repositories.security_state_repo import SecurityStateRepo
from app.repositories.signal_repo import SignalRepo
from app.repositories.stock_criteria_repo import StockCriteriaRepo
from app.screener_lock import screener_lock

NEW_YORK = ZoneInfo('America/New_York')
# a Monday, so every fifth weekday from it starts a new calendar week
FIRST = date(2025, 11, 17)
SESSIONS = 260


def weekdays(first: date, n: int) -> list[date]:
    days: list[date] = []
    day = first
    while len(days) < n:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


DAYS = weekdays(FIRST, SESSIONS)
LAST = DAYS[-1]
# both markets have published LAST's candles
EVENING = datetime.combine(LAST, time(20), tzinfo=NEW_YORK)
# the 51st calendar week starts on the 251st weekday, the first session with enough history for the 21-week EMA's
# rising check (the 200-day SMA's needs only 215)
FIRST_REPLAYABLE = 250
# every fifth day trades three times the usual volume, 3,000,000 against an average of 1,400,000 over the 20 days
# before it
SPIKE_EVERY = 5


def test_replay_thresholds():
    assert (REPLAY_MIN_SESSIONS, REPLAY_MIN_WEEKS) == (215, 51)


def test_replayable_sessions_start_once_the_window_holds_51_weeks():
    dates = {US: pd.Index(DAYS), CANADA: pd.Index(DAYS[:-1])}

    sessions = replayable_sessions(dates, EVENING)

    assert list(sessions) == DAYS[FIRST_REPLAYABLE:]
    assert all(markets == (US, CANADA) for markets in list(sessions.values())[:-1])
    # Canada has no row on LAST
    assert sessions[LAST] == (US,)


def test_replayable_sessions_need_215_sessions_however_many_weeks():
    # one session a week for 60 weeks
    dates = {US: pd.Index([FIRST + timedelta(weeks=i) for i in range(60)])}
    assert replayable_sessions(dates, EVENING) == {}


def test_replayable_sessions_leave_out_a_session_not_yet_final():
    market_hours = datetime.combine(LAST, time(11), tzinfo=NEW_YORK)
    assert list(replayable_sessions({US: pd.Index(DAYS)}, market_hours)) == DAYS[FIRST_REPLAYABLE:-1]


# each session's close above or below the trend, 100 + its row, by its place in its cycle of SPIKE_EVERY sessions
CYCLE_OFFSETS = (4.0, -3.0, -4.0, -3.0, -2.0)


def rows(ticker: str, days: list[date] = DAYS) -> list[PriceData]:
    """
    Closes rising by 1 a day from 100 in cycles of SPIKE_EVERY sessions (CYCLE_OFFSETS), each day's range from 2 below
    its close to half a point above it, on 1,000,000 shares a day and 3,000,000 on the first of each cycle. That spike
    day closes 4.5 above the previous spike's high, clearing it by more than the margin, and the next 2 days close below
    that high, ending the breakout's run: a confirmed breakout on every spike day with enough history.
    """
    closes = [100.0 + i + CYCLE_OFFSETS[i % SPIKE_EVERY] for i in range(len(days))]
    return [
        PriceData(
            id=0,
            subsector_id=1,
            ticker=ticker,
            date=day,
            close=close,
            high=close + 0.5,
            low=close - 2,
            volume=3_000_000 if i % SPIKE_EVERY == 0 else 1_000_000,
        )
        for i, (day, close) in enumerate(zip(days, closes, strict=True))
    ]


SPIKE_SESSIONS = [DAYS[i] for i in range(FIRST_REPLAYABLE, SESSIONS) if i % SPIKE_EVERY == 0]


@pytest.fixture
def path(tmp_path) -> str:
    path = str(tmp_path / 'stock_data.db')
    init_db(path)
    return path


@pytest.fixture
def container(path) -> SimpleNamespace:
    container = SimpleNamespace(db_path=path)
    container.price_data_repo = PriceDataRepo(container)
    container.benchmark_price_repo = BenchmarkPriceRepo(container)
    container.stock_criteria_repo = StockCriteriaRepo(container)
    container.screened_session_repo = ScreenedSessionRepo(container)
    container.signal_repo = SignalRepo(container)
    container.issuer_repo = IssuerRepo(container)
    container.security_state_repo = SecurityStateRepo(container)
    container.analyzer = Analyzer(container, now=lambda: EVENING)
    container.backfill = Backfill(container, now=lambda: EVENING)
    return container


def signals(path: str, kind: str | None = 'signal') -> list[dict]:
    """The archived records of 'kind', a signal unless told otherwise, of every kind when it is None."""
    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        return [
            dict(row)
            for row in con.execute(
                'SELECT * FROM signal WHERE ? IS NULL OR kind = ? ORDER BY ticker, session_date, id', (kind, kind)
            )
        ]


def test_backfill_archives_each_replayable_sessions_signals_once(path, container, capsys):
    container.price_data_repo.bulk_insert(rows('AAA'))
    container.price_data_repo.bulk_insert(rows('SHOP.TO'))

    result = container.backfill.run()

    archived = signals(path)
    assert [(s['ticker'], s['session_date']) for s in archived] == [
        (ticker, day.isoformat()) for ticker in ('AAA', 'SHOP.TO') for day in SPIKE_SESSIONS
    ]
    assert {(s['source'], s['run_id']) for s in archived} == {('backfill', None)}
    assert {s['status'] for s in archived} <= {'confirmed', 'high_conviction'}
    assert (result.sessions, result.signals, result.relinked) == (SESSIONS - FIRST_REPLAYABLE, len(archived), 0)
    records = signals(path, kind=None)
    assert result.records == dict(Counter(s['kind'] for s in records))
    out = capsys.readouterr().out
    assert f'backfill: replayed {SESSIONS - FIRST_REPLAYABLE} sessions, archived {len(archived)} new signals, ' in out
    # no line per ticker scored
    assert 'generating criteria' not in out

    # oldest first, each repeated signal links to the one before it
    aaa = [s for s in archived if s['ticker'] == 'AAA']
    assert [(s['previous_signal_id'], s['trading_days_since_previous']) for s in aaa] == [
        (None, None),
        *((earlier['id'], SPIKE_EVERY) for earlier in aaa[:-1]),
    ]

    again = container.backfill.run()

    assert (again.sessions, again.signals, again.relinked) == (SESSIONS - FIRST_REPLAYABLE, 0, 0)
    assert sum(again.records.values()) == 0
    assert signals(path, kind=None) == records


def test_replayed_signal_matches_the_prices_on_its_session(path, container):
    container.price_data_repo.bulk_insert(rows('AAA'))
    container.backfill.run()

    session = SPIKE_SESSIONS[0]
    signal = next(s for s in signals(path) if s['session_date'] == session.isoformat())
    prices = rows('AAA')
    i = DAYS.index(session)
    before = prices[i - 20 : i]
    assert signal['close'] == prices[i].close
    assert signal['high_20d'] == max(p.high for p in before)
    assert signal['high_60d'] == max(p.high for p in prices[i - 60 : i])
    high_120d = max(p.high for p in prices[i - 120 : i])
    assert signal['high_120d'] == high_120d
    assert signal['dist_high_120d_pct'] == pytest.approx((high_120d - prices[i].close) / high_120d * 100)
    assert signal['breakout_120d'] == 1
    assert signal['volume'] == prices[i].volume == 3_000_000
    assert signal['avg_vol_20d'] == sum(p.volume for p in before) / 20 == 1_400_000
    assert signal['volume_ratio'] == pytest.approx(3_000_000 / 1_400_000)
    assert (signal['breakout_20d'], signal['breakout_60d'], signal['breakout_confirmed']) == (1, 1, 1)


def test_backfill_relinks_a_live_signal_archived_before_it(path, container):
    # a run on the last spike session archived a live signal, linked to nothing
    last_spike = DAYS.index(SPIKE_SESSIONS[-1])
    container.price_data_repo.bulk_insert(rows('AAA', DAYS[: last_spike + 1]))
    container.analyzer.analyze(started_at=EVENING)
    container.price_data_repo.replace_for_ticker('AAA', rows('AAA'))
    [live] = signals(path)
    assert (live['source'], live['previous_signal_id']) == ('live', None)

    result = container.backfill.run()

    archived = signals(path)
    # the backfill leaves the live signal's session to it, and links it to the backfill signal before it
    assert [(s['session_date'], s['source']) for s in archived] == [
        *((day.isoformat(), 'backfill') for day in SPIKE_SESSIONS[:-1]),
        (SPIKE_SESSIONS[-1].isoformat(), 'live'),
    ]
    assert (archived[-1]['previous_signal_id'], archived[-1]['trading_days_since_previous']) == (
        archived[-2]['id'],
        SPIKE_EVERY,
    )
    assert result.relinked == 1
    assert container.backfill.run().relinked == 0


def test_backfill_leaves_another_versions_signals_unlinked(path, container, monkeypatch):
    # a run on the last spike session archived a live signal under an earlier screener version
    last_spike = DAYS.index(SPIKE_SESSIONS[-1])
    container.price_data_repo.bulk_insert(rows('AAA', DAYS[: last_spike + 1]))
    with monkeypatch.context() as patched:
        patched.setattr('app.analyzer.SCREENER_VERSION', '0.9.0')
        container.analyzer.analyze(started_at=EVENING)
    container.price_data_repo.replace_for_ticker('AAA', rows('AAA'))

    result = container.backfill.run()

    by_version: dict[str, list[dict]] = {}
    for s in signals(path):
        by_version.setdefault(s['screener_version'], []).append(s)
    [old] = by_version.pop('0.9.0')
    [current] = by_version.values()
    # the earlier version's signal is not relinked to the backfill signal before it, and the backfill archives its
    # own signal on that session, linked only to its own version's signals
    assert (old['source'], old['previous_signal_id'], old['trading_days_since_previous']) == ('live', None, None)
    assert result.relinked == 0
    assert [s['session_date'] for s in current] == [day.isoformat() for day in SPIKE_SESSIONS]
    assert [s['previous_signal_id'] for s in current] == [None, *(s['id'] for s in current[:-1])]


def test_backfill_with_no_prices_replays_nothing(container, capsys):
    assert container.backfill.run().sessions == 0
    assert 'nothing to replay' in capsys.readouterr().out


def test_command_exits_tempfail_while_a_refresh_holds_the_lock(path, container, capsys):
    with screener_lock(path), pytest.raises(SystemExit) as exit_info:
        BackfillCmd().handle(argparse.Namespace(), container)
    assert exit_info.value.code == os.EX_TEMPFAIL
    assert 'a refresh holds the lock' in capsys.readouterr().err
