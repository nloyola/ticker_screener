"""
Which tickers analyze scores, against a real database: a stale or frozen ticker keeps its previous stock_criteria
row, date_created included, but not its status or breakout position, freshness is judged per market, run-refresh
scores only the markets it fetched, and a run started before a market's day is complete scores that market's previous
session.
"""

import argparse
import json
import sqlite3
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
import requests

import app.analyzer as analyzer_module
from app.analyzer import SCREENER_VERSION, Analyzer, rule_settings, trading_days_between
from app.commands.run_refresh_cmd import RUN_ID_ENV, RunRefreshCmd, run_id_from_env
from app.db import init_db
from app.markets import CANADA, US, is_session
from app.models import PriceData
from app.price_apis import DailyPrice
from app.provenance import COMMIT_ENV
from app.repositories.benchmark_price_repo import BenchmarkPriceRepo
from app.repositories.company_name_repo import CompanyNameRepo
from app.repositories.issuer_repo import IssuerRepo
from app.repositories.price_data_repo import PriceDataRepo
from app.repositories.screened_session_repo import ScreenedSessionRepo
from app.repositories.signal_outcome_repo import SignalOutcomeRepo
from app.repositories.signal_path_repo import SignalPathRepo
from app.repositories.signal_repo import SignalRepo
from app.repositories.stock_criteria_repo import StockCriteriaRepo
from app.services.sector_service import SectorAggregate, SubsectorWithTickers
from app.signal_outcome_updates import SignalOutcomeUpdates
from app.signal_paths import HORIZONS, REMOVED, SignalPaths
from app.ticker_fetcher import HISTORY_DAYS, TickerFetcher, window_start
from tests.test_issuer_repo import add_issuer


def trades(day: date) -> bool:
    """Whether both markets trade on 'day', so a row of either market's ticker dated on it is a session's."""
    return is_session(US, day) and is_session(CANADA, day)


def shift(day: date, sessions: int) -> date:
    """The day 'sessions' days both markets trade after 'day' (before it when negative)."""
    step = timedelta(days=1 if sessions > 0 else -1)
    for _ in range(abs(sessions)):
        day += step
        while not trades(day):
            day += step
    return day


def session_days(last_day: date, n: int) -> list[date]:
    """The n days both markets trade ending on last_day, oldest first."""
    return [shift(last_day, i - n + 1) for i in range(n)]


# the newest day both markets traded: the fixtures' rows are dated by sessions, never by a weekend or a holiday, which
# a run drops, and stay near the real today, which the fetch's window is judged from
TODAY = shift(date.today() + timedelta(days=1), -1)
LAST_RUN = shift(TODAY, -3)

NEW_YORK = ZoneInfo('America/New_York')
# runs started while the US and Canadian markets are open, and in the evening once both have published TODAY's rows
MARKET_HOURS = datetime.combine(TODAY, time(11), tzinfo=NEW_YORK)
EVENING = datetime.combine(TODAY, time(20), tzinfo=NEW_YORK)


def price_rows(ticker: str, last_day: date, n: int = 100) -> list[PriceData]:
    """n rising closes on the sessions ending on last_day, with a volume spike on the last day."""
    days = session_days(last_day, n)
    return [
        PriceData(
            id=0,
            subsector_id=1,
            ticker=ticker,
            date=days[i],
            close=100.0 + i,
            volume=2_000_000 if i == n - 1 else 1_000_000,
        )
        for i in range(n)
    ]


def with_breakout(rows: list[PriceData]) -> list[PriceData]:
    """
    'rows' with the last close 4 above the one before it, over 2% above the high it clears (the close before, with no
    high stored), so the last session, on doubled volume, is a breakout clearing the 1% minimum margin.
    """
    return [*rows[:-1], replace(rows[-1], close=rows[-2].close + 4)]


def criteria(path: str) -> dict[str, tuple[float, str]]:
    with sqlite3.connect(path) as con:
        rows = con.execute('SELECT ticker, price, date_created FROM stock_criteria').fetchall()
    return {ticker: (price, date_created) for ticker, price, date_created in rows}


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
    container.company_name_repo = CompanyNameRepo(container)
    container.stock_criteria_repo = StockCriteriaRepo(container)
    container.screened_session_repo = ScreenedSessionRepo(container)
    container.signal_repo = SignalRepo(container)
    container.issuer_repo = IssuerRepo(container)
    container.signal_path_repo = SignalPathRepo(container)
    container.signal_paths = SignalPaths(container)
    container.signal_outcome_repo = SignalOutcomeRepo(container)
    container.signal_outcome_updates = SignalOutcomeUpdates(container)
    container.analyzer = Analyzer(container, now=lambda: EVENING)
    return container


def score_all(analyzer: Analyzer, date_created: date) -> None:
    """Scores every stored ticker as a run did, dating its rows date_created."""
    analyzer.stock_criteria_repo.bulk_insert(
        [analyzer.analyze_ticker(t, g) for t, g in analyzer.load_price_data().groupby(level='ticker')], date_created
    )


def summary_lines(out: str) -> list[str]:
    return [line for line in out.splitlines() if line.startswith(('US:', 'Canada:'))]


def flagged_lines(out: str) -> list[str]:
    return [line for line in out.splitlines() if line.startswith('FLAGGED')]


def test_stale_ticker_keeps_its_previous_row_and_date(path, container, capsys):
    analyzer = container.analyzer

    # the last run scored both tickers from rows ending yesterday
    container.price_data_repo.bulk_insert(price_rows('AAA', shift(TODAY, -1)))
    container.price_data_repo.bulk_insert(price_rows('STALE', shift(TODAY, -1)))
    score_all(analyzer, LAST_RUN)
    assert criteria(path) == {'AAA': (199.0, LAST_RUN.isoformat()), 'STALE': (199.0, LAST_RUN.isoformat())}

    # today's fetch got a new day for AAA only; STALE's download failed and kept its rows
    container.price_data_repo.replace_for_ticker('AAA', price_rows('AAA', TODAY, n=101))
    analyzer.analyze()

    assert criteria(path) == {'AAA': (200.0, TODAY.isoformat()), 'STALE': (199.0, LAST_RUN.isoformat())}
    out = capsys.readouterr().out
    # US-only data: one line, for the US, and nothing about Canada
    assert summary_lines(out) == [
        f'US: skipped 1 stale ticker (stopped trading, or last price before {TODAY}) and 0 frozen tickers'
    ]
    assert flagged_lines(out) == [f'FLAGGED ticker=STALE reason=stale last={shift(TODAY, -1)}']


def test_us_holiday_leaves_us_tickers_scored(path, container, capsys):
    # Canada traded today and the US did not: every US ticker's last price is yesterday's
    yesterday = shift(TODAY, -1)
    container.price_data_repo.bulk_insert(price_rows('AAA', yesterday))
    container.price_data_repo.bulk_insert(price_rows('BBB', shift(yesterday, -1)))
    container.price_data_repo.bulk_insert(price_rows('SHOP.TO', TODAY))

    container.analyzer.analyze()

    # each market's rows are dated by its own session, the US's yesterday's
    assert criteria(path) == {'AAA': (199.0, yesterday.isoformat()), 'SHOP.TO': (199.0, TODAY.isoformat())}
    assert summary_lines(capsys.readouterr().out) == [
        f'US: skipped 1 stale ticker (stopped trading, or last price before {yesterday}) and 0 frozen tickers',
        f'Canada: skipped 0 stale tickers (stopped trading, or last price before {TODAY}) and 0 frozen tickers',
    ]


def test_standalone_analyze_scores_both_markets(path, container, capsys):
    container.price_data_repo.bulk_insert(price_rows('AAA', TODAY))
    container.price_data_repo.bulk_insert(price_rows('SHOP.TO', shift(TODAY, -3)))
    container.price_data_repo.bulk_insert(price_rows('ABC.V', shift(TODAY, -3)))

    container.analyzer.analyze()

    # each market's rows are dated by its own newest session
    canada = shift(TODAY, -3).isoformat()
    assert {t: d for t, (_, d) in criteria(path).items()} == {
        'AAA': TODAY.isoformat(),
        'SHOP.TO': canada,
        'ABC.V': canada,
    }
    assert summary_lines(capsys.readouterr().out) == [
        f'US: skipped 0 stale tickers (stopped trading, or last price before {TODAY}) and 0 frozen tickers',
        f'Canada: skipped 0 stale tickers (stopped trading, or last price before {shift(TODAY, -3)}) '
        'and 0 frozen tickers',
    ]


class TiingoStub:
    """A Tiingo that answers every US ticker with 101 rising closes ending today."""

    name = 'Tiingo'
    setting = 'tiingo'
    markets = frozenset({US})
    api_key = 'test-key'

    def check_quota(self, needed: int) -> None:
        pass

    def company_name(self, ticker: str) -> str | None:
        return f'{ticker} Inc.'

    def daily_prices(self, ticker: str, start: date, end: date) -> list[DailyPrice]:
        rows = price_rows(ticker, TODAY, n=101)
        return [
            DailyPrice(date=p.date, close=p.close, volume=p.volume, high=p.close, low=p.close, open=p.close)
            for p in rows
        ]


class SectorServiceStub:
    def __init__(self, tickers: list[str]) -> None:
        self.tickers = tickers

    def get_all(self) -> list[SectorAggregate]:
        subsector = SubsectorWithTickers(id=1, subsector='Software', tickers=self.tickers)
        return [SectorAggregate(id=1, sector='Tech', created_at=None, subsectors=[subsector])]


def rows_of(path: str, table: str, ticker: str) -> list[tuple]:
    with sqlite3.connect(path) as con:
        return con.execute(f'SELECT * FROM {table} WHERE ticker = ? ORDER BY id', (ticker,)).fetchall()


def test_tiingo_run_refresh_leaves_canadian_scores_and_prices_as_they_were(path, container, capsys):
    # the last run, an EODHD one, scored both markets; SHOP.TO holds rows older than the window too
    container.price_data_repo.bulk_insert(price_rows('AAA', shift(TODAY, -1)))
    container.price_data_repo.bulk_insert(price_rows('SHOP.TO', shift(TODAY, -1), n=HISTORY_DAYS + 10))
    score_all(container.analyzer, LAST_RUN)
    shop_criteria = rows_of(path, 'stock_criteria', 'SHOP.TO')
    shop_prices = rows_of(path, 'price_data', 'SHOP.TO')
    assert min(row[3] for row in shop_prices) < window_start(TODAY).isoformat()

    container.price_api_client = TiingoStub()
    container.sector_service = SectorServiceStub(['AAA', 'SHOP.TO'])
    container.ticker_fetcher = TickerFetcher(container)
    cmd = RunRefreshCmd()
    parser = argparse.ArgumentParser()
    cmd.add_arguments(parser)
    cmd.handle(parser.parse_args([]), container)

    assert criteria(path)['AAA'] == (200.0, TODAY.isoformat())
    assert rows_of(path, 'stock_criteria', 'SHOP.TO') == shop_criteria
    assert rows_of(path, 'price_data', 'SHOP.TO') == shop_prices
    assert summary_lines(capsys.readouterr().out) == [
        f'US: skipped 0 stale tickers (stopped trading, or last price before {TODAY}) and 0 frozen tickers',
        'Canada: not scored, not fetched by this run (Tiingo)',
    ]


def test_stale_ticker_already_flagged_is_counted_but_not_flagged_again(path, container, capsys):
    container.price_data_repo.bulk_insert(price_rows('AAA', TODAY))
    container.price_data_repo.bulk_insert(price_rows('GONE', shift(TODAY, -2)))
    container.price_data_repo.bulk_insert(price_rows('LATE', shift(TODAY, -1)))

    container.analyzer.analyze(already_flagged={'GONE'})

    out = capsys.readouterr().out
    assert summary_lines(out) == [
        f'US: skipped 2 stale tickers (stopped trading, or last price before {TODAY}) and 0 frozen tickers'
    ]
    assert flagged_lines(out) == [f'FLAGGED ticker=LATE reason=stale last={shift(TODAY, -1)}']


class TiingoWithDeadTickers(TiingoStub):
    """TiingoStub, except that it does not know GONE (a 404) and has no prices in the window for EMPTY."""

    def daily_prices(self, ticker: str, start: date, end: date) -> list[DailyPrice]:
        if ticker == 'GONE':
            response = requests.Response()
            response.status_code = 404
            raise requests.HTTPError('404 Client Error: Not Found', response=response)
        if ticker == 'EMPTY':
            return []
        return super().daily_prices(ticker, start, end)


def test_run_refresh_flags_each_dead_or_stale_ticker_once(path, container, capsys):
    # every ticker was scored by the last run; GONE and EMPTY keep their rows, which analyze finds stale
    yesterday = shift(TODAY, -1)
    for ticker in ('AAA', 'GONE', 'EMPTY'):
        container.price_data_repo.bulk_insert(price_rows(ticker, yesterday))

    container.price_api_client = TiingoWithDeadTickers()
    container.sector_service = SectorServiceStub(['AAA', 'GONE', 'EMPTY'])
    container.ticker_fetcher = TickerFetcher(container)
    cmd = RunRefreshCmd()
    parser = argparse.ArgumentParser()
    cmd.add_arguments(parser)
    cmd.handle(parser.parse_args([]), container)

    out = capsys.readouterr().out
    assert flagged_lines(out) == [
        f'FLAGGED ticker=GONE reason=not_found last={yesterday}',
        f'FLAGGED ticker=EMPTY reason=no_prices last={yesterday}',
    ]
    assert summary_lines(out) == [
        f'US: skipped 2 stale tickers (stopped trading, or last price before {TODAY}) and 0 frozen tickers'
    ]


def add_signal(path: str, signal_id: int, ticker: str, session_date: date) -> None:
    with sqlite3.connect(path) as con:
        con.execute(
            'INSERT INTO signal (id, ticker, market, time_zone, session_date, close, status, screener_version, '
            "rule_settings, source) VALUES (?, ?, 'US', 'America/New_York', ?, 1, 'confirmed', '1.0.0', '{}', 'live')",
            (signal_id, ticker, session_date.isoformat()),
        )


def test_run_refresh_keeps_each_signals_path_and_ends_those_it_got_no_prices_for(path, container):
    # signals of a ticker the run fetches, of two it gets no prices for, and of one removed from every subsector
    yesterday = shift(TODAY, -1)
    for ticker in ('AAA', 'GONE', 'EMPTY'):
        container.price_data_repo.bulk_insert(price_rows(ticker, yesterday))
    session = shift(TODAY, -10)
    for signal_id, ticker in enumerate(('AAA', 'GONE', 'EMPTY', 'OLD'), start=1001):
        add_signal(path, signal_id, ticker, session)

    container.price_api_client = TiingoWithDeadTickers()
    container.sector_service = SectorServiceStub(['AAA', 'GONE', 'EMPTY'])
    container.ticker_fetcher = TickerFetcher(container)
    cmd = RunRefreshCmd()
    parser = argparse.ArgumentParser()
    cmd.add_arguments(parser)
    cmd.handle(parser.parse_args([]), container)

    # AAA's path is the fetch's rows from the signal session to today, on the fetch's basis
    fetched = {p.date: p.close for p in price_rows('AAA', TODAY, n=101)}
    path_rows = [(p.session_index, p.session_date, p.close) for p in container.signal_path_repo.path(1001)]
    assert path_rows == [(i, shift(session, i), fetched[shift(session, i)]) for i in range(11)]
    # analyze found GONE and EMPTY stale too, but the fetch's reasons win
    assert container.analyzer.flagged == {'GONE': 'stale', 'EMPTY': 'stale'}
    with sqlite3.connect(path) as con:
        ended = con.execute(
            "SELECT signal_id, ended_reason, COUNT(*) FROM signal_outcome WHERE status = 'ended' "
            'GROUP BY signal_id ORDER BY signal_id'
        ).fetchall()
    assert ended == [(1002, 'not_found', 6), (1003, 'no_prices', 6), (1004, REMOVED, len(HORIZONS))]
    # and last the outcomes: AAA's path reaches session 10, so its 5- and 10-session horizons are complete
    with sqlite3.connect(path) as con:
        complete = con.execute(
            "SELECT signal_id, horizon FROM signal_outcome WHERE status = 'complete' ORDER BY signal_id, horizon"
        ).fetchall()
    assert complete == [(1001, 5), (1001, 10)]


def frozen_rows(ticker: str, flat: int, n: int = 100, volume_on: int | None = None) -> list[PriceData]:
    """
    n rows ending today of a thinly traded ticker, with no volume every 7th day, whose last 'flat' days repeat one
    close with zero volume, as Tiingo answers for a ticker that stopped trading; 'volume_on' gives that many days
    before today a volume of 500 instead. The thin trading keeps it from being a padded stop.
    """
    rows = price_rows(ticker, TODAY, n)
    rows[: n - flat - 1] = [replace(row, volume=0) if i % 7 == 3 else row for i, row in enumerate(rows[: n - flat - 1])]
    close = rows[n - flat - 1].close
    rows[n - flat :] = [replace(row, close=close, volume=0) for row in rows[n - flat :]]
    if volume_on is not None:
        rows[n - 1 - volume_on] = replace(rows[n - 1 - volume_on], volume=500)
    return rows


def first_flat_day(flat: int) -> date:
    return shift(TODAY, -(flat - 1))


def test_frozen_ticker_keeps_its_previous_row_and_is_flagged(path, container, capsys, caplog):
    analyzer = container.analyzer
    container.price_data_repo.bulk_insert(price_rows('AAA', shift(TODAY, -1)))
    container.price_data_repo.bulk_insert(price_rows('ANSS', shift(TODAY, -1)))
    score_all(analyzer, LAST_RUN)

    # today's fetch got a new day for both, but ANSS's last 10 days are one close with no volume
    container.price_data_repo.replace_for_ticker('AAA', price_rows('AAA', TODAY, n=101))
    container.price_data_repo.replace_for_ticker('ANSS', frozen_rows('ANSS', flat=10))
    with caplog.at_level('WARNING', logger='app.analyzer'):
        analyzer.analyze()

    assert criteria(path) == {'AAA': (200.0, TODAY.isoformat()), 'ANSS': (199.0, LAST_RUN.isoformat())}
    out = capsys.readouterr().out
    assert summary_lines(out) == [
        f'US: skipped 0 stale tickers (stopped trading, or last price before {TODAY}) and 1 frozen ticker'
    ]
    assert flagged_lines(out) == [f'FLAGGED ticker=ANSS reason=frozen last={first_flat_day(10)}']
    assert [r.getMessage() for r in caplog.records if r.levelname == 'WARNING'] == [
        f'ANSS not scored: its price has been frozen, with no volume, since {first_flat_day(10)}'
    ]


def test_frozen_ticker_is_flagged_from_the_first_day_of_a_longer_flat_stretch(path, container, capsys):
    container.price_data_repo.bulk_insert(price_rows('AAA', TODAY))
    container.price_data_repo.bulk_insert(frozen_rows('NKLA', flat=40))

    container.analyzer.analyze()

    assert set(criteria(path)) == {'AAA'}
    assert flagged_lines(capsys.readouterr().out) == [f'FLAGGED ticker=NKLA reason=frozen last={first_flat_day(40)}']


@pytest.mark.parametrize(
    'rows',
    [
        pytest.param(frozen_rows('FL', flat=9), id='nine flat days'),
        pytest.param(frozen_rows('FL', flat=10, volume_on=0), id='volume today'),
        pytest.param(frozen_rows('FL', flat=10, volume_on=5), id='volume within the ten days'),
    ],
)
def test_ticker_short_of_frozen_is_scored_as_before(path, container, capsys, rows):
    container.price_data_repo.bulk_insert(price_rows('AAA', TODAY))
    container.price_data_repo.bulk_insert(rows)

    container.analyzer.analyze()

    assert set(criteria(path)) == {'AAA', 'FL'}
    assert {date_created for _, date_created in criteria(path).values()} == {TODAY.isoformat()}
    out = capsys.readouterr().out
    assert summary_lines(out) == [
        f'US: skipped 0 stale tickers (stopped trading, or last price before {TODAY}) and 0 frozen tickers'
    ]
    assert flagged_lines(out) == []


def test_frozen_ticker_already_flagged_is_counted_but_not_flagged_again(path, container, capsys):
    container.price_data_repo.bulk_insert(frozen_rows('GONE', flat=10))
    container.price_data_repo.bulk_insert(frozen_rows('FL', flat=10))

    container.analyzer.analyze(already_flagged={'GONE'})

    out = capsys.readouterr().out
    assert summary_lines(out) == [
        f'US: skipped 0 stale tickers (stopped trading, or last price before {TODAY}) and 2 frozen tickers'
    ]
    assert flagged_lines(out) == [f'FLAGGED ticker=FL reason=frozen last={first_flat_day(10)}']


def test_ticker_both_stale_and_frozen_is_flagged_once_as_stale(path, container, capsys):
    container.price_data_repo.bulk_insert(price_rows('AAA', TODAY))
    yesterday = shift(TODAY, -1)
    container.price_data_repo.bulk_insert([r for r in frozen_rows('VERO', flat=11) if r.date <= yesterday])

    container.analyzer.analyze()

    out = capsys.readouterr().out
    assert summary_lines(out) == [
        f'US: skipped 1 stale ticker (stopped trading, or last price before {TODAY}) and 0 frozen tickers'
    ]
    assert flagged_lines(out) == [f'FLAGGED ticker=VERO reason=stale last={yesterday}']


def test_run_during_market_hours_scores_the_previous_session(path, container, capsys):
    # the fetch stored a row for TODAY, a session still in progress, for both markets
    container.price_data_repo.bulk_insert(price_rows('AAA', TODAY))
    container.price_data_repo.bulk_insert(price_rows('SHOP.TO', TODAY))
    container.price_data_repo.bulk_insert(price_rows('LATE', shift(TODAY, -1)))

    container.analyzer.analyze(started_at=MARKET_HOURS)

    # 198.0 is the close of the session before TODAY, which dates every row; LATE, which has no row for TODAY, is
    # current, not stale
    previous = shift(TODAY, -1)
    assert criteria(path) == {
        'AAA': (198.0, previous.isoformat()),
        'LATE': (199.0, previous.isoformat()),
        'SHOP.TO': (198.0, previous.isoformat()),
    }
    # the last complete day is the calendar day before TODAY, a Sunday when TODAY is a Monday
    yesterday = TODAY - timedelta(days=1)
    out = capsys.readouterr().out
    assert summary_lines(out) == [
        f'US: dropped 1 row dated after {yesterday}, whose candles are final only after 19:00 America/New_York '
        'that day',
        f'US: skipped 0 stale tickers (stopped trading, or last price before {previous}) and 0 frozen tickers',
        f'Canada: dropped 1 row dated after {yesterday}, whose candles are final only after 19:00 America/Toronto '
        'that day',
        f'Canada: skipped 0 stale tickers (stopped trading, or last price before {previous}) and 0 frozen tickers',
    ]
    assert flagged_lines(out) == []


def test_run_in_the_evening_scores_that_day(path, container, capsys):
    container.price_data_repo.bulk_insert(price_rows('AAA', TODAY))
    container.price_data_repo.bulk_insert(price_rows('LATE', shift(TODAY, -1)))

    container.analyzer.analyze(started_at=EVENING)

    assert criteria(path) == {'AAA': (199.0, TODAY.isoformat())}
    out = capsys.readouterr().out
    assert summary_lines(out) == [
        f'US: skipped 1 stale ticker (stopped trading, or last price before {TODAY}) and 0 frozen tickers'
    ]
    assert flagged_lines(out) == [f'FLAGGED ticker=LATE reason=stale last={shift(TODAY, -1)}']


def test_run_refresh_judges_completeness_by_when_it_started(path, container):
    # the refresh starts during market hours and its fetch ends in the evening: TODAY's row is still not final
    clock = iter([MARKET_HOURS, EVENING])
    container.analyzer = Analyzer(container, now=lambda: next(clock))
    container.price_api_client = TiingoStub()
    container.sector_service = SectorServiceStub(['AAA'])
    container.ticker_fetcher = TickerFetcher(container)
    cmd = RunRefreshCmd()
    parser = argparse.ArgumentParser()
    cmd.add_arguments(parser)
    cmd.handle(parser.parse_args([]), container)

    # TiingoStub's closes run from 100.0 to 200.0 on TODAY; 199.0 is the session before's, which dates the row
    assert criteria(path) == {'AAA': (199.0, shift(TODAY, -1).isoformat())}


SECONDARY_COLUMNS = (
    'rsi_50_70',
    'macd_above_signal',
    'macd_hist',
    'macd_hist_prev',
    'macd_days_above',
    'macd_crossover',
    'macd_hist_rising',
    'dma_20',
    'dma_20_prev_15d',
    'dma_200',
    'dma_200_prev_15d',
    'price_gt_20dma',
    'dma_20_rising',
    'price_gt_200dma',
    'dma_200_rising',
    'ema_21',
    'ema_21_prev_5d',
    'price_gt_21ema',
    'ema_21_rising',
    'ema_21w',
    'ema_21w_prev_1w',
    'price_gt_21wema',
    'ema_21w_rising',
)


def test_analyze_stores_every_secondary_indicator(path, container):
    # a year of closes rising by 1 to 467: long enough for the 200-day average and the 21-week EMA
    container.price_data_repo.bulk_insert(price_rows('AAA', TODAY, n=368))

    container.analyzer.analyze(started_at=EVENING)

    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        (row,) = con.execute(f'SELECT core_criteria_score, {", ".join(SECONDARY_COLUMNS)} FROM stock_criteria')
    expected = container.analyzer.analyze_ticker(
        'AAA', next(g for _, g in container.analyzer.load_price_data().groupby(level='ticker'))
    )
    assert {column: row[column] for column in SECONDARY_COLUMNS} == {
        column: getattr(expected, column) for column in SECONDARY_COLUMNS
    }
    assert all(row[column] is not None for column in SECONDARY_COLUMNS)
    # the n-day averages of a close rising by 1 lag it by (n - 1) / 2, the 21-day EMA by 10
    assert (row['dma_20'], row['dma_200'], row['ema_21']) == (457.5, 367.5, 457.0)
    assert row['core_criteria_score'] == 4


PRICE_ACTION_COLUMNS = (
    'close_position',
    'close_upper_quarter',
    'consolidation_range_pct',
    'consolidation_tight',
    'consolidation_vol_ratio',
    'consolidation_vol_declining',
    'dist_high_20d_pct',
    'dist_high_60d_pct',
    'near_resistance',
)


def test_analyze_stores_every_price_action_indicator(path, container):
    # closes rising by 1 to 199, each day's range 2 below its close to 1 above it
    rows = [replace(row, high=row.close + 1, low=row.close - 2) for row in price_rows('AAA', TODAY)]
    container.price_data_repo.bulk_insert(rows)

    container.analyzer.analyze(started_at=EVENING)

    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        (row,) = con.execute(f'SELECT core_criteria_score, {", ".join(PRICE_ACTION_COLUMNS)} FROM stock_criteria')
    assert {column: row[column] for column in PRICE_ACTION_COLUMNS} == pytest.approx(
        {
            'close_position': 2 / 3,  # 199 in a range from 197 to 200
            'close_upper_quarter': 0,
            'consolidation_range_pct': (199 - 177) / 199 * 100,  # highs 180..199, lows 177..196
            'consolidation_tight': 0,
            'consolidation_vol_ratio': 1.0,
            'consolidation_vol_declining': 0,
            'dist_high_20d_pct': 0.0,  # the 199 close equals the 20 days' highest high, 198 + 1
            'dist_high_60d_pct': 0.0,
            'near_resistance': 1,
        }
    )
    assert row['core_criteria_score'] == 3  # no breakout: the close only equals the prior high


def test_analyze_stores_the_120_day_tier(path, container):
    # closes rising by 1 to 249, the last 4 above the day before: above the 248 high of the 120 days before it
    container.price_data_repo.bulk_insert(with_breakout(price_rows('AAA', TODAY, n=150)))
    container.price_data_repo.bulk_insert(price_rows('BBB', TODAY, n=100))

    container.analyzer.analyze(started_at=EVENING)

    with sqlite3.connect(path) as con:
        rows = con.execute(
            'SELECT ticker, high_120d, dist_high_120d_pct, breakout_120d FROM stock_criteria ORDER BY ticker'
        ).fetchall()
    # BBB has 99 sessions before today: no 120-day high
    assert rows == [('AAA', 248.0, pytest.approx(-400 / 248), 1), ('BBB', None, None, None)]


RELATIVE_STRENGTH_COLUMNS = (
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
)


def test_analyze_stores_every_relative_strength_value(path, container):
    # three tickers of one subsector, their closes rising by 1, 2 and 0.5 a day from 100, and the US benchmark
    # rising by 1 from 200, on the same 100 days ending today
    for ticker, step in (('AAA', 1.0), ('BBB', 2.0), ('CCC', 0.5)):
        rows = [replace(row, close=100.0 + step * i) for i, row in enumerate(price_rows(ticker, TODAY))]
        container.price_data_repo.bulk_insert(rows)
    container.benchmark_price_repo.replace(
        'SPY',
        [DailyPrice(d, 200.0 + i, 1, 200.0 + i, 200.0 + i, 200.0 + i) for i, d in enumerate(session_days(TODAY, 100))],
    )

    container.analyzer.analyze(started_at=EVENING)

    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        row = con.execute(
            f'SELECT core_criteria_score, {", ".join(RELATIVE_STRENGTH_COLUMNS)} FROM stock_criteria '
            "WHERE ticker = 'BBB'"
        ).fetchone()
    # BBB closes at 298, 256 21 rows earlier and 172 63 rows earlier; the benchmark at 299, 278 and 236; AAA's
    # returns, 199/178 and 199/136, are the subsector's medians
    assert {column: row[column] for column in RELATIVE_STRENGTH_COLUMNS} == pytest.approx(
        {
            'return_1m_pct': (298 / 256 - 1) * 100,
            'return_3m_pct': (298 / 172 - 1) * 100,
            'rs_market_1m': (298 / 256 - 299 / 278) * 100,
            'rs_market_3m': (298 / 172 - 299 / 236) * 100,
            'rs_sector_1m': (298 / 256 - 199 / 178) * 100,
            'rs_sector_3m': (298 / 172 - 199 / 136) * 100,
            'outperforms_market_1m': 1,
            'outperforms_market_3m': 1,
            'outperforms_sector_1m': 1,
            'outperforms_sector_3m': 1,
        }
    )
    assert row['core_criteria_score'] == 4


def test_analyze_stores_each_tickers_status(path, container):
    # BRK breaks out on doubled volume; NEAR closes at 197, 1.01% under its 199 high, on a tight base; FALL closes
    # lower every day
    container.price_data_repo.bulk_insert(with_breakout(price_rows('BRK', TODAY)))
    container.price_data_repo.bulk_insert(near_rows())
    fall = price_rows('FALL', TODAY)
    container.price_data_repo.bulk_insert([replace(row, close=300.0 - i) for i, row in enumerate(fall)])

    container.analyzer.analyze(started_at=EVENING)

    with sqlite3.connect(path) as con:
        rows = con.execute('SELECT ticker, core_criteria_score, secondary_passed, status FROM stock_criteria')
        by_ticker = {ticker: (score, passed, status) for ticker, score, passed, status in rows}
    # BRK and FALL have no high or low stored, so their closing strength and base range are empty, and 100 rows are
    # too few for the 200-day average and the 21-week EMA. BRK passes the MACD above its signal line and, its returns
    # above NEAR's, the subsector's median, so outperforms it; NEAR's drop puts its MACD under its signal line, and
    # it passes only its tight base
    assert by_ticker == {
        'BRK': (4, 2, 'confirmed'),
        'NEAR': (2, 1, 'pre_breakout'),
        'FALL': (1, 0, None),
    }


def test_analyze_stores_the_extended_status_of_a_session_after_a_breakout(path, container):
    # EXT broke out yesterday, closing at 202 over its 198 high, and closes at 203 today on its usual volume: still
    # above the 198 level, in the breakout's run
    yesterday = with_breakout(price_rows('EXT', shift(TODAY, -1)))
    today = replace(price_rows('EXT', TODAY, n=1)[0], close=203.0, volume=1_000_000)
    container.price_data_repo.bulk_insert([*yesterday, today])

    container.analyzer.analyze(started_at=EVENING)

    assert statuses(path)['EXT'] == (203.0, TODAY.isoformat(), 'extended')
    assert signal_count(path) == 0
    # measured against the run's 198 level, one session after its breakout
    ext = table_rows(path, 'stock_criteria')['EXT']
    assert tuple(ext[c] for c in ('breakout_margin_pct', 'invalidation_distance_pct', 'sessions_since_breakout')) == (
        pytest.approx(5 / 198 * 100),
        pytest.approx(5 / 203 * 100),
        1,
    )


def test_a_breakout_below_the_liquidity_floor_is_stored_with_its_values_but_no_status(path, container):
    # BRK, THIN and THIN.TO break out alike, closing at 202 over their 198 highs on doubled volume, but THIN and
    # THIN.TO trade 1,000 shares a day where BRK trades 1,000,000: over the 20 sessions before, closing at 179 to 198,
    # a mean of 188.5, that is $188,500 a day, under the $5M floor in either market, and $188.5M
    add_energy_oil(path)
    container.price_data_repo.bulk_insert(with_breakout(price_rows('BRK', TODAY)))
    for ticker in ('THIN', 'THIN.TO'):
        rows = with_breakout(price_rows(ticker, TODAY))
        container.price_data_repo.bulk_insert([replace(row, volume=row.volume // 1000) for row in rows])

    container.analyzer.analyze(started_at=EVENING)

    stored = table_rows(path, 'stock_criteria')
    assert {t: (r['core_criteria_score'], r['status']) for t, r in stored.items()} == {
        'BRK': (4, 'confirmed'),
        'THIN': (4, None),
        'THIN.TO': (4, None),
    }
    assert {t: r['avg_dollar_volume'] for t, r in stored.items()} == pytest.approx(
        {'BRK': 188_500_000.0, 'THIN': 188_500.0, 'THIN.TO': 188_500.0}
    )
    with sqlite3.connect(path) as con:
        archived = con.execute('SELECT ticker, avg_dollar_volume FROM signal').fetchall()
    assert archived == [('BRK', pytest.approx(188_500_000.0))]


# Friday 2026-11-06 in the evening, once the US market has published that day's rows
NOV_6_EVENING = datetime(2026, 11, 6, 20, tzinfo=NEW_YORK)
LXP_LAST_TRADE = date(2026, 10, 30)


def trading_days(last_day: date, n: int) -> list[date]:
    """The n US sessions ending on last_day, oldest first."""
    days: list[date] = []
    day = last_day
    while len(days) < n:
        if is_session(US, day):
            days.append(day)
        day -= timedelta(days=1)
    return days[::-1]


def daily_rows(ticker: str, days: list[date], volumes: list[int], closes: list[float] | None = None) -> list[PriceData]:
    closes = closes or [50.0 + i * 0.1 for i in range(len(days))]
    return [
        PriceData(id=0, subsector_id=1, ticker=ticker, date=d, close=c, volume=v)
        for d, c, v in zip(days, closes, volumes, strict=True)
    ]


def lxp_rows(padded: bool) -> list[PriceData]:
    """
    LXP's rows as each price API answered on 2026-11-08: traded every day until 2026-10-30, then, from Tiingo only,
    four rows for 2026-11-03 to 2026-11-06 repeating that close at zero volume.
    """
    days = trading_days(LXP_LAST_TRADE, 100)
    closes = [55.0 + i * 0.06 for i in range(99)] + [61.18]
    rows = daily_rows('LXP', days, [1_000_000] * 100, closes)
    if padded:
        rows += daily_rows('LXP', [date(2026, 11, d) for d in (3, 4, 5, 6)], [0] * 4, [61.18] * 4)
    return rows


def thin_rows(ticker: str) -> list[PriceData]:
    """
    100 days of a thinly traded ticker like THNCF: short runs of traded days between flat zero-volume runs of 1 to
    11 days, ending in a run of 3 flat days.
    """
    days = trading_days(NOV_6_EVENING.date(), 100)
    runs = [(3, 2), (2, 11), (4, 1), (3, 5), (5, 3), (2, 4), (3, 1), (4, 8), (2, 2), (5, 3)]
    closes: list[float] = []
    volumes: list[int] = []
    close = 0.5
    while len(closes) < 97:
        for traded, flat in runs:
            for _ in range(traded):
                close += 0.01
                closes.append(close)
                volumes.append(4000)
            closes += [close] * flat
            volumes += [0] * flat
    closes = closes[:97] + [closes[96]] * 3
    volumes = volumes[:96] + [4000] + [0] * 3
    return daily_rows(ticker, days, volumes, closes)


@pytest.mark.parametrize('padded', [True, False], ids=['Tiingo', 'EODHD'])
def test_ticker_that_stopped_trading_is_stale_on_its_last_trade_on_either_api(path, container, capsys, padded):
    container.price_data_repo.bulk_insert(daily_rows('AAA', trading_days(NOV_6_EVENING.date(), 100), [1000] * 100))
    container.price_data_repo.bulk_insert(lxp_rows(padded))

    Analyzer(container, now=lambda: NOV_6_EVENING).analyze()

    assert set(criteria(path)) == {'AAA'}
    out = capsys.readouterr().out
    assert summary_lines(out) == [
        'US: skipped 1 stale ticker (stopped trading, or last price before 2026-11-06) and 0 frozen tickers'
    ]
    assert flagged_lines(out) == [f'FLAGGED ticker=LXP reason=stale last={LXP_LAST_TRADE}']


def test_stale_ticker_ending_in_a_padded_stop_is_dated_by_its_last_trade(container, capsys):
    # DINE's EODHD rows: traded until 2026-10-30, then padded at zero volume until 2026-11-05, a day short
    container.price_data_repo.bulk_insert(daily_rows('AAA', trading_days(NOV_6_EVENING.date(), 100), [1000] * 100))
    days = trading_days(date(2026, 11, 5), 100)
    container.price_data_repo.bulk_insert(daily_rows('DINE', days, [500] * 96 + [0] * 4, [24.0] * 100))

    Analyzer(container, now=lambda: NOV_6_EVENING).analyze()

    out = capsys.readouterr().out
    assert summary_lines(out) == [
        'US: skipped 1 stale ticker (stopped trading, or last price before 2026-11-06) and 0 frozen tickers'
    ]
    assert flagged_lines(out) == ['FLAGGED ticker=DINE reason=stale last=2026-10-30']


def test_padded_stop_is_logged_with_its_last_trade(container, caplog):
    container.price_data_repo.bulk_insert(lxp_rows(padded=True))

    with caplog.at_level('WARNING', logger='app.analyzer'):
        Analyzer(container, now=lambda: NOV_6_EVENING).analyze()

    assert [r.getMessage() for r in caplog.records if r.levelname == 'WARNING'] == [
        f'LXP not scored: it last traded on {LXP_LAST_TRADE}, and its price has been repeated since with no volume'
    ]


def test_thinly_traded_ticker_ending_in_a_flat_run_is_scored(path, container, capsys):
    container.price_data_repo.bulk_insert(thin_rows('THNCF'))

    Analyzer(container, now=lambda: NOV_6_EVENING).analyze()

    assert set(criteria(path)) == {'THNCF'}
    out = capsys.readouterr().out
    assert summary_lines(out) == [
        'US: skipped 0 stale tickers (stopped trading, or last price before 2026-11-06) and 0 frozen tickers'
    ]
    assert flagged_lines(out) == []


def padded_rows(ticker: str) -> list[PriceData]:
    """price_rows ending today, whose last 2 days repeat the close of the day before them with no volume."""
    rows = price_rows(ticker, TODAY)
    close = rows[-3].close
    return [*rows[:-2], *(replace(row, close=close, volume=0) for row in rows[-2:])]


def statuses(path: str) -> dict[str, tuple[float, str, str | None]]:
    with sqlite3.connect(path) as con:
        rows = con.execute('SELECT ticker, price, date_created, status FROM stock_criteria').fetchall()
    return {ticker: (price, date_created, status) for ticker, price, date_created, status in rows}


def breakout_position(path: str, ticker: str) -> tuple[float | None, float | None, int | None]:
    """The ticker's stored breakout margin, invalidation distance and sessions since its breakout."""
    row = table_rows(path, 'stock_criteria')[ticker]
    return tuple(row[c] for c in ('breakout_margin_pct', 'invalidation_distance_pct', 'sessions_since_breakout'))


@pytest.mark.parametrize(
    'rows',
    [
        pytest.param(price_rows('BRK', shift(TODAY, -1)), id='stale'),
        pytest.param(padded_rows('BRK'), id='padded stop'),
        pytest.param(frozen_rows('BRK', flat=10), id='frozen'),
    ],
)
def test_ticker_skipped_after_being_scored_the_same_day_loses_its_status(path, container, capsys, rows):
    container.price_data_repo.bulk_insert(price_rows('AAA', TODAY))
    container.price_data_repo.bulk_insert(with_breakout(price_rows('BRK', TODAY)))
    container.analyzer.analyze()
    scored = statuses(path)
    assert scored['BRK'][2] is not None
    assert breakout_position(path, 'BRK')[2] == 0

    # a later run the same day, with another price API, stored rows BRK is skipped on
    container.price_data_repo.replace_for_ticker('BRK', rows)
    container.analyzer.analyze()

    price, date_created, _ = scored['BRK']
    assert statuses(path) == {**scored, 'BRK': (price, date_created, None)}
    # the breakout position goes with the status: only a breakout or extended session has one
    assert breakout_position(path, 'BRK') == (None, None, None)
    assert len(flagged_lines(capsys.readouterr().out)) == 1


@pytest.mark.parametrize(
    'rows',
    [pytest.param(price_rows('BRK', TODAY, n=10), id='too few rows'), pytest.param([], id='no rows')],
)
def test_ticker_left_unscored_after_being_scored_loses_its_status(path, container, capsys, rows):
    container.price_data_repo.bulk_insert(price_rows('AAA', TODAY))
    container.price_data_repo.bulk_insert(with_breakout(price_rows('BRK', TODAY)))
    container.analyzer.analyze()
    scored = statuses(path)
    assert scored['BRK'][2] is not None
    assert breakout_position(path, 'BRK')[2] == 0

    # a later run stored fewer rows for BRK than scoring needs, or none at all
    container.price_data_repo.replace_for_ticker('BRK', rows)
    container.analyzer.analyze()

    price, date_created, _ = scored['BRK']
    assert statuses(path) == {**scored, 'BRK': (price, date_created, None)}
    # the breakout position goes with the status: only a breakout or extended session has one
    assert breakout_position(path, 'BRK') == (None, None, None)
    assert flagged_lines(capsys.readouterr().out) == []


# screened_session: each market's session a run scored, with the rules it scored by

SESSION_BEFORE = shift(TODAY, -1)


def sessions(path: str) -> list[tuple]:
    with sqlite3.connect(path) as con:
        return con.execute(
            'SELECT market, session_date, screener_version, rule_settings, git_commit, run_id, tickers_scored '
            'FROM screened_session ORDER BY market, session_date'
        ).fetchall()


# Each market's rows are dated by its own last session, never by the run's day.


def market_rows(ticker: str, last_day: date) -> list[PriceData]:
    """100 rising closes on the sessions of the ticker's market ending on last_day, with a volume spike on the last."""
    market = US if '.' not in ticker else CANADA
    days: list[date] = []
    day = last_day
    while len(days) < 100:
        if is_session(market, day):
            days.append(day)
        day -= timedelta(days=1)
    return [
        PriceData(
            id=0, subsector_id=1, ticker=ticker, date=d, close=100.0 + i, volume=2_000_000 if i == 99 else 1_000_000
        )
        for i, d in enumerate(reversed(days))
    ]


@pytest.mark.parametrize(
    ('started_at', 'us_session', 'canada_session'),
    [
        # the run that labelled its output 'Data as of 2026-11-15': Saturday evening in New York, Sunday in UTC
        pytest.param(
            datetime(2026, 11, 14, 20, tzinfo=NEW_YORK),
            date(2026, 11, 13),
            date(2026, 11, 13),
            id='Saturday evening, Sunday in UTC',
        ),
        pytest.param(datetime(2026, 11, 15, 12, tzinfo=NEW_YORK), date(2026, 11, 13), date(2026, 11, 13), id='Sunday'),
        # US Thanksgiving: Toronto trades, New York does not
        pytest.param(
            datetime(2026, 11, 26, 20, tzinfo=NEW_YORK), date(2026, 11, 25), date(2026, 11, 26), id='US holiday'
        ),
        # Canadian Thanksgiving: New York trades, Toronto does not
        pytest.param(
            datetime(2026, 10, 12, 20, tzinfo=NEW_YORK), date(2026, 10, 12), date(2026, 10, 9), id='Canadian holiday'
        ),
    ],
)
def test_each_market_is_dated_by_its_last_session(path, container, started_at, us_session, canada_session):
    # every ticker's rows end on its market's last session, as a fetch stores them (app/price_apis on_sessions)
    for ticker, last in (
        ('AAA', us_session),
        ('BBB', us_session),
        ('SHOP.TO', canada_session),
        ('RY.TO', canada_session),
    ):
        container.price_data_repo.bulk_insert(market_rows(ticker, last))

    container.analyzer.analyze(started_at=started_at)

    # every ticker is scored on its market's last session, and dated by it
    us, canada = us_session.isoformat(), canada_session.isoformat()
    assert criteria(path) == {
        'AAA': (199.0, us),
        'BBB': (199.0, us),
        'SHOP.TO': (199.0, canada),
        'RY.TO': (199.0, canada),
    }
    assert [(market, session_date) for market, session_date, *_ in sessions(path)] == [(CANADA, canada), (US, us)]


def run_with_rules(container, monkeypatch, version: str, run_id: int) -> None:
    """Runs analyze as a screener at 'version', whose snapshot names it, so each row shows which run wrote it."""
    monkeypatch.setattr(analyzer_module, 'SCREENER_VERSION', version)
    monkeypatch.setattr(analyzer_module, 'rule_settings', lambda: {'version': version})
    container.analyzer.analyze(run_id=run_id)


def test_reruns_replace_a_session_and_the_next_session_adds_its_own(path, container, monkeypatch):
    monkeypatch.setenv(COMMIT_ENV, 'abc1234')
    container.price_data_repo.bulk_insert(price_rows('AAA', SESSION_BEFORE))
    container.price_data_repo.bulk_insert(price_rows('BBB', SESSION_BEFORE))
    container.price_data_repo.bulk_insert(price_rows('SHOP.TO', SESSION_BEFORE))

    run_with_rules(container, monkeypatch, '1.0.0', run_id=1)
    run_with_rules(container, monkeypatch, '1.1.0', run_id=2)
    # the next session: every ticker gets TODAY's row
    for ticker in ('AAA', 'BBB', 'SHOP.TO'):
        container.price_data_repo.bulk_insert(price_rows(ticker, TODAY))
    run_with_rules(container, monkeypatch, '2.0.0', run_id=3)

    v11, v20 = json.dumps({'version': '1.1.0'}), json.dumps({'version': '2.0.0'})
    assert sessions(path) == [
        (CANADA, SESSION_BEFORE.isoformat(), '1.1.0', v11, 'abc1234', 2, 1),
        (CANADA, TODAY.isoformat(), '2.0.0', v20, 'abc1234', 3, 1),
        (US, SESSION_BEFORE.isoformat(), '1.1.0', v11, 'abc1234', 2, 2),
        (US, TODAY.isoformat(), '2.0.0', v20, 'abc1234', 3, 2),
    ]


def test_session_carries_the_current_rules(path, container, monkeypatch):
    monkeypatch.delenv(COMMIT_ENV, raising=False)
    monkeypatch.setattr('app.provenance._git', lambda *args: {'rev-parse': 'f' * 40, 'status': ''}[args[0]])
    container.price_data_repo.bulk_insert(price_rows('AAA', TODAY))

    container.analyzer.analyze()

    [(market, day, version, settings, commit, run_id, scored)] = sessions(path)
    assert (market, day, version, commit, run_id, scored) == (
        US,
        TODAY.isoformat(),
        SCREENER_VERSION,
        'f' * 40,
        None,
        1,
    )
    assert json.loads(settings) == rule_settings()


def test_market_with_no_ticker_scored_records_no_session(path, container):
    # 10 rows are too few to score
    container.price_data_repo.bulk_insert(price_rows('AAA', TODAY, n=10))

    container.analyzer.analyze()

    assert sessions(path) == []


def test_run_refresh_records_its_run_id_for_the_markets_it_fetched(path, container, monkeypatch):
    monkeypatch.setenv(RUN_ID_ENV, '42')
    monkeypatch.setenv(COMMIT_ENV, 'unknown')
    container.price_data_repo.bulk_insert(price_rows('SHOP.TO', TODAY))
    container.price_api_client = TiingoStub()
    container.sector_service = SectorServiceStub(['AAA', 'SHOP.TO'])
    container.ticker_fetcher = TickerFetcher(container)
    cmd = RunRefreshCmd()
    parser = argparse.ArgumentParser()
    cmd.add_arguments(parser)
    cmd.handle(parser.parse_args([]), container)

    # Tiingo covers only the US; an image built without a revision records no commit
    assert [(m, d, c, r, n) for m, d, _, _, c, r, n in sessions(path)] == [(US, TODAY.isoformat(), None, 42, 1)]


@pytest.mark.parametrize(
    ('value', 'expected'), [('', None), ('7', 7), (' 7 ', 7), ('0', None), ('-3', None), ('x', None)]
)
def test_run_id_from_env(monkeypatch, value, expected):
    monkeypatch.setenv(RUN_ID_ENV, value)
    assert run_id_from_env() == expected


def test_run_id_unset(monkeypatch):
    monkeypatch.delenv(RUN_ID_ENV, raising=False)
    assert run_id_from_env() is None


def add_energy_oil(path: str) -> None:
    """Sector Energy (id 1) with subsector Oil (id 1), the subsector every price_rows row is in."""
    with sqlite3.connect(path) as con:
        con.execute("INSERT INTO sector (id, sector) VALUES (1, 'Energy')")
        con.execute("INSERT INTO subsector (id, sector_id, subsector, tickers) VALUES (1, 1, 'Oil', 'BRK,NEAR,FALL')")


def table_rows(path: str, table: str) -> dict[str, dict]:
    """Each row of 'table' by ticker, as a dict of its columns."""
    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        return {row['ticker']: dict(row) for row in con.execute(f'SELECT * FROM {table}')}


def signal_count(path: str, kind: str | None = None) -> int:
    """How many records the archive holds, of 'kind' alone when given."""
    with sqlite3.connect(path) as con:
        return con.execute('SELECT count(*) FROM signal WHERE ? IS NULL OR kind = ?', (kind, kind)).fetchone()[0]


def kinds(path: str) -> dict[str, str]:
    """Each archived ticker's kind of record, for a session archiving one record per ticker."""
    return {ticker: row['kind'] for ticker, row in table_rows(path, 'signal').items()}


def near_rows() -> list[PriceData]:
    """
    NEAR's 101 sessions to TODAY: closes rising by 1 to 199, then 197 on its usual volume, each day's high and low at
    its close. The 20 days before TODAY span (199 - 180) / 199, a tight base, and 197 is 2 under the 199 high.
    """
    near = [replace(row, high=row.close, low=row.close) for row in price_rows('NEAR', TODAY, n=101)]
    return [*near[:-1], replace(near[-1], close=197.0, high=197.0, low=197.0, volume=1_000_000)]


def breakout_rows(path: str, container) -> None:
    """
    BRK and SHOP.TO break out on doubled volume on TODAY, BRK closing at 202.456 and SHOP.TO at 202, clearing their
    198 highs by more than the 1% margin; NEAR (near_rows) is on the pre-breakout watchlist; FALL closes
    lower every day.
    """
    add_energy_oil(path)
    brk = price_rows('BRK', TODAY)
    container.price_data_repo.bulk_insert([*brk[:-1], replace(brk[-1], close=202.456)])
    container.price_data_repo.bulk_insert(with_breakout(price_rows('SHOP.TO', TODAY)))
    container.price_data_repo.bulk_insert(near_rows())
    fall = price_rows('FALL', TODAY)
    container.price_data_repo.bulk_insert([replace(row, close=300.0 - i) for i, row in enumerate(fall)])


def test_run_archives_each_confirmed_breakout_with_its_stored_values(path, container, monkeypatch):
    monkeypatch.setenv(COMMIT_ENV, 'abc1234')
    breakout_rows(path, container)

    container.analyzer.analyze(run_id=7)

    criteria_rows = table_rows(path, 'stock_criteria')
    signals = {t: r for t, r in table_rows(path, 'signal').items() if r['kind'] == 'signal'}
    assert {t: r['status'] for t, r in criteria_rows.items()} == {
        'BRK': 'confirmed',
        'SHOP.TO': 'confirmed',
        'NEAR': 'pre_breakout',
        'FALL': None,
    }
    # NEAR, on the watchlist, is archived beside the signals
    assert kinds(path) == {'BRK': 'signal', 'SHOP.TO': 'signal', 'NEAR': 'watchlist'}
    archived = set(criteria_rows['BRK']) - {'id', 'subsector_id', 'date_created'}
    for ticker, signal in signals.items():
        assert {c: signal[c] for c in archived} == {c: criteria_rows[ticker][c] for c in archived}, ticker
    settings = json.dumps(rule_settings(), sort_keys=True)
    common = {
        'session_date': TODAY.isoformat(),
        'sector': 'Energy',
        'subsector': 'Oil',
        'screener_version': SCREENER_VERSION,
        'rule_settings': settings,
        'git_commit': 'abc1234',
        'run_id': 7,
        'source': 'live',
        'previous_signal_id': None,
        'trading_days_since_previous': None,
    }
    own = [*common, 'market', 'exchange', 'time_zone', 'close', 'price']
    assert {c: signals['BRK'][c] for c in own} == {
        **common,
        'market': US,
        'exchange': None,
        'time_zone': 'America/New_York',
        'close': 202.456,
        'price': 202.46,
    }
    assert {c: signals['SHOP.TO'][c] for c in own} == {
        **common,
        'market': CANADA,
        'exchange': 'TSX',
        'time_zone': 'America/Toronto',
        'close': 202.0,
        'price': 202.0,
    }
    # a signal is a breakout session: BRK's unrounded close clears its 198 level by 4.456, its margin taken over the
    # level and its invalidation distance over the close
    position = ('breakout_margin_pct', 'invalidation_distance_pct', 'sessions_since_breakout')
    assert tuple(signals['BRK'][c] for c in position) == (
        pytest.approx(4.456 / 198 * 100),
        pytest.approx(4.456 / 202.456 * 100),
        0,
    )
    # the session's signals carry the rules its screened session records
    assert {(m, d, v, s, c, r) for m, d, v, s, c, r, _ in sessions(path)} == {
        (US, TODAY.isoformat(), SCREENER_VERSION, settings, 'abc1234', 7),
        (CANADA, TODAY.isoformat(), SCREENER_VERSION, settings, 'abc1234', 7),
    }


def every_kind_rows(path: str, container) -> None:
    """
    On TODAY: BRK breaks out (with_breakout), THREE rises to a new high on its usual volume, meeting three of the four
    core conditions outside any run, NEAR is on the watchlist (near_rows), and EXT, which broke out on SESSION_BEFORE
    on doubled volume clearing its 198 high by 2%, closes 1 higher on its usual volume, an Extended day of that run.
    """
    add_energy_oil(path)
    container.price_data_repo.bulk_insert(with_breakout(price_rows('BRK', TODAY)))
    three = price_rows('THREE', TODAY)
    container.price_data_repo.bulk_insert([*three[:-1], replace(three[-1], volume=1_000_000)])
    container.price_data_repo.bulk_insert(near_rows())
    ext = price_rows('EXT', TODAY)
    container.price_data_repo.bulk_insert(
        [
            *ext[:-2],
            replace(ext[-2], close=ext[-3].close + 4, volume=2_000_000),
            replace(ext[-1], close=ext[-3].close + 5, volume=1_000_000),
        ]
    )


def _all_records(path: str) -> list[dict]:
    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        return [dict(row) for row in con.execute('SELECT * FROM signal ORDER BY id')]


def test_a_session_archives_one_record_of_each_kind_and_a_rerun_adds_nothing(path, container):
    every_kind_rows(path, container)
    # the run during TODAY's session scores SESSION_BEFORE, EXT's breakout
    container.analyzer.analyze(started_at=MARKET_HOURS, run_id=1)
    [breakout] = [r for r in _all_records(path) if r['ticker'] == 'EXT' and r['kind'] == 'signal']
    assert breakout['session_date'] == SESSION_BEFORE.isoformat()

    container.analyzer.analyze(started_at=EVENING, run_id=2)

    today = {r['ticker']: r for r in _all_records(path) if r['session_date'] == TODAY.isoformat()}
    assert {t: (r['kind'], r['status']) for t, r in today.items()} == {
        'BRK': ('signal', 'confirmed'),
        'THREE': ('near_miss', None),
        'NEAR': ('watchlist', 'pre_breakout'),
        'EXT': ('continuation', 'extended'),
    }
    # THREE failed only the volume confirmation, read from its flags
    three = today['THREE']
    assert (three['core_criteria_score'], three['breakout_confirmed'], three['breakout_20d']) == (3, 0, 1)
    # the continuation links to its run's breakout, one session before it; only a signal links to a previous one
    assert (today['EXT']['breakout_id'], today['EXT']['sessions_since_breakout']) == (breakout['id'], 1)
    assert {t: r['breakout_id'] for t, r in today.items() if t != 'EXT'} == {'BRK': None, 'THREE': None, 'NEAR': None}
    assert {r['previous_signal_id'] for r in today.values()} == {None}
    assert {(r['run_id'], r['source']) for r in today.values()} == {(2, 'live')}
    archived = _all_records(path)

    container.analyzer.analyze(started_at=EVENING, run_id=3)

    assert _all_records(path) == archived


def test_an_extended_day_whose_breakout_is_not_archived_is_left_out(path, container):
    every_kind_rows(path, container)
    # no run scored SESSION_BEFORE, so EXT's breakout was never archived

    container.analyzer.analyze(started_at=EVENING, run_id=2)

    assert kinds(path) == {'BRK': 'signal', 'THREE': 'near_miss', 'NEAR': 'watchlist'}
    assert table_rows(path, 'stock_criteria')['EXT']['status'] == 'extended'


def test_a_new_high_on_doubled_volume_short_of_the_margin_is_a_near_miss(path, container):
    add_energy_oil(path)
    # 199 over the 198 high on doubled volume: all four conditions, but 0.5% over the level, short of the 1% margin
    container.price_data_repo.bulk_insert(price_rows('SHORT', TODAY))

    container.analyzer.analyze(started_at=EVENING)

    [record] = _all_records(path)
    assert (record['ticker'], record['kind'], record['status'], record['core_criteria_met']) == (
        'SHORT',
        'near_miss',
        None,
        1,
    )


def test_run_archives_only_the_primary_listing_of_an_issuer_and_keeps_the_others_status(path, container):
    breakout_rows(path, container)
    add_issuer(path, 'Breakers', 'BRK', 'BRK', 'SHOP.TO')

    container.analyzer.analyze(run_id=7)

    assert kinds(path) == {'BRK': 'signal', 'NEAR': 'watchlist'}
    # the other listing is scored and keeps its status, shown under the primary by the dashboard
    assert table_rows(path, 'stock_criteria')['SHOP.TO']['status'] == 'confirmed'


def test_run_archives_no_listing_whose_primary_is_another(path, container):
    breakout_rows(path, container)
    # NEAR, on the watchlist, is the primary: neither breakout of its issuer is archived, of any kind
    add_issuer(path, 'Breakers', 'NEAR', 'BRK', 'NEAR', 'SHOP.TO')

    container.analyzer.analyze(run_id=7)

    assert kinds(path) == {'NEAR': 'watchlist'}


def test_second_run_on_the_same_session_archives_nothing(path, container):
    breakout_rows(path, container)
    container.analyzer.analyze(run_id=1)
    first = table_rows(path, 'signal')

    container.analyzer.analyze(run_id=2)

    assert signal_count(path) == 3
    assert table_rows(path, 'signal') == first


def test_same_session_under_new_rules_is_archived_again(path, container, monkeypatch):
    breakout_rows(path, container)
    run_with_rules(container, monkeypatch, '1.0.0', run_id=1)
    run_with_rules(container, monkeypatch, '1.1.0', run_id=2)

    with sqlite3.connect(path) as con:
        rows = con.execute(
            'SELECT ticker, screener_version, run_id, previous_signal_id FROM signal '
            "WHERE kind = 'signal' ORDER BY ticker, screener_version"
        ).fetchall()
    # a session never links to a signal of the same session
    assert rows == [
        ('BRK', '1.0.0', 1, None),
        ('BRK', '1.1.0', 2, None),
        ('SHOP.TO', '1.0.0', 1, None),
        ('SHOP.TO', '1.1.0', 2, None),
    ]


def test_repeated_signal_links_to_the_previous_one(path, container):
    add_energy_oil(path)
    earlier = shift(TODAY, -3)
    first = with_breakout(price_rows('BRK', earlier))
    container.price_data_repo.bulk_insert(first)
    container.analyzer.analyze()
    # two closes below the 198 level it broke end its run, and BRK breaks out again on TODAY, clearing the 202 high
    # by 1.5% on twice the average volume
    later = [
        replace(row, id=0, close=close, volume=volume)
        for row, close, volume in zip(
            price_rows('BRK', TODAY, n=3), (197.0, 196.0, 205.0), (1_000_000, 1_000_000, 2_000_000), strict=True
        )
    ]
    container.price_data_repo.replace_for_ticker('BRK', [*first, *later])

    container.analyzer.analyze()

    with sqlite3.connect(path) as con:
        signals = con.execute(
            'SELECT id, session_date, previous_signal_id, trading_days_since_previous FROM signal ORDER BY id'
        ).fetchall()
    first_id = signals[0][0]
    assert [row[1:] for row in signals] == [(earlier.isoformat(), None, None), (TODAY.isoformat(), first_id, 3)]


@pytest.mark.parametrize(
    ('previous', 'expected'),
    [
        # Friday 2026-11-06 to Tuesday 2026-11-10, both stored: Monday and Tuesday
        pytest.param(date(2026, 11, 6), 2, id='within the stored rows'),
        # Friday 2026-10-30, before the oldest stored row, Thursday 2026-11-05: the weekdays Monday to Wednesday
        # before it, then Thursday to Tuesday's four stored rows
        pytest.param(date(2026, 10, 30), 7, id='before the stored rows'),
    ],
)
def test_trading_days_between(previous, expected):
    stored = pd.Index([date(2026, 11, 5), date(2026, 11, 6), date(2026, 11, 9), date(2026, 11, 10)])
    assert trading_days_between(stored, previous, date(2026, 11, 10)) == expected


# Scoring a past session: score_as_of reads only the rows dated on or before it.
AS_OF = shift(TODAY, -40)
LATER_DAYS = 40


def history_rows(ticker: str, step: float, last_day: date) -> list[PriceData]:
    """
    A year of closes rising by 'step' a day from 100 to last_day, each day's range 2 below its close to half a step
    above it, so every close breaks out above the day before's high, with a volume spike on last_day, whose close
    jumps 2% to clear that high by more than the 1% margin: a breakout, and long enough for every indicator, the
    200-day average and 21-week EMA included.
    """
    rows = []
    days = price_rows(ticker, last_day, n=368)
    for i, row in enumerate(days):
        close = (100.0 + step * i) * (1.02 if i == len(days) - 1 else 1.0)
        rows.append(replace(row, close=close, high=close + step / 2, low=close - 2))
    return rows


def benchmark_prices(last_day: date) -> list[DailyPrice]:
    return [
        DailyPrice(d, 200.0 + i, 1, 201.0 + i, 198.0 + i, 200.0 + i) for i, d in enumerate(session_days(last_day, 368))
    ]


def extreme_rows(ticker: str, first_day: date, n: int) -> list[PriceData]:
    """n days from first_day whose closes, ranges and volumes swing wildly, to move any indicator that reads them."""
    return [
        PriceData(
            id=0,
            subsector_id=1,
            ticker=ticker,
            date=shift(first_day, i),
            close=1e6 if i % 2 else 0.01,
            volume=10**12 if i % 2 else 0,
            high=2e6,
            low=0.001,
        )
        for i in range(n)
    ]


def extreme_benchmark(first_day: date, n: int) -> list[DailyPrice]:
    return [DailyPrice(shift(first_day, i), 1e9 if i % 2 else 1e-3, 1, 2e9, 1e-4, 1.0) for i in range(n)]


SUBSECTOR_STEPS = {'AAA': 1.0, 'BBB': 2.0, 'CCC': 0.5}


def store_history(container, last_day: date) -> None:
    """Three US tickers of one subsector and one Canadian, with both markets' benchmarks, ending on last_day."""
    for ticker, step in (*SUBSECTOR_STEPS.items(), ('SHOP.TO', 1.0)):
        container.price_data_repo.bulk_insert(history_rows(ticker, step, last_day))
    container.benchmark_price_repo.replace('SPY', benchmark_prices(last_day))
    container.benchmark_price_repo.replace('XIU.TO', benchmark_prices(last_day))


def scores(scored) -> dict[str, tuple]:
    return {s.criteria.ticker: (s.criteria, s.market, s.session_date, s.close, list(s.dates)) for s in scored}


def test_score_as_of_ignores_every_row_after_the_session(container):
    store_history(container, AS_OF)
    before = container.analyzer.score_as_of(AS_OF)

    # every indicator is filled in, the subsector comparison and the status included, so each one is checked
    aaa = next(s.criteria for s in before if s.criteria.ticker == 'AAA')
    assert [f for f, v in vars(aaa).items() if v is None] == []
    assert aaa.status == 'high_conviction'
    assert {s.criteria.ticker for s in before} == {*SUBSECTOR_STEPS, 'SHOP.TO'}

    # wild rows after the session for every ticker and benchmark, and a ticker of the same subsector with no row on
    # the session, stale as of it, whose later rows would move the subsector's medians if it were scored
    first_later = shift(AS_OF, 1)
    for ticker in (*SUBSECTOR_STEPS, 'SHOP.TO'):
        container.price_data_repo.bulk_insert(extreme_rows(ticker, first_later, LATER_DAYS))
    container.price_data_repo.bulk_insert(history_rows('GAP', 3.0, shift(AS_OF, -1)))
    container.price_data_repo.bulk_insert(extreme_rows('GAP', first_later, LATER_DAYS))
    for symbol in ('SPY', 'XIU.TO'):
        container.benchmark_price_repo.replace(
            symbol, [*benchmark_prices(AS_OF), *extreme_benchmark(first_later, LATER_DAYS)]
        )

    after = container.analyzer.score_as_of(AS_OF)

    assert scores(after) == scores(before)
    # the later rows are there to be read: the newest session scores differently
    latest = container.analyzer.score_as_of(shift(first_later, LATER_DAYS - 1))
    assert scores(latest).keys() == {*scores(before), 'GAP'}
    assert all(scores(latest)[t][0] != scores(before)[t][0] for t in scores(before))


def test_score_as_of_the_latest_session_matches_what_a_run_stores(path, container):
    store_history(container, TODAY)
    container.analyzer.analyze(started_at=EVENING)

    stored = table_rows(path, 'stock_criteria')
    scored = container.analyzer.score_as_of(TODAY)

    assert {s.criteria.ticker for s in scored} == set(stored) == {*SUBSECTOR_STEPS, 'SHOP.TO'}
    for s in scored:
        row = stored[s.criteria.ticker]
        # the breakout phase and whether the session is in a run are not stored; the status they decide is
        expected = {f: row[f] for f in vars(s.criteria) if f not in ('id', 'breakout_phase', 'in_breakout_run')}
        assert {f: getattr(s.criteria, f) for f in expected} == expected, s.criteria.ticker


def test_score_as_of_leaves_out_a_market_closed_that_session(container):
    # Canada traded on AS_OF and the US did not
    container.price_data_repo.bulk_insert(history_rows('AAA', 1.0, shift(AS_OF, -1)))
    container.price_data_repo.bulk_insert(history_rows('SHOP.TO', 1.0, AS_OF))

    scored = container.analyzer.score_as_of(AS_OF)

    assert [(s.criteria.ticker, s.session_date) for s in scored] == [('SHOP.TO', AS_OF)]
