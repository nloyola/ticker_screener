"""
Which tickers analyze scores, against a real database: a stale or frozen ticker keeps its previous stock_criteria
row, date_created included, but not its status, freshness is judged per market, run-refresh scores only the markets
it fetched, and a run started before a market's day is complete scores that market's previous session.
"""

import argparse
import sqlite3
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
import requests

from app.analyzer import Analyzer
from app.commands.run_refresh_cmd import RunRefreshCmd
from app.db import init_db
from app.markets import US
from app.models import PriceData
from app.price_apis import DailyPrice
from app.repositories.benchmark_price_repo import BenchmarkPriceRepo
from app.repositories.company_name_repo import CompanyNameRepo
from app.repositories.price_data_repo import PriceDataRepo
from app.repositories.stock_criteria_repo import StockCriteriaRepo
from app.services.sector_service import SectorAggregate, SubsectorWithTickers
from app.ticker_fetcher import HISTORY_DAYS, TickerFetcher, window_start

TODAY = date.today()
LAST_RUN = TODAY - timedelta(days=3)

NEW_YORK = ZoneInfo('America/New_York')
# runs started while the US and Canadian markets are open, and in the evening once both have published TODAY's rows
MARKET_HOURS = datetime.combine(TODAY, time(11), tzinfo=NEW_YORK)
EVENING = datetime.combine(TODAY, time(20), tzinfo=NEW_YORK)


def price_rows(ticker: str, last_day: date, n: int = 100) -> list[PriceData]:
    """n rising closes ending on last_day, with a volume spike on the last day."""
    return [
        PriceData(
            id=0,
            subsector_id=1,
            ticker=ticker,
            date=last_day - timedelta(days=n - 1 - i),
            close=100.0 + i,
            volume=2000 if i == n - 1 else 1000,
        )
        for i in range(n)
    ]


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
    container.analyzer = Analyzer(container, now=lambda: EVENING)
    return container


def score_all(analyzer: Analyzer, date_created: date) -> None:
    """Scores every stored ticker as a run on date_created did."""
    analyzer.save_to_repository(
        [analyzer.analyze_ticker(t, g) for t, g in analyzer.load_price_data().groupby(level='ticker')], date_created
    )


def summary_lines(out: str) -> list[str]:
    return [line for line in out.splitlines() if line.startswith(('US:', 'Canada:'))]


def flagged_lines(out: str) -> list[str]:
    return [line for line in out.splitlines() if line.startswith('FLAGGED')]


def test_stale_ticker_keeps_its_previous_row_and_date(path, container, capsys):
    analyzer = container.analyzer

    # the last run scored both tickers from rows ending yesterday
    container.price_data_repo.bulk_insert(price_rows('AAA', TODAY - timedelta(days=1)))
    container.price_data_repo.bulk_insert(price_rows('STALE', TODAY - timedelta(days=1)))
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
    assert flagged_lines(out) == [f'FLAGGED ticker=STALE reason=stale last={TODAY - timedelta(days=1)}']


def test_us_holiday_leaves_us_tickers_scored(path, container, capsys):
    # Canada traded today and the US did not: every US ticker's last price is yesterday's
    yesterday = TODAY - timedelta(days=1)
    container.price_data_repo.bulk_insert(price_rows('AAA', yesterday))
    container.price_data_repo.bulk_insert(price_rows('BBB', yesterday - timedelta(days=1)))
    container.price_data_repo.bulk_insert(price_rows('SHOP.TO', TODAY))

    container.analyzer.analyze()

    assert criteria(path) == {'AAA': (199.0, TODAY.isoformat()), 'SHOP.TO': (199.0, TODAY.isoformat())}
    assert summary_lines(capsys.readouterr().out) == [
        f'US: skipped 1 stale ticker (stopped trading, or last price before {yesterday}) and 0 frozen tickers',
        f'Canada: skipped 0 stale tickers (stopped trading, or last price before {TODAY}) and 0 frozen tickers',
    ]


def test_standalone_analyze_scores_both_markets(path, container, capsys):
    container.price_data_repo.bulk_insert(price_rows('AAA', TODAY))
    container.price_data_repo.bulk_insert(price_rows('SHOP.TO', TODAY - timedelta(days=3)))
    container.price_data_repo.bulk_insert(price_rows('ABC.V', TODAY - timedelta(days=3)))

    container.analyzer.analyze()

    assert set(criteria(path)) == {'AAA', 'SHOP.TO', 'ABC.V'}
    assert {date_created for _, date_created in criteria(path).values()} == {TODAY.isoformat()}
    assert summary_lines(capsys.readouterr().out) == [
        f'US: skipped 0 stale tickers (stopped trading, or last price before {TODAY}) and 0 frozen tickers',
        f'Canada: skipped 0 stale tickers (stopped trading, or last price before {TODAY - timedelta(days=3)}) '
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
        return [DailyPrice(date=p.date, close=p.close, volume=p.volume, high=p.close, low=p.close) for p in rows]


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
    container.price_data_repo.bulk_insert(price_rows('AAA', TODAY - timedelta(days=1)))
    container.price_data_repo.bulk_insert(price_rows('SHOP.TO', TODAY - timedelta(days=1), n=HISTORY_DAYS + 10))
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
    container.price_data_repo.bulk_insert(price_rows('GONE', TODAY - timedelta(days=2)))
    container.price_data_repo.bulk_insert(price_rows('LATE', TODAY - timedelta(days=1)))

    container.analyzer.analyze(already_flagged={'GONE'})

    out = capsys.readouterr().out
    assert summary_lines(out) == [
        f'US: skipped 2 stale tickers (stopped trading, or last price before {TODAY}) and 0 frozen tickers'
    ]
    assert flagged_lines(out) == [f'FLAGGED ticker=LATE reason=stale last={TODAY - timedelta(days=1)}']


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
    yesterday = TODAY - timedelta(days=1)
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
    return TODAY - timedelta(days=flat - 1)


def test_frozen_ticker_keeps_its_previous_row_and_is_flagged(path, container, capsys, caplog):
    analyzer = container.analyzer
    container.price_data_repo.bulk_insert(price_rows('AAA', TODAY - timedelta(days=1)))
    container.price_data_repo.bulk_insert(price_rows('ANSS', TODAY - timedelta(days=1)))
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
    yesterday = TODAY - timedelta(days=1)
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
    container.price_data_repo.bulk_insert(price_rows('LATE', TODAY - timedelta(days=1)))

    container.analyzer.analyze(started_at=MARKET_HOURS)

    # 198.0 is the close of the day before TODAY; LATE, which has no row for TODAY, is current, not stale
    assert criteria(path) == {
        'AAA': (198.0, TODAY.isoformat()),
        'LATE': (199.0, TODAY.isoformat()),
        'SHOP.TO': (198.0, TODAY.isoformat()),
    }
    yesterday = TODAY - timedelta(days=1)
    out = capsys.readouterr().out
    assert summary_lines(out) == [
        f'US: dropped 1 row dated after {yesterday}, whose candles are final only after 19:00 America/New_York '
        'that day',
        f'US: skipped 0 stale tickers (stopped trading, or last price before {yesterday}) and 0 frozen tickers',
        f'Canada: dropped 1 row dated after {yesterday}, whose candles are final only after 19:00 America/Toronto '
        'that day',
        f'Canada: skipped 0 stale tickers (stopped trading, or last price before {yesterday}) and 0 frozen tickers',
    ]
    assert flagged_lines(out) == []


def test_run_in_the_evening_scores_that_day(path, container, capsys):
    container.price_data_repo.bulk_insert(price_rows('AAA', TODAY))
    container.price_data_repo.bulk_insert(price_rows('LATE', TODAY - timedelta(days=1)))

    container.analyzer.analyze(started_at=EVENING)

    assert criteria(path) == {'AAA': (199.0, TODAY.isoformat())}
    out = capsys.readouterr().out
    assert summary_lines(out) == [
        f'US: skipped 1 stale ticker (stopped trading, or last price before {TODAY}) and 0 frozen tickers'
    ]
    assert flagged_lines(out) == [f'FLAGGED ticker=LATE reason=stale last={TODAY - timedelta(days=1)}']


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

    # TiingoStub's closes run from 100.0 to 200.0 on TODAY; 199.0 is the day before's
    assert criteria(path) == {'AAA': (199.0, TODAY.isoformat())}


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
        'SPY', [DailyPrice(TODAY - timedelta(days=99 - i), 200.0 + i, 1, 200.0 + i, 200.0 + i) for i in range(100)]
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
    # BRK breaks out on doubled volume; NEAR closes at 197, 1.5% under its 200 high; FALL closes lower every day
    container.price_data_repo.bulk_insert(price_rows('BRK', TODAY))
    near = price_rows('NEAR', TODAY, n=101)
    container.price_data_repo.bulk_insert([*near[:-1], replace(near[-1], close=197.0, volume=1000)])
    fall = price_rows('FALL', TODAY)
    container.price_data_repo.bulk_insert([replace(row, close=300.0 - i) for i, row in enumerate(fall)])

    container.analyzer.analyze(started_at=EVENING)

    with sqlite3.connect(path) as con:
        rows = con.execute('SELECT ticker, core_criteria_score, secondary_passed, status FROM stock_criteria')
        by_ticker = {ticker: (score, passed, status) for ticker, score, passed, status in rows}
    # no high or low stored: closing strength and the base's range are empty, and 100 rows are too few for the
    # 200-day average and the 21-week EMA. BRK passes the MACD above its signal line and, its returns above NEAR's,
    # the subsector's median, so outperforms it; NEAR's drop puts its MACD under its signal line
    assert by_ticker == {
        'BRK': (4, 2, 'confirmed'),
        'NEAR': (2, 0, 'pre_breakout'),
        'FALL': (1, 0, None),
    }


# Friday 2026-11-06 in the evening, once the US market has published that day's rows
NOV_6_EVENING = datetime(2026, 11, 6, 20, tzinfo=NEW_YORK)
LXP_LAST_TRADE = date(2026, 10, 30)


def trading_days(last_day: date, n: int) -> list[date]:
    """The n weekdays ending on last_day, oldest first."""
    days: list[date] = []
    day = last_day
    while len(days) < n:
        if day.weekday() < 5:
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


@pytest.mark.parametrize(
    'rows',
    [
        pytest.param(price_rows('BRK', TODAY - timedelta(days=1)), id='stale'),
        pytest.param(padded_rows('BRK'), id='padded stop'),
        pytest.param(frozen_rows('BRK', flat=10), id='frozen'),
    ],
)
def test_ticker_skipped_after_being_scored_the_same_day_loses_its_status(path, container, capsys, rows):
    container.price_data_repo.bulk_insert(price_rows('AAA', TODAY))
    container.price_data_repo.bulk_insert(price_rows('BRK', TODAY))
    container.analyzer.analyze()
    scored = statuses(path)
    assert scored['BRK'][2] is not None

    # a later run the same day, with another price API, stored rows BRK is skipped on
    container.price_data_repo.replace_for_ticker('BRK', rows)
    container.analyzer.analyze()

    price, date_created, _ = scored['BRK']
    assert statuses(path) == {**scored, 'BRK': (price, date_created, None)}
    assert len(flagged_lines(capsys.readouterr().out)) == 1


@pytest.mark.parametrize(
    'rows',
    [pytest.param(price_rows('BRK', TODAY, n=10), id='too few rows'), pytest.param([], id='no rows')],
)
def test_ticker_left_unscored_after_being_scored_loses_its_status(path, container, capsys, rows):
    container.price_data_repo.bulk_insert(price_rows('AAA', TODAY))
    container.price_data_repo.bulk_insert(price_rows('BRK', TODAY))
    container.analyzer.analyze()
    scored = statuses(path)
    assert scored['BRK'][2] is not None

    # a later run stored fewer rows for BRK than scoring needs, or none at all
    container.price_data_repo.replace_for_ticker('BRK', rows)
    container.analyzer.analyze()

    price, date_created, _ = scored['BRK']
    assert statuses(path) == {**scored, 'BRK': (price, date_created, None)}
    assert flagged_lines(capsys.readouterr().out) == []
