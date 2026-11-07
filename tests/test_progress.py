"""The PROGRESS and FLAGGED lines are parsed by the stock_analyzer admin pages: these tests pin their format."""

import re
from datetime import date
from types import SimpleNamespace

import pytest

from app.analyzer import Analyzer
from app.progress import format_flagged, format_progress, report_flagged
from app.services.sector_service import SectorAggregate, SubsectorWithTickers
from app.ticker_fetcher import TickerFetcher

# the grammar documented in README.md § Progress lines
LINE = re.compile(r'PROGRESS stage=[a-z]+( [a-z_]+=[A-Za-z0-9_./-]+)*')


def progress_lines(out: str) -> list[str]:
    lines = [line for line in out.splitlines() if line.startswith('PROGRESS')]
    for line in lines:
        assert LINE.fullmatch(line), line
    return lines


def test_format_progress_puts_stage_first_and_keeps_field_order():
    assert format_progress('fetch', subsector='3/11') == 'PROGRESS stage=fetch subsector=3/11'
    assert format_progress('x', b=1, a='y') == 'PROGRESS stage=x b=1 a=y'


@pytest.mark.parametrize('stage, fields', [('fe tch', {}), ('fetch', {'subsector': 'a b'}), ('fetch', {'k': 'a=b'})])
def test_format_progress_rejects_tokens_that_break_the_grammar(stage, fields):
    with pytest.raises(ValueError):
        format_progress(stage, **fields)


# the grammar documented in README.md § Flagged tickers
FLAGGED_LINE = re.compile(
    r'FLAGGED ticker=[A-Za-z0-9_./-]+ reason=(not_found|no_prices|stale|frozen)( last=\d{4}-\d{2}-\d{2})?'
)


@pytest.mark.parametrize(
    'ticker, reason, last, line',
    [
        ('BLD', 'no_prices', date(2026, 7, 2), 'FLAGGED ticker=BLD reason=no_prices last=2026-07-02'),
        ('SMOP', 'not_found', None, 'FLAGGED ticker=SMOP reason=not_found'),
        ('RCI-B.TO', 'stale', date(2026, 11, 5), 'FLAGGED ticker=RCI-B.TO reason=stale last=2026-11-05'),
        ('ANSS', 'frozen', date(2026, 7, 17), 'FLAGGED ticker=ANSS reason=frozen last=2026-07-17'),
    ],
)
def test_format_flagged(ticker, reason, last, line):
    assert format_flagged(ticker, reason, last) == line
    assert FLAGGED_LINE.fullmatch(line)


@pytest.mark.parametrize('ticker, reason', [('BRK B', 'stale'), ('A=B', 'stale'), ('CVX', 'gone')])
def test_format_flagged_rejects_what_breaks_the_grammar(ticker, reason):
    with pytest.raises(ValueError):
        format_flagged(ticker, reason, None)


def test_report_flagged_logs_a_ticker_the_grammar_cannot_carry(capsys, caplog):
    report_flagged('BRK B', 'stale', None)

    assert capsys.readouterr().out == ''
    assert 'BRK B' in caplog.text


class FakeSectorService:
    def __init__(self, sectors: list[SectorAggregate]) -> None:
        self.sectors = sectors

    def get_all(self) -> list[SectorAggregate]:
        return self.sectors

    def subsector_for_ticker(self, ticker: str):
        return SimpleNamespace(id=1)


def make_fetcher(sectors: list[SectorAggregate]) -> TickerFetcher:
    container = SimpleNamespace(
        price_api_client=SimpleNamespace(name='Tiingo', markets=frozenset({'US'}), check_quota=lambda needed: None),
        sector_service=FakeSectorService(sectors),
        price_data_repo=SimpleNamespace(replace_for_ticker=lambda ticker, rows: None, latest_for_ticker=lambda t: None),
    )
    fetcher = TickerFetcher(container)
    fetcher._fetch_ticker_data = lambda subsector_id, ticker: []
    return fetcher


def sector(sector_id: int, name: str, subsector_count: int) -> SectorAggregate:
    subsectors = [
        SubsectorWithTickers(id=sector_id * 10 + i, subsector=f'sub{i}', tickers=['T']) for i in range(subsector_count)
    ]
    return SectorAggregate(id=sector_id, sector=name, created_at=None, subsectors=subsectors)


def test_fetch_counts_subsectors_across_sectors(capsys):
    make_fetcher([sector(1, 'Tech', 2), sector(2, 'Energy', 1)]).fetch()

    assert progress_lines(capsys.readouterr().out) == [
        'PROGRESS stage=fetch subsector=0/3',
        'PROGRESS stage=fetch subsector=1/3',
        'PROGRESS stage=fetch subsector=2/3',
        'PROGRESS stage=fetch subsector=3/3',
    ]


def test_fetch_total_is_the_filtered_sectors_subsectors(capsys):
    make_fetcher([sector(1, 'Tech', 2), sector(2, 'Energy', 1)]).fetch(sector_filter='energy')

    assert progress_lines(capsys.readouterr().out) == [
        'PROGRESS stage=fetch subsector=0/1',
        'PROGRESS stage=fetch subsector=1/1',
    ]


def test_fetch_one_ticker_is_one_subsector(capsys):
    make_fetcher([sector(1, 'Tech', 2)]).fetch(ticker_filter='T')

    assert progress_lines(capsys.readouterr().out) == [
        'PROGRESS stage=fetch subsector=0/1',
        'PROGRESS stage=fetch subsector=1/1',
    ]


def test_fetch_with_no_subsectors_reports_zero_total(capsys):
    make_fetcher([]).fetch()

    assert progress_lines(capsys.readouterr().out) == ['PROGRESS stage=fetch subsector=0/0']


def test_analyze_reports_start_and_done(capsys):
    container = SimpleNamespace(stock_criteria_repo=None, price_data_repo=SimpleNamespace(all=lambda: []))
    Analyzer(container).analyze()

    assert progress_lines(capsys.readouterr().out) == [
        'PROGRESS stage=analyze status=start',
        'PROGRESS stage=analyze status=done',
    ]
