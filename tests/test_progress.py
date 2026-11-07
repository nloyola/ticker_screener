"""The PROGRESS lines are parsed by the stock_analyzer admin pages: these tests pin their format."""

import re
from types import SimpleNamespace

import pytest

from app.analyzer import Analyzer
from app.progress import format_progress
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
        price_data_repo=SimpleNamespace(replace_for_ticker=lambda ticker, rows: None),
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
