"""ticker-fetcher --ticker finds the ticker's subsector by parsing each subsector's ticker list."""

import argparse
from types import SimpleNamespace

import pytest

from app.commands.ticker_fetcher_cmd import TickerFetcherCmd
from app.db import init_db
from app.models import Sector, Subsector
from app.repositories.sector_repo import SectorRepo
from app.repositories.subsector_repo import SubsectorRepo
from app.services.sector_service import SectorService
from app.ticker_fetcher import FetchStats


@pytest.fixture
def service(tmp_path):
    path = str(tmp_path / 'stock_data.db')
    init_db(path)
    container = SimpleNamespace(db_path=path)
    container.sector_repo = SectorRepo(container)
    container.subsector_repo = SubsectorRepo(container)
    sector_id = container.sector_repo.insert(Sector(id=0, sector='Mixed'))
    for name, tickers in [('Lookalike', 'BRKXB,AAPL'), ('Holding', 'BRK_B'), ('Spaced', 'XOM, CVX , SU')]:
        container.subsector_repo.insert(Subsector(id=0, sector_id=sector_id, subsector=name, tickers=tickers))
    return SectorService(container)


@pytest.mark.parametrize(
    ('ticker', 'subsector'),
    [
        ('brk_b', 'Holding'),  # not Lookalike: '_' is matched literally
        ('BRKXB', 'Lookalike'),
        ('CVX', 'Spaced'),  # in a list written with spaces
        ('su', 'Spaced'),  # last in the list, lower-case
        (' xom ', 'Spaced'),
    ],
)
def test_finds_the_subsector_listing_the_ticker(service, ticker, subsector):
    assert service.subsector_for_ticker(ticker).subsector == subsector


@pytest.mark.parametrize('ticker', ['BRK', 'BRK_', 'XO', 'XOM,CVX', ''])
def test_finds_nothing_for_a_ticker_no_list_holds(service, ticker):
    assert service.subsector_for_ticker(ticker) is None


def test_searches_in_the_order_a_full_fetch_walks(service):
    """Sectors by name, then subsectors by id: the first listing a full fetch meets is the one --ticker finds."""
    mixed_id = next(iter(service._sectors.all())).id
    alpha_id = service._sectors.insert(Sector(id=0, sector='Alpha'))  # sorts before 'Mixed' though inserted later
    service._subsectors.insert(Subsector(id=0, sector_id=mixed_id, subsector='Later', tickers='CVX'))
    assert service.subsector_for_ticker('CVX').subsector == 'Spaced'  # same sector: lower id first

    service._subsectors.insert(Subsector(id=0, sector_id=alpha_id, subsector='First by name', tickers='CVX'))
    assert service.subsector_for_ticker('CVX').subsector == 'First by name'
    walk = [sub.subsector for sector in service.get_all() for sub in sector.subsectors if 'CVX' in sub.tickers]
    assert walk[0] == 'First by name'


@pytest.mark.parametrize(('argv', 'expected'), [(['--ticker', ' brk_b '], 'BRK_B'), ([], None)])
def test_ticker_fetcher_normalizes_the_ticker(argv, expected):
    calls = []

    def fetch(sector_filter=None, ticker_filter=None, strict=False):
        calls.append(ticker_filter)
        return FetchStats()

    parser = argparse.ArgumentParser()
    cmd = TickerFetcherCmd()
    cmd.add_arguments(parser)
    cmd.handle(parser.parse_args(argv), SimpleNamespace(ticker_fetcher=SimpleNamespace(fetch=fetch)))
    assert calls == [expected]


def test_ticker_fetcher_rejects_a_blank_ticker(capsys):
    parser = argparse.ArgumentParser()
    TickerFetcherCmd().add_arguments(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(['--ticker', ' '])
    assert 'must not be blank' in capsys.readouterr().err
