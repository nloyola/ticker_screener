# wiring.py
from app.config import Config
from app.container import Container, Singleton, Value
from app.db import SQLite
from app.repositories.metadata_repo import MetadataRepo
from app.repositories.price_data_repo import PriceDataRepo
from app.repositories.sector_repo import SectorRepo
from app.repositories.stock_criteria_repo import StockCriteriaRepo
from app.repositories.subsector_repo import SubsectorRepo
from app.repositories.ticker_repo import TickerRepo
from app.services.sector_service import SectorService
from app.ticker_fetcher import TickerFetcher
from app.ticker_sync_tiingo import TickerSync


def build_container() -> Container:
    c = Container()

    c.register('tiingo_api_key', Value(Config.get_tiingo_api_key()))

    # Config values can be simple factories/singletons too
    c.register('db_path', Value(Config.get_db_name()))

    # Core infra
    c.register('db', Singleton(lambda: SQLite(c.db_path)))  # one SQLite wrapper

    c.register('metadata_repo', Singleton(lambda: MetadataRepo(c)))
    c.register('sector_repo', Singleton(lambda: SectorRepo(c)))
    c.register('subsector_repo', Singleton(lambda: SubsectorRepo(c)))
    c.register('price_data_repo', Singleton(lambda: PriceDataRepo(c)))
    c.register('stock_criteria_repo', Singleton(lambda: StockCriteriaRepo(c)))
    c.register('ticker_repo', Singleton(lambda: TickerRepo(c)))

    c.register('sector_service', Singleton(lambda: SectorService(c)))
    c.register('ticker_fetcher', Singleton(lambda: TickerFetcher(c)))
    c.register('ticker_sync', Singleton(lambda: TickerSync(c)))
    return c
