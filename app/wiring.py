# wiring.py
from app.config import Config
from app.container import Container, Singleton
from app.db import SQLite
from app.repositories.price_data_repo import PriceDataRepo
from app.repositories.sector_repo import SectorRepo
from app.repositories.stock_criteria_repo import StockCriteriaRepo
from app.repositories.subsector_repo import SubsectorRepo
from app.services.sector_service import SectorService

# from app.price_data_repo import PriceDataRepo  # if you have one


def build_container() -> Container:
    c = Container()

    # Config values can be simple factories/singletons too
    c.register('db_path', Singleton(lambda: Config.get_db_name()))

    # Core infra
    c.register('db', Singleton(lambda: SQLite(c.db_path)))  # one SQLite wrapper

    c.register('sector_repo', Singleton(lambda: SectorRepo(c.db_path)))
    c.register('subsector_repo', Singleton(lambda: SubsectorRepo(c.db_path)))
    c.register('price_data_repo', Singleton(lambda: PriceDataRepo(c.db_path)))
    c.register('stock_criteria_repo', Singleton(lambda: StockCriteriaRepo(c.db_path)))

    c.register('sector_service', Singleton(lambda: SectorService(c)))
    return c
