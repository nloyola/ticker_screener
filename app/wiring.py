# wiring.py
from app.analyzer import Analyzer
from app.config import DB_NAME, EODHD, EODHD_API_KEY, PRICE_API, TIINGO, TIINGO_API_KEY, Config
from app.container import Container, Required, Singleton
from app.db import SQLite
from app.price_apis import PriceApi
from app.price_apis.eodhd import EodhdPrices
from app.price_apis.tiingo import TiingoPrices
from app.repositories.metadata_repo import MetadataRepo
from app.repositories.price_data_repo import PriceDataRepo
from app.repositories.sector_repo import SectorRepo
from app.repositories.stock_criteria_repo import StockCriteriaRepo
from app.repositories.subsector_repo import SubsectorRepo
from app.repositories.ticker_repo import TickerRepo
from app.services.sector_service import SectorService
from app.ticker_fetcher import TickerFetcher
from app.ticker_sync_tiingo import TickerSync


def build_container(config: Config) -> Container:
    c = Container()

    # every provider is lazy: only what a command reads from the container is built
    c.register('tiingo_api_key', Required(TIINGO_API_KEY, config.tiingo_api_key))
    c.register('eodhd_api_key', Required(EODHD_API_KEY, config.eodhd_api_key))
    c.register('price_api', Required(PRICE_API, config.price_api))
    c.register('db_path', Required(DB_NAME, config.db_name))

    # Core infra
    c.register('db', Singleton(lambda: SQLite(c.db_path)))  # one SQLite wrapper

    c.register('metadata_repo', Singleton(lambda: MetadataRepo(c)))
    c.register('sector_repo', Singleton(lambda: SectorRepo(c)))
    c.register('subsector_repo', Singleton(lambda: SubsectorRepo(c)))
    c.register('price_data_repo', Singleton(lambda: PriceDataRepo(c)))
    c.register('stock_criteria_repo', Singleton(lambda: StockCriteriaRepo(c)))
    c.register('ticker_repo', Singleton(lambda: TickerRepo(c)))

    c.register('price_api_client', Singleton(lambda: price_api_client(c)))
    c.register('sector_service', Singleton(lambda: SectorService(c)))
    c.register('ticker_fetcher', Singleton(lambda: TickerFetcher(c)))
    c.register('ticker_sync', Singleton(lambda: TickerSync(c)))
    c.register('analyzer', Singleton(lambda: Analyzer(c)))
    return c


def price_api_client(c: Container) -> PriceApi:
    """The client of the price API that PRICE_API names: the one place that picks between them."""
    if c.price_api == TIINGO:
        return TiingoPrices(c.tiingo_api_key)
    if c.price_api == EODHD:
        return EodhdPrices(c.eodhd_api_key)
    raise ValueError(f'unknown price API: {c.price_api}')
