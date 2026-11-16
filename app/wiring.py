# wiring.py
from app.analyzer import Analyzer
from app.backfill import Backfill
from app.breakoutiq.run import Analysis
from app.config import DB_NAME, EODHD, EODHD_API_KEY, HISTORY_DATABASE_FILE, PRICE_API, TIINGO, TIINGO_API_KEY, Config
from app.container import Container, Required, Singleton
from app.db import SQLite
from app.history_backfill import HistoryBackfill
from app.history_fetch import HistoryFetch
from app.history_paths import HistoryPaths
from app.price_apis import PriceApi
from app.price_apis.eodhd import EodhdPrices
from app.price_apis.tiingo import TiingoPrices
from app.repositories.analysis_data_repo import AnalysisDataRepo
from app.repositories.analysis_repo import AnalysisRepo
from app.repositories.backfill_run_repo import BackfillRunRepo
from app.repositories.benchmark_price_repo import BenchmarkPriceRepo
from app.repositories.company_name_repo import CompanyNameRepo
from app.repositories.history_repo import HistoryRepo
from app.repositories.metadata_repo import MetadataRepo
from app.repositories.price_data_repo import PriceDataRepo
from app.repositories.screened_session_repo import ScreenedSessionRepo
from app.repositories.sector_repo import SectorRepo
from app.repositories.signal_outcome_repo import SignalOutcomeRepo
from app.repositories.signal_path_repo import SignalPathRepo
from app.repositories.signal_repo import SignalRepo
from app.repositories.stock_criteria_repo import StockCriteriaRepo
from app.repositories.subsector_repo import SubsectorRepo
from app.repositories.ticker_repo import TickerRepo
from app.services.sector_service import SectorService
from app.signal_outcome_updates import SignalOutcomeUpdates
from app.signal_paths import SignalPaths
from app.ticker_fetcher import TickerFetcher
from app.ticker_sync_tiingo import TickerSync


def build_container(config: Config) -> Container:
    c = Container()

    # every provider is lazy: only what a command reads from the container is built
    c.register('tiingo_api_key', Required(TIINGO_API_KEY, config.tiingo_api_key))
    c.register('eodhd_api_key', Required(EODHD_API_KEY, config.eodhd_api_key))
    c.register('price_api', Required(PRICE_API, config.price_api))
    c.register('db_path', Required(DB_NAME, config.db_name))
    c.register('history_db_path', Required(HISTORY_DATABASE_FILE, config.history_database_file))

    # Core infra
    c.register('db', Singleton(lambda: SQLite(c.db_path)))  # one SQLite wrapper

    c.register('metadata_repo', Singleton(lambda: MetadataRepo(c)))
    c.register('sector_repo', Singleton(lambda: SectorRepo(c)))
    c.register('subsector_repo', Singleton(lambda: SubsectorRepo(c)))
    c.register('price_data_repo', Singleton(lambda: PriceDataRepo(c)))
    c.register('benchmark_price_repo', Singleton(lambda: BenchmarkPriceRepo(c)))
    c.register('stock_criteria_repo', Singleton(lambda: StockCriteriaRepo(c)))
    c.register('ticker_repo', Singleton(lambda: TickerRepo(c)))
    c.register('company_name_repo', Singleton(lambda: CompanyNameRepo(c)))
    c.register('screened_session_repo', Singleton(lambda: ScreenedSessionRepo(c)))
    c.register('signal_repo', Singleton(lambda: SignalRepo(c)))
    c.register('signal_path_repo', Singleton(lambda: SignalPathRepo(c)))
    c.register('signal_outcome_repo', Singleton(lambda: SignalOutcomeRepo(c)))
    c.register('history_repo', Singleton(lambda: HistoryRepo(c)))
    c.register('backfill_run_repo', Singleton(lambda: BackfillRunRepo(c)))
    c.register('analysis_repo', Singleton(lambda: AnalysisRepo(c)))
    c.register('analysis_data_repo', Singleton(lambda: AnalysisDataRepo(c)))

    c.register('price_api_client', Singleton(lambda: price_api_client(c)))
    c.register('sector_service', Singleton(lambda: SectorService(c)))
    c.register('ticker_fetcher', Singleton(lambda: TickerFetcher(c)))
    c.register('ticker_sync', Singleton(lambda: TickerSync(c)))
    c.register('analyzer', Singleton(lambda: Analyzer(c)))
    c.register('backfill', Singleton(lambda: Backfill(c)))
    c.register('signal_paths', Singleton(lambda: SignalPaths(c)))
    c.register('signal_outcome_updates', Singleton(lambda: SignalOutcomeUpdates(c)))
    c.register('history_fetch', Singleton(lambda: HistoryFetch(c)))
    c.register('history_paths', Singleton(lambda: HistoryPaths(c)))
    c.register('history_backfill', Singleton(lambda: HistoryBackfill(c)))
    c.register('breakoutiq_analysis', Singleton(lambda: Analysis(c)))
    return c


def price_api_client(c: Container) -> PriceApi:
    """The client of the price API that PRICE_API names: the one place that picks between them."""
    if c.price_api == TIINGO:
        return TiingoPrices(c.tiingo_api_key)
    if c.price_api == EODHD:
        return EodhdPrices(c.eodhd_api_key)
    raise ValueError(f'unknown price API: {c.price_api}')
