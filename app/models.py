from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class Ticker:
    ticker: str
    last: float | None
    prev_close: float | None
    volume: int | None
    bid_price: float | None
    ask_price: float | None
    timestamp: str | None


@dataclass(frozen=True)
class Sector:
    id: int
    sector: str
    created_at: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class Subsector:
    id: int
    sector_id: int
    subsector: str
    tickers: str
    created_at: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class PriceData:
    id: int
    subsector_id: int
    ticker: str
    date: date
    close: float
    volume: int
    # adjusted as the close is; None in rows stored before the screener kept them, until the next fetch replaces them
    high: float | None = None
    low: float | None = None


@dataclass(frozen=True)
class StockCriteria:
    id: int
    subsector_id: int
    ticker: str
    price: float | None = None
    core_criteria_met: bool | None = None
    core_criteria_score: int | None = None
    high_20d: float | None = None
    high_60d: float | None = None
    volume: int | None = None
    avg_vol_20d: int | None = None
    # volume / avg_vol_20d, None when the average is 0
    volume_ratio: float | None = None
    dma_50: float | None = None
    dma_50_prev_15d: float | None = None
    rsi: float | None = None
    macd: float | None = None
    signal: float | None = None
    breakout_20d: int | None = None
    breakout_60d: int | None = None
    breakout_confirmed: int | None = None
    price_gt_50dma: int | None = None
    dma_50_rising: int | None = None
    rsi_50_75: int | None = None
    macd_bullish: int | None = None
