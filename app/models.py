from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True)
class Sector:
    sector: str
    subsector: str
    tickers: str
    created_at: datetime = datetime.now()


@dataclass(frozen=True)
class PriceData:
    ticker: str
    date: date
    close: float
    volume: int


@dataclass(frozen=True)
class StockCriteria:
    ticker: str
    price: float | None = None
    core_criteria_met: bool | None = None
    core_criteria_score: int | None = None
    high_20d: float | None = None
    high_60d: float | None = None
    volume: int | None = None
    avg_vol_20d: int | None = None
    max_vol_10d: int | None = None
    dma_50: float | None = None
    ema_10: float | None = None
    dma_50_prev_15d: float | None = None
    dma_50_25d_ago: float | None = None
    rsi: float | None = None
    macd: float | None = None
    signal: float | None = None
    breakout_20d: int | None = None
    breakout_60d: int | None = None
    breakout_confirmed: int | None = None
    price_gt_50dma: int | None = None
    price_gt_10ema: int | None = None
    dma_50_rising: int | None = None
    rsi_50_75: int | None = None
    macd_bullish: int | None = None
