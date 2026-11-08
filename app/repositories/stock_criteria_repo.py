from collections.abc import Iterable
from datetime import date

from app.container import Container
from app.db import connect
from app.models import StockCriteria

COLUMNS = (
    'subsector_id',
    'ticker',
    'price',
    'core_criteria_met',
    'core_criteria_score',
    'high_20d',
    'high_60d',
    'volume',
    'avg_vol_20d',
    'volume_ratio',
    'dma_50',
    'dma_50_prev_15d',
    'rsi',
    'macd',
    'signal',
    'breakout_20d',
    'breakout_60d',
    'breakout_confirmed',
    'price_gt_50dma',
    'dma_50_rising',
    'rsi_50_70',
    'macd_above_signal',
    'date_created',
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
    'close_position',
    'close_upper_quarter',
    'consolidation_range_pct',
    'consolidation_tight',
    'consolidation_vol_ratio',
    'consolidation_vol_declining',
    'dist_high_20d_pct',
    'dist_high_60d_pct',
    'near_resistance',
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
    'secondary_passed',
    'status',
)
PLACEHOLDERS = ','.join(['?'] * len(COLUMNS))


def _as_int(v) -> int | None:
    if v is None:
        return None
    if isinstance(v, bytes):  # avoid X'03' blobs
        return int.from_bytes(v, 'big')
    # bool/np.int/float will coerce via int(...)
    return int(v)


def _as_float(v) -> float | None:
    if v is None:
        return None
    return float(v)


def _row_values(sc: StockCriteria, date_created: date) -> tuple:
    # Convert to plain Python ints/floats; booleans -> 0/1
    return (
        _as_int(sc.subsector_id),  # INTEGER
        sc.ticker,
        _as_float(sc.price),
        _as_int(sc.core_criteria_met),  # INTEGER
        _as_int(sc.core_criteria_score),  # INTEGER
        _as_float(sc.high_20d),
        _as_float(sc.high_60d),
        _as_int(sc.volume),  # INTEGER
        _as_int(sc.avg_vol_20d),  # INTEGER
        _as_float(sc.volume_ratio),
        _as_float(sc.dma_50),
        _as_float(sc.dma_50_prev_15d),
        _as_float(sc.rsi),
        _as_float(sc.macd),
        _as_float(sc.signal),
        _as_int(sc.breakout_20d),  # INTEGER
        _as_int(sc.breakout_60d),  # INTEGER
        _as_int(sc.breakout_confirmed),  # INTEGER
        _as_int(sc.price_gt_50dma),  # INTEGER
        _as_int(sc.dma_50_rising),  # INTEGER
        _as_int(sc.rsi_50_70),  # INTEGER
        _as_int(sc.macd_above_signal),  # INTEGER
        date_created,
        _as_float(sc.macd_hist),
        _as_float(sc.macd_hist_prev),
        _as_int(sc.macd_days_above),  # INTEGER
        _as_int(sc.macd_crossover),  # INTEGER
        _as_int(sc.macd_hist_rising),  # INTEGER
        _as_float(sc.dma_20),
        _as_float(sc.dma_20_prev_15d),
        _as_float(sc.dma_200),
        _as_float(sc.dma_200_prev_15d),
        _as_int(sc.price_gt_20dma),  # INTEGER
        _as_int(sc.dma_20_rising),  # INTEGER
        _as_int(sc.price_gt_200dma),  # INTEGER
        _as_int(sc.dma_200_rising),  # INTEGER
        _as_float(sc.ema_21),
        _as_float(sc.ema_21_prev_5d),
        _as_int(sc.price_gt_21ema),  # INTEGER
        _as_int(sc.ema_21_rising),  # INTEGER
        _as_float(sc.ema_21w),
        _as_float(sc.ema_21w_prev_1w),
        _as_int(sc.price_gt_21wema),  # INTEGER
        _as_int(sc.ema_21w_rising),  # INTEGER
        _as_float(sc.close_position),
        _as_int(sc.close_upper_quarter),  # INTEGER
        _as_float(sc.consolidation_range_pct),
        _as_int(sc.consolidation_tight),  # INTEGER
        _as_float(sc.consolidation_vol_ratio),
        _as_int(sc.consolidation_vol_declining),  # INTEGER
        _as_float(sc.dist_high_20d_pct),
        _as_float(sc.dist_high_60d_pct),
        _as_int(sc.near_resistance),  # INTEGER
        _as_float(sc.return_1m_pct),
        _as_float(sc.return_3m_pct),
        _as_float(sc.rs_market_1m),
        _as_float(sc.rs_market_3m),
        _as_float(sc.rs_sector_1m),
        _as_float(sc.rs_sector_3m),
        _as_int(sc.outperforms_market_1m),  # INTEGER
        _as_int(sc.outperforms_market_3m),  # INTEGER
        _as_int(sc.outperforms_sector_1m),  # INTEGER
        _as_int(sc.outperforms_sector_3m),  # INTEGER
        _as_int(sc.secondary_passed),  # INTEGER
        sc.status,
    )


class StockCriteriaRepo:
    def __init__(self, container: Container):
        self.db_path = container.db_path

    def bulk_insert(self, items: Iterable[StockCriteria], date_created: date) -> None:
        sql = f"""
        INSERT INTO stock_criteria ({','.join(COLUMNS)})
        VALUES ({PLACEHOLDERS})
        ON CONFLICT(ticker) DO UPDATE SET
        {','.join(f'{c}=excluded.{c}' for c in COLUMNS if c != 'ticker')}
        """
        values = [_row_values(sc, date_created) for sc in items]
        with connect(self.db_path) as con:
            con.executemany(sql, values)
