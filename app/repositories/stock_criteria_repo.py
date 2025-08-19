import sqlite3
from collections.abc import Iterable, Iterator

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
    'max_vol_10d',
    'dma_50',
    'ema_10',
    'dma_50_prev_15d',
    'dma_50_25d_ago',
    'rsi',
    'macd',
    'signal',
    'breakout_20d',
    'breakout_60d',
    'breakout_confirmed',
    'price_gt_50dma',
    'price_gt_10ema',
    'dma_50_rising',
    'rsi_50_75',
    'macd_bullish',
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


def _row_values(sc: StockCriteria) -> tuple:
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
        _as_int(sc.max_vol_10d),  # INTEGER
        _as_float(sc.dma_50),
        _as_float(sc.ema_10),
        _as_float(sc.dma_50_prev_15d),
        _as_float(sc.dma_50_25d_ago),
        _as_float(sc.rsi),
        _as_float(sc.macd),
        _as_float(sc.signal),
        _as_int(sc.breakout_20d),  # INTEGER
        _as_int(sc.breakout_60d),  # INTEGER
        _as_int(sc.breakout_confirmed),  # INTEGER
        _as_int(sc.price_gt_50dma),  # INTEGER
        _as_int(sc.price_gt_10ema),  # INTEGER
        _as_int(sc.dma_50_rising),  # INTEGER
        _as_int(sc.rsi_50_75),  # INTEGER
        _as_int(sc.macd_bullish),  # INTEGER
    )


class StockCriteriaRepo:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def insert(self, sc: StockCriteria) -> None:
        sql = f"""
            INSERT INTO stock_criteria ({','.join(COLUMNS)})
            VALUES ({PLACEHOLDERS})
            ON CONFLICT(ticker) DO UPDATE SET
            {','.join(f'{c}=excluded.{c}' for c in COLUMNS if c != 'ticker')}
        """
        with self._connect() as con:
            con.execute(sql, _row_values(sc))

    def bulk_insert(self, items: Iterable[StockCriteria]) -> None:
        sql = f"""
            INSERT INTO stock_criteria ({','.join(COLUMNS)})
            VALUES ({PLACEHOLDERS})
            ON CONFLICT(ticker) DO UPDATE SET
            {','.join(f'{c}=excluded.{c}' for c in COLUMNS if c != 'ticker')}
        """
        values = [_row_values(sc) for sc in items]
        with self._connect() as con:
            con.executemany(sql, values)

    def get(self, ticker: str) -> StockCriteria | None:
        with self._connect() as con:
            row = con.execute(
                'SELECT * FROM stock_criteria WHERE ticker = ?',
                (ticker,),
            ).fetchone()
            return StockCriteria(**row) if row else None

    def all(self) -> Iterator[StockCriteria]:
        with self._connect() as con:
            rows = con.execute('SELECT * FROM stock_criteria ORDER BY ticker').fetchall()
            for r in rows:
                yield StockCriteria(**r)

    def delete(self, ticker: str) -> None:
        with self._connect() as con:
            con.execute('DELETE FROM stock_criteria WHERE ticker = ?', (ticker,))
