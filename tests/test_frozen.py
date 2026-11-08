"""
frozen_since and padded_stop judge one ticker's rows; tests/test_analyzer_db.py covers what analyze does with their
answers.
"""

from datetime import date, timedelta

import pandas as pd
import pytest

from app.analyzer import FROZEN_DAYS, PADDED_MIN_DAYS, PADDED_WINDOW, frozen_since, padded_stop

START = date(2026, 1, 1)


def rows(closes: list[float | None], volumes: list[int | None]) -> pd.DataFrame:
    """One ticker's rows, oldest first, indexed by ticker and date as analyze reads them."""
    dates = [START + timedelta(days=i) for i in range(len(closes))]
    index = pd.MultiIndex.from_arrays([['FL'] * len(dates), dates], names=['ticker', 'date'])
    return pd.DataFrame(
        {'close': pd.array(closes, dtype='Float64'), 'volume': pd.array(volumes, dtype='Int64')}, index=index
    )


def test_stretch_after_a_different_close_starts_on_its_first_flat_day():
    df = rows([9.0] * 5 + [7.0] * FROZEN_DAYS, [0] * (5 + FROZEN_DAYS))

    assert frozen_since(df) == START + timedelta(days=5)


def test_ticker_with_fewer_rows_than_the_stretch_is_not_frozen():
    df = rows([7.0] * (FROZEN_DAYS - 1), [0] * (FROZEN_DAYS - 1))

    assert frozen_since(df) is None


@pytest.mark.parametrize('column', ['close', 'volume'])
@pytest.mark.parametrize('days_back', [0, 4], ids=['last day', 'within the stretch'])
def test_missing_value_ends_the_stretch(column, days_back):
    n = FROZEN_DAYS + 5
    closes: list[float | None] = [7.0] * n
    volumes: list[int | None] = [0] * n
    (closes if column == 'close' else volumes)[n - 1 - days_back] = None

    assert frozen_since(rows(closes, volumes)) is None


def traded_then_flat(flat: int, traded: int = PADDED_WINDOW + 5) -> tuple[list[float | None], list[int | None]]:
    """'traded' days of rising closes, each with volume, then 'flat' days repeating the last close at zero volume."""
    closes: list[float | None] = [10.0 + i for i in range(traded)]
    volumes: list[int | None] = [1000] * traded
    return closes + [closes[-1]] * flat, volumes + [0] * flat


@pytest.mark.parametrize('flat', [PADDED_MIN_DAYS, 4, FROZEN_DAYS + 5])
def test_padded_stop_is_dated_by_its_last_trade(flat):
    closes, volumes = traded_then_flat(flat)
    df = rows(closes, volumes)

    assert padded_stop(df) == START + timedelta(days=len(closes) - flat - 1)


def test_single_flat_day_is_not_a_padded_stop():
    assert padded_stop(rows(*traded_then_flat(PADDED_MIN_DAYS - 1))) is None


def test_day_without_volume_in_the_window_is_not_a_padded_stop():
    closes, volumes = traded_then_flat(4)
    volumes[-4 - PADDED_WINDOW] = 0

    assert padded_stop(rows(closes, volumes)) is None


def test_day_without_volume_before_the_window_still_makes_a_padded_stop():
    closes, volumes = traded_then_flat(4)
    volumes[-4 - PADDED_WINDOW - 1] = 0

    assert padded_stop(rows(closes, volumes)) == START + timedelta(days=len(closes) - 5)


def test_missing_volume_in_the_window_is_not_a_padded_stop():
    closes, volumes = traded_then_flat(4)
    volumes[-6] = None

    assert padded_stop(rows(closes, volumes)) is None


def test_fewer_traded_rows_than_the_window_is_not_a_padded_stop():
    assert padded_stop(rows(*traded_then_flat(4, traded=PADDED_WINDOW - 1))) is None


def test_flat_stretch_at_another_close_than_the_last_trade_is_not_a_padded_stop():
    closes, volumes = traded_then_flat(4)
    closes[-4:] = [closes[-5] + 1] * 4

    assert padded_stop(rows(closes, volumes)) is None


def test_ticker_trading_today_is_not_a_padded_stop():
    assert padded_stop(rows(*traded_then_flat(0))) is None
