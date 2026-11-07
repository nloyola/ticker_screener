"""frozen_since judges one ticker's rows; tests/test_analyzer_db.py covers what analyze does with its answer."""

from datetime import date, timedelta

import pandas as pd
import pytest

from app.analyzer import FROZEN_DAYS, frozen_since

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
