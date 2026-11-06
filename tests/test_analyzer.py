"""The core criteria Analyzer.analyze_ticker computes, pinned to today's behaviour.

Each test builds a made-up price series, loads it through Analyzer.load_price_data exactly as a run does, and
checks the stored values against numbers worked out by hand. Close prices are linear or zigzag so the averages,
highs and indicators have closed forms:

- A close rising by s a day has a 10-day EMA lagging it by s * (10 - 1) / 2 = 4.5s once the start has decayed,
  a 12-day EMA by 5.5s and a 26-day EMA by 12.5s, so MACD = 7s and its 9-day signal line is also 7s.
- RSI uses Wilder's smoothing (alpha = 1/14). For closes alternating +2 and -1, the smoothed gain just after an up
  day settles at 2a / (1 - (1 - a)^2) = 28/27 and the smoothed loss at (1 - a)a / (1 - (1 - a)^2) = 13/27, so
  RSI = 100 * 28 / 41 = 68.29. A close that never falls has RSI 100.
"""

import math
from datetime import date, timedelta
from types import SimpleNamespace

import pytest

from app.analyzer import Analyzer
from app.models import PriceData, StockCriteria

SUBSECTOR_ID = 7
BASE_VOLUME = 1000


class FakePriceDataRepo:
    def __init__(self, rows: list[PriceData]) -> None:
        self.rows = rows

    def all(self):
        return iter(self.rows)


class FakeStockCriteriaRepo:
    def __init__(self) -> None:
        self.saved: list[tuple[list[StockCriteria], date]] = []

    def bulk_insert(self, items, date_created) -> None:
        self.saved.append((list(items), date_created))


def price_rows(ticker: str, closes: list[float], volumes: list[int] | None = None) -> list[PriceData]:
    volumes = volumes if volumes is not None else [BASE_VOLUME] * len(closes)
    assert len(volumes) == len(closes)
    start = date(2026, 1, 1)
    return [
        PriceData(
            id=i + 1,
            subsector_id=SUBSECTOR_ID,
            ticker=ticker,
            date=start + timedelta(days=i),
            close=close,
            volume=volume,
        )
        for i, (close, volume) in enumerate(zip(closes, volumes, strict=True))
    ]


def make_analyzer(rows: list[PriceData]) -> tuple[Analyzer, FakeStockCriteriaRepo]:
    criteria_repo = FakeStockCriteriaRepo()
    container = SimpleNamespace(price_data_repo=FakePriceDataRepo(rows), stock_criteria_repo=criteria_repo)
    return Analyzer(container), criteria_repo


def analyze(closes: list[float], volumes: list[int] | None = None) -> StockCriteria | None:
    """Runs analyze_ticker on one ticker's rows, grouped as Analyzer._analyze groups them."""
    analyzer, _ = make_analyzer(price_rows('TEST', closes, volumes))
    df = analyzer.load_price_data()
    ((ticker, group_df),) = list(df.groupby(level='ticker'))
    return analyzer.analyze_ticker(ticker, group_df)


def rising(n: int, start: float = 100.0, step: float = 1.0) -> list[float]:
    return [start + step * i for i in range(n)]


def zigzag(n: int, start: float = 100.0, up: float = 2.0, down: float = 1.0) -> list[float]:
    """Alternates +up and -down, ending on an up day (a new high when up > down)."""
    closes = [start]
    for i in range(1, n):
        moves_up = (n - 1 - i) % 2 == 0
        closes.append(closes[-1] + (up if moves_up else -down))
    return closes


def volumes_with_last(n: int, last: int, spikes: dict[int, int] | None = None) -> list[int]:
    volumes = [BASE_VOLUME] * n
    for index, volume in (spikes or {}).items():
        volumes[index] = volume
    volumes[-1] = last
    return volumes


def core_flags(sc: StockCriteria) -> dict[str, int]:
    return {
        'breakout_20d': sc.breakout_20d,
        'breakout_60d': sc.breakout_60d,
        'breakout_confirmed': sc.breakout_confirmed,
        'price_gt_50dma': sc.price_gt_50dma,
        'price_gt_10ema': sc.price_gt_10ema,
        'dma_50_rising': sc.dma_50_rising,
    }


# --- A clean breakout -------------------------------------------------------


def test_clean_breakout_meets_every_core_criterion():
    # closes 100..199, volume 1000 a day and 2000 on the last day
    sc = analyze(rising(100), volumes_with_last(100, 2000))

    assert sc.ticker == 'TEST'
    assert sc.subsector_id == SUBSECTOR_ID
    assert sc.price == 199.0
    assert sc.high_20d == 198.0  # the highest close of the 20 days before today
    assert sc.high_60d == 198.0
    assert sc.dma_50 == 174.5  # mean of 150..199
    assert sc.dma_50_prev_15d == 160.5  # 14 days earlier: mean of 136..185
    assert sc.dma_50_25d_ago == 150.5  # 24 days earlier: mean of 126..175
    assert sc.ema_10 == 194.5  # 199 - 4.5
    assert sc.volume == 2000
    assert sc.avg_vol_20d == 1050  # (19 * 1000 + 2000) / 20
    assert sc.max_vol_10d == 1000  # the 10 days before today

    assert core_flags(sc) == dict.fromkeys(core_flags(sc), 1)
    assert sc.core_criteria_score == 5
    assert sc.core_criteria_met is True


def test_macd_and_signal_of_a_steady_rise():
    sc = analyze(rising(100), volumes_with_last(100, 2000))

    assert sc.macd == pytest.approx(7.0, abs=0.01)
    assert sc.signal == pytest.approx(7.0, abs=0.01)


def test_values_are_rounded_to_two_places():
    sc = analyze(rising(100, start=100.123, step=1.0), volumes_with_last(100, 2000))

    assert sc.price == 199.12
    assert sc.high_20d == 198.12
    assert sc.dma_50 == 174.62
    assert sc.ema_10 == 194.62


def test_20_day_breakout_without_a_60_day_breakout():
    # closes 100..179, then a fall to 160 for 20 days: the 60-day high (179) stays out of reach of a jump to 175
    closes = rising(80) + [160.0] * 20 + [175.0]
    sc = analyze(closes, volumes_with_last(len(closes), 2000))

    assert sc.high_20d == 160.0
    assert sc.high_60d == 179.0
    assert sc.breakout_20d == 1
    assert sc.breakout_60d == 0
    assert sc.dma_50 == 163.2  # (151..179 + 20 * 160 + 175) / 50
    assert sc.dma_50_prev_15d == 158.28  # (137..179 + 7 * 160) / 50
    assert sc.dma_50_25d_ago == 151.5  # mean of 127..176
    # either breakout is enough
    assert sc.core_criteria_score == 5
    assert sc.core_criteria_met is True


def test_no_breakout_at_the_previous_high():
    # today's close equals the 20-day high: not above it
    closes = rising(99) + [198.0]
    sc = analyze(closes, volumes_with_last(100, 2000))

    assert sc.high_20d == 198.0
    assert sc.breakout_20d == 0
    assert sc.breakout_60d == 0
    assert sc.core_criteria_met is False
    assert sc.core_criteria_score == 4


# --- Volume confirmation ----------------------------------------------------


def test_breakout_on_low_volume_is_unconfirmed():
    # 1500 < 1.5 * (19 * 1000 + 1500) / 20 = 1537.5
    sc = analyze(rising(100), volumes_with_last(100, 1500))

    assert sc.avg_vol_20d == 1025
    assert sc.breakout_confirmed == 0
    assert sc.core_criteria_score == 4
    assert sc.core_criteria_met is False


def test_breakout_below_a_recent_volume_spike_is_unconfirmed():
    # 2000 >= 1.5 * the 20-day average of 1250, but below the 5000 of five days ago
    sc = analyze(rising(100), volumes_with_last(100, 2000, spikes={-6: 5000}))

    assert sc.avg_vol_20d == 1250  # (18 * 1000 + 5000 + 2000) / 20
    assert sc.max_vol_10d == 5000
    assert sc.breakout_confirmed == 0
    assert sc.core_criteria_met is False


@pytest.mark.parametrize(('spike_at', 'confirmed'), [(-11, 0), (-12, 1)])
def test_the_10_day_volume_high_covers_the_10_days_before_today(spike_at, confirmed):
    # a 5000 spike 10 days before today is in the window; 11 days before is not
    sc = analyze(rising(100), volumes_with_last(100, 2000, spikes={spike_at: 5000}))

    assert sc.max_vol_10d == (5000 if confirmed == 0 else 1000)
    assert sc.breakout_confirmed == confirmed


def test_volume_exactly_at_both_thresholds_confirms():
    # 1500 is exactly 1.5 * the 20-day average of (17 * 1000 + 0 + 1500 + 1500) / 20 = 1000,
    # and equal to yesterday's 1500, the 10-day high
    volumes = volumes_with_last(100, 1500, spikes={-2: 1500, -3: 0})
    sc = analyze(rising(100), volumes)

    assert sc.avg_vol_20d == 1000
    assert sc.max_vol_10d == 1500
    assert sc.breakout_confirmed == 1


# --- The 50-day and 10-day averages -----------------------------------------


def test_close_under_the_50_day_average():
    # closes 100..198, then a drop to 150
    closes = rising(99) + [150.0]
    sc = analyze(closes, volumes_with_last(100, 2000))

    assert sc.dma_50 == 173.52  # (sum of 150..198 + 150) / 50 = (49 * 174 + 150) / 50
    assert sc.price_gt_50dma == 0
    assert sc.price_gt_10ema == 0
    assert sc.breakout_20d == 0
    assert sc.breakout_60d == 0
    assert sc.breakout_confirmed == 1
    assert sc.dma_50_rising == 1
    assert sc.core_criteria_score == 2
    assert sc.core_criteria_met is False


def test_falling_50_day_average():
    # closes 200 down to 102, then a jump to 170: a breakout over a falling 50-day average
    closes = rising(99, start=200.0, step=-1.0) + [170.0]
    sc = analyze(closes, volumes_with_last(100, 2000))

    assert sc.high_20d == 121.0  # 200 - 79
    assert sc.high_60d == 161.0  # 200 - 39
    assert sc.dma_50 == 126.88  # (49 * 126 + 170) / 50
    assert sc.dma_50_prev_15d == 139.5  # mean of 200 - 36 .. 200 - 85
    assert sc.dma_50_25d_ago == 149.5
    assert sc.dma_50_rising == 0
    assert {k: v for k, v in core_flags(sc).items() if k != 'dma_50_rising'} == {
        'breakout_20d': 1,
        'breakout_60d': 1,
        'breakout_confirmed': 1,
        'price_gt_50dma': 1,
        'price_gt_10ema': 1,
    }
    assert sc.core_criteria_score == 4
    assert sc.core_criteria_met is False


def test_50_day_average_rising_lately_but_not_before():
    # falling for 70 days then rising for 30: today's average is above 14 days ago, which is below 24 days ago
    closes = rising(70, start=200.0, step=-1.0) + rising(30, start=132.0, step=2.0)
    sc = analyze(closes, volumes_with_last(100, 2000))

    assert sc.dma_50 > sc.dma_50_prev_15d
    assert sc.dma_50_prev_15d < sc.dma_50_25d_ago
    assert sc.dma_50_rising == 0


def test_50_day_average_rising_before_but_not_lately():
    # rising for 86 days then falling for 14: 14 days ago is above 24 days ago, but today is below 14 days ago
    closes = rising(86) + rising(14, start=180.0, step=-10.0)
    sc = analyze(closes, volumes_with_last(100, 2000))

    assert sc.dma_50 == 152.8  # (150..185 + 180, 170, .., 50) / 50
    assert sc.dma_50_prev_15d == 160.5
    assert sc.dma_50_25d_ago == 150.5
    assert sc.dma_50_rising == 0


def test_ema_10_weights_every_row_as_pandas_adjust_true():
    # One huge first close makes the start of the series visible in today's EMA. With adjust=True each close i of n
    # weighs (1 - a)^(n - 1 - i), a = 2 / 11, over the sum of all the weights. Step 4 of the screener review
    # rollout may switch to adjust=False, which would give 107.21 here.
    closes = [1_000_000.0] + [100.0] * 59
    n, a = len(closes), 2 / 11
    weights = [(1 - a) ** (n - 1 - i) for i in range(n)]
    expected = sum(w * c for w, c in zip(weights, closes, strict=True)) / sum(weights)
    sc = analyze(closes, volumes_with_last(n, 2000))

    assert round(expected, 2) == 101.31
    assert sc.ema_10 == 101.31


def test_close_above_the_50_day_average_but_under_the_10_day_ema():
    # closes 100..194, then five days at 185: under the 10-day EMA, still above the 50-day average
    closes = rising(95) + [185.0] * 5
    sc = analyze(closes, volumes_with_last(100, 2000))

    assert sc.dma_50 < 185.0 < sc.ema_10
    assert sc.price_gt_50dma == 1
    assert sc.price_gt_10ema == 0


# --- RSI and MACD, which are stored but not core criteria -------------------


def test_rsi_inside_50_to_75():
    sc = analyze(zigzag(200), volumes_with_last(200, 2000))

    assert sc.rsi == pytest.approx(68.29, abs=0.01)
    assert sc.rsi_50_75 == 1


def test_rsi_above_75():
    sc = analyze(rising(100), volumes_with_last(100, 2000))

    assert sc.rsi == 100.0
    assert sc.rsi_50_75 == 0
    # RSI is not a core criterion
    assert sc.core_criteria_met is True


def test_rsi_below_50():
    # alternating +1 and -2: gain 14/27, loss 26/27, RSI = 100 * 14 / 40
    sc = analyze(zigzag(200, start=500.0, up=1.0, down=2.0), volumes_with_last(200, 2000))

    assert sc.rsi == pytest.approx(35.0, abs=0.01)
    assert sc.rsi_50_75 == 0


def test_macd_above_its_signal_line_when_the_rise_speeds_up():
    closes = rising(80) + rising(20, start=181.0, step=3.0)
    sc = analyze(closes, volumes_with_last(100, 2000))

    assert sc.macd > sc.signal
    assert sc.macd_bullish == 1


def test_macd_below_its_signal_line_when_the_rise_turns():
    closes = rising(95) + [190.0, 186.0, 182.0, 178.0, 174.0]
    sc = analyze(closes, volumes_with_last(100, 2000))

    assert sc.macd < sc.signal
    assert sc.macd_bullish == 0
    assert sc.rsi_50_75 == 0  # still near the top of a long rise, so not below 50


# --- How much history is needed ---------------------------------------------


def test_fewer_than_60_rows_is_not_scored():
    assert analyze(rising(59), volumes_with_last(59, 2000)) is None


def test_no_rows_is_not_scored():
    analyzer, _ = make_analyzer([])
    assert analyzer.load_price_data() is None


@pytest.mark.parametrize('rows', [60, 64])
def test_60_to_64_rows_leaves_both_earlier_averages_unknown(rows):
    # today's behaviour, which step 4 of the screener review rollout changes
    sc = analyze(rising(rows), volumes_with_last(rows, 2000))

    assert sc.dma_50_prev_15d is None
    assert sc.dma_50_25d_ago is None
    assert sc.dma_50_rising == 0
    assert sc.core_criteria_met is False


@pytest.mark.parametrize('rows', [65, 74])
def test_65_to_74_rows_leaves_the_25_day_average_unknown(rows):
    # today's behaviour, which step 4 of the screener review rollout changes
    sc = analyze(rising(rows), volumes_with_last(rows, 2000))

    assert sc.dma_50_prev_15d is not None
    assert sc.dma_50_25d_ago is None
    assert sc.dma_50_rising == 0
    assert sc.core_criteria_score == 4
    assert sc.core_criteria_met is False


def test_60_rows_leaves_the_60_day_high_unknown():
    # the 60-day high is taken from yesterday's window, which needs 61 rows; today it is stored as NaN
    sc = analyze(rising(60), volumes_with_last(60, 2000))

    assert math.isnan(sc.high_60d)
    assert sc.breakout_60d == 0
    assert sc.breakout_20d == 1


def test_75_rows_is_enough_to_score_fully():
    sc = analyze(rising(75), volumes_with_last(75, 2000))

    assert sc.dma_50_prev_15d is not None
    assert sc.dma_50_25d_ago is not None
    assert sc.dma_50_rising == 1
    assert sc.core_criteria_score == 5
    assert sc.core_criteria_met is True


# --- A whole run --------------------------------------------------------------


def test_analyze_saves_one_row_per_ticker_with_enough_history():
    rows = (
        price_rows('AAA', rising(100), volumes_with_last(100, 2000))
        + price_rows('BBB', rising(100), volumes_with_last(100, 1500))
        + price_rows('CCC', rising(59))
    )
    analyzer, criteria_repo = make_analyzer(rows)

    analyzer.analyze()

    ((items, date_created),) = criteria_repo.saved
    assert date_created == date.today()
    by_ticker = {sc.ticker: sc for sc in items}
    assert sorted(by_ticker) == ['AAA', 'BBB']
    assert by_ticker['AAA'].core_criteria_met is True
    assert by_ticker['BBB'].core_criteria_met is False
    assert by_ticker['BBB'].breakout_confirmed == 0


def test_analyze_skips_a_ticker_whose_prices_end_before_the_newest_market_date(caplog, capsys):
    # STALE's download failed in the last fetch, so its rows end a day before everyone else's
    rows = (
        price_rows('AAA', rising(100), volumes_with_last(100, 2000))
        + price_rows('BBB', rising(100), volumes_with_last(100, 1500))
        + price_rows('STALE', rising(99), volumes_with_last(99, 2000))
    )
    analyzer, criteria_repo = make_analyzer(rows)

    with caplog.at_level('WARNING', logger='app.analyzer'):
        analyzer.analyze()

    ((items, _),) = criteria_repo.saved
    assert sorted(sc.ticker for sc in items) == ['AAA', 'BBB']
    newest, last = date(2026, 1, 1) + timedelta(days=99), date(2026, 1, 1) + timedelta(days=98)
    warnings = [r for r in caplog.records if r.levelname == 'WARNING']
    assert [r.getMessage() for r in warnings] == [
        f'STALE not scored: its last price is from {last}, before the newest market date {newest}'
    ]
    assert 'skipped 1 stale ticker' in capsys.readouterr().out


def test_analyze_with_every_ticker_current_skips_none(capsys):
    rows = price_rows('AAA', rising(100), volumes_with_last(100, 2000)) + price_rows(
        'BBB', rising(100), volumes_with_last(100, 2000)
    )
    analyzer, criteria_repo = make_analyzer(rows)

    analyzer.analyze()

    ((items, _),) = criteria_repo.saved
    assert sorted(sc.ticker for sc in items) == ['AAA', 'BBB']
    assert 'skipped 0 stale tickers' in capsys.readouterr().out
