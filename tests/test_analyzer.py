"""The core criteria and secondary indicators Analyzer.analyze_ticker computes.

Each test builds a made-up price series, loads it through Analyzer.load_price_data exactly as a run does, and
checks the stored values against numbers worked out by hand. Close prices are linear or zigzag so the averages,
highs and indicators have closed forms. Each day's high equals its close unless a test says otherwise.

- A close rising by s a day has a 12-day EMA lagging it by 5.5s once the start has decayed and a 26-day EMA by
  12.5s, so MACD = 7s and its 9-day signal line is also 7s.
- RSI uses Wilder's smoothing (alpha = 1/14). For closes alternating +2 and -1, the smoothed gain just after an up
  day settles at 2a / (1 - (1 - a)^2) = 28/27 and the smoothed loss at (1 - a)a / (1 - (1 - a)^2) = 13/27, so
  RSI = 100 * 28 / 41 = 68.29. A close that never falls has RSI 100.
- An n-day SMA of a close rising by s a day lags it by (n - 1)s / 2. A 21-period EMA, started at the first close,
  lags such a close by 10s once the start has decayed, the start's share after k more values being 10s * (10/11)^k.
- The rows' dates are consecutive calendar days from Thursday 1 January 2026, so the weekly closes are the Sundays'
  closes and the scored day's own, 7 days of a daily rise apart.
- Where no closed form is at hand (the MACD's crossovers and histogram), the tests compare with reference_ema and
  reference_macd below, written out from the definitions rather than with the library the analyzer uses.
"""

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


def price_rows(
    ticker: str, closes: list[float], volumes: list[int] | None = None, highs: list[float | None] | None = None
) -> list[PriceData]:
    volumes = volumes if volumes is not None else [BASE_VOLUME] * len(closes)
    highs = highs if highs is not None else closes
    start = date(2026, 1, 1)
    return [
        PriceData(
            id=i + 1,
            subsector_id=SUBSECTOR_ID,
            ticker=ticker,
            date=start + timedelta(days=i),
            close=close,
            volume=volume,
            high=high,
            low=None if high is None else min(close, high) - 1,
        )
        for i, (close, volume, high) in enumerate(zip(closes, volumes, highs, strict=True))
    ]


def make_analyzer(rows: list[PriceData]) -> tuple[Analyzer, FakeStockCriteriaRepo]:
    criteria_repo = FakeStockCriteriaRepo()
    container = SimpleNamespace(price_data_repo=FakePriceDataRepo(rows), stock_criteria_repo=criteria_repo)
    return Analyzer(container), criteria_repo


def analyze(
    closes: list[float], volumes: list[int] | None = None, highs: list[float | None] | None = None
) -> StockCriteria | None:
    """Runs analyze_ticker on one ticker's rows, grouped as Analyzer._analyze groups them."""
    analyzer, _ = make_analyzer(price_rows('TEST', closes, volumes, highs))
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
        'dma_50_rising': sc.dma_50_rising,
    }


# --- A clean breakout -------------------------------------------------------


def test_clean_breakout_meets_every_core_criterion():
    # closes 100..199, volume 1000 a day and 2000 on the last day
    sc = analyze(rising(100), volumes_with_last(100, 2000))

    assert sc.ticker == 'TEST'
    assert sc.subsector_id == SUBSECTOR_ID
    assert sc.price == 199.0
    assert sc.high_20d == 198.0  # the highest high of the 20 days before today
    assert sc.high_60d == 198.0
    assert sc.dma_50 == 174.5  # mean of 150..199
    assert sc.dma_50_prev_15d == 159.5  # 15 trading days earlier: mean of 135..184
    assert sc.volume == 2000
    assert sc.avg_vol_20d == 1000  # the 20 days before today, not including its 2000
    assert sc.volume_ratio == 2.0

    assert core_flags(sc) == dict.fromkeys(core_flags(sc), 1)
    assert sc.core_criteria_score == 4
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


# --- The price breakout -----------------------------------------------------


def test_20_day_breakout_without_a_60_day_breakout():
    # closes 100..179, then a fall to 160 for 20 days: the 60-day high (179) stays out of reach of a jump to 175
    closes = rising(80) + [160.0] * 20 + [175.0]
    sc = analyze(closes, volumes_with_last(len(closes), 2000))

    assert sc.high_20d == 160.0
    assert sc.high_60d == 179.0
    assert sc.breakout_20d == 1
    assert sc.breakout_60d == 0
    assert sc.dma_50 == 163.2  # (151..179 + 20 * 160 + 175) / 50
    assert sc.dma_50_prev_15d == 157.8  # (136..179 + 6 * 160) / 50
    # either breakout is enough
    assert sc.core_criteria_score == 4
    assert sc.core_criteria_met is True


def test_no_breakout_at_the_previous_high():
    # today's close equals the 20-day high: not above it
    closes = rising(99) + [198.0]
    sc = analyze(closes, volumes_with_last(100, 2000))

    assert sc.high_20d == 198.0
    assert sc.breakout_20d == 0
    assert sc.breakout_60d == 0
    assert sc.core_criteria_met is False
    assert sc.core_criteria_score == 3


def test_the_breakout_is_measured_against_the_highs_not_the_closes():
    # five days ago the price reached 250 during the day but closed at 194: today's 199 close is above every close
    # before it, yet below that high
    closes = rising(100)
    highs = closes[:-6] + [250.0] + closes[-5:]
    sc = analyze(closes, volumes_with_last(100, 2000), highs)

    assert sc.high_20d == 250.0
    assert sc.high_60d == 250.0
    assert sc.breakout_20d == 0
    assert sc.breakout_60d == 0


@pytest.mark.parametrize(('high_at', 'breakout_20d'), [(-21, 0), (-22, 1)])
def test_the_20_day_high_covers_the_20_days_before_today(high_at, breakout_20d):
    # a 250 high 20 days before today is in the window; 21 days before is not
    closes = rising(100)
    highs = list(closes)
    highs[high_at] = 250.0
    sc = analyze(closes, volumes_with_last(100, 2000), highs)

    assert sc.breakout_20d == breakout_20d
    assert sc.breakout_60d == 0


def test_todays_own_high_does_not_raise_the_bar():
    closes = rising(100)
    highs = closes[:-1] + [300.0]
    sc = analyze(closes, volumes_with_last(100, 2000), highs)

    assert sc.high_20d == 198.0
    assert sc.breakout_20d == 1


def test_a_row_without_a_high_counts_its_close():
    # rows stored before the screener kept highs have none until the next fetch
    sc = analyze(rising(100), volumes_with_last(100, 2000), [None] * 100)

    assert sc.high_20d == 198.0
    assert sc.high_60d == 198.0
    assert sc.core_criteria_met is True


# --- Volume confirmation ----------------------------------------------------


def test_breakout_on_low_volume_is_unconfirmed():
    # 1499 < 1.5 * the 20-day average of 1000
    sc = analyze(rising(100), volumes_with_last(100, 1499))

    assert sc.avg_vol_20d == 1000
    assert sc.volume_ratio == pytest.approx(1.499)  # stored unrounded, so it never reads as 1.5
    assert sc.breakout_confirmed == 0
    assert sc.core_criteria_score == 3
    assert sc.core_criteria_met is False


def test_the_breakout_days_own_volume_does_not_raise_the_average():
    # exactly 1.5 times the 20 days before it; counted in its own average it would fall short
    sc = analyze(rising(100), volumes_with_last(100, 1500))

    assert sc.avg_vol_20d == 1000
    assert sc.volume_ratio == 1.5
    assert sc.breakout_confirmed == 1
    assert sc.core_criteria_met is True


def test_a_recent_volume_spike_raises_the_average_but_does_not_block_the_breakout():
    # 2000 >= 1.5 * the 20-day average of 1200, though below the 5000 of five days ago
    sc = analyze(rising(100), volumes_with_last(100, 2000, spikes={-6: 5000}))

    assert sc.avg_vol_20d == 1200  # (19 * 1000 + 5000) / 20
    assert sc.volume_ratio == pytest.approx(2000 / 1200)
    assert sc.breakout_confirmed == 1


@pytest.mark.parametrize(('spike_at', 'confirmed'), [(-21, 0), (-22, 1)])
def test_the_average_volume_covers_the_20_days_before_today(spike_at, confirmed):
    # an 11000 spike 20 days before today raises the average to 1500, above 2000 / 1.5; 21 days before it is out
    sc = analyze(rising(100), volumes_with_last(100, 2000, spikes={spike_at: 11000}))

    assert sc.avg_vol_20d == (1500 if confirmed == 0 else 1000)
    assert sc.breakout_confirmed == confirmed


def test_no_volume_before_today_has_no_ratio():
    volumes = [0] * 99 + [2000]
    sc = analyze(rising(100), volumes)

    assert sc.avg_vol_20d == 0
    assert sc.volume_ratio is None
    assert sc.breakout_confirmed == 0


# --- The 50-day average -----------------------------------------------------


def test_close_under_the_50_day_average():
    # closes 100..198, then a drop to 150
    closes = rising(99) + [150.0]
    sc = analyze(closes, volumes_with_last(100, 2000))

    assert sc.dma_50 == 173.52  # (sum of 150..198 + 150) / 50 = (49 * 174 + 150) / 50
    assert sc.price_gt_50dma == 0
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
    assert sc.dma_50_prev_15d == 140.5  # mean of 200 - 35 .. 200 - 84
    assert sc.dma_50_rising == 0


def test_50_day_average_rising_before_but_not_lately():
    # rising for 86 days then falling for 14: today's average is below the one 15 trading days earlier
    closes = rising(86) + rising(14, start=180.0, step=-10.0)
    sc = analyze(closes, volumes_with_last(100, 2000))

    assert sc.dma_50 == 152.8  # (150..185 + 180, 170, .., 50) / 50
    assert sc.dma_50_prev_15d == 159.5
    assert sc.dma_50_rising == 0


# --- Exactly the four mandatory conditions ----------------------------------

# Each fixture fails one mandatory condition and passes the other three: the core criteria count the four
# conditions alone, so it scores 3 and is not core criteria met.
SINGLE_FAILURES = {
    # today's close equals the 20-day high
    'breakout': (rising(99) + [198.0], volumes_with_last(100, 2000)),
    'breakout_confirmed': (rising(100), volumes_with_last(100, 1499)),
    # 100 for 50 days, 300 for 29, 200 for 20, then 210: above the 20-day high of 200, below the 50-day average of
    # (29 * 300 + 20 * 200 + 210) / 50 = 258.2, which is above the 228 of 15 days earlier
    'price_gt_50dma': ([100.0] * 50 + [300.0] * 29 + [200.0] * 20 + [210.0], volumes_with_last(100, 2000)),
    # closes 200 down to 102, then a jump to 170
    'dma_50_rising': (rising(99, start=200.0, step=-1.0) + [170.0], volumes_with_last(100, 2000)),
}


@pytest.mark.parametrize('failing', SINGLE_FAILURES)
def test_failing_any_single_condition_is_not_core_criteria_met(failing):
    closes, volumes = SINGLE_FAILURES[failing]
    sc = analyze(closes, volumes)

    passed = {
        'breakout': bool(sc.breakout_20d or sc.breakout_60d),
        'breakout_confirmed': bool(sc.breakout_confirmed),
        'price_gt_50dma': bool(sc.price_gt_50dma),
        'dma_50_rising': bool(sc.dma_50_rising),
    }
    assert passed == {condition: condition != failing for condition in passed}
    assert sc.core_criteria_score == 3
    assert sc.core_criteria_met is False


def test_the_price_under_the_50_day_average_fixture_is_worked_out_right():
    sc = analyze(*SINGLE_FAILURES['price_gt_50dma'])

    assert sc.high_20d == 200.0
    assert sc.dma_50 == 258.2
    assert sc.dma_50_prev_15d == 228.0  # (15 * 100 + 29 * 300 + 6 * 200) / 50


# --- Secondary indicators, which are stored but never counted in the score --


def reference_ema(values: list[float], span: int) -> list[float]:
    """The exponential moving average with alpha = 2 / (span + 1), started at the first value."""
    alpha = 2 / (span + 1)
    averages = [values[0]]
    for value in values[1:]:
        averages.append(alpha * value + (1 - alpha) * averages[-1])
    return averages


def reference_macd(closes: list[float]) -> tuple[list[float], list[float]]:
    """
    MACD (the 12-day EMA minus the 26-day EMA) and its signal line (the 9-day EMA of MACD, started at its first
    value), from the 26th close on, where the 26-day EMA has 26 values; the signal line counts from its 9th value.
    """
    macd = [fast - slow for fast, slow in zip(reference_ema(closes, 12), reference_ema(closes, 26), strict=True)][25:]
    return macd, reference_ema(macd, 9)


def lagging_ema(last: float, step: float, values: int) -> float:
    """A 21-period EMA of 'values' values rising by 'step' each, ending at 'last' (see the module docstring)."""
    return last - 10 * step + 10 * step * (10 / 11) ** (values - 1)


def test_rsi_inside_50_to_70():
    sc = analyze(zigzag(200), volumes_with_last(200, 2000))

    assert sc.rsi == pytest.approx(68.29, abs=0.01)
    assert sc.rsi_50_70 == 1


def test_rsi_between_70_and_75_is_outside_the_band():
    # alternating +2.5 and -1: RSI = 100 * 2.5 / (2.5 + 13/14) = 72.92
    sc = analyze(zigzag(200, up=2.5), volumes_with_last(200, 2000))

    assert sc.rsi == pytest.approx(72.92, abs=0.01)
    assert sc.rsi_50_70 == 0


def test_rsi_above_70():
    sc = analyze(rising(100), volumes_with_last(100, 2000))

    assert sc.rsi == 100.0
    assert sc.rsi_50_70 == 0
    # RSI is not a core criterion
    assert sc.core_criteria_met is True


def test_rsi_below_50():
    # alternating +1 and -2: gain 14/27, loss 26/27, RSI = 100 * 14 / 40
    sc = analyze(zigzag(200, start=500.0, up=1.0, down=2.0), volumes_with_last(200, 2000))

    assert sc.rsi == pytest.approx(35.0, abs=0.01)
    assert sc.rsi_50_70 == 0


def recovery(days_up: int, step: float) -> list[float]:
    """100 for 60 days, down 1 a day for 10 days, then up 'step' a day for 'days_up' days."""
    return [100.0] * 60 + rising(10, start=100.0, step=-1.0) + rising(days_up, start=91.0 + step, step=step)


@pytest.mark.parametrize(
    ('closes', 'days_above', 'crossover'),
    [(recovery(5, 3.0), 3, 1), (recovery(8, 2.0), 5, 1), (recovery(8, 3.0), 6, 0)],
    ids=['3 days ago', '5 days ago', '6 days ago'],
)
def test_macd_crossover_counts_the_last_5_sessions(closes, days_above, crossover):
    sc = analyze(closes, volumes_with_last(len(closes), 2000))
    macd, signal = reference_macd(closes)

    assert sc.macd == pytest.approx(macd[-1], abs=0.005)
    assert sc.signal == pytest.approx(signal[-1], abs=0.005)
    assert sc.macd_hist == pytest.approx(macd[-1] - signal[-1], abs=0.005)
    assert sc.macd_hist_prev == pytest.approx(macd[-2] - signal[-2], abs=0.005)
    # the crossover: MACD above its signal line on the last 'days_above' sessions, and not on the one before them
    assert all(m > s for m, s in zip(macd[-days_above:], signal[-days_above:], strict=True))
    assert macd[-1 - days_above] <= signal[-1 - days_above]
    assert sc.macd_days_above == days_above
    assert sc.macd_crossover == crossover
    assert sc.macd_above_signal == 1
    # the histogram is still widening after the turn
    assert 0 < macd[-2] - signal[-2] < macd[-1] - signal[-1]
    assert sc.macd_hist_rising == 1


def test_macd_long_above_its_signal_line_with_a_narrowing_histogram():
    # a rise that sped up 20 days ago: MACD has been above its signal line since well before, and the gap is closing
    closes = rising(80) + rising(20, start=181.0, step=3.0)
    sc = analyze(closes, volumes_with_last(100, 2000))
    macd, signal = reference_macd(closes)

    assert sc.macd > sc.signal
    assert sc.macd_above_signal == 1
    assert sc.macd_days_above == sum(1 for m, s in zip(macd[8:], signal[8:], strict=True) if m > s) == 67
    assert sc.macd_crossover == 0
    assert 0 < macd[-1] - signal[-1] < macd[-2] - signal[-2]
    assert sc.macd_hist_rising == 0


def test_macd_below_its_signal_line_when_the_rise_turns():
    closes = rising(95) + [190.0, 186.0, 182.0, 178.0, 174.0]
    sc = analyze(closes, volumes_with_last(100, 2000))

    assert sc.macd < sc.signal
    assert sc.macd_above_signal == 0
    assert sc.macd_days_above == 0
    assert sc.macd_crossover == 0
    assert sc.macd_hist < 0
    assert sc.macd_hist_rising == 0
    assert sc.rsi_50_70 == 0  # still near the top of a long rise, so not below 50


# 368 rows end on Sunday 3 January 2027, closing 467: 53 weekly closes, from 103 on Sunday 4 January 2026 up by 7 a week
YEAR = rising(368)


def test_moving_average_alignment_of_a_steady_rise():
    sc = analyze(YEAR, volumes_with_last(368, 2000))

    assert sc.price == 467.0
    assert (sc.dma_20, sc.dma_20_prev_15d) == (457.5, 442.5)  # mean of 448..467, and of 433..452
    assert (sc.dma_200, sc.dma_200_prev_15d) == (367.5, 352.5)  # mean of 268..467, and of 253..452
    assert (sc.price_gt_20dma, sc.dma_20_rising) == (1, 1)
    assert (sc.price_gt_50dma, sc.dma_50_rising) == (1, 1)
    assert (sc.price_gt_200dma, sc.dma_200_rising) == (1, 1)


def test_21_day_and_21_week_emas_of_a_steady_rise():
    sc = analyze(YEAR, volumes_with_last(368, 2000))

    assert sc.ema_21 == round(lagging_ema(467.0, 1.0, 368), 2) == 457.0
    assert sc.ema_21_prev_5d == round(lagging_ema(462.0, 1.0, 363), 2) == 452.0
    assert (sc.price_gt_21ema, sc.ema_21_rising) == (1, 1)
    # 53 weekly closes, the first week's start not yet decayed away
    assert sc.ema_21w == round(lagging_ema(467.0, 7.0, 53), 2) == 397.49
    assert sc.ema_21w_prev_1w == round(lagging_ema(460.0, 7.0, 52), 2) == 390.54
    assert (sc.price_gt_21wema, sc.ema_21w_rising) == (1, 1)


def test_the_21_week_ema_ends_on_the_scored_days_close_mid_week():
    # 365 rows end on Thursday 31 December 2026: 52 Sundays' closes (103, 110, .., 460), then the Thursday's 464
    sc = analyze(rising(365), volumes_with_last(365, 2000))
    weekly = reference_ema([100.0 + day for day in range(3, 364, 7)] + [464.0], 21)

    assert sc.ema_21w == round(weekly[-1], 2) == 397.22
    assert sc.ema_21w_prev_1w == round(weekly[-2], 2) == 390.54  # last Sunday's, as in the steady rise
    assert sc.ema_21w_rising == 1


def test_averages_and_emas_of_a_steady_fall():
    # closes 500 down to 133: below every average, and every one falling
    sc = analyze(rising(368, start=500.0, step=-1.0), volumes_with_last(368, 2000))

    assert (sc.dma_20, sc.dma_20_prev_15d) == (142.5, 157.5)
    assert (sc.dma_200, sc.dma_200_prev_15d) == (232.5, 247.5)
    assert sc.ema_21 == round(lagging_ema(133.0, -1.0, 368), 2) == 143.0
    assert sc.ema_21w == round(lagging_ema(133.0, -7.0, 53), 2) == 202.51
    for flag in SECONDARY_FLAGS[4:]:
        assert getattr(sc, flag) == 0, flag


def test_a_ticker_without_the_history_for_the_long_averages_leaves_them_empty():
    sc = analyze(rising(100), volumes_with_last(100, 2000))

    assert (sc.dma_200, sc.dma_200_prev_15d, sc.price_gt_200dma, sc.dma_200_rising) == (None, None, None, None)
    # 100 days are 15 weekly closes, short of the 50 a 21-week EMA needs
    assert (sc.ema_21w, sc.ema_21w_prev_1w, sc.price_gt_21wema, sc.ema_21w_rising) == (None, None, None, None)
    # the short ones are there, and the ticker is scored
    assert sc.dma_20 is not None
    assert sc.ema_21 is not None
    assert sc.core_criteria_score == 4


@pytest.mark.parametrize(
    ('rows', 'dma_200', 'dma_200_rising'),
    [(199, None, None), (200, 199.5, None), (214, 213.5, None), (215, 214.5, 1)],
)
def test_the_200_day_average_needs_200_rows_and_15_more_to_rise(rows, dma_200, dma_200_rising):
    sc = analyze(rising(rows), volumes_with_last(rows, 2000))

    assert sc.dma_200 == dma_200
    assert sc.dma_200_rising == dma_200_rising
    assert sc.price_gt_200dma == (None if dma_200 is None else 1)


@pytest.mark.parametrize(
    ('rows', 'weekly_closes', 'has_ema', 'has_prior'),
    [(340, 49, False, False), (346, 50, True, False), (347, 50, True, False), (348, 51, True, True)],
)
def test_the_21_week_ema_needs_50_weekly_closes(rows, weekly_closes, has_ema, has_prior):
    # 340 rows end on the 49th Sunday; 346 on the following Saturday, mid-week 50; 347 on its Sunday; 348 on a Monday
    days = [date(2026, 1, 1) + timedelta(days=i) for i in range(rows)]
    assert len({day.isocalendar()[:2] for day in days}) == weekly_closes
    sc = analyze(rising(rows), volumes_with_last(rows, 2000))

    assert (sc.ema_21w is not None, sc.ema_21w_prev_1w is not None) == (has_ema, has_prior)
    assert sc.ema_21w_rising == (1 if has_prior else None)


SECONDARY_FLAGS = (
    'rsi_50_70',
    'macd_above_signal',
    'macd_crossover',
    'macd_hist_rising',
    'price_gt_20dma',
    'dma_20_rising',
    'price_gt_200dma',
    'dma_200_rising',
    'price_gt_21ema',
    'ema_21_rising',
    'price_gt_21wema',
    'ema_21w_rising',
)


@pytest.mark.parametrize(
    ('closes', 'volumes'),
    [
        (YEAR, volumes_with_last(368, 2000)),
        (rising(368, start=500.0, step=-1.0), volumes_with_last(368, 2000)),
        (recovery(5, 3.0), volumes_with_last(75, 2000)),
        *SINGLE_FAILURES.values(),
    ],
    ids=['steady rise', 'steady fall', 'recovery', *SINGLE_FAILURES],
)
def test_the_core_score_counts_only_the_core_conditions(closes, volumes):
    sc = analyze(closes, volumes)

    core = [sc.breakout_20d or sc.breakout_60d, sc.breakout_confirmed, sc.price_gt_50dma, sc.dma_50_rising]
    assert sc.core_criteria_score == sum(core)
    assert sc.core_criteria_met is all(core)


def test_the_secondary_indicators_never_change_the_core_score():
    # both pass the four core conditions; the zigzag also passes every secondary indicator, the steady rise fails
    # the RSI band (100), the MACD crossover and the widening histogram
    rise = analyze(YEAR, volumes_with_last(368, 2000))
    zig = analyze(zigzag(368), volumes_with_last(368, 2000))

    assert {flag: getattr(zig, flag) for flag in SECONDARY_FLAGS} == dict.fromkeys(SECONDARY_FLAGS, 1)
    assert (rise.rsi_50_70, rise.macd_crossover, rise.macd_hist_rising) == (0, 0, 0)
    assert rise.core_criteria_score == zig.core_criteria_score == 4
    assert rise.core_criteria_met is zig.core_criteria_met is True


# --- How much history is needed ---------------------------------------------


@pytest.mark.parametrize('rows', [59, 61, 64])
def test_fewer_than_65_rows_is_not_scored(rows):
    # dma_50_rising compares today's 50-day average with the one 15 trading days earlier, which needs 65 rows
    assert analyze(rising(rows), volumes_with_last(rows, 2000)) is None


def test_no_rows_is_not_scored():
    analyzer, _ = make_analyzer([])
    assert analyzer.load_price_data() is None


def test_65_rows_is_enough_to_score_fully():
    sc = analyze(rising(65), volumes_with_last(65, 2000))

    assert sc.high_60d == 163.0  # the highest high of rows 4..63
    assert sc.dma_50_prev_15d == 124.5  # mean of 100..149, the first 50 rows
    assert sc.dma_50_rising == 1
    assert sc.core_criteria_score == 4
    assert sc.core_criteria_met is True


# --- A whole run --------------------------------------------------------------


def test_analyze_saves_one_row_per_ticker_with_enough_history():
    rows = (
        price_rows('AAA', rising(100), volumes_with_last(100, 2000))
        + price_rows('BBB', rising(100), volumes_with_last(100, 1499))
        + price_rows('CCC', rising(64))
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
        f'STALE not scored: its last price is from {last}, before the newest US market date {newest}'
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
