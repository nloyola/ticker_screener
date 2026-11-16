"""
The performance of an archived signal, computed from its own price path (app/signal_paths.py) alone: compute_outcomes
is a pure function, with no I/O, which the outcome updates call for each signal.

Entry is the open of the session after the signal session (path index 1), each horizon's exit the close of the
horizon's session (path index N), and the benchmark is measured over the same entry open to exit close. Every price
in a path shares one adjustment basis, so a split inside it moves nothing; only the breakout level, stored with the
signal in the basis of the run that scored it, is rescaled to the path's basis.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from app.models import PathSession
from app.signal_paths import HORIZONS

# the round-trip cost, commission and slippage together, in basis points, taken off every gross return
COST_BPS = 20
# a breakout fails when any close in this many sessions after the signal session is below the level it cleared
FAILURE_SESSIONS = 10


@dataclass(frozen=True)
class HorizonOutcome:
    """
    A signal's outcome at one horizon. complete is True only once the horizon's exit session is in the path; an
    incomplete outcome carries no figures. Returns are percentages: the net return is the gross return less the
    round-trip cost, the excess return the net return less the benchmark's, None with the benchmark's when the path
    lacks the benchmark's entry or exit price. mfe_pct and mae_pct are the highest high and the lowest low from the
    entry session to the exit session, against the entry price.
    """

    horizon: int
    complete: bool
    entry_date: date | None = None
    entry_price: float | None = None
    exit_date: date | None = None
    exit_price: float | None = None
    cost_bps: int | None = None
    gross_return_pct: float | None = None
    net_return_pct: float | None = None
    benchmark_return_pct: float | None = None
    excess_return_pct: float | None = None
    mfe_pct: float | None = None
    mae_pct: float | None = None


@dataclass(frozen=True)
class BreakoutFailure:
    """
    Whether the breakout failed: level is the breakout level in the path's adjustment basis. failed is True, with the
    first session closing below the level, as soon as one of the FAILURE_SESSIONS sessions after the signal session
    does; False once all of them are in the path and none did; None while undecided.
    """

    level: float
    failed: bool | None
    failure_date: date | None


@dataclass(frozen=True)
class SignalOutcomes:
    """One outcome per horizon in HORIZONS, in that order, and the breakout failure."""

    horizons: list[HorizonOutcome]
    failure: BreakoutFailure


def breakout_level(high_20d: float, high_60d: float, breakout_60d: bool) -> float:
    """The level a breakout cleared: the prior 60-day high when the 60-day breakout passed, the prior 20-day high
    otherwise."""
    return high_60d if breakout_60d else high_20d


def compute_outcomes(
    path: Sequence[PathSession], signal_close: float, level: float, cost_bps: int = COST_BPS
) -> SignalOutcomes:
    """
    The outcomes of a signal from its path (index 0 the signal session, in index order with no gaps), its close on
    the signal session and its breakout level, both as the signal was archived with them, and the round-trip cost.

    The level is rescaled by the ratio of the path's signal session close to 'signal_close', so a split or dividend
    restated after the signal compares the closes with the level on one basis.

    When the entry session has no open, its close is the entry, and that session's high and low, which may have come
    before it, are left out of MFE and MAE; the benchmark's entry follows the same rule. A session missing its high
    or low counts its close instead, as analyze does.
    """
    _check(path)
    scale = path[0].close / signal_close
    return SignalOutcomes(
        horizons=[_horizon(path, h, cost_bps) for h in HORIZONS],
        failure=_failure(path, level * scale),
    )


def _check(path: Sequence[PathSession]) -> None:
    if not path:
        raise ValueError('a path holds at least the signal session')
    for i, s in enumerate(path):
        if s.session_index != i:
            raise ValueError(f'path session {i} has index {s.session_index}')


def _pct(start: float, end: float) -> float:
    return (end / start - 1) * 100


def _horizon(path: Sequence[PathSession], horizon: int, cost_bps: int) -> HorizonOutcome:
    if len(path) <= horizon:
        return HorizonOutcome(horizon=horizon, complete=False)
    entry, exit_ = path[1], path[horizon]
    entry_at_open = entry.open is not None
    entry_price = entry.open if entry.open is not None else entry.close

    held = path[1 : horizon + 1]
    highs = [s.high if s.high is not None else s.close for s in held]
    lows = [s.low if s.low is not None else s.close for s in held]
    if not entry_at_open:
        highs[0] = lows[0] = entry.close

    gross = _pct(entry_price, exit_.close)
    net = gross - cost_bps / 100
    bench_entry = entry.benchmark_open if entry.benchmark_open is not None else entry.benchmark_close
    benchmark = None
    if bench_entry is not None and exit_.benchmark_close is not None:
        benchmark = _pct(bench_entry, exit_.benchmark_close)
    return HorizonOutcome(
        horizon=horizon,
        complete=True,
        entry_date=entry.session_date,
        entry_price=entry_price,
        exit_date=exit_.session_date,
        exit_price=exit_.close,
        cost_bps=cost_bps,
        gross_return_pct=gross,
        net_return_pct=net,
        benchmark_return_pct=benchmark,
        excess_return_pct=None if benchmark is None else net - benchmark,
        mfe_pct=_pct(entry_price, max(highs)),
        mae_pct=_pct(entry_price, min(lows)),
    )


def _failure(path: Sequence[PathSession], level: float) -> BreakoutFailure:
    window = path[1 : FAILURE_SESSIONS + 1]
    below = next((s for s in window if s.close < level), None)
    if below is not None:
        return BreakoutFailure(level=level, failed=True, failure_date=below.session_date)
    if len(window) == FAILURE_SESSIONS:
        return BreakoutFailure(level=level, failed=False, failure_date=None)
    return BreakoutFailure(level=level, failed=None, failure_date=None)
