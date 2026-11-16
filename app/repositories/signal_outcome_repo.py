from collections.abc import Iterable
from dataclasses import dataclass

from app.container import Container
from app.db import connect
from app.signal_outcomes import BreakoutFailure, HorizonOutcome
from app.signal_paths import HORIZONS


@dataclass(frozen=True)
class OutcomeSignal:
    """
    A signal with a path whose outcomes are not all decided, with what its archived row says about the breakout,
    the horizons it already has complete, those of them stored without the benchmark's return that its path now
    holds the benchmark prices for, and whether its breakout failure is decided.
    """

    id: int
    close: float
    high_20d: float | None
    high_60d: float | None
    breakout_60d: bool
    complete: frozenset[int]
    failure_decided: bool
    benchmark_due: frozenset[int] = frozenset()


class SignalOutcomeRepo:
    """Each signal's outcome per horizon (signal_outcome) and its breakout failure (signal_track)."""

    def __init__(self, container: Container):
        self.db_path = container.db_path

    def undecided(self) -> list[OutcomeSignal]:
        """
        Every signal with a path in signal_price that lacks a complete outcome at some horizon, whose breakout
        failure is not decided yet, or with a complete outcome stored without the benchmark's return whose entry and
        exit sessions now have the benchmark prices in the path, oldest session first.
        """
        with connect(self.db_path) as con:
            rows = con.execute(
                f"""
                WITH benchmark_due AS (
                    SELECT o.signal_id, group_concat(o.horizon) AS horizons
                    FROM signal_outcome o
                    JOIN signal_price pe ON pe.signal_id = o.signal_id AND pe.session_date = o.entry_date
                    JOIN signal_price px ON px.signal_id = o.signal_id AND px.session_date = o.exit_date
                    WHERE o.status = 'complete' AND o.benchmark_return_pct IS NULL
                      AND COALESCE(pe.benchmark_open, pe.benchmark_close) IS NOT NULL
                      AND px.benchmark_close IS NOT NULL
                    GROUP BY o.signal_id
                )
                SELECT s.id, s.close, s.high_20d, s.high_60d, s.breakout_60d, t.breakout_failed,
                       (SELECT group_concat(o.horizon) FROM signal_outcome o
                        WHERE o.signal_id = s.id AND o.status = 'complete') AS complete,
                       d.horizons AS benchmark_due
                FROM signal s
                JOIN signal_track t ON t.signal_id = s.id
                LEFT JOIN benchmark_due d ON d.signal_id = s.id
                WHERE EXISTS (SELECT 1 FROM signal_price p WHERE p.signal_id = s.id)
                  AND (t.breakout_failed IS NULL
                       OR d.signal_id IS NOT NULL
                       OR (SELECT COUNT(*) FROM signal_outcome o
                           WHERE o.signal_id = s.id AND o.status = 'complete') < {len(HORIZONS)})
                ORDER BY s.session_date, s.id
                """
            ).fetchall()
        return [
            OutcomeSignal(
                id=row['id'],
                close=row['close'],
                high_20d=row['high_20d'],
                high_60d=row['high_60d'],
                breakout_60d=bool(row['breakout_60d']),
                complete=_horizons(row['complete']),
                failure_decided=row['breakout_failed'] is not None,
                benchmark_due=_horizons(row['benchmark_due']),
            )
            for row in rows
        ]

    def store(
        self,
        signal_id: int,
        completed: Iterable[HorizonOutcome],
        failure: BreakoutFailure | None,
        benchmarks: Iterable[HorizonOutcome] = (),
    ) -> None:
        """
        In one transaction: stores each of 'completed', the signal's newly complete horizons, as complete, leaving a
        horizon already complete as it was; for each of 'benchmarks', a horizon stored complete without the
        benchmark's return, stores that return and the excess return over it, the stored net return less it, when
        its entry and exit sessions are the stored row's, and changes nothing else in the row; and records the
        breakout failure in signal_track while it is not decided: the level as it stands in the path's basis, and
        once 'failure' is decided, whether it failed and on which session. A decided failure is never changed.
        """
        rows = [
            (
                signal_id,
                o.horizon,
                o.entry_date.isoformat(),
                o.entry_price,
                o.exit_date.isoformat(),
                o.exit_price,
                o.cost_bps,
                o.gross_return_pct,
                o.net_return_pct,
                o.benchmark_return_pct,
                o.excess_return_pct,
                o.mfe_pct,
                o.mae_pct,
            )
            for o in completed
            if o.complete
        ]
        with connect(self.db_path) as con:
            con.executemany(
                """
                INSERT INTO signal_outcome (signal_id, horizon, status, entry_date, entry_price, exit_date, exit_price,
                    cost_bps, gross_return_pct, net_return_pct, benchmark_return_pct, excess_return_pct, mfe_pct,
                    mae_pct, completed_at)
                VALUES (?, ?, 'complete', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT (signal_id, horizon) DO UPDATE SET
                    status = 'complete', entry_date = excluded.entry_date, entry_price = excluded.entry_price,
                    exit_date = excluded.exit_date, exit_price = excluded.exit_price, cost_bps = excluded.cost_bps,
                    gross_return_pct = excluded.gross_return_pct, net_return_pct = excluded.net_return_pct,
                    benchmark_return_pct = excluded.benchmark_return_pct,
                    excess_return_pct = excluded.excess_return_pct, mfe_pct = excluded.mfe_pct,
                    mae_pct = excluded.mae_pct, ended_reason = NULL, last_session_date = NULL,
                    completed_at = excluded.completed_at, updated_at = datetime('now')
                WHERE signal_outcome.status <> 'complete'
                """,
                rows,
            )
            con.executemany(
                """
                UPDATE signal_outcome SET benchmark_return_pct = ?, excess_return_pct = net_return_pct - ?,
                    updated_at = datetime('now')
                WHERE signal_id = ? AND horizon = ? AND status = 'complete' AND benchmark_return_pct IS NULL
                  AND entry_date = ? AND exit_date = ?
                """,
                [
                    (
                        o.benchmark_return_pct,
                        o.benchmark_return_pct,
                        signal_id,
                        o.horizon,
                        o.entry_date.isoformat(),
                        o.exit_date.isoformat(),
                    )
                    for o in benchmarks
                    if o.complete and o.benchmark_return_pct is not None
                ],
            )
            if failure is None:
                return
            con.execute(
                """
                UPDATE signal_track SET breakout_level = ?, breakout_failed = ?, failure_date = ?,
                    failure_decided_at = CASE WHEN ? IS NULL THEN NULL ELSE datetime('now') END
                WHERE signal_id = ? AND breakout_failed IS NULL
                """,
                (
                    failure.level,
                    None if failure.failed is None else int(failure.failed),
                    failure.failure_date.isoformat() if failure.failure_date else None,
                    failure.failed,
                    signal_id,
                ),
            )


def _horizons(listed: str | None) -> frozenset[int]:
    """The horizons in a group_concat list, empty for NULL."""
    return frozenset(int(h) for h in listed.split(',')) if listed else frozenset()
