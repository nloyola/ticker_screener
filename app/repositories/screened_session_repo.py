from __future__ import annotations

from collections.abc import Iterable
from datetime import date

from app.container import Container
from app.db import connect
from app.models import ScreenedSession


class ScreenedSessionRepo:
    """The sessions each market was scored on, one row per market and session date."""

    def __init__(self, container: Container):
        self.db_path = container.db_path

    def record(self, sessions: Iterable[ScreenedSession]) -> None:
        """Records each session, replacing the row of a session scored before: the last run to score it wins."""
        with connect(self.db_path) as con:
            con.executemany(
                """
                INSERT INTO screened_session
                    (market, session_date, screener_version, rule_settings, git_commit, run_id, tickers_scored)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(market, session_date) DO UPDATE SET
                    screener_version = excluded.screener_version,
                    rule_settings = excluded.rule_settings,
                    git_commit = excluded.git_commit,
                    run_id = excluded.run_id,
                    tickers_scored = excluded.tickers_scored,
                    screened_at = datetime('now')
                """,
                [
                    (
                        s.market,
                        s.session_date.isoformat(),
                        s.screener_version,
                        s.rule_settings,
                        s.git_commit,
                        s.run_id,
                        s.tickers_scored,
                    )
                    for s in sessions
                ],
            )

    def all(self) -> list[ScreenedSession]:
        """Every recorded session, by market and then date."""
        with connect(self.db_path) as con:
            rows = con.execute(
                'SELECT market, session_date, screener_version, rule_settings, git_commit, run_id, tickers_scored '
                'FROM screened_session ORDER BY market, session_date'
            ).fetchall()
        return [
            ScreenedSession(
                market=row['market'],
                session_date=date.fromisoformat(row['session_date']),
                screener_version=row['screener_version'],
                rule_settings=row['rule_settings'],
                git_commit=row['git_commit'],
                run_id=row['run_id'],
                tickers_scored=row['tickers_scored'],
            )
            for row in rows
        ]
