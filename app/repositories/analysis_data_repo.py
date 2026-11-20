from dataclasses import dataclass

import pandas as pd

from app.container import Container
from app.db import connect
from app.models import SIGNAL_KIND

# the signal columns that say how a row was produced rather than what the market did, and those every signal has
# alike (its kind, and no breakout of its own to link to): the analysis leaves them out
_PROVENANCE = ('time_zone', 'rule_settings', 'git_commit', 'run_id', 'archived_at', 'kind', 'breakout_id')


@dataclass(frozen=True)
class Archive:
    """
    What the analysis reads of one screener version's signals, the archive's records of kind signal alone:
    - signals: every archived signal, oldest session first, each column of signal but those in _PROVENANCE, with its
      id as signal_id, the MACD signal line (signal.signal) as macd_signal, and breakout_failed from signal_track (1,
      0, or None while undecided or without a track)
    - outcomes: every resolved outcome, one row per signal and horizon: complete, with its figures, or ended
      (censored), with its reason and last session. Pending outcomes are left out
    - path_sessions: every session in a signal's price path, once per market: columns market and session_date
    """

    signals: pd.DataFrame
    outcomes: pd.DataFrame
    path_sessions: pd.DataFrame


class AnalysisDataRepo:
    """
    What BreakoutIQ's analysis reads from the signal archive and its tracking tables, as data frames. It only reads.
    """

    def __init__(self, container: Container):
        self.db_path = container.db_path

    def archive(self, screener_version: str) -> Archive:
        """
        The signals archived by 'screener_version', with their outcomes and path sessions, read in one transaction,
        so a run that commits meanwhile cannot leave the three out of step. A signal archived by several versions
        has one row per version, each with that version's values, so the analysis reads one version only.
        """
        with connect(self.db_path) as con:
            con.execute('BEGIN')  # one read snapshot for the three queries
            signals = pd.read_sql_query(
                'SELECT s.*, t.breakout_failed FROM signal s LEFT JOIN signal_track t ON t.signal_id = s.id '
                'WHERE s.screener_version = ? AND s.kind = ? ORDER BY s.session_date, s.id',
                con,
                params=(screener_version, SIGNAL_KIND),
            )
            outcomes = pd.read_sql_query(
                'SELECT o.signal_id, o.horizon, o.status AS outcome_status, o.entry_date, o.exit_date, '
                'o.gross_return_pct, o.net_return_pct, o.benchmark_return_pct, o.excess_return_pct, o.mfe_pct, '
                'o.mae_pct, o.ended_reason, o.last_session_date FROM signal_outcome o '
                'JOIN signal s ON s.id = o.signal_id '
                "WHERE s.screener_version = ? AND s.kind = ? AND o.status IN ('complete', 'ended') "
                'ORDER BY o.signal_id, o.horizon',
                con,
                params=(screener_version, SIGNAL_KIND),
            )
            path_sessions = pd.read_sql_query(
                'SELECT DISTINCT s.market, p.session_date FROM signal_price p JOIN signal s ON s.id = p.signal_id '
                'WHERE s.screener_version = ? AND s.kind = ?',
                con,
                params=(screener_version, SIGNAL_KIND),
            )
        signals = signals.drop(columns=list(_PROVENANCE)).rename(columns={'id': 'signal_id', 'signal': 'macd_signal'})
        return Archive(signals, outcomes, path_sessions)
