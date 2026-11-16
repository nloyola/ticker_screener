import pandas as pd

from app.container import Container
from app.db import connect

# the signal columns that say how a row was produced rather than what the market did: the analysis leaves them out
_PROVENANCE = ('time_zone', 'rule_settings', 'git_commit', 'run_id', 'archived_at')


class AnalysisDataRepo:
    """
    What BreakoutIQ's analysis reads from the signal archive and its tracking tables, as data frames. It only reads.
    """

    def __init__(self, container: Container):
        self.db_path = container.db_path

    def signals(self) -> pd.DataFrame:
        """
        Every archived signal, oldest session first: each column of signal but those in _PROVENANCE, with its id as
        signal_id, the MACD signal line (signal.signal) as macd_signal, and breakout_failed from signal_track (1, 0,
        or None while undecided or without a track).
        """
        with connect(self.db_path) as con:
            frame = pd.read_sql_query(
                'SELECT s.*, t.breakout_failed FROM signal s LEFT JOIN signal_track t ON t.signal_id = s.id '
                'ORDER BY s.session_date, s.id',
                con,
            )
        return frame.drop(columns=list(_PROVENANCE)).rename(columns={'id': 'signal_id', 'signal': 'macd_signal'})

    def outcomes(self) -> pd.DataFrame:
        """
        Every resolved outcome, one row per signal and horizon: complete, with its figures, or ended (censored), with
        its reason and last session. Pending outcomes are left out.
        """
        with connect(self.db_path) as con:
            return pd.read_sql_query(
                'SELECT signal_id, horizon, status AS outcome_status, entry_date, exit_date, gross_return_pct, '
                'net_return_pct, benchmark_return_pct, excess_return_pct, mfe_pct, mae_pct, ended_reason, '
                "last_session_date FROM signal_outcome WHERE status IN ('complete', 'ended') "
                'ORDER BY signal_id, horizon',
                con,
            )

    def path_sessions(self) -> pd.DataFrame:
        """Every session in a signal's price path, once per market: columns market and session_date."""
        with connect(self.db_path) as con:
            return pd.read_sql_query(
                'SELECT DISTINCT s.market, p.session_date FROM signal_price p JOIN signal s ON s.id = p.signal_id',
                con,
            )
