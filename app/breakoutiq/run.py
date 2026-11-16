"""
The whole BreakoutIQ analysis as one run: the dataset built once from the archive, every section run on it in order
(SECTIONS), and the run stored in analysis_run with every section's results in analysis_result, in one transaction,
so a reader never sees a run with only some of its results.

The run records the settings it ran with (settings()): every named constant a section reads, the indicators,
outcomes, splits and controls each section covers, and each section's payload version, so a stored result can always
be read against the methods that produced it. The run reads the signals of the screener's own version
(SCREENER_VERSION, stored with the run), with the regime of a backfilled signal from the price history.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

import pandas as pd

from app.analyzer import SCREENER_VERSION
from app.breakoutiq import baseline, candidates, confounders, indicators, mandatory, stats
from app.breakoutiq import dataset as data
from app.container import Container
from app.models import AnalysisResult, AnalysisRun
from app.provenance import git_commit
from app.signal_outcomes import COST_BPS, FAILURE_SESSIONS
from app.signal_paths import HORIZONS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Section:
    name: str
    payload_version: int
    analyse: Callable[[pd.DataFrame], list[AnalysisResult]]


# every section, in the order it runs and its results are stored
SECTIONS = (
    Section(baseline.SECTION, baseline.PAYLOAD_VERSION, baseline.baseline),
    Section(indicators.SECTION, indicators.PAYLOAD_VERSION, indicators.indicators),
    Section(mandatory.SECTION, mandatory.PAYLOAD_VERSION, mandatory.mandatory),
    Section(confounders.SECTION, confounders.PAYLOAD_VERSION, confounders.confounders),
    Section(candidates.SECTION, candidates.PAYLOAD_VERSION, candidates.candidates),
)


@dataclass
class AnalysisSummary:
    """
    What a run read and stored: the run's id, its data's extent (the signals with a resolved horizon), how many of
    them have no regime, and each section's result count and seconds.
    """

    run_id: int
    signals: int
    rows: int
    censored: int
    no_regime: int
    first_session: date | None
    last_session: date | None
    results: dict[str, int] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)


def settings() -> dict:
    """Every setting the run's results depend on, as stored in analysis_run.settings."""
    return {
        'horizons': list(HORIZONS),
        'cost_bps': COST_BPS,
        'failure_sessions': FAILURE_SESSIONS,
        'development_share': data.DEVELOPMENT_SHARE,
        'validation_share': data.VALIDATION_SHARE,
        'gap_sessions': data.GAP_SESSIONS,
        'regime_sma_days': data.REGIME_SMA_DAYS,
        'min_signal_dates': stats.MIN_SIGNAL_DATES,
        'min_blocks': stats.MIN_BLOCKS,
        'confidence': stats.CONFIDENCE,
        'bootstrap_resamples': stats.BOOTSTRAP_RESAMPLES,
        'bootstrap_seed': stats.BOOTSTRAP_SEED,
        'fdr': stats.FDR,
        'clusters': list(stats.CLUSTERS),
        'baseline_splits': list(baseline.SPLITS),
        'baseline_measures': list(baseline.MEASURES),
        'bands': indicators.BANDS,
        'main_outcome': indicators.MAIN_OUTCOME,
        'horizon_outcomes': list(indicators.HORIZON_OUTCOMES),
        'signal_outcomes': list(indicators.SIGNAL_OUTCOMES),
        'values': list(indicators.VALUES),
        'flags': list(indicators.FLAGS),
        'composites': list(indicators.COMPOSITES),
        'strengths': list(mandatory.STRENGTHS),
        'confounder_indicators': list(confounders.INDICATORS),
        'confounder_controls': list(confounders.CONTROLS),
        'min_rows_per_column': confounders.MIN_ROWS_PER_COLUMN,
        'max_missing_share': confounders.MAX_MISSING_SHARE,
        'redundant_correlation': confounders.REDUNDANT_CORRELATION,
        'redundant_vif': confounders.REDUNDANT_VIF,
        'min_net_excess_return': {str(h): bar for h, bar in candidates.MIN_NET_EXCESS_RETURN.items()},
        'min_kept_share': candidates.MIN_KEPT_SHARE,
        'max_candidates': candidates.MAX_CANDIDATES,
        'pair_pool': candidates.PAIR_POOL,
        'payload_versions': {s.name: s.payload_version for s in SECTIONS},
    }


def analyse(rows: pd.DataFrame, seconds: dict[str, float] | None = None) -> list[AnalysisResult]:
    """
    Every section's results from the analysis dataset (dataset.build_dataset), SECTIONS in order; none when the
    dataset is empty. Each section's run time goes into 'seconds' when given.
    """
    if rows.empty:
        return []
    results = []
    for section in SECTIONS:
        start = time.monotonic()
        found = section.analyse(rows)
        elapsed = time.monotonic() - start
        if seconds is not None:
            seconds[section.name] = elapsed
        logger.info('%s: %d results in %.0fs', section.name, len(found), elapsed)
        results += found
    return results


class Analysis:
    """The work behind the breakoutiq-analysis command."""

    def __init__(self, container: Container):
        self.container = container

    def run(self) -> AnalysisSummary:
        """
        Builds the dataset from the archive, the regime of a backfilled signal read from the price history
        (MissingHistory without it), runs every section and stores the run with its results.
        """
        started_at = datetime.now(UTC)
        rows = data.load_dataset(self.container, with_history=True)
        seconds: dict[str, float] = {}
        results = analyse(rows, seconds)
        signals = rows.drop_duplicates('signal_id')
        first = signals['session_date'].min().date() if len(signals) else None
        last = signals['session_date'].max().date() if len(signals) else None
        run = AnalysisRun(
            started_at=started_at,
            finished_at=datetime.now(UTC),
            first_session=first,
            last_session=last,
            signals=len(signals),
            settings=settings(),
            screener_version=SCREENER_VERSION,
            git_commit=git_commit(),
        )
        run_id = self.container.analysis_repo.add(run, results)
        counts = {s.name: sum(1 for r in results if r.section == s.name) for s in SECTIONS}
        return AnalysisSummary(
            run_id=run_id,
            signals=len(signals),
            rows=len(rows),
            censored=int(rows['censored'].sum()) if len(rows) else 0,
            no_regime=int(signals['regime'].isna().sum()) if len(signals) else 0,
            first_session=first,
            last_session=last,
            results=counts,
            seconds=seconds,
        )


def summarize(summary: AnalysisSummary) -> None:
    """Prints what a run read and stored."""
    extent = f' ({summary.first_session} to {summary.last_session})' if summary.first_session else ''
    print(
        f'breakoutiq-analysis: read {summary.signals} signal{"" if summary.signals == 1 else "s"} with a resolved '
        f'horizon{extent}, {summary.rows} resolved horizon{"" if summary.rows == 1 else "s"}, {summary.censored} of '
        f'them censored, {summary.no_regime} signal{"" if summary.no_regime == 1 else "s"} with no regime'
    )
    for name, count in summary.results.items():
        took = f' in {summary.seconds[name]:.0f}s' if name in summary.seconds else ''
        print(f'breakoutiq-analysis: {name}: {count} result{"" if count == 1 else "s"}{took}')
    print(f'breakoutiq-analysis: stored run {summary.run_id}')
