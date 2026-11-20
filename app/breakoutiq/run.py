"""
The whole BreakoutIQ analysis as one run: the dataset built once from the archive, every section run on it in order
(SECTIONS), the strength test with the near-miss dataset beside it, and the run stored in analysis_run with every
section's results in analysis_result, in one transaction, so a reader never sees a run with only some of its results.

The run records the settings it ran with (settings()): its thresholds (thresholds.py, the named defaults unless the
command was told otherwise), every other named constant a section reads, the indicators, outcomes, splits and
controls each section covers, and each section's payload version, so a stored result can always be read against the
methods that produced it. The run reads the records of one screener version, the newest that archived a record
unless told another (dataset.analysed_version), and stores that version with it, so records scored by different rules
never mix in a result; the regime of a backfilled signal comes from the price history.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

import pandas as pd

from app.breakoutiq import baseline, candidates, confounders, indicators, mandatory, predictive, stats, walkforward
from app.breakoutiq import dataset as data
from app.breakoutiq.thresholds import DEFAULTS, Thresholds
from app.container import Container
from app.models import AnalysisResult, AnalysisRun
from app.provenance import git_commit
from app.signal_outcomes import COST_BPS, FAILURE_SESSIONS
from app.signal_paths import HORIZONS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Section:
    """
    A section: its name, payload version and analysis, which takes the dataset and the thresholds, and, when
    'reads_earlier', the results of the sections before it in the run, or, when 'reads_near_misses', the near-miss
    dataset.
    """

    name: str
    payload_version: int
    analyse: Callable[..., list[AnalysisResult]]
    reads_earlier: bool = False
    reads_near_misses: bool = False


# every section, in the order it runs and its results are stored; the models read the rules candidates validated
SECTIONS = (
    Section(baseline.SECTION, baseline.PAYLOAD_VERSION, baseline.baseline),
    Section(indicators.SECTION, indicators.PAYLOAD_VERSION, indicators.indicators),
    Section(mandatory.SECTION, mandatory.PAYLOAD_VERSION, mandatory.mandatory, reads_near_misses=True),
    Section(confounders.SECTION, confounders.PAYLOAD_VERSION, confounders.confounders),
    Section(candidates.SECTION, candidates.PAYLOAD_VERSION, candidates.candidates),
    Section(predictive.SECTION, predictive.PAYLOAD_VERSION, predictive.predictive, reads_earlier=True),
)


@dataclass
class AnalysisSummary:
    """
    What a run read and stored: the run's id, the screener version it read, its data's extent (the signals with a
    resolved horizon), how many of them have no regime, the near-miss events with a resolved horizon, and each
    section's result count and seconds.
    """

    run_id: int
    screener_version: str
    signals: int
    near_misses: int
    rows: int
    censored: int
    no_regime: int
    first_session: date | None
    last_session: date | None
    results: dict[str, int] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)


def settings(thresholds: Thresholds = DEFAULTS) -> dict:
    """Every setting the run's results depend on, with 'thresholds', as stored in analysis_run.settings."""
    return {
        **thresholds.settings(),
        'horizons': list(HORIZONS),
        'cost_bps': COST_BPS,
        'failure_sessions': FAILURE_SESSIONS,
        'development_share': data.DEVELOPMENT_SHARE,
        'validation_share': data.VALIDATION_SHARE,
        'gap_sessions': data.GAP_SESSIONS,
        'regime_sma_days': data.REGIME_SMA_DAYS,
        'bootstrap_seed': stats.BOOTSTRAP_SEED,
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
        'near_miss_conditions': list(mandatory.CONDITIONS),
        'near_miss_horizons': list(mandatory.NEAR_MISS_HORIZONS),
        'confounder_indicators': list(confounders.INDICATORS),
        'confounder_controls': list(confounders.CONTROLS),
        'initial_train_share': walkforward.INITIAL_TRAIN_SHARE,
        'folds': walkforward.FOLDS,
        'top_share': walkforward.TOP_SHARE,
        'models': list(predictive.MODELS),
        'model_features': list(predictive.FEATURES),
        'model_categories': list(predictive.CATEGORIES),
        'model_l2_penalty': predictive.L2_PENALTY,
        'calibration_bins': predictive.CALIBRATION_BINS,
        'model_tree_iterations': predictive.TREE_ITERATIONS,
        'model_tree_max_depth': predictive.TREE_MAX_DEPTH,
        'model_tree_min_leaf': predictive.TREE_MIN_LEAF,
        'model_tree_learning_rate': predictive.TREE_LEARNING_RATE,
        'model_tree_l2_penalty': predictive.TREE_L2_PENALTY,
        'attribution_repeats': predictive.ATTRIBUTION_REPEATS,
        'attribution_seed': predictive.ATTRIBUTION_SEED,
        'payload_versions': {s.name: s.payload_version for s in SECTIONS},
    }


def analyse(
    rows: pd.DataFrame,
    thresholds: Thresholds = DEFAULTS,
    seconds: dict[str, float] | None = None,
    near_misses: pd.DataFrame | None = None,
) -> list[AnalysisResult]:
    """
    Every section's results from the analysis dataset (dataset.build_dataset), judged by 'thresholds', SECTIONS in
    order, the strength test comparing the near-miss dataset 'near_misses' (dataset.near_miss_dataset) with the
    signals when given; none when the analysis dataset is empty. Each section's run time goes into 'seconds' when
    given.
    """
    if rows.empty:
        return []
    results = []
    for section in SECTIONS:
        start = time.monotonic()
        if section.reads_earlier:
            found = section.analyse(rows, thresholds, list(results))
        elif section.reads_near_misses:
            found = section.analyse(rows, thresholds, near_misses)
        else:
            found = section.analyse(rows, thresholds)
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

    def run(self, thresholds: Thresholds = DEFAULTS, screener_version: str | None = None) -> AnalysisSummary:
        """
        Builds the dataset and the near-miss dataset from the records 'screener_version' archived (the newest version
        when None; UnknownVersion when it archived no record), the regime of a backfilled signal and the sessions of a
        backfilled near-miss read from the price history (MissingHistory without it), runs every section by
        'thresholds' and stores the run, with the version it read, and its results.
        """
        started_at = datetime.now(UTC)
        version = data.analysed_version(self.container, screener_version)
        rows, near_misses = data.load_datasets(self.container, with_history=True, screener_version=version)
        seconds: dict[str, float] = {}
        results = analyse(rows, thresholds, seconds, near_misses)
        signals = rows.drop_duplicates('signal_id')
        first = signals['session_date'].min().date() if len(signals) else None
        last = signals['session_date'].max().date() if len(signals) else None
        run = AnalysisRun(
            started_at=started_at,
            finished_at=datetime.now(UTC),
            first_session=first,
            last_session=last,
            signals=len(signals),
            settings=settings(thresholds),
            screener_version=version,
            git_commit=git_commit(),
        )
        run_id = self.container.analysis_repo.add(run, results)
        counts = {s.name: sum(1 for r in results if r.section == s.name) for s in SECTIONS}
        return AnalysisSummary(
            run_id=run_id,
            screener_version=version,
            signals=len(signals),
            near_misses=near_misses['signal_id'].nunique(),
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
        f'breakoutiq-analysis: read {summary.signals} signal{"" if summary.signals == 1 else "s"} of screener version '
        f'{summary.screener_version} with a resolved horizon{extent}, {summary.rows} resolved '
        f'horizon{"" if summary.rows == 1 else "s"}, {summary.censored} of them censored, {summary.no_regime} '
        f'signal{"" if summary.no_regime == 1 else "s"} with no regime, '
        f'{summary.near_misses} near-miss event{"" if summary.near_misses == 1 else "s"} with a resolved horizon'
    )
    for name, count in summary.results.items():
        took = f' in {summary.seconds[name]:.0f}s' if name in summary.seconds else ''
        print(f'breakoutiq-analysis: {name}: {count} result{"" if count == 1 else "s"}{took}')
    print(f'breakoutiq-analysis: stored run {summary.run_id}')
