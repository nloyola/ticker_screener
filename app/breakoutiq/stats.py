"""
The statistics every BreakoutIQ section shares: the minimum sample a result needs to be shown, and confidence
intervals bootstrapped by blocks of signal dates.

Signals on the same day move together, and signals a few sessions apart share most of their outcome window, so
resampling single signals would overstate certainty. The bootstrap resamples whole blocks of consecutive sessions
instead, each block as long as the horizon, so signals whose windows overlap are drawn together.

A draw is how many times each block was drawn, and a figure's value on a draw is computed from per-block summaries
rather than by copying rows, so a draw costs the same for a hundred thousand signals as for a hundred.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

# a result needs complete outcomes on at least this many distinct signal dates to be shown at all
MIN_SIGNAL_DATES = 30
# the confidence level of every interval, and the bootstrap's resamples and seed (fixed, so a run is reproducible)
CONFIDENCE = 0.95
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 51
# the false discovery rate every family of tests (one per section and horizon) is corrected to, by Benjamini-Hochberg
FDR = 0.10
# the columns a regression's standard errors are clustered by, two-way: signals of one date share the market's
# move, and signals of one ticker share its own
CLUSTERS = ('session_date', 'ticker')

# the draws computed at once, which bounds the memory a figure's draws take
_CHUNK = 200
# the rank bins a median first finds its value's bin in, before finding the value among the bin's members
_MEDIAN_BINS = 1024

# a figure over draws: given the block counts of k draws (k x blocks), its value on each draw (k)
Figure = Callable[[np.ndarray], np.ndarray]


def date_blocks(dates: pd.Series, block_sessions: int) -> np.ndarray:
    """
    Each row's block, aligned with 'dates': consecutive runs of 'block_sessions' business days from the first date,
    numbered 0, 1, ... in date order with no number skipped. Rows on the same date are always in the same block.
    """
    days = pd.to_datetime(pd.Series(dates)).to_numpy().astype('datetime64[D]')
    if len(days) == 0:
        return np.array([], dtype=int)
    runs = np.busday_count(days.min(), days) // max(block_sessions, 1)
    return np.unique(runs, return_inverse=True)[1]


def block_bootstrap(
    blocks: np.ndarray, figures: list[Figure], resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED
) -> tuple[np.ndarray, np.ndarray]:
    """
    The CONFIDENCE percentile interval of each of 'figures', from 'resamples' draws of as many blocks as there are,
    with replacement. 'blocks' is each row's block (date_blocks). Each figure's bounds are NaN when fewer than two
    blocks exist or no draw gave it a value.
    """
    count = int(blocks.max()) + 1 if len(blocks) else 0
    low, high = np.full(len(figures), np.nan), np.full(len(figures), np.nan)
    if count < 2:
        return low, high
    rng = np.random.default_rng(seed)
    draws = np.empty((resamples, len(figures)))
    for start in range(0, resamples, _CHUNK):
        k = min(_CHUNK, resamples - start)
        chosen = rng.integers(count, size=(k, count))
        counts = np.zeros((k, count))
        np.add.at(counts, (np.repeat(np.arange(k), count), chosen.ravel()), 1)
        for i, figure in enumerate(figures):
            draws[start : start + k, i] = figure(counts)
    tail = (1 - CONFIDENCE) / 2 * 100
    valid = ~np.isnan(draws).all(axis=0)
    if valid.any():
        low[valid] = np.nanpercentile(draws[:, valid], tail, axis=0)
        high[valid] = np.nanpercentile(draws[:, valid], 100 - tail, axis=0)
    return low, high


def block_mean(values: np.ndarray, blocks: np.ndarray) -> Figure:
    """The mean of 'values' (NaN where a row has no observation) on each draw; NaN on a draw with no observation."""
    present = ~np.isnan(values)
    count = int(blocks.max()) + 1 if len(blocks) else 0
    sums = np.bincount(blocks[present], weights=values[present], minlength=count)
    sizes = np.bincount(blocks[present], minlength=count).astype(float)

    def figure(counts: np.ndarray) -> np.ndarray:
        total = counts @ sizes
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(total > 0, (counts @ sums) / total, np.nan)

    return figure


def block_median(values: np.ndarray, blocks: np.ndarray) -> Figure:
    """
    The median of 'values' (NaN where a row has no observation) on each draw, as np.median gives it on the drawn
    rows: the middle value, or the mean of the two middle ones. NaN on a draw with no observation.

    The observations are sorted once and cut into rank bins, with each block's count in each bin; a draw finds the
    bin holding its middle copy from those counts, then the value among that bin's members.
    """
    present = ~np.isnan(values)
    order = np.argsort(values[present], kind='stable')
    sorted_values = values[present][order]
    sorted_blocks = blocks[present][order]
    count = int(blocks.max()) + 1 if len(blocks) else 0
    n = len(sorted_values)
    if n == 0:
        return lambda counts: np.full(len(counts), np.nan)

    bins = min(_MEDIAN_BINS, n)
    bin_of = np.arange(n) * bins // n
    per_block = np.zeros((count, bins))
    np.add.at(per_block, (sorted_blocks, bin_of), 1)
    cumulative = np.cumsum(per_block, axis=1)
    # each bin's members, as positions in the sorted values, padded with n (a member of weight 0)
    starts = np.searchsorted(bin_of, np.arange(bins))
    width = int(np.diff(np.r_[starts, n]).max())
    members = starts[:, None] + np.arange(width)
    members = np.where(members < np.r_[starts[1:], n][:, None], members, n)
    member_blocks = np.r_[sorted_blocks, 0][members]
    padded_values = np.r_[sorted_values, np.nan]

    def copy_at(counts: np.ndarray, by_bin: np.ndarray, position: np.ndarray) -> np.ndarray:
        """The value of each draw's 'position'-th copy, counting from 1 in sorted order."""
        rows = np.arange(len(counts))
        bin_index = np.minimum((by_bin < position[:, None]).sum(axis=1), bins - 1)
        before = np.where(bin_index > 0, by_bin[rows, np.maximum(bin_index - 1, 0)], 0)
        bin_members = members[bin_index]
        weights = np.where(bin_members < n, counts[rows[:, None], member_blocks[bin_index]], 0)
        within = before[:, None] + np.cumsum(weights, axis=1)
        at = np.minimum((within < position[:, None]).sum(axis=1), width - 1)
        return padded_values[bin_members[rows, at]]

    def figure(counts: np.ndarray) -> np.ndarray:
        by_bin = counts @ cumulative
        total = by_bin[:, -1]
        lower = copy_at(counts, by_bin, np.maximum(np.ceil(total / 2), 1))
        upper = copy_at(counts, by_bin, np.floor(total / 2) + 1)
        return np.where(total > 0, (lower + upper) / 2, np.nan)

    return figure


@dataclass(frozen=True)
class Coefficient:
    """One regression coefficient: its estimate, standard error, CONFIDENCE interval and two-sided p-value."""

    estimate: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float


def cluster_codes(rows: pd.DataFrame) -> np.ndarray:
    """Each row's CLUSTERS as integer codes, one column per cluster (rows x 2), as clustered_ols takes them."""
    return np.column_stack([pd.factorize(rows[column])[0] for column in CLUSTERS])


def clustered_ols(y: np.ndarray, exog: pd.DataFrame, clusters: np.ndarray) -> dict[str, Coefficient] | None:
    """
    The least-squares fit of 'y' on the columns of 'exog' (which must include any constant), with standard errors
    clustered two-way by the two columns of 'clusters' (integer codes, cluster_codes), Cameron, Gelbach and Miller's
    sum of the two one-way covariances less the one by their intersection, with statsmodels' small-sample
    correction. Intervals and p-values use Student's t with one degree of freedom fewer than the smaller cluster
    count.

    Each column's Coefficient, or None when the fit cannot be clustered: fewer than two clusters on a side, or a
    variance that comes out not positive (the two-way sum is not always positive semi-definite).
    """
    codes = np.asarray(clusters)
    if min(len(np.unique(codes[:, i])) for i in range(2)) < 2:
        return None
    fit = sm.OLS(np.asarray(y, dtype=float), exog.astype(float)).fit(
        cov_type='cluster', cov_kwds={'groups': codes}, use_t=True
    )
    se = np.asarray(fit.bse, dtype=float)
    if not np.all(np.isfinite(se)) or np.any(se <= 0):
        return None
    interval = np.asarray(fit.conf_int(alpha=1 - CONFIDENCE), dtype=float)
    params, pvalues = np.asarray(fit.params, dtype=float), np.asarray(fit.pvalues, dtype=float)
    return {
        name: Coefficient(*(float(v) for v in (params[i], se[i], interval[i, 0], interval[i, 1], pvalues[i])))
        for i, name in enumerate(exog.columns)
    }


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """
    Each p-value adjusted by Benjamini-Hochberg over the family 'p_values'; a finding is significant at FDR when its
    adjusted value is at most FDR.
    """
    if not p_values:
        return []
    return [float(p) for p in multipletests(p_values, alpha=FDR, method='fdr_bh')[1]]
