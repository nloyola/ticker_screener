"""
The indicators' effects controlled for confounders and for each other: whether an indicator still predicts the main
outcome once the others, the sector, the market and the regime are held fixed, and which indicators only repeat
what another already says.

At each horizon the net excess return is regressed on every secondary indicator's raw value and every mandatory
strength together (INDICATORS), each standardized so its effect is per standard deviation, with sector and market
fixed effects and the regime, and with errors clustered two-way by signal date and ticker (stats.clustered_ols). Each
indicator's adjusted effect is reported beside its unadjusted one, the slope of the outcome on it alone over the same
rows, so a reader sees how much of an effect the others explain. Every adjusted effect of a horizon is one family,
corrected by Benjamini-Hochberg at the run's fdr. The thresholds named here are the run's (thresholds.py).

Redundancy is read from the correlations between the indicators, with each one's variance inflation factor (VIF)
reported beside them. Indicators correlated at least redundant_correlation say the same thing, and their adjusted
effects share it: collinearity widens both errors, so neither may be significant alone though together they matter.
Each indicator with correlated ones is therefore also tested jointly with them (a Wald test of their effects all
being zero, with the clustered covariance), so an effect they share is not lost. An indicator whose adjusted effect is
significant is never redundant; one that is not is redundant with each correlated indicator that is significant, or,
when none is, with the one among them whose unadjusted effect has the larger |t|: that one is credited, the rest
named redundant. A high VIF alone names nothing redundant.

Each indicator's effect is also compared across sectors and regimes: the same regression fitted on each sector's and
each regime's rows alone, with the indicators on the pooled scale so the subgroup effects compare with the pooled one.
The subgroup effects are descriptive, with intervals and no tests.

A regression is fitted only on min_signal_dates distinct signal dates and min_rows_per_column rows per column of its
design; a model or subgroup not fitted says why (its 'reason').

Like the ranking, the section describes the whole archive, gap included, and chooses no rule. The regression reads
complete outcomes only; ended (censored) horizons are counted, never dropped silently, and so are complete outcomes
with no excess return (no_outcome).
"""

import numpy as np
import pandas as pd

from app.breakoutiq.dataset import COMPLETE, ENDED
from app.breakoutiq.indicators import MAIN_OUTCOME, OUTCOMES, VALUE, VALUES, correct, rank, value_frame
from app.breakoutiq.mandatory import STRENGTH, STRENGTHS
from app.breakoutiq.stats import CLUSTERS, Coefficient, cluster_codes, clustered_fit, clustered_ols
from app.breakoutiq.thresholds import DEFAULTS, Thresholds
from app.models import AnalysisResult

SECTION = 'confounders'
PAYLOAD_VERSION = 1

# the findings: the horizon's regression as a whole, and each indicator's adjusted effect ('adjusted:<indicator>')
MODEL = 'model'
ADJUSTED = 'adjusted'

# the indicators regressed together, each with its kind: every secondary indicator's raw value and every mandatory
# strength. The flags are thresholds of these same values, so they are left out of the regression, and so is
# secondary_passed, a count of those flags
INDICATORS = {
    **dict.fromkeys((v for v in VALUES if v != 'secondary_passed'), VALUE),
    **dict.fromkeys(STRENGTHS, STRENGTH),
}

# the confounders held fixed, each a fixed effect per level, the first level (in sorted order) the baseline
CONTROLS = ('market', 'sector', 'regime')
# the subgroups each indicator's effect is compared across
SPLITS = ('sector', 'regime')
# the level of a control a row has no value for
UNKNOWN = 'unknown'

# why an indicator is left out of a horizon's regression
MISSING = 'missing'
CONSTANT = 'constant'
COLLINEAR = 'collinear'

# why a model or subgroup was not fitted
NO_OUTCOMES = 'no complete outcomes'
TOO_FEW_DATES = 'too few signal dates'
TOO_FEW_ROWS = 'too few rows'
NOT_CLUSTERED = 'errors could not be clustered'
NO_INDICATOR = 'no indicator to regress'

# a column whose part not explained by the columns before it is below this share of its length is collinear with them
_COLLINEAR_TOLERANCE = 1e-6


def confounders(rows: pd.DataFrame, thresholds: Thresholds = DEFAULTS) -> list[AnalysisResult]:
    """
    The section's results from the analysis dataset (dataset.build_dataset): per horizon, the MODEL finding, then one
    'adjusted:<indicator>' finding per indicator of INDICATORS, in order, including those left out of the regression.
    """
    frame = value_frame(rows)
    results = []
    for horizon, at_horizon in frame.groupby('horizon', sort=True):
        model, payloads = _horizon(at_horizon.reset_index(drop=True), thresholds)
        results.append(AnalysisResult(SECTION, MODEL, int(horizon), PAYLOAD_VERSION, model))
        results += [
            AnalysisResult(SECTION, f'{ADJUSTED}:{p["indicator"]}', int(horizon), PAYLOAD_VERSION, p) for p in payloads
        ]
    return results


def _horizon(rows: pd.DataFrame, thresholds: Thresholds) -> tuple[dict, list[dict]]:
    """One horizon's model payload and its indicators' payloads."""
    y = pd.to_numeric(rows[OUTCOMES[MAIN_OUTCOME].column], errors='coerce').to_numpy(dtype=float)
    complete = (rows['outcome_status'] == COMPLETE).to_numpy()
    observed = complete & ~np.isnan(y)
    values = rows[list(INDICATORS)].apply(pd.to_numeric, errors='coerce').astype(float)
    controls = pd.DataFrame({c: _levels(rows[c]) for c in CONTROLS})

    missing_share = {
        name: float(values.loc[observed, name].isna().mean()) if observed.any() else None for name in INDICATORS
    }
    left_out = {name: MISSING for name in INDICATORS if (missing_share[name] or 0) > thresholds.max_missing_share}
    candidates = [name for name in INDICATORS if name not in left_out]
    used = observed & values[candidates].notna().all(axis=1).to_numpy()

    fitted = rows[used].reset_index(drop=True)
    y_used = y[used]
    x = values[used].reset_index(drop=True)
    fixed = controls[used].reset_index(drop=True)
    dates = _distinct(fitted['session_date'])
    means, sds = x[candidates].mean(), x[candidates].std()

    reason, fit, regressed = None, None, []
    z = pd.DataFrame(index=fitted.index)
    if not observed.any():
        reason = NO_OUTCOMES
    elif dates < thresholds.min_signal_dates:
        reason = TOO_FEW_DATES
    else:
        for name in candidates:
            if sds[name] == 0:
                left_out[name] = CONSTANT
        scaled = [name for name in candidates if name not in left_out]
        z = (x[scaled] - means[scaled]) / sds[scaled]
        design = pd.concat([_fixed_effects(fixed, CONTROLS), z], axis=1)
        if len(design) < thresholds.min_rows_per_column * design.shape[1]:
            reason = TOO_FEW_ROWS
        else:
            kept = _independent(design)
            for name in scaled:
                if name not in kept:
                    left_out[name] = COLLINEAR
            regressed = [name for name in scaled if name in kept]
            if not regressed:
                reason = NO_INDICATOR
            else:
                fit = clustered_fit(
                    y_used, design[kept], cluster_codes(fitted), of=regressed, confidence=thresholds.confidence
                )
                reason = NOT_CLUSTERED if fit is None else None
    shown = fit is not None
    if not shown:
        regressed = []
    z = z[regressed]

    correlations, vif = _redundancy(z)
    model = {
        'outcome': MAIN_OUTCOME,
        'shown': shown,
        'reason': reason,
        'n': int(used.sum()),
        'signals': _distinct(fitted['signal_id']),
        'signal_dates': dates,
        'censored': int((rows['outcome_status'] == ENDED).sum()),
        'no_outcome': int((complete & np.isnan(y)).sum()),
        'missing': int((observed & ~used).sum()),
        'missing_share': missing_share,
        'indicators': regressed,
        'left_out': {name: left_out[name] for name in INDICATORS if name in left_out},
        'controls': {c: sorted(fixed[c].unique().tolist()) for c in CONTROLS},
        'correlations': {'indicators': regressed, 'matrix': correlations},
        'vif': vif,
        'scale': 'per standard deviation',
        'max_missing_share': thresholds.max_missing_share,
        'redundant_correlation': thresholds.redundant_correlation,
        'redundant_vif': thresholds.redundant_vif,
        'min_rows_per_column': thresholds.min_rows_per_column,
        'min_signal_dates': thresholds.min_signal_dates,
        'confidence': thresholds.confidence,
        'fdr': thresholds.fdr,
        'clusters': list(CLUSTERS),
    }

    subgroups = {split: _subgroups(fitted, y_used, z, fixed, split, thresholds) for split in SPLITS}
    payloads = []
    for name, kind in INDICATORS.items():
        payload = {
            'indicator': name,
            'kind': kind,
            'outcome': MAIN_OUTCOME,
            'main': True,
            'shown': False,
            'left_out': left_out.get(name),
            'scale': 'per standard deviation',
            'mean': _float(means.get(name)),
            'sd': _float(sds.get(name)),
            'n': model['n'],
            'signals': model['signals'],
            'signal_dates': dates,
            'controls': list(CONTROLS),
            'adjusted': None,
            'unadjusted': None,
            'p_value': None,
            'p_adjusted': None,
            'significant': False,
            'rank': None,
            'vif': vif.get(name),
            'high_vif': vif[name] >= thresholds.redundant_vif if name in vif else None,
            'correlated': [],
            'joint': None,
            'redundant': None,
            'redundant_with': [],
            'by_sector': [],
            'by_regime': [],
            'sign_agreement': {split: None for split in SPLITS},
            'min_signal_dates': thresholds.min_signal_dates,
            'confidence': thresholds.confidence,
            'fdr': thresholds.fdr,
            'clusters': list(CLUSTERS),
        }
        if shown and name in regressed:
            alone = clustered_ols(
                y_used,
                pd.DataFrame({'const': 1.0, name: z[name]}),
                cluster_codes(fitted),
                of=[name],
                confidence=thresholds.confidence,
            )
            payload['shown'] = True
            payload['adjusted'] = _effect(fit.coefficients[name])
            payload['unadjusted'] = _effect(alone[name]) if alone else None
            payload['p_value'] = fit.coefficients[name].p_value
            index = regressed.index(name)
            payload['correlated'] = [
                {'indicator': other, 'r': correlations[index][j]}
                for j, other in enumerate(regressed)
                if j != index and abs(correlations[index][j]) >= thresholds.redundant_correlation
            ]
            if payload['correlated']:
                block = [name, *(c['indicator'] for c in payload['correlated'])]
                payload['joint'] = {'indicators': block, 'p_value': fit.joint_p_value(block)}
            for split in SPLITS:
                payload[f'by_{split}'] = [
                    {**g['counts'], **g['effects'].get(name, _NO_EFFECT)} for g in subgroups[split]
                ]
                payload['sign_agreement'][split] = _sign_agreement(payload['adjusted'], payload[f'by_{split}'])
        payloads.append(payload)

    correct(payloads, thresholds.fdr)
    rank(payloads, effect='adjusted')
    _name_redundant(payloads)
    return model, payloads


_NO_EFFECT = {'estimate': None, 'se': None, 'ci_low': None, 'ci_high': None}


def _subgroups(
    rows: pd.DataFrame, y: np.ndarray, z: pd.DataFrame, fixed: pd.DataFrame, split: str, thresholds: Thresholds
) -> list[dict]:
    """
    The regression fitted on each level of 'split' alone, holding the other controls fixed: each level's counts
    and, when it meets the minimum sample, each indicator's effect there.
    """
    others = tuple(c for c in CONTROLS if c != split)
    groups = []
    for level in sorted(fixed[split].unique()):
        mask = (fixed[split] == level).to_numpy()
        at = rows[mask].reset_index(drop=True)
        dates = _distinct(at['session_date'])
        counts = {
            'group': level,
            'n': int(mask.sum()),
            'signals': _distinct(at['signal_id']),
            'signal_dates': dates,
            'shown': False,
            'reason': None,
        }
        effects = {}
        design = pd.concat(
            [_fixed_effects(fixed[mask].reset_index(drop=True), others), z[mask].reset_index(drop=True)], axis=1
        )
        if dates < thresholds.min_signal_dates:
            counts['reason'] = TOO_FEW_DATES
        elif len(design) < thresholds.min_rows_per_column * design.shape[1]:
            counts['reason'] = TOO_FEW_ROWS
        else:
            kept = _independent(design)
            reported = [name for name in z.columns if name in kept]
            fit = (
                clustered_ols(y[mask], design[kept], cluster_codes(at), of=reported, confidence=thresholds.confidence)
                if reported
                else None
            )
            if fit is None:
                counts['reason'] = NOT_CLUSTERED if reported else NO_INDICATOR
            else:
                counts['shown'] = True
                effects = {name: _effect(fit[name]) for name in reported}
        groups.append({'counts': counts, 'effects': effects})
    return groups


def _levels(values: pd.Series) -> pd.Series:
    """A control's values as text, UNKNOWN where a row has none."""
    return values.astype(object).where(values.notna(), UNKNOWN).astype(str)


def _fixed_effects(fixed: pd.DataFrame, controls: tuple[str, ...]) -> pd.DataFrame:
    """The constant, then a dummy per level of each of 'controls' but its first."""
    parts = [pd.DataFrame({'const': np.ones(len(fixed))})]
    parts += [pd.get_dummies(fixed[c], prefix=c, prefix_sep='=', drop_first=True, dtype=float) for c in controls]
    return pd.concat(parts, axis=1)


def _independent(design: pd.DataFrame) -> list[str]:
    """
    The columns of 'design' kept, in order, each one not a linear combination of those kept before it: a fixed effect
    no row varies on, or an indicator the fixed effects or the indicators before it fully explain, is dropped.
    """
    basis = np.empty((len(design), 0))
    kept = []
    for name in design.columns:
        column = design[name].to_numpy(dtype=float)
        length = np.linalg.norm(column)
        if length == 0:
            continue
        rest = column - basis @ (basis.T @ column)
        rest = rest - basis @ (basis.T @ rest)
        if np.linalg.norm(rest) <= _COLLINEAR_TOLERANCE * length:
            continue
        basis = np.column_stack([basis, rest / np.linalg.norm(rest)])
        kept.append(name)
    return kept


def _redundancy(z: pd.DataFrame) -> tuple[list[list[float]], dict[str, float]]:
    """The indicators' correlation matrix and each one's variance inflation factor, the diagonal of its inverse."""
    if z.shape[1] == 0 or len(z) < 3:
        return [], {}
    matrix = np.atleast_2d(np.corrcoef(z.to_numpy(dtype=float), rowvar=False))
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        inverse = np.linalg.pinv(matrix)
    vif = {name: float(inverse[i, i]) for i, name in enumerate(z.columns)}
    return [[float(v) for v in row] for row in matrix], vif


def _effect(coefficient: Coefficient) -> dict:
    return {
        'estimate': coefficient.estimate,
        'se': coefficient.se,
        'ci_low': coefficient.ci_low,
        'ci_high': coefficient.ci_high,
    }


def _sign_agreement(pooled: dict, groups: list[dict]) -> float | None:
    """The share of the subgroups with an effect whose estimate has the pooled effect's sign; None with none."""
    estimates = [g['estimate'] for g in groups if g['estimate'] is not None]
    if not estimates:
        return None
    return sum(np.sign(e) == np.sign(pooled['estimate']) for e in estimates) / len(estimates)


def _t(effect: dict | None) -> float:
    return abs(effect['estimate'] / effect['se']) if effect else 0.0


def _name_redundant(payloads: list[dict]) -> None:
    """
    Names each shown indicator redundant or not. One whose adjusted effect is significant never is. Otherwise it is
    redundant with each highly correlated indicator that is significant; when none is, with the one whose unadjusted
    effect has the larger |t|, if that is not itself. A high VIF alone names nothing redundant.
    """
    by_name = {p['indicator']: p for p in payloads if p['shown']}
    for payload in by_name.values():
        if payload['significant']:
            payload['redundant'] = False
            continue
        correlated = [by_name[c['indicator']] for c in payload['correlated']]
        significant = [p['indicator'] for p in correlated if p['significant']]
        if significant:
            payload['redundant_with'] = significant
        elif correlated:
            credited = max([payload, *correlated], key=lambda p: _t(p['unadjusted']))
            payload['redundant_with'] = [] if credited is payload else [credited['indicator']]
        payload['redundant'] = bool(payload['redundant_with'])


def _distinct(values: pd.Series) -> int:
    return int(values.nunique())


def _float(value) -> float | None:
    return float(value) if value is not None and np.isfinite(value) else None
