"""Fixed-horizon, two-arm, independent-user binary-outcome MVP."""
from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class Business:
    monthly_users: int = 200000
    reach: float = .8
    margin: float = 12.
    implementation: float = 25000.
    maintenance: float = 0.
    months: int = 6
    persistence: float = 1.
    margin_uncertainty: float = .2

    def validate(self):
        values = list(asdict(self).values())
        if not all(np.isfinite(v) for v in values):
            raise ValueError('Assumptions must be finite.')
        if self.monthly_users <= 0 or self.margin <= 0 or self.months < 1:
            raise ValueError('Population, margin and horizon must be positive.')
        if not 0 < self.reach <= 1 or not 0 < self.persistence <= 1:
            raise ValueError('Reach and persistence must be in (0, 1].')
        if min(self.implementation, self.maintenance, self.margin_uncertainty) < 0:
            raise ValueError('Costs and uncertainty cannot be negative.')


def demo(scenario='A', seed=42):
    rng = np.random.default_rng(seed)
    n = 40000
    t = rng.random(n) < (.65 if scenario == 'B' else .5)
    os = rng.choice(['Android', 'iOS', 'Web'], n)
    effect = {'A': .006, 'B': .012, 'C': .001}[scenario]
    return pd.DataFrame({'unit_id': np.arange(n).astype(str),
                         'variant': np.where(t, 'treatment', 'control'),
                         'converted_7d': rng.binomial(1, .08 + t * effect),
                         'crashes_7d': rng.binomial(1, .02 + t * .0005),
                         'os': os})


def validate(df, expected=.5):
    required = {'unit_id', 'variant', 'converted_7d'}
    errors, warnings = [], []
    if not required <= set(df.columns):
        return {'errors': [f'Missing columns: {sorted(required - set(df.columns))}'], 'warnings': [], 'srm_p': None}
    if not 0 < expected < 1:
        raise ValueError('Expected allocation must be in (0,1).')
    if df.unit_id.isna().any() or df.unit_id.duplicated().any():
        errors.append('Missing or duplicate IDs: one row per randomized unit is required.')
    if set(df.variant.dropna()) != {'control', 'treatment'} or df.variant.isna().any():
        errors.append('Exactly two arms, control and treatment, are required.')
    for name in ['converted_7d', 'crashes_7d']:
        if name in df and (df[name].isna().any() or not df[name].isin([0, 1]).all()):
            errors.append(f'{name}: complete binary values (0/1) are required; missing is not zero.')
    counts = df.variant.value_counts().reindex(['control', 'treatment'], fill_value=0)
    # Invalid arm labels must not make scipy raise before diagnostics can be shown.
    p = (float(stats.chisquare(counts, np.array([1-expected, expected])*len(df)).pvalue)
         if len(df) and counts.sum() == len(df) else None)
    if p is not None and p < .001:
        errors.append('Sample ratio mismatch (SRM p < 0.001): investigate assignment before deciding.')
    if min(counts) < 30:
        errors.append('This approximation requires at least 30 units per arm.')
    warnings.append('Design assumptions: complete assignments, independent users, mature outcome windows and a fixed stopping rule. Outcome columns alone cannot verify these.')
    return {'errors': errors, 'warnings': warnings, 'srm_p': p, 'counts': counts.to_dict()}


def estimate(df, metric='converted_7d', sesoi=.002):
    c = df.loc[df.variant == 'control', metric].to_numpy(dtype=float)
    t = df.loc[df.variant == 'treatment', metric].to_numpy(dtype=float)
    if min(len(c), len(t)) < 30:
        raise ValueError('Insufficient sample for the binary approximation.')
    pc, pt = c.mean(), t.mean()
    effect = pt - pc
    se = np.sqrt(pc*(1-pc)/len(c) + pt*(1-pt)/len(t))
    # Newcombe interval from independent Wilson score intervals: remains nondegenerate at boundaries.
    def wilson(p, n):
        z = stats.norm.ppf(.975)
        center = (p + z*z/(2*n))/(1+z*z/n)
        half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
        return center-half, center+half
    lc, uc = wilson(pc, len(c))
    lt, ut = wilson(pt, len(t))
    low = effect - np.sqrt((pt-lt)**2+(uc-pc)**2)
    high = effect + np.sqrt((ut-pt)**2+(pc-lc)**2)
    pooled = (c.sum()+t.sum())/(len(c)+len(t))
    test_se = np.sqrt(pooled*(1-pooled)*(1/len(c)+1/len(t)))
    p = float(2*stats.norm.sf(abs(effect/test_se))) if test_se else 1.
    return {'metric': metric, 'n_control': len(c), 'n_treatment': len(t), 'control': pc,
            'treatment': pt, 'effect': effect, 'se': se, 'ci_low': low, 'ci_high': high,
            'p_value': p, 'sesoi': sesoi,
            'practically_equivalent': bool(low > -sesoi and high < sesoi)}


def adjust(pvalues, method='holm'):
    p = np.asarray(pvalues, dtype=float)
    if not len(p):
        return p
    order = np.argsort(p)
    sorted_p = p[order]
    n = len(p)
    adjusted = (np.maximum.accumulate(sorted_p*(n-np.arange(n))) if method == 'holm'
                else np.minimum.accumulate((sorted_p*n/np.arange(1,n+1))[::-1])[::-1])
    result = np.empty(n)
    result[order] = np.minimum(adjusted, 1.)
    return result


def segments(df, columns, method='holm'):
    rows = []
    for column in columns:
        for value, group in df.groupby(column, dropna=False):
            if min(group.variant.value_counts().reindex(['control','treatment'], fill_value=0)) < 30:
                continue
            row = estimate(group)
            row.update(segment=column, level=str(value))
            rows.append(row)
    result = pd.DataFrame(rows)
    if len(result):
        result['p_adjusted'] = adjust(result.p_value, method)
    return result


def posterior(df, prior=.5, draws=20000, seed=17):
    rng = np.random.default_rng(seed)
    samples = []
    for arm in ['control', 'treatment']:
        y = df.loc[df.variant == arm, 'converted_7d']
        samples.append(rng.beta(y.sum()+prior, len(y)-y.sum()+prior, draws))
    return samples[1]-samples[0]


def simulate(effects, business, seed=23):
    business.validate()
    rng = np.random.default_rng(seed)
    # Lognormal marginal contribution; CV is explicit, independent of experimental effect.
    sigma = np.sqrt(np.log1p(business.margin_uncertainty**2))
    margin = rng.lognormal(np.log(business.margin)-sigma*sigma/2, sigma, len(effects))
    exposure = sum(business.persistence**month for month in range(business.months))
    scale = business.monthly_users*business.reach*exposure
    costs = business.implementation + business.maintenance*business.months
    value = effects*scale*margin-costs
    return value, {'expected_value': float(value.mean()), 'probability_loss': float((value<0).mean()),
                   'p05': float(np.quantile(value,.05)), 'p95': float(np.quantile(value,.95)),
                   'break_even_effect': costs/(scale*business.margin),
                   'evpi': float(np.maximum(value,0).mean()-max(0,value.mean()))}


def decision(diagnostics, primary, guardrail, economics, risk_tolerance=.1):
    if diagnostics['errors']:
        return 'Investigate', 'Validity issues block a launch recommendation.'
    if guardrail is None:
        return 'Investigate', 'The per-user binary crash guardrail is missing.'
    if guardrail['ci_high'] >= .01:
        return 'Continue testing', 'A guardrail deterioration of 1 percentage point has not been ruled out.'
    if primary['ci_high'] < primary['sesoi']:
        return 'Iterate', 'The confidence interval falls below the minimum relevant effect.'
    if economics['expected_value'] > 0 and economics['probability_loss'] <= risk_tolerance:
        return 'Ship · conditional', 'Meets the economic policy under the declared assumptions.'
    return 'Continue testing', 'Expected return or downside risk does not meet the current policy.'
