"""Within-unit time contrasts and explicit target-population standardization."""
import numpy as np
from edi.inference import infer, effect_draws
from edi.engine import adjust


def temporal(df,early,late,metric,kind,denominator=None):
    if kind=='ratio':raise ValueError('Time profile requires comparable per-user outcomes, not an undecomposed ratio.')
    data=df.copy();data['_time_change']=data[late]-data[early]
    contrast=infer(data,'_time_change','numeric')
    early_result=infer(data,early,'numeric')
    late_result=infer(data,late,'numeric')
    return {'early_column':early,'late_column':late,'early':early_result,'late':late_result,'change_in_treatment_effect':contrast,
            'interpretation':'Welch test on each user’s late-minus-early difference retains within-user covariance. Requires equal-length, fully observed windows and comparable definitions. Timing/substitution and cohort effects can resemble novelty; this is not proof of decay or extrapolation to future months.'}


def cohort_effects(df,metric,kind,denominator=None):
    """Overall effects for Monday–Sunday assignment cohorts, never filtered by a segment."""
    import pandas as pd
    if 'assigned_at' not in df:return []
    assigned=pd.to_datetime(df.assigned_at,utc=True,errors='coerce')
    week=assigned.dt.normalize()-pd.to_timedelta(assigned.dt.weekday,unit='D')
    rows=[]
    for start,group in df.groupby(week,sort=True):
        if group.variant.value_counts().reindex(['control','treatment'],fill_value=0).min()<30:continue
        if kind=='ratio' and any(group.loc[group.variant==a,denominator].sum()<=0 for a in ['control','treatment']):continue
        end=start+pd.Timedelta(days=6)
        label=(start.strftime('%d %b')+' – '+end.strftime('%d %b %Y') if start.year==end.year else start.strftime('%d %b %Y')+' – '+end.strftime('%d %b %Y'))
        r=infer(group,metric,kind,denominator)
        r.update(cohort=start.strftime('%Y-%W'),week_start=start.date().isoformat(),week_end=end.date().isoformat(),week_label=label,
                 observed_assignment_start=assigned.loc[group.index].min().date().isoformat(),observed_assignment_end=assigned.loc[group.index].max().date().isoformat(),
                 population='All assigned users in this assignment week, across all segments')
        rows.append(r)
    for r,p in zip(rows,adjust([r['p_value'] for r in rows],'holm')):r['p_holm']=float(p)
    return rows


def transport(df,attribute,weights,metric,kind,denominator=None,prior_kwargs=None,beneficial_direction=1):
    if kind=='ratio':raise ValueError('Ratio transport needs target denominator weights; user-share weighting is not valid.')
    if any(not np.isfinite(v) or v<0 for v in weights.values()) or not np.isclose(sum(weights.values()),1.,atol=1e-6):
        raise ValueError('Target shares must be nonnegative, finite and sum to 1.')
    effects=np.zeros(20000);rows=[]
    for i,(level,weight) in enumerate(weights.items()):
        if weight==0:continue
        group=df.loc[df[attribute].astype(str)==str(level)]
        if group.variant.value_counts().reindex(['control','treatment'],fill_value=0).min()<30:
            raise ValueError(f'Insufficient overlap for target group {level}; at least 30 users per arm required.')
        result=infer(group,metric,kind)
        draws,model=effect_draws(group,result,seed=200+i,**(prior_kwargs or {}))
        effects+=weight*draws
        rows.append({'group':str(level),'sample_share':len(group)/len(df),'target_share':weight,
                     'effect':result['effect'],'ci_low':result['ci_low'],'ci_high':result['ci_high']})
    return effects,{'attribute':attribute,'groups':rows,'expected_effect':float(effects.mean()),
                    'credible_low':float(np.quantile(effects,.025)),'credible_high':float(np.quantile(effects,.975)),
                    'raw_probability_positive':float((effects>0).mean()),
                    'probability_benefit':float((beneficial_direction*effects>0).mean()),
                    'probability_harm':float((beneficial_direction*effects<0).mean()),
                    'beneficial_direction':'increase' if beneficial_direction==1 else 'decrease',
                    'assumptions':'Within-group transportability, valid target shares, overlap and unchanged metric definition. Does not correct unobserved effect modifiers or prove generalization.'}
