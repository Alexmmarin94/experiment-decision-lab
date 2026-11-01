"""Independent-unit binary, numeric and ratio inference; explicit posterior models."""
import numpy as np
import pandas as pd
from scipy import stats
from edi.engine import estimate, adjust


def detect_type(series):
    values=pd.to_numeric(series,errors='coerce').dropna()
    return 'binary' if len(values) and values.isin([0,1]).all() else 'numeric'


def arm_summary(group,metric,kind,denominator=None):
    y=group[metric].to_numpy(dtype=float)
    if kind=='ratio':
        d=group[denominator].to_numpy(dtype=float)
        if d.mean()<=0: raise ValueError('Ratio denominator has zero total in an arm or group.')
        value=y.sum()/d.sum()
        influence=(y-value*d)/d.mean()
        variance=float(influence.var(ddof=1)/len(y))
    else:
        value=float(y.mean())
        variance=float(y.var(ddof=1)/len(y))
    return value,variance,len(y)


def infer(df,metric,kind='numeric',denominator=None,sesoi=.002,alpha=.05):
    if kind=='binary':
        r=estimate(df,metric,sesoi)
        # Newcombe intervals with configurable confidence level (guardrail family control).
        c=df.loc[df.variant=='control',metric].to_numpy(float)
        t=df.loc[df.variant=='treatment',metric].to_numpy(float)
        def wilson(y):
            p=y.mean(); n=len(y); z=stats.norm.ppf(1-alpha/2)
            mid=(p+z*z/(2*n))/(1+z*z/n)
            half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
            return mid-half,mid+half
        lc,uc=wilson(c);lt,ut=wilson(t)
        r['ci_low']=float(r['effect']-np.hypot(t.mean()-lt,uc-c.mean()))
        r['ci_high']=float(r['effect']+np.hypot(ut-t.mean(),c.mean()-lc))
        pooled=(c.sum()+t.sum())/(len(c)+len(t))
        expected=np.array([len(c)*pooled,len(c)*(1-pooled),len(t)*pooled,len(t)*(1-pooled)])
        if expected.min()<5:
            _,p=stats.fisher_exact([[int(c.sum()),len(c)-int(c.sum())],[int(t.sum()),len(t)-int(t.sum())]])
            r['p_value']=float(p); method='Fisher exact test; Newcombe-Wilson interval'
        else: method='Pooled two-proportion z test; Newcombe-Wilson interval'
    else:
        groups=[df.loc[df.variant==arm] for arm in ['control','treatment']]
        (mc,vc,nc),(mt,vt,nt)=[arm_summary(g,metric,kind,denominator) for g in groups]
        se=np.sqrt(vc+vt)
        dfree=(vc+vt)**2/(vc*vc/(nc-1)+vt*vt/(nt-1)) if vc+vt>0 else np.inf
        effect=mt-mc
        p=float(2*stats.t.sf(abs(effect/se),dfree)) if se>0 else float(effect==0)
        half=stats.t.ppf(1-alpha/2,dfree)*se
        r={'metric':metric,'n_control':nc,'n_treatment':nt,'control':mc,'treatment':mt,
           'effect':effect,'se':float(se),'ci_low':float(effect-half),'ci_high':float(effect+half),'p_value':p,'sesoi':sesoi,'arm_variance_control':float(vc*nc),'arm_variance_treatment':float(vt*nt)}
        method='Welch unequal-variance t test' if kind=='numeric' else 'Ratio of sums; unit-level delta linearization; Welch-Satterthwaite interval/test'
    r.update(kind=kind,denominator=denominator,method=method,confidence_level=1-alpha,
             practically_equivalent=bool(r['ci_low']>-sesoi and r['ci_high']<sesoi))
    return r


def winsor_sensitivity(df,metric,sesoi):
    cutoff=float(df[metric].quantile(.99))
    capped=df.copy(); capped[metric]=capped[metric].clip(upper=cutoff)
    r=infer(capped,metric,'numeric',sesoi=sesoi)
    r.update(cutoff=cutoff,capped_rows=int((df[metric]>cutoff).sum()),
             interpretation='Exploratory capped-mean estimand; shared pooled cutoff, not arm-specific. Interval conditions on estimated cutoff; not full cutoff uncertainty.')
    return r


def effect_draws(df,result,prior_strategy='weak',historical_rate=.08,prior_strength=20.,prior_scale=.01,draws=20000,seed=17,beneficial_direction=1,relative_goal_pct=None):
    rng=np.random.default_rng(seed)
    if result['kind']=='binary':
        a,b=(.5,.5) if prior_strategy=='weak' else (historical_rate*prior_strength,(1-historical_rate)*prior_strength)
        samples=[];params=[]
        for arm in ['control','treatment']:
            y=df.loc[df.variant==arm,result['metric']]
            pa,pb=float(y.sum()+a),float(len(y)-y.sum()+b)
            params.append((pa,pb)); samples.append(rng.beta(pa,pb,draws))
        delta=samples[1]-samples[0]
        model={'name':'Independent beta-binomial','prior':{'alpha':a,'beta':b,'source':'Jeffreys weak prior' if prior_strategy=='weak' else 'User-declared independent historical evidence'},
               'arm_posterior_beta':params,'effect_mean':params[1][0]/sum(params[1])-params[0][0]/sum(params[0])}
    else:
        # Normal likelihood with plug-in sampling variance; explicitly an approximation.
        if prior_scale<=0: raise ValueError('Prior effect scale must be positive.')
        se=result['se'];tau=prior_scale
        variance=1/(1/tau**2+1/max(se**2,1e-24))
        mean=variance*result['effect']/max(se**2,1e-24)
        delta=rng.normal(mean,np.sqrt(variance),draws)
        model={'name':'Approximate normal effect likelihood with zero-centered normal prior',
               'prior':{'mean':0.,'sd':tau,'source':'Declared effect scale; not optimized for observed uplift'},
               'posterior_mean':float(mean),'posterior_variance':float(variance),'effect_mean':float(mean),
               'likelihood_variance':float(se**2),'approximation':'Plug-in sampling variance; caution with heavy tails, small samples and unstable denominators.'}
    threshold=(samples[0]*relative_goal_pct/100 if result['kind']=='binary' and relative_goal_pct is not None else
               abs(result['control'])*relative_goal_pct/100 if relative_goal_pct is not None else result['sesoi'])
    benefit=beneficial_direction*delta
    model.update(raw_probabilities={'effect_above_zero':float((delta>0).mean()),'effect_below_zero':float((delta<0).mean())},
        directional_probabilities={'beneficial_direction':'increase' if beneficial_direction==1 else 'decrease',
            'probability_benefit':float((benefit>0).mean()),'probability_harm':float((benefit<0).mean()),
            'probability_useful_benefit':float((benefit>threshold).mean()),'probability_material_harm':float((benefit<-threshold).mean())},
        goal={'relative_percent':relative_goal_pct,'absolute_reference_threshold':result['sesoi'],
              'basis':'Posterior control draws' if result['kind']=='binary' and relative_goal_pct is not None else 'Conditional on observed control reference'},
        credible_low=float(np.quantile(delta,.025)),credible_high=float(np.quantile(delta,.975)),draws=draws,seed=seed)
    return delta,model


def segment_effects(df,columns,metric,kind,denominator=None,sesoi=.002):
    rows=[];omitted=[]
    for column in columns:
        for level,group in df.groupby(column,dropna=False):
            counts=group.variant.value_counts().reindex(['control','treatment'],fill_value=0)
            if counts.min()<30:
                omitted.append(f'{column}={level}');continue
            if kind=='ratio' and any(group.loc[group.variant==a,denominator].sum()<=0 for a in ['control','treatment']):
                omitted.append(f'{column}={level} (zero denominator)');continue
            r=infer(group,metric,kind,denominator,sesoi)
            r.update(segment=column,level=str(level)); rows.append(r)
    for r,p in zip(rows,adjust([r['p_value'] for r in rows],'holm')):r['p_holm']=float(p)
    for r,p in zip(rows,adjust([r['p_value'] for r in rows],'bh')):r['p_bh']=float(p)
    # Omnibus heterogeneity within each disjoint partition; never compare subgroup significance.
    interactions=[]
    for column in columns:
        parts=[r for r in rows if r['segment']==column and r['se']>0]
        if len(parts)<2:continue
        weights=np.array([1/r['se']**2 for r in parts]);effects=np.array([r['effect'] for r in parts])
        average=np.average(effects,weights=weights)
        q=float(np.sum(weights*(effects-average)**2))
        interactions.append({'attribute':column,'q':q,'df':len(parts)-1,'p_value':float(stats.chi2.sf(q,len(parts)-1)),
                             'method':'Approximate inverse-variance heterogeneity Q across disjoint groups'})
    for r,p in zip(interactions,adjust([r['p_value'] for r in interactions],'holm')):r['p_holm']=float(p)
    return rows,interactions,omitted


def relative_effect_draws(model,raw_draws,reference):
    """Paired-rate relative draws for binary outcomes; fixed-reference approximation otherwise."""
    if 'arm_posterior_beta' in model and model['goal']['basis']=='Posterior control draws':
        rng=np.random.default_rng(model['seed'])
        control=rng.beta(*model['arm_posterior_beta'][0],len(raw_draws))
        treatment=rng.beta(*model['arm_posterior_beta'][1],len(raw_draws))
        return (treatment-control)/np.maximum(control,np.finfo(float).tiny)
    return raw_draws/reference
