"""Evidence-based blockers, warnings and unavailable checks. Never cleans silently."""
import numpy as np
import pandas as pd
from scipy import stats


def audit(df,metric,kind,denominator=None,expected=.5,covariates=(),platform=None,
          design_confirmed=False,randomization='user',analysis='ITT',stopping='fixed',ratio_bounded=False,nonnegative_metrics=()):
    checks=[]
    def add(check,status,detail):checks.append({'check':check,'status':status,'detail':detail})
    required={'unit_id','variant',metric}|({denominator} if denominator else set())
    missing=required-set(df.columns)
    if missing:
        add('Required fields','block',f'Missing columns: {sorted(missing)}')
        return finish(checks,None,{})
    blank=df.unit_id.isna() | df.unit_id.astype(str).str.strip().eq('')
    add('Missing unit IDs','block' if blank.any() else 'pass',f'{int(blank.sum())} empty IDs.')
    duplicates=int(df.unit_id.duplicated(keep=False).sum())
    crossing=int((df.groupby('unit_id').variant.nunique()>1).sum())
    add('Repeated units','block' if duplicates else 'pass',f'{duplicates} rows share an ID; {crossing} units occur in multiple arms. No deduplication is applied.')
    add('Exact repeated records','block' if df.duplicated().any() else 'pass',f'{int(df.duplicated().sum())} exact repeated records.')
    arms=set(df.variant.dropna())
    goodarms=arms=={'control','treatment'} and not df.variant.isna().any()
    add('Assignment labels','pass' if goodarms else 'block','Exactly control and treatment are required after mapping.')
    counts=df.variant.value_counts().reindex(['control','treatment'],fill_value=0)
    p=float(stats.chisquare(counts,np.array([1-expected,expected])*len(df)).pvalue) if goodarms and len(df) else None
    add('Sample ratio mismatch','block' if p is not None and p<.001 else ('pass' if p is not None else 'unavailable'),f'Observed {counts.to_dict()}; expected treatment share {expected:.1%}; p={p}. Blocking threshold 0.001.')
    add('Independent sample size','pass' if counts.min()>=30 else 'block',f'Minimum assigned users per arm: {int(counts.min())}; required ≥30. Sparse binary events still trigger exact testing.')
    metrics=[metric]+([denominator] if denominator else [])+[c for c in df if c.endswith('_guardrail')]
    critical=set(metrics)
    metrics += [c for c in df if c not in {'unit_id','variant'} and pd.api.types.is_numeric_dtype(df[c])]
    for name in dict.fromkeys(metrics):
        y=pd.to_numeric(df[name],errors='coerce')
        bad=int((~np.isfinite(y)).sum())
        add(f'{name}: observed finite numbers',('block' if name in critical else 'warning') if bad else 'pass',f'{bad} missing, nonnumeric or infinite values; never imputed to zero.')
        if bad:continue
        if name==metric and kind=='binary':
            add(f'{name}: binary domain','pass' if y.isin([0,1]).all() else 'block','Binary outcomes must be exactly 0 or 1.')
        if name==denominator:
            impossible=(y<0)|((y==0)&(pd.to_numeric(df[metric],errors='coerce')!=0))
            add('Ratio consistency','block' if impossible.any() or any(y[df.variant==a].sum()<=0 for a in ['control','treatment']) else 'pass',f'{int(impossible.sum())} negative denominators or nonzero numerators with zero denominator. Individual zero/zero is allowed; ratios use arm sums.')
            unstable=[]
            for arm in ['control','treatment']:
                d=y[df.variant==arm]
                if len(d)>1 and d.mean()>0:
                    relative_se=float(d.std(ddof=1)/np.sqrt(len(d))/d.mean())
                    if relative_se>.1:unstable.append(f'{arm}: denominator mean relative SE={relative_se:.1%}')
            add('Ratio denominator stability','warning' if unstable else 'pass',
                '; '.join(unstable)+'. Delta-method approximations can be unreliable when denominator uncertainty is large.' if unstable else 'Arm denominator mean relative SE ≤10%; this is a heuristic, not a proof of reliable linearization.')
            if ratio_bounded:
                broken=(df[metric]<0)|(df[metric]>y)
                add('Bounded ratio numerator ≤ denominator','block' if broken.any() else 'pass',f'{int(broken.sum())} invalid rows. This constraint applies only to declared bounded count ratios.')
        if y.nunique()==1 and name in critical:
            add(f'{name}: variation','warning',f'All rows have the same observed value {y.iloc[0]}. This may be real, but verify logging and metric coverage.')
        if not y.isin([0,1]).all():
            q1,q3=y.quantile([.25,.75]);iqr=q3-q1
            extreme=(y>q3+3*iqr)|(y<q1-3*iqr) if iqr>0 else y!=y.median()
            p99=float(y.quantile(.99));total=float(y.abs().sum())
            tailshare=float(y[y>p99].abs().sum()/total) if total else 0.
            flag=tailshare>.10 or abs(float(y.skew()))>3
            add(f'{name}: tails and unusual values','warning' if flag else 'pass',f'min={y.min():.5g}, max={y.max():.5g}, p99={p99:.5g}; {int(extreme.sum())} values outside 3×IQR fences; upper 1% contributes {tailshare:.1%} of absolute total. Flags are not automatic exclusions.')
        if name in nonnegative_metrics:
            add(f'{name}: nonnegative domain','block' if (y<0).any() else 'pass',f'{int((y<0).sum())} negative values in a user-declared nonnegative metric. Net/refund outcomes may legitimately be negative; declare their domain correctly.')
    # Missingness by arm for the primary metric remains visible even when it blocks inference.
    rates=df.assign(_missing=pd.to_numeric(df[metric],errors='coerce').isna()).groupby('variant')._missing.mean().to_dict()
    add('Outcome logging coverage','block' if any(rates.values()) else 'pass',f'Primary missing share by arm: {rates}. An observed zero must mean measured absence, not missing logging.')
    timestamps={}
    for name in ['assigned_at','exposed_at','window_end_at','observed_through']:
        if name in df:
            timestamps[name]=pd.to_datetime(df[name],utc=True,errors='coerce')
            invalid=int(timestamps[name].isna().sum())
            # Absent exposure is a coverage warning in ITT, never a reason to filter users.
            add(f'{name}: timestamp parsing','warning' if name=='exposed_at' and invalid else ('block' if invalid else 'pass'),f'{invalid} absent or unparseable UTC timestamps.')
    if 'assigned_at' in timestamps and 'window_end_at' in timestamps:
        reversed_=timestamps['window_end_at']<=timestamps['assigned_at']
        add('Observation ordering','block' if reversed_.any() else 'pass',f'{int(reversed_.sum())} windows end at/before assignment.')
        window=(timestamps['window_end_at']-timestamps['assigned_at']).dt.total_seconds()/86400
        add('Comparable window lengths','warning' if window.max()-window.min()>.01 else 'pass',f'Window duration range: {window.min():.3g}–{window.max():.3g} days.')
    if {'window_end_at','observed_through'}<=timestamps.keys():
        incomplete=timestamps['window_end_at']>timestamps['observed_through']
        add('Window maturity','block' if incomplete.any() else 'pass',f'{int(incomplete.sum())} windows are incomplete at data cutoff. No mature-user filtering is applied.')
    else:add('Window maturity','unavailable','Provide window_end_at and observed_through. Owner confirmation is required if unavailable.')
    if {'assigned_at','exposed_at'}<=timestamps.keys():
        before=timestamps['exposed_at']<timestamps['assigned_at']
        add('Exposure ordering','block' if before.any() else 'pass',f'{int(before.sum())} exposures precede assignment.')
        groups=[('all',df.index)]
        if platform and platform in df:groups += [(str(k),g.index) for k,g in df.groupby(platform)]
        details=[];flag=False
        coverage=timestamps['exposed_at'].notna()
        for label,idx in groups:
            armrates=coverage.loc[idx].groupby(df.loc[idx,'variant']).mean()
            if len(armrates)==2:
                gap=float(armrates.max()-armrates.min());flag |= gap>.02
                details.append(f'{label}: {armrates.to_dict()}')
        add('Exposure coverage by arm/platform','warning' if flag or not coverage.all() else 'pass','; '.join(details)+'. Gaps >2 pp flagged heuristically. Keep ITT; investigate rather than conditioning on exposure.')
        delay=(timestamps['exposed_at']-timestamps['assigned_at']).dt.total_seconds()/3600
        if 'observed_through' in timestamps:
            future=timestamps['exposed_at']>timestamps['observed_through']
            add('Exposure before data cutoff','block' if future.any() else 'pass',f'{int(future.sum())} exposures occur after the declared export cutoff.')
        if 'window_end_at' in timestamps:
            late=timestamps['exposed_at']>timestamps['window_end_at']
            add('Exposure inside outcome window','warning' if late.any() else 'pass',f'{int(late.sum())} first exposures occur after the outcome window. They remain in ITT; investigate opportunity to receive treatment.')
        armmeans=delay.groupby(df.variant).mean().to_dict()
        add('Exposure delay','warning' if len(armmeans)==2 and abs(armmeans.get('control',0)-armmeans.get('treatment',0))>1 else 'pass',f'Mean hours from assignment by arm: {armmeans}. >1 hour gap flagged heuristically.')
    else:add('Exposure diagnostics','unavailable','Provide assigned_at and exposed_at. Exposure comparability cannot be established from outcome columns.')
    balance=[]
    for column in covariates:
        if column not in df:continue
        missing=df[column].isna().groupby(df.variant).mean().to_dict()
        if any(missing.values()):add(f'{column}: baseline completeness','warning',f'Missing shares by arm: {missing}. Balance summaries use observed values; missingness may itself be imbalanced.')
        if pd.api.types.is_numeric_dtype(df[column]):
            c=df.loc[df.variant=='control',column].dropna().to_numpy(float)
            t=df.loc[df.variant=='treatment',column].dropna().to_numpy(float)
            if min(len(c),len(t))<2:continue
            scale=np.sqrt((c.var(ddof=1)+t.var(ddof=1))/2)
            separated=bool(scale==0 and t.mean()!=c.mean())
            smd=float((t.mean()-c.mean())/scale) if scale else (None if separated else 0.)
            balance.append({'covariate':column,'level':'numeric','standardized_difference':smd,'complete_separation':separated})
        else:
            levels=df[column].fillna('[missing]').astype(str)
            for level in levels.unique():
                pc=float((levels[df.variant=='control']==level).mean());pt=float((levels[df.variant=='treatment']==level).mean())
                scale=np.sqrt((pc*(1-pc)+pt*(1-pt))/2)
                separated=bool(scale==0 and pt!=pc)
                smd=float((pt-pc)/scale) if scale else (None if separated else 0.)
                balance.append({'covariate':column,'level':str(level),'standardized_difference':smd,'complete_separation':separated})
    add('Pre-treatment covariate balance','warning' if any(r['complete_separation'] or (r['standardized_difference'] is not None and abs(r['standardized_difference'])>.1) for r in balance) else ('pass' if balance else 'unavailable'),
        'Absolute standardized difference >0.1 or complete separation (undefined pooled-variance SMD) flagged as a diagnostic, not a significance test or automatic invalidation. Only genuinely pre-treatment covariates belong here.' if balance else 'Select measured pre-treatment attributes; otherwise balance cannot be checked.')
    add('Design declaration','pass' if design_confirmed else 'block','Owner must confirm the randomization unit, complete assignment export, consistent metric definitions, exclusions and stopping plan.')
    if 'cluster_id' in df:
        add('Possible clustered observations','warning','cluster_id is present. Confirm whether assignment or outcome dependence is clustered; individual-user errors may be invalid even if IDs are unique.')
    unsupported=randomization!='user' or analysis!='ITT' or stopping!='fixed'
    add('Supported design','block' if unsupported else 'pass',f'Unit={randomization}; population={analysis}; stopping={stopping}. Clustered, triggered/per-protocol and sequential designs need different estimators; this demo abstains.')
    result=finish(checks,p,counts.to_dict());result['balance']=balance
    return result


def finish(checks,p,counts):
    errors=[f'{c["check"]}: {c["detail"]}' for c in checks if c['status']=='block']
    warnings=[f'{c["check"]}: {c["detail"]}' for c in checks if c['status']=='warning']
    unavailable=[c['check'] for c in checks if c['status']=='unavailable']
    verdict=('Not decision-ready: resolve blocking issues' if errors else
             'Proceed cautiously: investigate warnings / unverified checks' if warnings or unavailable else
             'No detected blockers: inference is conditionally usable')
    return {'checks':checks,'errors':errors,'warnings':warnings,'unavailable':unavailable,
            'srm_p':p,'counts':counts,'verdict':verdict,'balance':[]}
