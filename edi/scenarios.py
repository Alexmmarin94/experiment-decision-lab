"""Validated, versionable business scenarios. Statistical settings are never writable here."""
from dataclasses import asdict
import hashlib
import json
import numpy as np
import pandas as pd
from edi.engine import Business
from edi.uncertainty import Uncertainty
from edi.decision_policy import Preferences,action_worlds,summarize_actions,evaluate,utility,eligible_names,OBJECTIVE

BUSINESS_LIMITS={'monthly_users':(1,1e10),'reach':(.01,1.),'margin':(.000001,1e7),'implementation':(0,1e9),
    'maintenance':(0,1e8),'months':(1,36),'persistence':(.1,1.),'margin_uncertainty':(0,1.)}
OPTION_LIMITS={'pilot_fraction':(.01,.9),'pilot_cost':(0,1e9),'iteration_cost':(0,1e9),'opportunity_cost':(0,1e9)}

LABELS={'monthly_users':'Eligible people / month','reach':'Share reached','margin':'Contribution per extra outcome',
    'implementation':'Global launch budget','maintenance':'Monthly maintenance (entered annually)','months':'Planning months',
    'persistence':'Improvement carried into the next month (not user retention)','margin_uncertainty':'Contribution uncertainty','pilot_fraction':'Gradual rollout share',
    'pilot_cost':'Gradual rollout budget','iteration_cost':'Iteration budget','opportunity_cost':'Displaced-project cost',
    'include_gradual':'Consider a gradual rollout','global_reversible':'Global rollback possible','population_cv':'Traffic uncertainty','reach_sd':'Reach uncertainty',
    'cost_cv':'Implementation cost uncertainty','retention_sd':'Uncertainty in improvement durability','generalization_cv':'Unmeasured transport uncertainty',
    'margin_reach_correlation':'Contribution/reach dependence'}

SOURCES=['Observed history','Observed experiment','Finance estimate','Expert judgment','Hypothetical scenario','AI suggestion (unverified)']


def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def starter(df,metric,kind,denominator=None):
    n=len(df);pace=None
    volume=float(pd.to_numeric(df[denominator],errors='coerce').sum()) if kind=='ratio' and denominator in df else n
    if not np.isfinite(volume) or volume<=0:volume=max(1,n)
    if 'assigned_at' in df:
        times=pd.to_datetime(df.assigned_at,utc=True,errors='coerce');days=(times.max()-times.min()).total_seconds()/86400
        if days>0 and np.isfinite(volume) and volume>0:pace=max(1,int(round(volume/days*30)))
    users=(pace or max(1,int(round(volume)))) if kind=='ratio' else (pace or max(n,10000))
    margin=12. if kind=='binary' else .3 if 'revenue' in metric.lower() else 1.
    business=asdict(Business(monthly_users=users,margin=margin,implementation=9000.,maintenance=500.,persistence=.9))
    options={'pilot_fraction':.25,'pilot_cost':4000.,'iteration_cost':8000.,'opportunity_cost':0.,'global_reversible':True,'include_gradual':False}
    provenance={}
    explanations={'monthly_users':('Observed experiment' if pace else 'Hypothetical scenario','Medium' if pace else 'Low',
        ('30-day extrapolation of observed denominator volume, not a validated future forecast.' if kind=='ratio' else '30-day extrapolation of assignment pace; trial enrollment is not necessarily future eligible traffic.') if pace else 'Illustrative traffic assumption; confirm eligible future users.'),
        'reach':('Hypothetical scenario','Low','Assume 80% of eligible users can receive the change.'),
        'margin':('Hypothetical scenario','Low','€12 contribution per extra purchase; gross spend does not identify profit.' if kind=='binary' else 'Illustrative value per extra metric unit; confirm the conversion to contribution.'),
        'implementation':('Expert judgment','Low','Illustrative budget: 10 engineering days × €600 plus €3,000 launch support; not inferred from the CSV.'),
        'maintenance':('Hypothetical scenario','Low','€500/month placeholder for ongoing engineering/support.'),
        'months':('Hypothetical scenario','Low','Six-month planning horizon, not six months of observed effects.'),
        'persistence':('Hypothetical scenario','Low','The product improvement keeps 90% of its previous month’s size. This is benefit durability, not customer retention, and is not estimated from a seven-day test.'),
        'margin_uncertainty':('Expert judgment','Low','20% relative margin uncertainty; widen if finance has little evidence.')}
    for key in business:
        source,quality,why=explanations[key];provenance[key]={'source':source,'quality':quality,'reason':why}
    for key in options:provenance[key]={'source':'Hypothetical scenario','quality':'Low','reason':{
        'pilot_fraction':'Start with a quarter of the intended reach.','pilot_cost':'€4,000 budget for a limited feature-flag rollout.',
        'iteration_cost':'€8,000 redesign budget; no assumed future uplift.','opportunity_cost':'Unknown displacement cost starts at zero, not evidence that it is free.',
        'global_reversible':'Assumes a feature flag permits rollback; verify this.',
        'include_gradual':'Optional alternative, disabled until the team explicitly chooses to consider a partial launch.'}[key]}
    for key in asdict(Uncertainty()):
        provenance[key]={'source':'Hypothetical scenario','quality':'Low','reason':'Illustrative uncertainty width or dependence; not estimated by this experiment.'}
    return {'business':business,'uncertainty':asdict(Uncertainty()),'options':options,'preferences':asdict(Preferences()),'provenance':provenance}


def validate_patch(patch):
    if not isinstance(patch,dict) or not patch:raise ValueError('Provide a nonempty object of permitted business changes.')
    allowed=set(BUSINESS_LIMITS)|set(OPTION_LIMITS)|{'annual_maintenance_eur','global_reversible','include_gradual'}
    unknown=set(patch)-allowed
    if unknown:raise ValueError('Protected or unknown fields: '+', '.join(sorted(unknown)))
    if 'annual_maintenance_eur' in patch and 'maintenance' in patch:raise ValueError('Supply annual or monthly maintenance, not both.')
    validated={}
    for key,value in patch.items():
        if key in ('global_reversible','include_gradual'):
            if not isinstance(value,bool):raise ValueError(f'{key} must be true or false.')
            validated[key]=value;continue
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value):raise ValueError(f'{key}: a finite number is required.')
        field='maintenance' if key=='annual_maintenance_eur' else key
        number=float(value)/12 if key=='annual_maintenance_eur' else float(value)
        lo,hi=(BUSINESS_LIMITS|OPTION_LIMITS)[field]
        if not lo<=number<=hi:raise ValueError(f'{key}: value outside the permitted range.')
        if field in ('months','monthly_users'):
            if not number.is_integer():raise ValueError(f'{field} must be an integer.')
            number=int(number)
        validated[field]=number
    return validated


def proposed_scenario(current,patch,reason,source='AI suggestion (unverified)'):
    normalized=validate_patch(patch)
    result=json.loads(json.dumps(current))
    for key,value in normalized.items():
        group='business' if key in BUSINESS_LIMITS else 'options'
        result[group][key]=value
        result['provenance'][key]={'source':source,'quality':'Low','reason':str(reason)[:2000]}
    Business(**result['business']).validate()
    return result,normalized


def run(effects,scenario,quality,guardrails):
    b=Business(**scenario['business']);u=Uncertainty(**scenario['uncertainty']);p=Preferences(**scenario['preferences'])
    worlds,summary,extra=action_worlds(effects,b,u,scenario['options'])
    rows=summarize_actions(worlds,b,scenario['options'],p)
    assessment=evaluate(quality,guardrails,rows,p)
    allowed=eligible_names(scenario['options'],p,assessment['allow_rollout'])
    utility_worlds=np.array([utility(worlds[name],p) for name in allowed])
    assessment['perfect_information_value']=float(utility_worlds.max(axis=0).mean()-utility_worlds.mean(axis=1).max())
    return {'assessment':assessment,'actions':rows,'full_summary':summary,'trajectory':extra['trajectory']},worlds,extra


def shared_evsi(model,result,scenario,allow_rollout,direction=1,n_per_arm=2000,study_cost=3000.,delay_cost=1000.,outer=180,inner=1200,seed=91):
    """Nested posterior predictive EVSI using the SAME actions, payoff and risk utility.

    Iteration has no supplied future-effect model and is unranked in both optimizers.
    Evidence readiness and reversibility remain fixed; guardrail/economic learning excluded.
    """
    rng=np.random.default_rng(seed);b=Business(**scenario['business']);u=Uncertainty(**scenario['uncertainty']);p=Preferences(**scenario['preferences'])
    names=eligible_names(scenario['options'],p,allow_rollout)
    count=outer*inner
    if result['kind']=='binary':
        c,t=model['arm_posterior_beta']
        observed_c=rng.binomial(n_per_arm,rng.beta(*c,outer));observed_t=rng.binomial(n_per_arm,rng.beta(*t,outer))
        posterior_c=rng.beta((c[0]+observed_c)[:,None],(c[1]+n_per_arm-observed_c)[:,None],(outer,inner))
        posterior_t=rng.beta((t[0]+observed_t)[:,None],(t[1]+n_per_arm-observed_t)[:,None],(outer,inner))
        future=(posterior_t-posterior_c).reshape(-1)*direction
        baseline=(rng.beta(*t,20000)-rng.beta(*c,20000))*direction
        method='Beta-binomial future samples with nested conditional effect draws'
    else:
        v=model['posterior_variance'];m=model['posterior_mean']
        future_v=(result['arm_variance_control']+result['arm_variance_treatment'])/n_per_arm
        predictive=rng.normal(m,np.sqrt(v+future_v),outer)
        weight=v/max(v+future_v,1e-24);updated_m=m+weight*(predictive-m);updated_v=max(0.,v*(1-weight))
        future=rng.normal(updated_m[:,None],np.sqrt(updated_v),(outer,inner)).reshape(-1)*direction
        baseline=rng.normal(m,np.sqrt(v),20000)*direction
        method='Approximate normal-normal predictive update with nested conditional effect draws'
    base_worlds,_,_=action_worlds(baseline,b,u,scenario['options'],seed=seed+1)
    base_scores={name:float(utility(base_worlds[name],p).mean()) for name in names}
    current=max(base_scores.values())
    future_worlds,_,_=action_worlds(future,b,u,scenario['options'],seed=seed+2)
    scores=np.vstack([utility(future_worlds[name],p).reshape(outer,inner).mean(axis=1) for name in names])
    posterior_best=scores.max(axis=0)
    evsi=float(posterior_best.mean()-current)
    cash_cost=study_cost+delay_cost
    net_scores=np.vstack([utility(future_worlds[name]-cash_cost,p).reshape(outer,inner).mean(axis=1) for name in names])
    net_value=float(net_scores.max(axis=0).mean()-current)
    se=float(posterior_best.std(ddof=1)/np.sqrt(outer))
    inner_noise=max(float(utility(future_worlds[name],p).reshape(outer,inner).std(axis=1).mean()/np.sqrt(inner)) for name in names)
    return {'evsi':evsi,'net_information_value':net_value,'mc_outer_se':se,'typical_inner_mc_se':inner_noise,
        'study_cost':study_cost,'delay_cost':delay_cost,'users_per_arm':n_per_arm,'current_expected_utility':current,
        'expected_best_utility_after_sample':float(posterior_best.mean()),'alternatives':names,'objective':OBJECTIVE,
        'cost_handling':'Study/delay costs are subtracted from every future action payoff BEFORE applying the shared downside utility.',
        'loss_weight':p.loss_weight,'outer_samples':outer,'inner_worlds':inner,'seed':seed,'method':method,
        'limitations':'Same business distributions and quantified alternatives as Monte Carlo. Learns only the primary effect; guardrail readiness/reversibility stay fixed. No economic learning or transported posterior. Nested maximization has finite-inner-sample optimism; MC SE does not bound that bias. Small/negative estimates may be numerical noise. Delay losses are entered explicitly.'}
