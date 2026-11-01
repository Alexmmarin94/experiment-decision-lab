"""Common-random-number economic scenarios and explicit value-of-information model."""
from dataclasses import asdict, dataclass
import numpy as np
from scipy import stats
from edi.engine import Business


@dataclass(frozen=True)
class Uncertainty:
    population_cv: float=.10
    reach_sd: float=.05
    cost_cv: float=.20
    retention_sd: float=.05
    generalization_cv: float=.15
    margin_reach_correlation: float=0.


def lognormal(mean,cv,z):
    sigma=np.sqrt(np.log1p(cv*cv))
    return mean*np.exp(sigma*z-sigma*sigma/2)


def bounded(mean,sd,z):
    if sd<=0 or mean in (0.,1.):return np.full(len(z),mean)
    if sd*sd>=mean*(1-mean):raise ValueError('Bounded uncertainty is too wide for its mean.')
    k=mean*(1-mean)/(sd*sd)-1
    return stats.beta.ppf(stats.norm.cdf(z),mean*k,(1-mean)*k)


def economic_simulation(effects,business,uncertainty=Uncertainty(),seed=23):
    business.validate()
    rng=np.random.default_rng(seed);n=len(effects)
    z=rng.normal(size=(7,n))
    rho=uncertainty.margin_reach_correlation
    if not -1<rho<1:raise ValueError('Correlation must be strictly between -1 and 1.')
    # Gaussian copula correlation applies to latent margin/reach shocks, not effects.
    reach_z=rho*z[0]+np.sqrt(1-rho*rho)*z[1]
    margin=lognormal(business.margin,business.margin_uncertainty,z[0])
    reach=bounded(business.reach,uncertainty.reach_sd,reach_z)
    population=lognormal(business.monthly_users,uncertainty.population_cv,z[2])
    cost_factor=lognormal(1.,uncertainty.cost_cv,z[3])
    cost=business.implementation*cost_factor
    retention=bounded(business.persistence,uncertainty.retention_sd,z[4])
    transport=lognormal(1.,uncertainty.generalization_cv,z[5])
    horizon=sum(retention**month for month in range(business.months))
    gross=effects*transport*population*reach*margin*horizon
    recurring=business.maintenance*business.months
    value=gross-cost-recurring
    evpi=float(np.maximum(value,0).mean()-max(0.,value.mean()))
    summary={'expected_value':float(value.mean()),'median':float(np.median(value)),
        'p05':float(np.quantile(value,.05)),'p95':float(np.quantile(value,.95)),
        'probability_loss':float((value<0).mean()),'expected_downside':float(np.maximum(-value,0).mean()),
        'worst_5pct_mean':float(value[value<=np.quantile(value,.05)].mean()),'evpi':evpi,
        'break_even_effect':(business.implementation+recurring)/(business.monthly_users*business.reach*business.margin*sum(business.persistence**m for m in range(business.months))),
        'mean_mc_se':float(value.std(ddof=1)/np.sqrt(n)),
        'probability_loss_mc_se':float(np.sqrt((value<0).mean()*(value>=0).mean()/n))}
    paths=np.array([effects*transport*population*reach*margin*sum(retention**m for m in range(h))-cost-business.maintenance*h for h in range(1,business.months+1)])
    trajectory=[{'month':i+1,'mean':float(v.mean()),'p05':float(np.quantile(v,.05)),'p95':float(np.quantile(v,.95))} for i,v in enumerate(paths)]
    return value,summary,{'gross':gross,'cost_factor':cost_factor,'trajectory':trajectory}


def alternatives(effects,business,uncertainty,pilot_fraction=.25,pilot_cost=5000.):
    full,_,world=economic_simulation(effects,business,uncertainty)
    # A literal fraction of the same rollout opportunity in every shared world.
    gradual=world['gross']*pilot_fraction-pilot_cost*world['cost_factor']-business.maintenance*business.months
    paths=np.vstack([np.zeros(len(effects)),gradual,full]);best=paths.max(axis=0)
    rows=[]
    for label,values in zip(['Keep current experience','Limited rollout','Full rollout'],paths):
        rows.append({'action':label,'expected_value':float(values.mean()),'loss_probability':float((values<0).mean()),
                     'expected_regret':float((best-values).mean()),'p05':float(np.quantile(values,.05)),'p95':float(np.quantile(values,.95))})
    return rows


def sensitivity(effects,business,uncertainty):
    specs=[('Contribution margin',business.margin*.7,business.margin*1.3,'margin'),
           ('Eligible population',business.monthly_users*.7,business.monthly_users*1.3,'monthly_users'),
           ('Reach',max(.05,business.reach-.15),min(.99,business.reach+.15),'reach'),
           ('Effect retention',max(.1,business.persistence-.25),min(1.,business.persistence+.15),'persistence'),
           ('Implementation cost',business.implementation*.5,business.implementation*1.5,'implementation'),
           ('Maintenance per month',max(0.,business.maintenance*.5),max(1000.,business.maintenance*1.5),'maintenance')]
    rows=[]
    for name,lo,hi,field in specs:
        values=[]
        for x in [lo,hi]:
            b=Business(**{**asdict(business),field:x})
            # Keep bounded SD feasible near 0/1 without changing requested mean.
            u=Uncertainty(**{**asdict(uncertainty),'reach_sd':min(uncertainty.reach_sd,.45*np.sqrt(b.reach*(1-b.reach))),
                            'retention_sd':min(uncertainty.retention_sd,.45*np.sqrt(b.persistence*(1-b.persistence)))})
            _,s,_=economic_simulation(effects,b,u)
            values.append(s)
        rows.append({'assumption':name,'low_input':lo,'high_input':hi,'low_expected_value':values[0]['expected_value'],
                     'high_expected_value':values[1]['expected_value'],'low_loss_risk':values[0]['probability_loss'],
                     'high_loss_risk':values[1]['probability_loss'],'swing':abs(values[1]['expected_value']-values[0]['expected_value'])})
    return sorted(rows,key=lambda r:r['swing'],reverse=True)


def sample_information(model,result,business,new_users_per_arm=5000,test_cost=5000.,delay_cost=0.,seed=61,draws=30000,direction=1):
    """EVSI with conjugate exact conditional means (binary), normal approximation otherwise.

    Risk-neutral launch/no-launch objective; fixed central economics independent of effect.
    Not the risk-constrained multi-action rollout policy or a promise of test outcomes.
    """
    rng=np.random.default_rng(seed)
    scale=business.monthly_users*business.reach*business.margin*sum(business.persistence**m for m in range(business.months))
    fixed=business.implementation+business.maintenance*business.months
    mean=model['effect_mean']
    if result['kind']=='binary':
        pc,pt=model['arm_posterior_beta']
        cc=rng.binomial(new_users_per_arm,rng.beta(*pc,draws))
        ct=rng.binomial(new_users_per_arm,rng.beta(*pt,draws))
        next_mean=(pt[0]+ct)/(sum(pt)+new_users_per_arm)-(pc[0]+cc)/(sum(pc)+new_users_per_arm)
        method='Beta-binomial posterior predictive; exact conjugate posterior means'
        # Common predictive draws permit a consistent control-variate EVSI estimator.
    else:
        variance=model['posterior_variance'];likelihood=model['likelihood_variance']
        current_n=(result['n_control']+result['n_treatment'])/2
        future_variance=((result['arm_variance_control']+result['arm_variance_treatment'])/new_users_per_arm
                         if 'arm_variance_control' in result else likelihood*current_n/new_users_per_arm)
        next_mean=rng.normal(mean,np.sqrt(variance*variance/max(variance+future_variance,1e-24)),draws)
        method='Normal-normal posterior predictive with fixed plug-in sampling variance'
    payoff=scale*direction*next_mean-fixed
    base=scale*direction*mean-fixed
    # Subtract a zero-mean control variate to reduce noise when current choice is launch.
    improvement=np.maximum(payoff,0)-max(0.,base)
    if base>0:improvement-=payoff-base
    raw=float(improvement.mean());se=float(improvement.std(ddof=1)/np.sqrt(draws))
    evsi=max(0.,raw)
    return {'evsi':evsi,'mc_se':se,'net_information_value':evsi-test_cost-delay_cost,
            'test_cost':test_cost,'delay_cost':delay_cost,'new_users_per_arm':new_users_per_arm,
            'current_risk_neutral_value':max(0.,base),'expected_value_after_information':max(0.,base)+evsi,
            'method':method,'draws':draws,'seed':seed,
            'scope':'Fixed central economic assumptions, linear money utility, launch/no-launch only; excludes economic/transport learning and the downside-risk constraint.'}
