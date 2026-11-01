"""Normal-normal partial pooling with hyperparameter uncertainty integrated by quadrature.

Approximate likelihood on disjoint subgroup effect estimates, not an exact binary model.
"""
import numpy as np
from scipy.special import logsumexp


def partial_pool(rows,mu_sd,tau_sd,threshold,direction=1,draws=20000,seed=71,grid_size=1025):
    if len(rows)<2:raise ValueError('Partial pooling requires at least two disjoint groups.')
    if mu_sd<=0 or tau_sd<=0:raise ValueError('Hyperprior scales must be positive.')
    y=np.array([r['effect'] for r in rows])
    se=np.array([max(r['se'],(r['ci_high']-r['ci_low'])/3.92,1e-10) for r in rows])
    upper=max(8*tau_sd,4*float(np.sqrt(np.mean(y*y))),4*float(se.max()))
    core=np.linspace(0.,min(8*tau_sd,upper),grid_size//2+1)
    tail=np.geomspace(max(tau_sd/1000,1e-12),upper,grid_size//2)
    tau=np.unique(np.r_[core,tail]);grid_size=len(tau)
    v=se[None,:]**2+tau[:,None]**2
    precision=1/mu_sd**2+(1/v).sum(axis=1)
    b=(y[None,:]/v).sum(axis=1)
    log_marginal=-.5*(np.log(v).sum(axis=1)+np.log(mu_sd**2)+np.log(precision)+(y[None,:]**2/v).sum(axis=1)-b*b/precision)
    log_prior=-tau*tau/(2*tau_sd*tau_sd)
    spacing=np.diff(tau)
    quadrature=np.r_[spacing[0]/2,(spacing[:-1]+spacing[1:])/2,spacing[-1]/2]
    logw=log_marginal+log_prior+np.log(quadrature);weights=np.exp(logw-logsumexp(logw))
    rng=np.random.default_rng(seed);index=rng.choice(grid_size,draws,p=weights)
    sampled_tau=tau[index]
    mu=rng.normal(b[index]/precision[index],np.sqrt(1/precision[index]))
    output=[]
    for j,r in enumerate(rows):
        weight=sampled_tau**2/(sampled_tau**2+se[j]**2)
        mean=weight*y[j]+(1-weight)*mu
        posterior=rng.normal(mean,np.sqrt(weight)*se[j])
        benefit=direction*posterior
        output.append({'group':r['level'],'unpooled_effect':float(y[j]),'pooled_effect':float(posterior.mean()),
            'credible_low':float(np.quantile(posterior,.025)),'credible_high':float(np.quantile(posterior,.975)),
            'probability_benefit':float((benefit>0).mean()),'probability_useful_benefit':float((benefit>threshold).mean())})
    edges=np.linspace(0.,upper,41)
    posterior_mass,_=np.histogram(tau,bins=edges,weights=weights)
    prior_weights=np.exp(log_prior)*quadrature;prior_weights/=prior_weights.sum()
    prior_mass,_=np.histogram(tau,bins=edges,weights=prior_weights)
    curve=[{'spread':float((edges[i]+edges[i+1])/2),'prior_mass':float(prior_mass[i]),'posterior_mass':float(posterior_mass[i])} for i in range(len(posterior_mass))]
    return {'spread_curve':curve,'groups':output,'heterogeneity':{'tau_median':float(np.median(sampled_tau)),
        'tau_low':float(np.quantile(sampled_tau,.025)),'tau_high':float(np.quantile(sampled_tau,.975)),
        'probability_tau_above_useful_effect':float((sampled_tau>threshold).mean())},
        'model':'effect_s ~ Normal(mu, tau²); observed effect_s ~ Normal(effect_s, estimated SE_s²)',
        'hyperpriors':{'mu':f'Normal(0, {mu_sd}²)','tau':f'HalfNormal({tau_sd})'},
        'method':'Analytic integration of mu; trapezoid quadrature over tau; conditional posterior simulation',
        'grid_size':grid_size,'upper_grid_mass':float(weights[-20:].sum()),'seed':seed,
        'limitations':'Normal effect likelihood with plug-in SEs; disjoint groups only. Few groups leave tau prior-sensitive. Shrinkage does not replace frequentist multiplicity control or validate post hoc targeting.'}
