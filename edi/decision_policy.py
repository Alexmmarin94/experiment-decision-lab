"""One reusable definition of alternatives, risk preferences and decision readiness."""
from dataclasses import dataclass
import numpy as np
from edi.uncertainty import economic_simulation


OBJECTIVE='E[net value − loss_weight × max(−net value, 0)]'

@dataclass(frozen=True)
class Preferences:
    loss_weight: float=1.
    require_reversible: bool=False


def utility(values,preferences):
    """Money-equivalent score: net value minus extra weight on negative outcomes."""
    return values-preferences.loss_weight*np.maximum(-values,0.)


def action_worlds(effects,business,uncertain,options,seed=23):
    full,summary,world=economic_simulation(effects,business,uncertain,seed)
    opportunity=options.get('opportunity_cost',0.)
    full=full-opportunity
    summary.update(expected_value=float(full.mean()),median=float(np.median(full)),p05=float(np.quantile(full,.05)),
        p95=float(np.quantile(full,.95)),probability_loss=float((full<0).mean()),
        expected_downside=float(np.maximum(-full,0).mean()),evpi=float(np.maximum(full,0).mean()-max(0.,full.mean())))
    summary['worst_5pct_mean']-=opportunity
    summary['break_even_effect']+=opportunity/(business.monthly_users*business.reach*business.margin*sum(business.persistence**m for m in range(business.months)))
    summary.pop('evpi',None)  # the shared-policy EVPI is computed over all feasible actions below
    world['trajectory']=[{**r,'mean':r['mean']-opportunity,'p05':r['p05']-opportunity,'p95':r['p95']-opportunity} for r in world['trajectory']]
    gradual=world['gross']*options['pilot_fraction']-options['pilot_cost']*world['cost_factor']-business.maintenance*business.months-opportunity
    actions={'Keep control':np.zeros(len(effects)),'Global launch':full}
    if options.get('include_gradual',False):actions['Gradual rollout']=gradual
    return actions,summary,world


def summarize_actions(worlds,business,options,preferences):
    best=np.max(np.array(list(worlds.values())),axis=0)
    rows=[]
    for name,values in worlds.items():
        cost=0. if name=='Keep control' else options['pilot_cost'] if name=='Gradual rollout' else business.implementation
        reversible=options.get('global_reversible',True) if name=='Global launch' else True
        rows.append({'action':name,'technical_cost_eur':cost,'reversible':reversible,
            'new_evidence':{'Keep control':'None by itself','Gradual rollout':'Possible with a planned randomized holdout','Global launch':'Limited; monitoring alone is not causal evidence'}[name],
            'expected_net_value':float(values.mean()),'loss_probability':float((values<0).mean()),
            'expected_regret':float((best-values).mean()),'decision_score':float(utility(values,preferences).mean()),
            'p05':float(np.quantile(values,.05)),'p95':float(np.quantile(values,.95)),
            'eligible':bool(not preferences.require_reversible or reversible)})
    rows.append({'action':'Iterate the feature','technical_cost_eur':options['iteration_cost'],'reversible':True,
        'new_evidence':'Requires a new experiment','expected_net_value':None,'loss_probability':None,
        'expected_regret':None,'decision_score':None,'p05':None,'p95':None,'eligible':False,
        'reason':'Technical budget is a scenario; a future effect is unknown, so no invented return is ranked.'})
    return rows


def eligible_names(options,preferences,allow_rollout=True):
    names=['Keep control']
    if allow_rollout:
        if options.get('include_gradual',False):names.append('Gradual rollout')
        if not preferences.require_reversible or options.get('global_reversible',True):names.append('Global launch')
    return names


def evaluate(quality,guardrails,actions,preferences):
    reasons=[]
    safety=bool(guardrails) and all(g['safe'] for g in guardrails)
    if quality['errors']:reasons.append('Resolve blocking data/design issues before acting.')
    if not guardrails:reasons.append('No guardrails supplied: safety is unassessed.')
    elif not safety:reasons.append('At least one guardrail has not ruled out the declared deterioration.')
    readiness=not quality['errors'] and safety
    candidates=[r for r in actions if r['decision_score'] is not None and r['eligible'] and (readiness or r['action']=='Keep control')]
    best=max(candidates,key=lambda r:r['decision_score']) if candidates else None
    preferred=best['action'] if best else 'Keep control'
    if not readiness:status='More evidence required'
    elif preferred=='Keep control':status='Keep control / investigate cheaper next steps'
    else:status='Conditional candidate'
    if quality['warnings'] or quality['unavailable']:reasons.append('Investigate the flagged or unavailable quality checks; the recommendation remains conditional.')
    reasons.append('Scores use net value with an explicit downside penalty; they are not observed profit or an automatic approval.')
    return {'status':status,'preferred_action':preferred,'allow_rollout':readiness,'reasons':reasons,
            'preferences':{'loss_weight':preferences.loss_weight,'require_reversible':preferences.require_reversible},
            'objective':OBJECTIVE,
            'conditions':['Owner confirms design and logging','Guardrails satisfy declared limits','Validate material economic assumptions','Confirm feasibility and reversibility']}
