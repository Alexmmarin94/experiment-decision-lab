"""Explain the existing policy using bounded, reproducible, read-only comparisons.

This module never defines a second recommendation rule. Every tested scenario is
evaluated by scenarios.run, with the same evidence, preferences and safety gates.
"""
import copy
import numpy as np
from edi.scenarios import run, proposed_scenario, BUSINESS_LIMITS, fingerprint
from edi.decision_policy import Preferences, utility

VERSION = 'decision-drivers-2'
FIELDS = {'Contribution margin':'margin', 'Eligible population':'monthly_users',
          'Reach':'reach', 'Effect retention':'persistence',
          'Implementation cost':'implementation', 'Maintenance per month':'maintenance'}
TITLES = {'margin':'Value per extra outcome', 'monthly_users':'Eligible audience',
          'reach':'Share receiving the change', 'persistence':'How long the improvement lasts',
          'implementation':'One-off launch cost', 'maintenance':'Ongoing maintenance'}


def _ranking(result, worlds, scenario, feasible=True):
    rows = [r for r in result['actions'] if r['decision_score'] is not None and r['eligible']
            and (not feasible or result['assessment']['allow_rollout'] or r['action']=='Keep control')]
    rows = sorted(rows, key=lambda r:r['decision_score'], reverse=True)
    gap = None
    if len(rows)>1:
        p = Preferences(**scenario['preferences'])
        paired = utility(worlds[rows[0]['action']],p)-utility(worlds[rows[1]['action']],p)
        se = float(paired.std(ddof=1)/np.sqrt(len(paired)))
        gap = {'leader':rows[0]['action'], 'runner_up':rows[1]['action'],
               'score_gap':float(paired.mean()), 'paired_mc_se':se,
               'separation_clear_of_mc_noise':bool(paired.mean()>2*se)}
    return rows, gap


def _point(result, worlds, scenario, value=None):
    feasible, gap = _ranking(result,worlds,scenario)
    economic, economic_gap = _ranking(result,worlds,scenario,False)
    launch = next(r for r in result['actions'] if r['action']=='Global launch')
    return {'input':value, 'preferred_feasible_action':result['assessment']['preferred_action'],
            'economic_leader_ignoring_safety_gate':economic[0]['action'],
            'rollout_allowed':result['assessment']['allow_rollout'],
            'global_launch_expected_value':launch['expected_net_value'],
            'global_launch_mean_mc_se':float(worlds['Global launch'].std(ddof=1)/np.sqrt(len(worlds['Global launch']))),
            'global_launch_loss_probability':launch['loss_probability'],
            'global_launch_score':launch['decision_score'],
            'preferred_action_expected_net_value':feasible[0]['expected_net_value'],
            'preferred_action_loss_probability':feasible[0]['loss_probability'],
            'feasible_gap':gap, 'economic_gap':economic_gap}


def _brackets(points, key):
    gap_key='feasible_gap' if key=='preferred_feasible_action' else 'economic_gap'
    return [{'lower_input':a['input'], 'upper_input':b['input'],
             'at_lower':a[key], 'at_upper':b[key],
             'endpoint_scores_separated_from_mc_noise':bool(a[gap_key] and b[gap_key]
                 and a[gap_key]['separation_clear_of_mc_noise'] and b[gap_key]['separation_clear_of_mc_noise'])}
            for a,b in zip(points,points[1:]) if a[key]!=b[key]]


def _impact_months(effects,scenario,quality,guards,baseline):
    """Test every supported whole-month horizon with shared worlds and policy.

    Longer horizons accumulate both fading benefit and maintenance. The choice
    can therefore switch more than once; do not assume a minimum guarantees launch.
    """
    lo,hi=map(int,BUSINESS_LIMITS['months'])
    points=[]
    for months in range(lo,hi+1):
        if months==scenario['business']['months']:
            points.append({**baseline,'input':months});continue
        changed,_=proposed_scenario(scenario,{'months':months},'Read-only whole-month impact comparison.',source='Hypothetical scenario')
        result,worlds,_=run(effects,changed,quality,guards)
        points.append(_point(result,worlds,changed,months))
    clear=[p for p in points if p['feasible_gap'] is None or p['feasible_gap']['separation_clear_of_mc_noise']]
    return {'current_months':scenario['business']['months'],'tested_range':[lo,hi],
            'source':scenario.get('provenance',{}).get('months',{}),'tested_points':points,
            'switch_brackets':_brackets(points,'preferred_feasible_action'),
            'clear_switch_windows':_brackets(clear,'preferred_feasible_action'),
            'ambiguous_months':[p['input'] for p in points if p not in clear],
            'scope':'Whole months only, same effect draws, shared seed 23, unchanged monthly benefit durability, costs, risk preferences and eligibility checks. Both benefit and maintenance accumulate over the selected period.',
            'limitation':'A modeled horizon is not observed follow-up. Longer impact must be justified; it is not evidence the improvement persists. Multiple preference switches are possible. Simulation uncertainty remains distinct from model uncertainty.'}


def build_decision_drivers(evidence, effects, max_worlds=4000, grid_points=5):
    """A bounded grid inside the already-declared sensitivity ranges, not a forecast.

    All comparisons use the same selected posterior draws and seed 23. Report
    brackets, never an interpolated exact switch or a claimed global monotonicity.
    Monte Carlo standard errors describe simulation precision, not model validity.
    """
    assessment=evidence['assessment']; scenario=evidence['scenario']; quality=evidence['quality']
    guards=evidence.get('guardrails',[])
    constraints=[]
    for finding in quality.get('errors',[]):
        constraints.append({'kind':'Data/design', 'binding':True, 'detail':str(finding)})
    if not guards:constraints.append({'kind':'Safety', 'binding':True, 'detail':'No guardrails supplied; safety is unassessed.'})
    for g in guards:
        if not g['safe']:constraints.append({'kind':'Safety', 'binding':True,
            'detail':str(g.get('metric','A guardrail'))+' has not ruled out the declared deterioration.'})
    if scenario['preferences']['require_reversible'] and not scenario['options']['global_reversible']:
        constraints.append({'kind':'Reversibility', 'binding':True, 'detail':'Global launch is excluded by the declared rollback requirement.'})
    for text in quality.get('warnings',[])+quality.get('unavailable',[]):
        constraints.append({'kind':'Trust condition', 'binding':False, 'detail':str(text)})
    result={'schema_version':VERSION, 'scenario_fingerprint':fingerprint(scenario), 'current_decision':copy.deepcopy(assessment),
            'constraints':constraints, 'counterfactuals':[],
            'assumption_provenance':copy.deepcopy(scenario.get('provenance',{})),
            'evsi_scope':{'result':evidence.get('evsi'), 'learns':'The primary treatment effect only.',
                'cannot_learn':'Engineering costs, benefit durability, future reach, guardrail readiness, or a redesigned feature effect.'},
            'limitations':['All financial results are conditional on the stated model and assumptions.',
                'A grid switch is a bracket between tested settings, not an exact threshold. Untested switches and non-monotonicity remain possible.',
                'Expected-profit break-even, downside-adjusted preference and rollout eligibility are different questions.',
                'Monte Carlo errors do not account for model misspecification or uncertain user-reported context.']}
    excluded=[]
    for row in evidence.get('actions',[]):
        if row['decision_score'] is None:why=row.get('reason','No supported future-effect/payoff model.')
        elif not row['eligible']:why='Excluded by the declared reversibility requirement.'
        elif not assessment['allow_rollout'] and row['action']!='Keep control':why='Unavailable until data/design and safety gates are satisfied.'
        else:continue
        excluded.append({'action':row['action'],'reason':why})
    result['excluded_actions']=excluded
    if effects is None or quality.get('errors'):
        result.update(eligible_actions=['Keep control'], economic_ranking=[], leading_gap=None,
                      next_action='Resolve the blocking data/design findings before evaluating a launch. More primary observations do not repair an invalid export or design.')
        return result
    effects=np.asarray(effects,dtype=float)
    if len(effects)<2 or not np.isfinite(effects).all():raise ValueError('Finite posterior effect draws are required.')
    n=min(int(max_worlds),len(effects))
    # Evenly spaced selection avoids using only one contiguous part of a stored chain.
    selected=effects[np.linspace(0,len(effects)-1,n,dtype=int)]
    base,worlds,_=run(selected,scenario,quality,guards)
    feasible,gap=_ranking(base,worlds,scenario)
    result.update(eligible_actions=[r['action'] for r in feasible], economic_ranking=feasible,
                  leading_gap=gap, comparison_baseline=_point(base,worlds,scenario),
                  simulation={'worlds':n,'seed':23,'grid_points_per_assumption':int(grid_points),
                              'shared_random_numbers':True,'authoritative_action':assessment['preferred_action'],
                              'comparison_action_matches_full_run':base['assessment']['preferred_action']==assessment['preferred_action']})
    result['months_of_impact']=_impact_months(selected,scenario,quality,guards,result['comparison_baseline'])
    launch_values=worlds['Global launch']
    result['launch_payoff_balance']={'expected_gain_component':float(np.maximum(launch_values,0).mean()),
        'expected_loss_component':float(np.maximum(-launch_values,0).mean()),
        'loss_probability':float((launch_values<0).mean()),
        'loss_probability_mc_se':float(np.sqrt((launch_values<0).mean()*(launch_values>=0).mean()/n)),
        'loss_weight':scenario['preferences']['loss_weight']}
    for row in evidence.get('sensitivity',[])[:6]:
        field=FIELDS.get(row['assumption'])
        if not field:continue
        lo=max(float(row['low_input']),BUSINESS_LIMITS[field][0])
        hi=min(float(row['high_input']),BUSINESS_LIMITS[field][1])
        if hi<lo:continue
        values=list(np.linspace(lo,hi,int(grid_points)))
        central=scenario['business'][field]
        if lo<=central<=hi:values.append(central)
        values=sorted(set(int(round(v)) if field=='monthly_users' else float(v) for v in values))
        points=[];unsupported=[]
        for value in values:
            try:
                changed,_=proposed_scenario(scenario,{field:value},'Read-only comparison inside the declared sensitivity range.',source='Hypothetical scenario')
                tested,w,_=run(selected,changed,quality,guards)
                points.append(_point(tested,w,changed,value))
            except ValueError:
                unsupported.append(value)
        if not points:continue
        preference=_brackets(points,'preferred_feasible_action')
        economic=_brackets(points,'economic_leader_ignoring_safety_gate')
        profit=[{'lower_input':a['input'],'upper_input':b['input']} for a,b in zip(points,points[1:])
                if (a['global_launch_expected_value']>0)!=(b['global_launch_expected_value']>0)]
        result['counterfactuals'].append({'field':field, 'label':TITLES[field],
            'declared_range':[lo,hi], 'current_input':central,
            'source':scenario.get('provenance',{}).get(field,{}), 'tested_points':points,
            'feasible_preference_switch_brackets':preference,
            'economic_switch_brackets_ignoring_safety_gate':economic,
            'expected_profit_break_even_brackets':profit,
            'switch_statement':'A feasible preference switch occurs between tested settings.' if preference else 'No feasible preference switch found at the tested settings; no extrapolation beyond this range.',
            'unsupported_settings':unsupported,
            'precision_note':'Use the paired score-gap MC errors at each endpoint. Near ties are numerically inconclusive; these are brackets, not exact thresholds.'})
    # Prioritize hard constraints, then actual feasible switches, not the longest sensitivity bar.
    binding=[c for c in constraints if c['binding']]
    switches=[c for c in result['counterfactuals'] if any(b['endpoint_scores_separated_from_mc_noise'] for b in c['feasible_preference_switch_brackets'])]
    if binding:
        if binding[0]['kind']=='Reversibility':
            result['next_action']='Keep global launch excluded under the rollback requirement. Confirm whether a reversible implementation is feasible, or evaluate the remaining eligible choices; better economics do not override this constraint.'
        else:result['next_action']='Resolve the binding '+binding[0]['kind'].lower()+' constraint: '+binding[0]['detail']+' Economic improvements alone do not remove it.'
    elif switches:
        item=switches[0]
        result['next_action']='Validate '+item['label'].lower()+' with its owner/source: tested settings change the preferred feasible action. More primary-metric data alone will not establish this business assumption.'
    elif gap and not gap['separation_clear_of_mc_noise']:
        result['next_action']='The leading choices are too close to Monte Carlo precision. Improve numerical precision before interpreting their small score difference.'
    else:
        result['next_action']='Confirm the least-supported material business inputs and execution constraints. No feasible switch was found on the tested grid; this is not proof the decision is robust to every scenario.'
    result['opportunity_cost_note']=('Displacement cost is currently unpriced (zero is a placeholder, not evidence that roadmap delay is free).'
        if scenario['options']['opportunity_cost']==0 else 'A priced displaced-project cost is included in rollout payoffs.')
    return result


def risk_explanation(drivers):
    """A probability counts worlds; the ranking also uses payoff sizes and preferences."""
    balance=drivers.get('launch_payoff_balance')
    if not balance:return 'Economic risk is not assessed while data/design blocks remain.'
    p=balance['loss_probability'];gain=balance['expected_gain_component'];loss=balance['expected_loss_component']
    text=(f'About {p:.0%} of the {drivers["simulation"]["worlds"]:,} shared comparison worlds lose money. This counts outcomes, not their size: '
          f'the average gain component is €{gain:,.0f} and the average loss component is €{loss:,.0f} '
          f'across all worlds. The choice score subtracts the loss component {1+balance["loss_weight"]:g} times '
          'from the gain component. A launch can lead on this score even with loss risk near 50%; that is an uncertain trade-off, not near-certain profit.')
    durability=next((c for c in drivers.get('counterfactuals',[]) if c['field']=='persistence'),None)
    if durability and len(durability['tested_points'])>1:
        a,b=durability['tested_points'][0],durability['tested_points'][-1]
        text+=(f' In the tested faster-fading versus longer-lasting improvement scenarios, modeled launch loss risk is '
               f'{a["global_launch_loss_probability"]:.1%} versus {b["global_launch_loss_probability"]:.1%}, '
               f'and expected value is €{a["global_launch_expected_value"]:,.0f} versus €{b["global_launch_expected_value"]:,.0f}. '
               'These vary the assumed duration of the product improvement—not user retention. '
               'Confirming durability can justify an assumption; selecting a favorable value is not new evidence. '
               'Sampling uncertainty and the other business risks still remain.')
    return text


def counterfactual_readout(drivers):
    """One computed business condition for UI/brief; full grids stay in the audit."""
    binding=[c for c in drivers.get('constraints',[]) if c['binding']]
    if binding:return 'First resolve '+binding[0]['kind'].lower()+': '+binding[0]['detail']+' Changing the economics does not remove this constraint.'
    tests=drivers.get('counterfactuals',[])
    if not tests:return 'No supported business counterfactual is available; resolve the missing evidence before interpreting a switch.'
    item=next((c for c in tests if c['feasible_preference_switch_brackets']),tests[0])
    brackets=item['feasible_preference_switch_brackets']
    bracket=next((b for b in brackets if b['endpoint_scores_separated_from_mc_noise']),next(iter(brackets),None))
    def display(value):
        field=item['field']
        if field=='maintenance':return f'€{value*12:,.0f}/year'
        if field=='persistence':return f'{value:.0%} monthly carryover of the improvement'
        if field=='reach':return f'{value:.0%} affected share'
        if field=='monthly_users':return f'{value:,.0f} eligible units/month'
        return f'€{value:,.2f}'
    if bracket:
        text=(f'{item["label"]}: tested settings of {display(bracket["lower_input"])} and {display(bracket["upper_input"])} '
                f'favor {bracket["at_lower"]} and {bracket["at_upper"]}, respectively. This brackets a modeled preference change; it is not an exact threshold or new evidence.')
        if not bracket['endpoint_scores_separated_from_mc_noise']:text+=' At least one endpoint is too close to Monte Carlo noise to establish a reliable switch.'
        return text
    a,b=item['tested_points'][0],item['tested_points'][-1]
    return (f'{item["label"]}: {display(a["input"])} versus {display(b["input"])} gives modeled launch loss risks '
            f'{a["global_launch_loss_probability"]:.1%} versus {b["global_launch_loss_probability"]:.1%}. '
            'No feasible preference switch was found on this tested range. Other assumptions and effect uncertainty remain.')
