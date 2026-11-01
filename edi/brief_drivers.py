"""Plain-English decision conditions grounded in the existing comparison results."""


def input_value(field,value,kind=None):
    if field=='persistence':return f'{value*100:.1f}% of the previous month’s improvement'
    if field=='monthly_users':return f'{value:,.0f} eligible '+('denominator units/month' if kind=='ratio' else 'people/month')
    if field=='reach':return f'{value:.1%} of eligible people receiving the change'
    if field=='maintenance':return f'€{value*12:,.0f}/year'
    if field=='margin':return f'€{value:,.2f} contribution per extra outcome'
    return f'€{value:,.0f}'


def ranked_conditions(evidence,limit=3):
    """Constraints first, then credible switches, then declared score swing.

    Score swing is a measured consequence over an explicit scenario range, not a
    universal importance score or the research value of verifying an assumption.
    No exact threshold or behavior beyond tested settings is inferred.
    """
    drivers=evidence.get('decision_drivers') or {}
    constraints=[]
    for kind,title in [('Data/design','Resolve the data/design blockers'),('Safety','Establish safety before launch'),('Reversibility','Respect the rollback requirement')]:
        items=[r['detail'] for r in drivers.get('constraints',[]) if r['binding'] and r['kind']==kind]
        if items:constraints.append({'id':kind,'title':title,'text':' '.join(items)+' Better financial assumptions cannot remove this condition.',
            'kind':'constraint','changes_preference':False,'score_swing':None})
    current=evidence['assessment']['preferred_action']
    same_baseline=drivers.get('simulation',{}).get('comparison_action_matches_full_run',True)
    rows=[]
    for item in drivers.get('counterfactuals',[]):
        points=item['tested_points']
        if not points:continue
        source=item.get('source') or {};field=item['field']
        show=lambda value:input_value(field,value,evidence.get('metric',{}).get('kind'))
        origin=source.get('source','Unverified assumption').lower()
        confidence=source.get('quality','Unknown').lower()
        text=f'Current assumption: **{show(item["current_input"])}** ({origin}; {confidence} confidence). '
        if field=='persistence':text+='This is how long the product improvement lasts, not customer retention. '
        if field=='monthly_users' and source.get('source')=='Observed experiment':text+=source.get('reason','Trial enrollment is not a validated forecast of future traffic.')+' '
        strong=[b for b in item['feasible_preference_switch_brackets'] if b.get('endpoint_scores_separated_from_mc_noise',False) and current in (b['at_lower'],b['at_upper'])]
        strong.sort(key=lambda b:abs((b['lower_input']+b['upper_input'])/2-item['current_input']))
        if strong and same_baseline:
            b=strong[0]
            alt=b['lower_input'] if b['at_lower']!=current else b['upper_input']
            old=b['upper_input'] if b['at_lower']!=current else b['lower_input']
            point=next(p for p in points if p['input']==alt)
            text+=(f'At **{show(alt)}**, **{point["preferred_feasible_action"]}** is preferred '
                   f'(launch expected net value €{point["global_launch_expected_value"]:,.0f}; loss risk {point["global_launch_loss_probability"]:.1%}). ')
            if point['preferred_feasible_action']=='Keep control' and point['global_launch_expected_value']>0:
                text+='Average profit remains positive here, but the selected extra weight on losses makes control preferable. '
            text+=f'At {show(old)}, {current} still leads. The change is between these tested settings, not an exact cutoff.'
        else:
            a,b=points[0],points[-1]
            text+=(f'Testing {show(a["input"])} to {show(b["input"])} gives launch expected values '
                   f'€{a["global_launch_expected_value"]:,.0f} to €{b["global_launch_expected_value"]:,.0f}. ')
            if not same_baseline:text+='The smaller comparison run disagrees with the full decision run; improve simulation precision before interpreting a turning point.'
            elif item['feasible_preference_switch_brackets']:text+='A possible change of choice is too close to simulation noise to locate confidently.'
            else:text+=f'No change from {current} was found at the tested settings.'
        scores=[p['global_launch_score'] for p in points]
        rows.append({'id':field,'title':item['label'],'text':text,'kind':'business assumption',
                     'changes_preference':bool(strong and same_baseline),'score_swing':max(scores)-min(scores),
                     'source':source,'tested_range':item['declared_range']})
    rows.sort(key=lambda r:(not r['changes_preference'],-r['score_swing'],r['id']))
    # Economic sensitivity is secondary while a hard constraint determines readiness.
    if constraints:return constraints[:limit]
    if rows:return rows[:limit]
    if evidence['quality'].get('errors'):
        return [{'id':'data','title':'Resolve the data/design blockers','text':'Correct '+str(evidence['quality']['errors'][0])+'. The current export cannot support a launch decision.','kind':'constraint','changes_preference':False,'score_swing':None}]
    return [{'id':'missing','title':'Confirm the missing decision inputs','text':'No supported decision-changing comparison is available. Review safety, costs and achievable reach before treating the financial scenario as a launch recommendation.','kind':'missing evidence','changes_preference':False,'score_swing':None}]


def render_conditions(evidence):
    rows=evidence.get('decision_conditions') or ranked_conditions(evidence)
    # Always show impact months, even when it is outside the three largest drivers.
    rows=[*rows,impact_months_condition(evidence)]
    return '\n\n'.join(f'{i}. **{row["title"]}:** {row["text"]}' for i,row in enumerate(rows,1))


def impact_months_condition(evidence):
    drivers=evidence.get('decision_drivers') or {}
    months=(evidence.get('scenario') or {}).get('business',{}).get('months')
    record=drivers.get('months_of_impact')
    item={'id':'months','title':'Months of impact','kind':'comparison period','changes_preference':False,'score_swing':None}
    intro=f'The current business case adds benefits and maintenance over **{months} months**. '
    if not record:
        item['text']=intro+('Fix the data/design blockers before calculating a decision-changing period.' if evidence['quality'].get('errors') else 'A supported month-by-month comparison is not available in this saved evidence; rerun the analysis to calculate it.')
        return item
    lo,hi=record['tested_range'];current=evidence['assessment']['preferred_action']
    blocked=not evidence['assessment'].get('allow_rollout',False) or (
        drivers.get('eligible_actions')==['Keep control'] and any(c['binding'] for c in drivers.get('constraints',[])))
    if blocked:
        item['text']=intro+f'Testing {lo}–{hi} whole months does not remove the data, safety or rollback requirement. No period alone makes an excluded launch eligible.'
        return item
    same_baseline=drivers.get('simulation',{}).get('comparison_action_matches_full_run',True)
    windows=[b for b in record['clear_switch_windows'] if current in (b['at_lower'],b['at_upper'])]
    windows.sort(key=lambda b:abs((b['lower_input']+b['upper_input'])/2-months))
    if windows and same_baseline:
        switch=windows[0]
        alternatives=[p for p in record['tested_points'] if p['input'] in (switch['lower_input'],switch['upper_input'])]
        descriptions=[]
        for p in alternatives:
            gradual=p['preferred_feasible_action']=='Gradual rollout'
            label='gradual-rollout' if gradual else 'global-launch'
            value=p['preferred_action_expected_net_value'] if gradual else p['global_launch_expected_value']
            risk=p['preferred_action_loss_probability'] if gradual else p['global_launch_loss_probability']
            descriptions.append(f'**{p["input"]} months → {p["preferred_feasible_action"]}** ({label} expected net value €{value:,.0f}; loss risk {risk:.1%})')
        text=intro+'; '.join(descriptions)+'. '
        if switch['upper_input']-switch['lower_input']>1:text+='Intermediate months are too close to simulation noise to locate the change more precisely. '
        else:text+='These are tested whole-month choices, not an exact continuous-time cutoff. '
        if len(record['clear_switch_windows'])>1:text+='The preferred action changes again elsewhere in the tested range: more months are not always better. '
        item['changes_preference']=True
    elif not same_baseline:text=intro+'The smaller comparison run disagrees with the full decision run; improve simulation precision before identifying a decision-changing month. '
    elif record['ambiguous_months']:text=intro+f'No reliable preference switch was established across {lo}–{hi} months; some choices are too close to simulation noise. '
    else:text=intro+f'No change from **{current}** was found across the tested {lo}–{hi} whole months. '
    item['text']=text+'This is the financial comparison period, not experiment length or proof of lasting benefit. The assumed improvement durability stays unchanged.'
    return item


def segment_summary(evidence):
    rows=evidence.get('segments',[])
    if not rows:return 'No usable group comparison was available; a group-specific launch is not assessed.'
    posterior=evidence.get('posterior') or {}
    direction=1 if posterior.get('directional_probabilities',{}).get('beneficial_direction','increase')=='increase' else -1
    positive=[f'{r["segment"]}={r["level"]}' for r in rows if r.get('p_holm',1)<.05 and direction*r['effect']>0]
    negative=[f'{r["segment"]}={r["level"]}' for r in rows if r.get('p_holm',1)<.05 and direction*r['effect']<0]
    parts=[]
    if positive:parts.append('Promising groups after the multiple-group check: '+', '.join(positive[:3])+'.')
    if negative:parts.append('Possible deterioration after that check: '+', '.join(negative[:3])+'.')
    if not parts:parts.append('No group has clear evidence after accounting for the search across groups; uncertain groups are not proven ineffective.')
    differences=any(r.get('p_holm',1)<.05 for r in evidence.get('heterogeneity_tests',[]))
    parts.append('The separate comparison finds exploratory differences between groups.' if differences else 'Different responses between groups are not established.')
    parts.append('No group-specific launch is validated.')
    return ' '.join(parts)
