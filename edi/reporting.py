"""Readable decision documents and local, versioned aggregate-evidence storage."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sqlite3
from edi.scenarios import fingerprint,LABELS
from edi.ui import clean
from edi.interpretation import segment_readout
from edi.decision_drivers import counterfactual_readout,risk_explanation
from edi.brief_drivers import render_conditions,ranked_conditions,segment_summary



def render_detailed_brief(evidence,context='',owner_note='',version=None,ai_summary=None):
    a=evidence['assessment'];primary=evidence.get('primary');model=evidence.get('posterior');scenario=evidence['scenario']
    lines=[f'# Experiment Decision Brief',f'\n{evidence["experiment_name"]} · {evidence["experiment_id"]}',
        '\nIllustrative synthetic example' if evidence['data_kind']=='synthetic example' else '\nUploaded experiment evidence',
        f'\nDecision version: {version or "Working draft"}',f'\n**Decision status:** {a["status"]}',f'\n**Preferred modeled option:** {a["preferred_action"]}',
        '\n## 1. Observed evidence',f'\n{evidence["rows"]:,} assigned units. Trust: {evidence["quality"]["verdict"]}.']
    if primary:
        lines += [f'\nPrimary outcome: {primary["metric"]}. Control: {primary["control"]:.6g}; treatment: {primary["treatment"]:.6g}.',
            f'\nRaw treatment-minus-control effect: {primary["effect"]:.6g}; 95% interval [{primary["ci_low"]:.6g}, {primary["ci_high"]:.6g}]; two-sided p={primary["p_value"]:.4g}.',
            '\nThe primary result does not establish a conventionally significant difference at 5%.' if primary['p_value']>=.05 else '\nThe primary test rejects zero at 5% under the stated design; business value still requires separate assessment.']
    lines+=['\n## 2. Inference and model-conditioned probabilities']
    if model:
        d=model['directional_probabilities']
        lines +=[f'\nBeneficial direction: {d["beneficial_direction"]}. P(benefit): {d["probability_benefit"]:.1%}; P(harm): {d["probability_harm"]:.1%}; P(useful benefit): {d["probability_useful_benefit"]:.1%}.',
            f'\nModel: {model["name"]}. Goal: {evidence["relative_goal_percent"]:g}% relative improvement. Probabilities depend on this model/prior; they are not observed success rates.']
    else:lines+=['\nInference is unavailable because required checks failed.']
    lines+=['\n## 3. Economic assumptions and evidence quality','\n| Parameter | Value | Source | Confidence | Rationale |','|---|---:|---|---|---|']
    all_values={**scenario['business'],**scenario['uncertainty'],**scenario['options']}
    for key,prov in scenario['provenance'].items():
        escape=lambda x:str(x).replace('|','/').replace('\n',' ')
        lines.append('| '+' | '.join(escape(x) for x in [LABELS.get(key,key),all_values.get(key,''),prov['source'],prov['quality'],prov['reason']])+' |')
    lines+=['\n## 4. Alternatives','\n| Option | Technical cost | Reversible | Expected net value | Loss risk | New evidence |','|---|---:|---|---:|---:|---|']
    for row in evidence.get('actions',[]):
        val='Not estimated' if row['expected_net_value'] is None else f'EUR {row["expected_net_value"]:,.0f}'
        risk='Not estimated' if row['loss_probability'] is None else f'{row["loss_probability"]:.1%}'
        lines.append(f'| {row["action"]} | EUR {row["technical_cost_eur"]:,.0f} | {"Yes" if row["reversible"] else "No / unconfirmed"} | {val} | {risk} | {row["new_evidence"]} |')
    global_option=next((r for r in evidence.get('actions',[]) if r['action']=='Global launch'),None)
    if global_option:
        lines.append(f'\nEconomic assessment: global-launch expected net value is EUR {global_option["expected_net_value"]:,.0f}, with a modeled 90% range from EUR {global_option["p05"]:,.0f} to EUR {global_option["p95"]:,.0f} and loss probability {global_option["loss_probability"]:.1%}. Expected value alone does not establish an acceptable decision.')
    if evidence.get('decision_drivers'):
        lines+=['\n### Why upside and loss probability can coexist','\n'+risk_explanation(evidence['decision_drivers']),
                '\n### Computed decision condition','\n'+counterfactual_readout(evidence['decision_drivers'])]
    lines+=['\nIteration is costed but not ranked: the effect of a redesigned feature is unknown. Monetary values are modeled, not production outcomes.',
        '\n## 5. Risks and constraints']
    for reason in a['reasons']:lines.append('\n- '+reason)
    guards=evidence.get('guardrails',[])
    lines+=['\nGuardrails: '+('; '.join(f'{g["metric"]}: {"within declared bound" if g["safe"] else "harm not ruled out"}' for g in guards) if guards else 'Not supplied; safety is unassessed.')]
    for g in guards:
        lines.append(f'\n- {g["metric"]}: harmful direction {g["harm_direction"].lower()}; upper bound on harm {g.get("harm_upper_bound",float("nan")):.6g}; declared tolerated harm {g["maximum_harm"]:.6g}. These are original metric-scale quantities for audit, not extra configurable assumptions.')
    pooling=evidence.get('partial_pooling')
    if pooling:
        h=pooling['heterogeneity']
        lines.append(f'\nHeterogeneity: the exploratory hierarchy over {pooling.get("attribute","the selected attribute")} estimates a between-group SD with median {h["tau_median"]:.6g}, 95% credible range [{h["tau_low"]:.6g}, {h["tau_high"]:.6g}]. This quantifies uncertainty about differences; it does not independently validate targeting.')
    time=evidence.get('time_profile')
    if time:
        t=time['change_in_treatment_effect']
        lines.append(f'\nTiming: the within-user early/late effect contrast is {t["effect"]:.6g}, p={t["p_value"]:.4g}. Purchase acceleration and instrument changes remain alternative explanations; persistence is still an economic assumption.')
    lines+=['\n### Segment implications', '\n'+segment_readout(evidence)]
    lines+=['\n## 6. What is not justified',
        '\n- A favorable subgroup does not rescue an uncertain primary metric or justify targeted rollout.',
        '\n- Partial pooling is not a substitute for multiplicity correction or independent confirmation.',
        '\n- Short-run timing differences do not establish novelty or persistence over months.',
        '\n- Hypothetical economics and AI suggestions are not validated finance forecasts.']
    lines+=['\n## 7. Decision conditions and next action']
    for condition in a['conditions']:lines.append('\n- '+condition)
    sensitivity=evidence.get('sensitivity',[])
    if sensitivity:lines.append('\nLargest declared scenario swing: '+('how long the improvement lasts' if sensitivity[0]['assumption']=='Effect retention' else sensitivity[0]['assumption'].lower())+'. This is not a value-of-information ranking or a reason to override a binding constraint.')
    voi=evidence.get('evsi')
    if voi:lines.append(f'\nAdditional sample ({voi["users_per_arm"]:,}/arm): estimated information value {voi["evsi"]:,.0f} score units; net value after study/delay costs {voi["net_information_value"]:,.0f}. Same quantified actions and utility as Monte Carlo; inspect numerical error and nested-sampling limits before planning a study.')
    lines+=['\n## 8. Missing evidence / context','\nConfirm incremental contribution, long-run persistence, maintenance and displaced-project costs, rollout reversibility and future eligible population. Resolve all unverified quality checks.']
    priority=[]
    if evidence['quality']['errors']:priority.append('Resolve the first blocker: '+evidence['quality']['errors'][0])
    elif not guards or not all(g['safe'] for g in guards):priority.append('Define or strengthen safety evidence before committing to a rollout.')
    else:
        low=[LABELS.get(k,k) for k,v in scenario['provenance'].items() if v['quality']=='Low' and k in ['margin','implementation','maintenance','persistence','opportunity_cost']]
        if low:priority.append('Validate weakly supported assumptions: '+', '.join(low)+'.')
        if voi:
            priority.append('Plan the specified follow-up only if feasibility and precision checks support its positive modeled net value.' if voi['net_information_value']>0 else 'Prefer cheaper financial/context validation before paying for this particular additional sample, whose modeled net value is not positive.')
    if priority:lines+=['\n### Case-specific next steps']+['\n- '+item for item in priority]
    if context:lines+=['\n### User-reported context (unverified)', '\n'+context]
    if owner_note:lines+=['\n### Owner judgment / decision note','\n'+owner_note]
    lines+=['\n## 9. Provenance and audit',f'\nData fingerprint: {evidence["data_hash"]}. Scenario fingerprint: {fingerprint(scenario)}. Engine {evidence["engine_version"]}.',
        '\nThe JSON includes methods, priors, guardrail limits, seed values, scenario provenance and confirmed scenario-change events. AI conversation text is not part of the audit record.']
    lines+=['\n## 10. Prioritized decision conditions', '\n'+render_conditions(evidence),
        '\nRanking rule: hard constraints first; then preference changes whose endpoint score gaps exceed two Monte Carlo standard errors, then the largest change in the global-launch downside-adjusted score over the declared range. These are one-input scenarios, not a research-value ranking. Untested switches and non-monotonic behavior remain possible.']
    if ai_summary:lines+=['\n## Optional AI explanation (interpretation, not additional evidence)', '\n'+ai_summary['text']]
    return '\n'.join(lines)+'\n'


def render_brief(evidence,context='',owner_note='',version=None,ai_summary=None):
    """Plain-English decision record with computed, prioritized conditions."""
    a=evidence['assessment'];primary=evidence.get('primary');model=evidence.get('posterior')
    scenario=evidence['scenario'];b=scenario['business'];guards=evidence.get('guardrails',[])
    rows=evidence.get('actions',[]);full=next((r for r in rows if r['action']=='Global launch'),None)
    preferred=next((r for r in rows if r['action']==a['preferred_action']),None)
    ref=max(abs(evidence.get('goal_reference',1.)),1e-12)
    direction=1 if not model or model['directional_probabilities']['beneficial_direction']=='increase' else -1
    lines=['# Decision brief',f'\n**{evidence["experiment_name"]} · {evidence["experiment_id"]}**',
        f'\n**Decision: {a["preferred_action"]} — {a["status"]}.**',f'\nVersion: {version or "Working draft"} · Business comparison: {b["months"]} months.']
    if evidence['data_kind']=='synthetic example':lines.append('\nIllustrative example; not evidence of production impact.')
    lines.append('\n## Why this decision')
    if evidence['quality']['errors']:lines.append('\nFix the data/design blockers before relying on the result. The current export cannot justify launch.')
    elif not guards or not all(g['safe'] for g in guards):lines.append('\nSafety is not yet established against the declared limits. A stronger financial case cannot remove that requirement.')
    elif a['preferred_action']=='Keep control':lines.append('\nThe expected upside does not justify the possible losses under the current assumptions and risk preferences. Keep the current experience while checking what could change that choice.')
    elif preferred:lines.append(f'\nThis option offers the best modeled balance of gains and losses among the available choices. Its expected net value is €{preferred["expected_net_value"]:,.0f}, with {preferred["loss_probability"]:.1%} risk of losing money. This remains a conditional recommendation, not guaranteed profit.')
    if ai_summary:lines+=['\n### AI explanation', '\n'+ai_summary['text'], '\nAI interpretation of this version; it does not change the evidence or decision rules.']
    lines.append('\n## Evidence that matters')
    if primary:
        change=direction*primary['effect']/ref*100
        lines.append(f'\n- **Observed result:** {change:+.1f}% relative improvement across {evidence["rows"]:,} assigned users. '+('The conventional test is inconclusive; this does not mean the effect is zero.' if primary['p_value']>=.05 else 'The conventional test detects a difference; launch value is a separate question.'))
    if model:
        d=model['directional_probabilities']
        lines.append(f'\n- **Effect uncertainty:** {d["probability_benefit"]:.1%} chance of benefit and {d["probability_useful_benefit"]:.1%} chance of meeting the {evidence["relative_goal_percent"]:g}% improvement goal, conditional on the Bayesian model and prior. These are not probabilities of financial success.')
    if full:lines.append(f'\n- **Launch economics:** expected net value €{full["expected_net_value"]:,.0f}; {full["loss_probability"]:.1%} loss risk; middle 90% of modeled outcomes €{full["p05"]:,.0f} to €{full["p95"]:,.0f}. These depend on business assumptions.')
    lines.append('\n- **Safety:** '+('the supplied guardrails meet their declared limits.' if guards and all(g['safe'] for g in guards) else 'not yet established.'))
    if evidence['quality']['warnings'] or evidence['quality']['unavailable']:lines.append('\n- **Data caution:** confirm the unresolved quality checks with the experiment owner.')
    lines.append('\n- **User groups:** '+segment_summary(evidence))
    lines+=['\n## What would change the choice?', '\nHere are the key drivers of the current decision:', '\n'+render_conditions(evidence)]
    if evidence.get('decision_drivers',{}).get('counterfactuals') and not any(c['kind']=='constraint' for c in ranked_conditions(evidence)):
        lines.append('\nPriority: assumptions that change the choice first, then those with the largest change in value after allowing for losses over the tested ranges. Each check changes one assumption at a time; this is not a ranking of research projects.')
    lines.append('\n## Next step')
    conditions=evidence.get('decision_conditions') or ranked_conditions(evidence)
    if conditions[0]['kind']=='constraint':lines.append('\n'+conditions[0]['title']+'. Financial scenarios cannot substitute for that evidence or requirement.')
    else:lines.append('\nConfirm **'+conditions[0]['title'].lower()+'** with its owner and evidence source before committing. A more favorable assumption is not new evidence.')
    voi=evidence.get('evsi')
    if voi:
        if abs(voi['net_information_value'])<2*voi.get('mc_outer_se',voi.get('mc_se',0)):lines.append('\nThe value of another primary-metric sample is too close to simulation noise for a clear study recommendation.')
        elif voi['net_information_value']>0:lines.append(f'\nA planned sample of {voi["users_per_arm"]:,} additional users per arm may repay its costs under the model. It would not establish engineering costs, long-term durability or missing safety evidence.')
        else:lines.append('\nThe planned extra primary-metric sample does not show enough modeled benefit to repay its costs; do not extend the test automatically.')
    iteration=next((r for r in rows if r['action']=='Iterate the feature'),None)
    if iteration:lines.append(f'\nIteration/retest budget: **€{iteration["technical_cost_eur"]:,.0f}**; the redesigned feature’s benefit remains unknown.')
    if owner_note:lines+=['\n**Owner judgment:** '+owner_note[:500]+('… (full note in appendix)' if len(owner_note)>500 else '')]
    if context:lines+=['\n**Reported context (unverified):** '+context[:300]+('… (full context in appendix)' if len(context)>300 else '')]
    lines.append('\nMethods, all assumptions, source quality and numerical details remain in the separate appendix and JSON audit.')
    return '\n'.join(lines)+'\n'


def save_version(path,evidence,context='',owner_note='',ai_summary=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=clean({'evidence':evidence,'context':context,'owner_note':owner_note,'ai_summary':ai_summary})
    digest=fingerprint(payload)
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE IF NOT EXISTS versions (experiment_id TEXT, version INTEGER, created_at TEXT, digest TEXT, payload TEXT, PRIMARY KEY(experiment_id,version))')
        conn.execute('BEGIN IMMEDIATE')
        row=conn.execute('SELECT version,digest FROM versions WHERE experiment_id=? ORDER BY version DESC LIMIT 1',(evidence['experiment_id'],)).fetchone()
        if row and row[1]==digest:return row[0]
        version=(row[0]+1) if row else 1
        conn.execute('INSERT INTO versions VALUES (?,?,?,?,?)',(evidence['experiment_id'],version,datetime.now(timezone.utc).isoformat(),digest,json.dumps(payload,allow_nan=False)))
        return version


def versions(path,experiment_id):
    if not Path(path).exists():return []
    with sqlite3.connect(path) as conn:
        rows=conn.execute('SELECT version,created_at,payload FROM versions WHERE experiment_id=? ORDER BY version',(experiment_id,)).fetchall()
    return [{'version':v,'created_at':when,**json.loads(payload)} for v,when,payload in rows]


def compare_versions(before,after):
    rows=[]
    for section in ['business','options','uncertainty','preferences']:
        a=before['scenario'][section];b=after['scenario'][section]
        for key in sorted(set(a)|set(b)):
            if a.get(key)!=b.get(key):rows.append({'field':section+'.'+key,'before':a.get(key),'after':b.get(key)})
    for key in sorted(set(before['scenario']['provenance'])|set(after['scenario']['provenance'])):
        a=before['scenario']['provenance'].get(key);b=after['scenario']['provenance'].get(key)
        if a!=b:rows.append({'field':'source / confidence: '+LABELS.get(key,key),'before':str(a),'after':str(b)})
    for key in ['status','preferred_action']:
        if before['assessment'][key]!=after['assessment'][key]:rows.append({'field':'decision.'+key,'before':before['assessment'][key],'after':after['assessment'][key]})
    for key in ['metric','relative_goal_percent','goal_reference','engine_version']:
        if before.get(key)!=after.get(key):rows.append({'field':key,'before':str(before.get(key)),'after':str(after.get(key))})
    for label,key in [('prior','posterior'),('guardrails','guardrails')]:
        a=before.get(key);b=after.get(key)
        if key=='posterior':
            a=a.get('prior') if a else None;b=b.get('prior') if b else None
        else:
            a=[(r['metric'],r['maximum_harm'],r['harm_direction']) for r in a or []]
            b=[(r['metric'],r['maximum_harm'],r['harm_direction']) for r in b or []]
        if a!=b:rows.append({'field':label,'before':str(a),'after':str(b)})
    if before['data_hash']!=after['data_hash']:rows.append({'field':'data_hash','before':before['data_hash'],'after':after['data_hash']})
    return rows
