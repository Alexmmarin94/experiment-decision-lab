"""Business-unit formatting and a decision-first preview; no state mutations here."""
import pandas as pd
import streamlit as st
from edi.scenarios import LABELS


def assumption_display(field,value,months=6):
    if field=='maintenance':return 'Annual maintenance',f'€{value*12:,.0f}/year'
    if field=='monthly_users':return 'Total eligible audience',f'{value:,.0f} eligible units/month'
    if field=='persistence':return 'How long the improvement lasts',f'{value:.0%} of the previous month’s improvement; {value**(months-1):.0%} of the initial effect in month {months}'
    if field in ('reach','pilot_fraction','margin_uncertainty'):return LABELS[field],f'{value:.1%}'
    if field=='margin':return 'Contribution per extra outcome',f'€{value:,.2f}/extra outcome unit'
    if field in ('implementation','pilot_cost','iteration_cost','opportunity_cost'):return LABELS[field],f'€{value:,.0f}'
    if field=='months':return 'Business comparison period',f'{value:g} months'
    if isinstance(value,bool):return LABELS.get(field,'Rollback constraint'),'Yes' if value else 'No'
    return LABELS.get(field,field.replace('_',' ').capitalize()),f'{value:g}'


def preview_description(current,proposal):
    rows=[]
    for field,value in proposal['changes'].items():
        group='business' if field in current['business'] else 'options'
        name,before=assumption_display(field,current[group][field],current['business']['months'])
        _,after=assumption_display(field,value,proposal['scenario']['business']['months'])
        category='Audience & time' if field in ('monthly_users','reach','months','persistence','pilot_fraction') else 'Constraints' if isinstance(value,bool) else 'Financial assumptions'
        rows.append({'category':category,'name':name,'before':before,'after':after})
    return rows


def preview_interpretation(current,preview):
    before=current['assessment'];after=preview['assessment']
    old=next(r for r in current['actions'] if r['action']=='Global launch')
    new=next(r for r in preview['actions'] if r['action']=='Global launch')
    if not before['allow_rollout']:
        return 'The business case changes, but rollout is still unavailable: '+next((r for r in after['reasons'] if 'guardrail' in r.lower() or 'blocking' in r.lower()),'the current readiness constraints remain unresolved.')+' This business update does not resolve that blocker.'
    if before['preferred_action']!=after['preferred_action']:
        return f'The preferred feasible action changes from {before["preferred_action"]} to {after["preferred_action"]}. This reflects the same downside-adjusted policy applied to different business assumptions; it is not a change in the experimental effect.'
    delta=new['decision_score']-old['decision_score']
    if abs(delta)<1:return 'The preferred feasible action is unchanged, and the modeled global-launch score changes by less than one money-equivalent unit. Review whether this assumption matters operationally despite that small modeled impact.'
    return f'{after["preferred_action"]} remains preferred among eligible actions. Global launch becomes '+('more' if delta>0 else 'less')+' attractive on the downside-adjusted score; the preference does not switch.'


def render_proposal(current,proposal):
    """Return a requested action; caller retains confirmation, audit and stale checks."""
    st.subheader('What would change if we used this assumption?')
    st.info('This is a preview. Your current scenario has not changed.')
    changes=preview_description(current['scenario'],proposal)
    for category in ('Financial assumptions','Audience & time','Constraints'):
        items=[r for r in changes if r['category']==category]
        if items:
            st.markdown('**'+category+'**')
            for r in items:st.markdown(f'- **{r["name"]}:** {r["before"]} → **{r["after"]}**')
    st.write('**Why this was proposed:** '+proposal['reason'])
    old=next(r for r in current['actions'] if r['action']=='Global launch')
    new=next(r for r in proposal['preview']['actions'] if r['action']=='Global launch')
    st.markdown('**What happens to the business case?** The financial figures below refer to global launch, even when control is the only currently eligible action.')
    st.table(pd.DataFrame([
        {'Measure':'Modeled expected net value','Current scenario':f'€{old["expected_net_value"]:,.0f}','With this change':f'€{new["expected_net_value"]:,.0f}'},
        {'Measure':'Modeled probability of losing money','Current scenario':f'{old["loss_probability"]:.1%}','With this change':f'{new["loss_probability"]:.1%}'},
        {'Measure':'Downside-adjusted score (money-equivalent)','Current scenario':f'{old["decision_score"]:,.0f}','With this change':f'{new["decision_score"]:,.0f}'},
        {'Measure':'Preferred eligible action','Current scenario':current['assessment']['preferred_action'],'With this change':proposal['preview']['assessment']['preferred_action']}]).set_index('Measure'))
    st.write(preview_interpretation(current,proposal['preview']))
    st.caption('The statistical evidence is unchanged: same data, treatment-effect estimates, priors, significance rules, metric direction and guardrail limits. Decision preferences also stay protected. Only the listed business assumptions can change after confirmation.')
    with st.expander('See detailed comparison'):
        st.dataframe(pd.DataFrame([{'Assumption':r['name'],'Current':r['before'],'Preview':r['after']} for r in changes]),hide_index=True,width='stretch')
        st.dataframe(pd.DataFrame([{'Alternative':r['action'],'Expected value (€)':r['expected_net_value'],'Loss probability':r['loss_probability'],'Downside-adjusted score':r['decision_score'],'Rollback requirement satisfied':r['eligible']} for r in proposal['preview']['actions']]),hide_index=True,width='stretch')
        st.write('The same simulation seed is reused. Small differences can reflect Monte Carlo precision; this preview does not establish that an uncertain business assumption is true.')
    a,b=st.columns(2)
    with a:confirm=st.button('Use this scenario',key='confirm_proposal',type='primary')
    with b:discard=st.button('Keep my current scenario',key='discard_proposal')
    return 'confirm' if confirm else 'discard' if discard else None
