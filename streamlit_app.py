"""A PM-friendly, six-step experiment decision workspace."""
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import json
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from edi.engine import Business
from edi.example_data import FILENAME,DICTIONARY
from edi.inference import detect_type,infer,winsor_sensitivity,effect_draws,segment_effects,arm_summary,relative_effect_draws
from edi.quality import audit
from edi.hierarchy import partial_pool
from edi.context_analysis import temporal,cohort_effects,transport
from edi.scenarios import starter,run,shared_evsi,fingerprint,SOURCES,proposed_scenario,LABELS,validate_patch
from edi.decision_policy import Preferences
from edi.uncertainty import sensitivity,Uncertainty
from edi.reporting import render_brief,render_detailed_brief,save_version,versions,compare_versions
from edi.tools import ToolSession,confirm_proposal
from edi.copilot import ask,explain_quality,configuration,DEFAULT_MODEL,DEFAULT_BASE_URL,failure_message,STRATEGIC_REVIEW,review_fingerprint
from edi.decision_drivers import build_decision_drivers,risk_explanation,counterfactual_readout,VERSION as DRIVER_VERSION
from edi.proposal_ui import render_proposal
from edi.brief_drivers import ranked_conditions,impact_months_condition
from edi.brief_ai import render_ai_option,cached_explanation
from edi.explanations import check_explanation,check_title,BRIEF_DISCLAIMER
from edi.ui import clean,dumps
from edi.presentation import posterior_chart,probability_chart,pooling_chart,sensitivity_view,reach_threshold,observed_effect_chart,weekly_effect_chart,scenario_change,financial_outcome_chart
from edi.onboarding import hero,render_onboarding,STEPS,next_step,workflow_panels
from edi.input_help import input_help

ROOT=Path(__file__).resolve().parent
DB=ROOT/'.local'/'decisions.sqlite3'
st.set_page_config(page_title='Experiment Decision Intelligence',page_icon='◈',layout='wide')


def setting(name,default=''):
    try:return st.secrets.get(name,os.getenv(name,default))
    except (FileNotFoundError,st.errors.StreamlitSecretNotFoundError):return os.getenv(name,default)


@st.cache_data
def load_example(mtime):return pd.read_csv(ROOT/'data'/FILENAME,dtype={'unit_id':str})


@st.cache_data
def calculated(df,metric,kind,denominator,goal,direction,relative_percent,prior):
    result=infer(df,metric,kind,denominator,goal)
    draws,model=effect_draws(df,result,**prior,beneficial_direction=direction,relative_goal_pct=relative_percent)
    return result,draws,model


@st.cache_data
def scenario_results(effects,scenario,quality,guards):
    result,worlds,extra=run(effects,scenario,quality,guards)
    ranked=sensitivity(effects[:5000],Business(**scenario['business']),Uncertainty(**scenario['uncertainty']))
    return result,worlds,extra,ranked


@st.cache_data
def information(model,result,scenario,allowed,direction,n,cost,delay):
    return shared_evsi(model,result,scenario,allowed,direction,n,cost,delay)


@st.cache_data
def decision_drivers(evidence,effects,version):return build_decision_drivers(evidence,effects)


def effect_chart(rows,reference,label='level'):
    # Original-scale intervals are preserved; the reference argument remains for call compatibility.
    return observed_effect_chart(rows,kind,label=label,metric_name=metric)


def show_cards(items):
    for title,text in items:
        with st.container(border=True):st.markdown('**'+title+'**');st.write(text)


def api_failure(error):st.warning(failure_message(error))


key=setting('OPENROUTER_API_KEY');model_name=setting('LLM_MODEL',DEFAULT_MODEL);base_url=setting('OPENROUTER_BASE_URL',DEFAULT_BASE_URL)
api_problem=configuration(key,model_name)
st.caption('EXPERIMENT DECISION INTELLIGENCE')
with st.sidebar:
    st.header('Workspace')
    st.caption('Evidence → decisions → contextual AI → accountable owner.')
    st.caption('AI is optional. Enable it in Welcome; provider requests happen only when you ask for an explanation, review or chat response.')
ai_enabled=bool(st.session_state.get('ai_enabled',True))
ai_active=ai_enabled and not api_problem
tabs=workflow_panels()
with tabs[0]:
    hero()
    configured=render_onboarding(ROOT,load_example)
if configured is None:
    for tab in tabs[1:]:
        with tab:st.info('Load a file and complete the guided mapping in Welcome & Upload. Setup problems are explained there; no estimates are produced from an incomplete configuration.')
    st.stop()
if not configured['analysis_started']:
    for tab in tabs[1:]:
        with tab:st.info('Ready when you are. In Welcome, click Start the example walkthrough or Start analysis of my CSV. Statistical calculations and AI actions remain inactive until you start this dataset.')
    st.stop()
df=configured['df'];source=configured['source'];source_hash=configured['source_hash']
experiment_name=configured['experiment_name'];experiment_id=configured['experiment_id']
metric=configured['metric'];kind=configured['kind'];denominator=configured['denominator'];bounded_ratio=configured['bounded_ratio']
direction=configured['direction'];sign=configured['sign'];role=configured['role'];definition=configured['definition']
relative_goal=configured['relative_goal'];reference=configured['reference'];ref=configured['ref'];zero_reference=configured['zero_reference']
goal=configured['goal'];design=configured['design'];allocation=configured['allocation'];numeric=[c for c in df if c not in ['unit_id','variant'] and pd.api.types.is_numeric_dtype(df[c]) and not c.endswith('_guardrail')]
data_hash=hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()
catcols=[c for c in df if c not in ['unit_id','variant','assigned_at','exposed_at','window_end_at','observed_through'] and c!=metric and not c.endswith('_guardrail') and not pd.api.types.is_numeric_dtype(df[c]) and 1<df[c].nunique()<=20]
precols=catcols+[c for c in df if c.endswith('_pre_7d') or c.startswith('pre_')]
with tabs[1]:
    st.header('Can we trust this experiment?')
    st.write('Green means a specific check found no issue. Amber/red findings explain what to investigate. Blue means evidence was not supplied. None of these checks can certify the entire experiment by themselves.')
    with st.expander('Optional baseline checks'):
        covariates=st.multiselect('Attributes measured before assignment',precols,default=[c for c in precols if c in {'platform','country','os','customer_type'} or c.startswith('pre_') or c.endswith('_pre_7d')],key='covariates', help=input_help('Attributes measured before assignment'))
        st.caption('Baseline attributes are optional. We compare their distributions before treatment; we do not search for variables that improve significance.')
    quality=audit(df,metric,kind,denominator,allocation,covariates,'platform' if 'platform' in df else None,design,
                  ratio_bounded=bounded_ratio)
    valid=not quality['errors']
    qkey=fingerprint(clean(quality));qcache=st.session_state.setdefault('quality_ai',{})
    request_quality=st.button('Explain these checks with AI',disabled=not ai_active,help='Makes a provider request using computed aggregate facts. The explanation cannot change the check statuses; the core checks already work without AI.')
    if request_quality:
        try:
            with st.spinner('Translating flagged checks into practical next steps…'):qcache[qkey]=explain_quality(clean(quality),key,model_name,base_url)
        except Exception as error:qcache[qkey]=None;api_failure(error)
    ai_quality=qcache.get(qkey)
    if ai_active and qkey in qcache and ai_quality is None and st.button('Retry AI check explanations'):
        try:qcache[qkey]=explain_quality(clean(quality),key,model_name,base_url);st.rerun()
        except Exception as error:api_failure(error)
    st.subheader(quality['verdict'])
    passed=[c for c in quality['checks'] if c['status']=='pass'];flagged=[c for c in quality['checks'] if c['status']!='pass']
    a,b,c=st.columns(3);a.metric('Checks passed',len(passed));b.metric('Blocking findings',len(quality['errors']));c.metric('Concerns / unavailable checks',len(flagged)-len(quality['errors']))
    st.caption('Detected faults block supported inference. Warnings need investigation. A check marked unavailable was not performed; it is not evidence that the missing optional field is defective.')
    for check in flagged:
        icon={'block':'🔴','warning':'🟠','unavailable':'🔵'}[check['status']]
        text=(ai_quality or {}).get('details',{}).get(check['check'],check_explanation(check))
        with st.container(border=True):st.markdown(f'{icon} **{check_title(check["check"])}**');st.write(text)
    with st.expander(f'✅ Passed checks ({len(passed)})'):
        for check in passed:st.markdown('✅ **'+check_title(check['check'])+'** — '+check_explanation(check))
    st.subheader('Trust assessment for this export')
    if ai_quality:st.write(ai_quality['trust_comment']);st.caption('AI interpretation; the computed status above remains authoritative.')
    elif not valid:st.error('Do not use effect estimates yet. Resolve the blocking checks above; a model or AI explanation cannot repair these design/data issues.')
    elif flagged:st.write('The supported calculations can run, but the decision is conditional. Investigate '+', '.join(c['check'] for c in flagged[:3])+'. Missing checks are not proof of a fault; they identify what the owner still needs to confirm.')
    else:st.success('No issues were detected in the supplied fields. Confirm the real assignment and measurement process with the experiment owner before acting.')
    if quality['balance']:
        imbalanced=[r for r in quality['balance'] if r.get('complete_separation') or abs(r.get('standardized_difference') or 0)>.1]
        st.write('**Were groups comparable before treatment?** '+('Some baseline differences deserve investigation: '+', '.join(sorted({r['covariate'] for r in imbalanced}))+'.' if imbalanced else 'No large measured baseline differences were flagged. This is a diagnostic, not proof of correct randomization.'))
    with st.expander('Exact computed facts and technical diagnostics'):
        st.dataframe(pd.DataFrame(quality['checks']),hide_index=True,width='stretch')
        if quality['balance']:st.dataframe(pd.DataFrame(quality['balance']),hide_index=True,width='stretch')

    next_step(2,'Next: interpret effect uncertainty and safety. Resolve blockers first; warnings remain visible.',disabled=not valid)

primary=None;draws=None;posterior=None;prior_sensitivity=[];prior_origin='Weak reference prior';winsor=None;guards=[];segments=[];interactions=[];pooled=None;time_profile=None;cohorts=[];target=None;target_draws=None;use_target=False;prior={}
with tabs[2]:
    st.header('What did we learn—and what is still unresolved?')
    st.write('Start with the overall outcome, then check safety. Segment and time views help investigate consistency and possible differences; they do not turn a non-significant primary result into a confirmed win.')
    if not valid:st.warning('Supported inference is blocked. Resolve the findings in data quality before interpreting an effect.')
    else:
        st.write(f'**Readiness for interpretation:** the supported calculations can run under the confirmed scope. The business goal is a {relative_goal:g}% improvement; the safety assessment is shown separately below. Remaining data-quality concerns stay conditional, not resolved by the model.')
    overall_tab,guard_tab,segment_tab,time_tab=st.tabs(['Overall outcome','Could it cause harm?','Differences across users','Weeks & future population'])
    with overall_tab:
        st.write('We use two complementary views. **The experiment comparison** shows what was observed and its uncertainty. **The Bayesian view** asks how plausible different improvements are after combining a stated starting assumption with these data.')
        with st.expander('What is a prior, and which starting assumption are we using?'):
            st.markdown('**Prior** means the model’s starting assumptions before this experiment. **Data** are the user outcomes in the CSV. **Posterior** means the updated model after seeing those outcomes. A prior is not another batch of observed users; it makes the model’s assumptions explicit.')
            if kind=='binary':
                st.write('Default: binomial outcomes with independent Beta(0.5, 0.5) reference priors, contributing one total pseudo-observation per arm. This prior is not fitted to the observed uplift. An independent historical prior can be declared below.')
                historical=st.checkbox('Use independently sourced historical conversion evidence',value=False, help=input_help('Use independently sourced historical conversion evidence'))
                if historical:
                    historical_rate=st.number_input('Historical conversion rate (%)',min_value=.01,max_value=99.99,value=8., help=input_help('Historical conversion rate (%)'))/100
                    strength=st.number_input('Weight of that evidence (equivalent users per arm)',min_value=1.,max_value=10000.,value=20., help=input_help('Weight of that evidence (equivalent users per arm)'))
                    origin=st.text_input('Source and period of the independent evidence', help=input_help('Source and period of the independent evidence'))
                    independent=st.checkbox('This source does not reuse outcomes from this experiment.', help=input_help('This source does not reuse outcomes from this experiment.'))
                    if independent and origin.strip():
                        prior={'prior_strategy':'historical','historical_rate':historical_rate,'prior_strength':strength}
                        prior_origin=origin
                    else:st.info('Using the weak default until independence and source are supplied.')
            else:
                prior_sd=st.number_input('Prior plausible effect scale (% of current value)',min_value=.01,value=relative_goal*5, help=input_help('Prior plausible effect scale (% of current value)'))/100*ref
                prior={'prior_scale':prior_sd}
                prior_origin='Declared zero-centered effect scale; not selected for observed uplift'
                st.write('Approximate normal effect likelihood with estimated sampling variance, and a zero-centered normal prior at the declared scale. This is not a full heavy-tail Bayesian model. Original numeric results are supplemented with a capped-mean sensitivity.')
        if valid:
            primary,draws,posterior=calculated(df,metric,kind,denominator,goal,sign,None if zero_reference or reference<0 else relative_goal,prior)
            pbenefit=posterior['directional_probabilities']
            a,b,c=st.columns(3)
            a.metric('Observed relative improvement',f'{sign*primary["effect"]/ref*100:+.1f}%')
            b.metric('Chance of benefit · model',f'{pbenefit["probability_benefit"]:.1%}',help='Posterior probability that this metric changes in its beneficial direction, conditional on the Bayesian model and prior. This is not the probability that a launch succeeds financially.')
            c.metric(f'Chance of ≥{relative_goal:g}% useful improvement',f'{pbenefit["probability_useful_benefit"]:.1%}')
            st.info('These benefit probabilities are conditional on the Bayesian model, prior and metric direction. They are not the observed share of users who benefited, and not the probability that a launch will succeed financially.')
            if kind=='binary':st.write(f'**Observed absolute change:** {primary["effect"]*100:+.2f} percentage points. For example, a move from 8% to 8.3% is +0.3 percentage points, or +3.75% relative—not a 3.75-point increase.')
            else:st.write(f'**Observed absolute change:** {primary["effect"]:+,.4g} in the defined {metric} units; relative improvement uses the observed control reference.')
            current=f'{primary["control"]:.2%}' if kind=='binary' else f'{primary["control"]:,.4g}'
            changed=f'{primary["treatment"]:.2%}' if kind=='binary' else f'{primary["treatment"]:,.4g}'
            st.write(f'**Observed:** control {current} → treatment {changed}. **Two-sided p-value:** {primary["p_value"]:.4g}. '+('The conventional 5% test does not establish a difference. That does not prove equivalence or rule out a worthwhile effect.' if primary['p_value']>=.05 else 'The test detects a difference under the declared assumptions; business value is a separate question.'))
            st.subheader('1. What does the experiment comparison establish?')
            st.write('The **dark diamond is the estimated change**. The **blue line with end caps is its 95% confidence interval**. The orange dashed line marks no change. If the confidence interval crosses zero, this conventional comparison has not clearly separated an increase from a decrease.')
            st.plotly_chart(effect_chart([{**primary,'level':'Overall outcome'}],ref),use_container_width=True)
            st.caption('The plotted interval uses the outcome’s original scale: percentage points for binary outcomes. It describes effects compatible with this procedure, not a probability distribution over effects. Relative improvement is shown separately; no rescaled interval is claimed to be an exact relative-effect interval.')
            st.subheader('2. How plausible is a useful improvement?')
            st.caption('Bayesian view: probabilities depend on the selected model and prior.')
            st.write('We include this view because “not statistically significant” does not tell us how plausible useful upside or harmful downside is. The model turns the evidence into a range of possible effects that the economic simulations can use—without replacing the original test.')
            st.markdown('**How we got here:** start with an explicit prior → update it with the experiment outcomes → obtain the posterior shown below. Taller bars mean that part of the effect range appears more often in the model’s possible outcomes. These bars are **not counts of users**.')
            if kind=='binary':
                pr=posterior['prior'];weight=pr['alpha']+pr['beta']
                st.write(f'The selected starting prior has a weight of **{weight:g} equivalent observations per arm**. '+('The default Beta(0.5, 0.5) is a standard weak reference for yes/no outcomes: it contributes one total pseudo-observation per arm, so a large experiment supplies almost all the information. It is not a claim about your company’s historical conversion.' if prior.get('prior_strategy','weak')=='weak' else 'It uses the independent historical rate and strength you declared. Stronger historical assumptions can have more influence; check the sensitivity comparison below.'))
            else:
                st.write('For this outcome, the model starts around no change and allows a declared range of plausible effects. It then updates using the estimated effect and its uncertainty. This normal approximation needs caution with heavy tails or unstable ratios; it is not a complete model of every user outcome.')
            st.plotly_chart(posterior_chart(sign*relative_effect_draws(posterior,draws,ref),relative_goal),use_container_width=True)
            st.caption('Orange: possible harm. Grey: an improvement that falls short of your business goal. Teal: meets the goal. These are model-conditioned possibilities, not a guarantee of future performance.')
            if kind=='numeric':
                winsor=winsor_sensitivity(df,metric,goal)
                st.subheader('Could a few very large values drive the average?')
                st.write(f'The p99-capped sensitivity changes {winsor["capped_rows"]:,} records using the same pooled upper cutoff for both arms. The original relative change is {sign*primary["effect"]/ref:.1%}; the capped-mean change is {sign*winsor["effect"]/ref:.1%}.')
                st.caption(winsor['interpretation']+' The original outcome remains the basis for decisions.')
            choices=[('Weak reference',{}),('Uniform Beta(1,1)',{'prior_strategy':'historical','historical_rate':.5,'prior_strength':2}),('Selected prior',prior)] if kind=='binary' else [('Half effect scale',{'prior_scale':prior['prior_scale']/2}),('Selected effect scale',prior),('Double effect scale',{'prior_scale':prior['prior_scale']*2})]
            for label,kwargs in choices:
                _,m=effect_draws(df,primary,**kwargs,beneficial_direction=sign,relative_goal_pct=None if zero_reference or reference<0 else relative_goal)
                prior_sensitivity.append({'Prior':label,**m['directional_probabilities']})
            with st.expander('Would a different starting assumption change the answer?'):
                st.write('We repeat the model with a small set of alternative priors. Similar answers are reassuring for these choices; large differences mean the assumptions deserve more investigation. Do not select whichever version looks most favorable.')
                comparison=pd.DataFrame([{'Starting assumption':r['Prior'],'Chance of benefit (%)':r['probability_benefit']*100,'Chance of meeting the goal (%)':r['probability_useful_benefit']*100} for r in prior_sensitivity])
                st.plotly_chart(px.bar(comparison,x='Starting assumption',y=['Chance of benefit (%)','Chance of meeting the goal (%)'],barmode='group',range_y=[0,100],labels={'value':'Model probability (%)','variable':'Question'}),use_container_width=True)
                st.caption('Historical source / selected rationale: '+prior_origin)
            with st.expander('How to read the probabilities and the method'):
                st.write('These are different questions, not percentages that should add up: meeting your goal is a subset of helping at all. Meaningful harm means deterioration exceeding the declared useful-effect scale.')
                st.plotly_chart(probability_chart(pbenefit),use_container_width=True)
                st.write('**What counts as benefit?** '+('A higher outcome is better.' if sign==1 else 'A lower outcome is better. An increase is not automatically a benefit.'))
                st.write('**Experiment method:** '+primary['method']+'. It compares the assigned groups using the outcome type you selected.')
                st.write('**Bayesian method:** '+posterior['name']+'. Its assumptions and prior are visible above; the exact settings remain in the downloadable appendix.')
                rel=sign*relative_effect_draws(posterior,draws,ref)*100
                st.write(f'Under this model, the middle 95% of possible relative improvements lies between **{np.quantile(rel,.025):+.1f}% and {np.quantile(rel,.975):+.1f}%**. This is a credible range under the model, not the frequentist interval shown earlier.')
        else:st.warning('Resolve the blocking checks before estimating effects.')
    with guard_tab:
        st.subheader('Can we rule out deterioration that matters?')
        st.write('Every column ending `_guardrail` appears here. Set limits from product risk—not from whichever result looks reassuring. The AI cannot change these limits.')
        guardcols=[c for c in df if c.endswith('_guardrail')]
        if not guardcols:st.warning('No guardrails were supplied. Financial attractiveness is not evidence of safety.')
        for name in guardcols:
            with st.container(border=True):
                st.markdown('**'+name+'**')
                harmful=st.selectbox('Which change would be harmful?',['Increase','Decrease'],key='harm_'+name, help=input_help('Which change would be harmful?'))
                max_relative=st.number_input('Largest tolerable relative deterioration (%)',min_value=.01,value=20.,key='guard_limit_'+name, help=input_help('Largest tolerable relative deterioration (%)'))
                if valid:
                    r=infer(df,name,detect_type(df[name]),alpha=.05/len(guardcols));gref=abs(r['control'])
                    if gref==0:
                        absolute=st.number_input('Current guardrail is zero: specify a direct deterioration limit',min_value=.000001,value=.001,key='absolute_'+name, help=input_help('Current guardrail is zero: specify a direct deterioration limit'))
                    else:absolute=gref*max_relative/100
                    bound=r['ci_high'] if harmful=='Increase' else -r['ci_low']
                    r.update(harm_direction=harmful,maximum_harm=absolute,harm_upper_bound=bound,safe=bool(bound<absolute));guards.append(r)
                    (st.success if r['safe'] else st.warning)('Within the declared deterioration limit.' if r['safe'] else 'Meaningful harm has not been ruled out. More safety evidence is needed.')
        st.caption('Joint guardrail protection uses Bonferroni-adjusted confidence intervals. “No significant deterioration” alone is not a safety assessment. Limits are protected from AI tool updates.')
    with segment_tab:
        splitcols=[c for c in covariates if c in catcols]
        st.subheader('1. Do the group estimates tell a broadly consistent story?')
        st.write('Do estimates point broadly in the same direction? Are differences mostly noise or tiny groups? This descriptive consistency check is distinct from evidence that causal effects actually differ.')
        if valid and splitcols:
            segments,interactions,omitted=segment_effects(df,splitcols,metric,kind,denominator,goal)
            if segments:
                st.caption('Dark diamonds = estimated changes. Blue lines = individual 95% confidence intervals, before multiple-comparison correction. A positive point alone is not a confirmed group win; the corrected check below evaluates the full search across groups.')
                st.plotly_chart(effect_chart([{**r,'level':r['segment']+' = '+r['level']} for r in segments],ref),use_container_width=True)
                focus=st.selectbox('Investigate one baseline attribute',splitcols, help=input_help('Investigate one baseline attribute'))
                selected=[r for r in segments if r['segment']==focus]
                st.caption('The overview uses original-scale effects; all estimates are exploratory. Baseline attributes must be measured before treatment.')
                st.subheader('2. Could a promising group just be a lucky finding?')
                st.write('**Why can the chart look positive while the conclusion is uncertain?** A diamond to the right of zero means a positive estimate. A blue interval entirely to the right supports that one comparison at its individual 5% level. Searching across many groups gives noise more opportunities to produce an apparent winner—even when no group truly benefits.')
                st.write(f'**What we correct:** the Holm procedure checks all {len(segments)} subgroup tests together. It sorts their p-values from smallest to largest and first requires the smallest to pass 0.05/{len(segments)}; later thresholds relax only while earlier tests pass. The adjusted p-values express this stricter requirement on the familiar 0.05 scale. The estimates and the blue, individual intervals do not move or disappear.')
                raw_wins=sum(r['p_value']<.05 for r in segments);corrected_wins=sum(r['p_holm']<.05 for r in segments)
                st.info(f'In this analysis: {raw_wins} of {len(segments)} group comparisons pass the individual 5% test; {corrected_wins} pass Holm across the full group family. A result that does not pass is uncertain, not proven ineffective. This family contains every group in the overview, including groups outside the selected attribute.')
                group_rows=[]
                for r in selected:
                    interpretation=('Improvement detected after correction' if sign*r['effect']>0 else 'Deterioration detected after correction') if r['p_holm']<.05 else 'Still uncertain after correction'
                    group_rows.append({'Group':r['level'],'Assigned users':r['n_control']+r['n_treatment'],'Estimated improvement (% reference)':sign*r['effect']/ref*100,'Individual test p':f'{r["p_value"]:.4g}','After Holm p':f'{r["p_holm"]:.4g}','What the corrected test supports':interpretation})
                st.dataframe(pd.DataFrame(group_rows),hide_index=True,width='stretch')
                st.caption(f'The correction includes all {len(segments)} groups, not just the rows above. “Still uncertain” does not mean “no effect”. Every finding remains exploratory.')
                st.write('**Which should guide a decision?** Use the corrected result when selecting apparent winners from this multi-group exploration: under the test assumptions, Holm limits the chance of even one false detection in this family to 5%. Individual results remain useful for generating hypotheses. A genuinely pre-specified single-group question can have a different testing plan, but you cannot choose it after seeing the chart to avoid correction. Neither result alone proves that a group-specific launch is worthwhile.')
                st.subheader('3. Is there evidence the effects really differ between groups?')
                st.write('A result detected in one group and not another does not prove the groups respond differently. This separate check asks whether their effect estimates differ more than sampling noise would explain.')
                hetero=next((r for r in interactions if r['attribute']==focus),None)
                if hetero:
                    if hetero['p_holm']<.05:st.info('The corrected comparison detects differences between these groups. That supports targeted follow-up—not an automatic platform-specific launch or an explanation of why the effects differ.')
                    else:st.info('The corrected comparison does not clearly establish different effects between these groups. That is not proof that every group responds identically.')
                else:st.info('There is not enough usable information for this comparison of group effects.')
                with st.expander('Which tests and corrections support these statements?'):
                    st.write('Holm protects against at least one false rejection across the group-test family and permits arbitrary dependence. BH controls a different false-discovery objective and needs suitable dependence assumptions. The separate across-group tests also receive Holm correction. The overview intervals remain unadjusted.')
                    st.dataframe(pd.DataFrame([{'Group':r['level'],'Original p-value':r['p_value'],'Holm-adjusted p-value':r['p_holm'],'BH-adjusted p-value':r['p_bh']} for r in selected]),hide_index=True,width='stretch')
                    if hetero:st.write(f'Across-group comparison for {focus}: Holm-adjusted p-value = {hetero["p_holm"]:.4g}. This is an approximate inverse-variance heterogeneity test, not a causal explanation.')
                if len(selected)>=2:
                    pooled=partial_pool(selected,mu_sd=max(5*goal,1e-8),tau_sd=max(2*goal,1e-8),threshold=goal,direction=sign)
                    pooled['attribute']=focus
                    st.subheader('4. Can related groups help us avoid overreacting to noise?')
                    st.caption('Method: hierarchical partial pooling. This complements, rather than replaces, the testing corrections.')
                    st.write('An extreme estimate from a small or noisy group is easy to overread. Partial pooling lets related groups inform one another. Uncertain estimates tend to move toward the shared pattern; precise groups can retain stronger differences. This is an additional model—not additional experimental data.')
                    st.markdown('**Read the chart:** the hollow grey point uses that group alone. The teal point shares information with the other groups. The connecting line shows how much the estimate changed; the teal horizontal line is its 95% credible range after sharing information. A range crossing zero still allows both benefit and harm.')
                    st.plotly_chart(pooling_chart(pooled['groups'],ref,sign),use_container_width=True)
                    moved=max(pooled['groups'],key=lambda r:abs(r['unpooled_effect']-r['pooled_effect']))
                    st.write(f'**Largest adjustment: {moved["group"]}.** Its estimate moves from {sign*moved["unpooled_effect"]/ref:+.1%} to {sign*moved["pooled_effect"]/ref:+.1%} of the overall control reference. This reduces the influence of an uncertain standalone estimate; it does not prove the original result was wrong.')
                    st.dataframe(pd.DataFrame([{'Group':r['group'],'Chance of meeting the useful-effect threshold':f'{r["probability_useful_benefit"]:.1%}'} for r in pooled['groups']]),hide_index=True,width='stretch')
                    st.write('**Would separate implementations be worth the complexity?** That needs independently credible differences plus segment value and maintenance cost. This demo does not calculate a segmented-launch payoff; discuss strategic importance in AI rather than treating the largest estimate as an automatic target.')
                    h=pooled['heterogeneity']
                    st.write(f'**How much might groups genuinely differ?** The model gives a {h["probability_tau_above_useful_effect"]:.1%} chance that their spread around the shared pattern exceeds the useful-effect scale. This describes overall variation, not the probability that a specific targeted launch is justified.')
                    with st.expander('What assumptions let groups share information?'):
                        show_cards([('One shared pattern, room for differences','Each group can have its own effect. The model also learns a common pattern and how widely group effects vary around it.'),('A starting assumption about that variation','A hyperprior is simply a prior about the shared pattern or the amount of variation. We allow differences without assuming they must be large; the experiment updates that starting view.'),('What to be cautious about','This uses an approximate normal model for group estimates. With only a few groups, the starting assumptions can still influence how much information is shared. Shrinkage does not replace the multiple-testing corrections or validate post hoc targeting.')])
                        curve=pd.DataFrame(pooled['spread_curve']);curve['Possible group spread (% reference)']=curve.spread/ref*100
                        curve['Starting assumption']=curve.prior_mass*100;curve['After the experiment']=curve.posterior_mass*100
                        st.write('The curves below compare what the model allowed before looking at these group estimates with what it supports afterward. More overlap means the experiment has not taught us much about the amount of group variation.')
                        st.plotly_chart(px.line(curve,x='Possible group spread (% reference)',y=['Starting assumption','After the experiment'],labels={'value':'Share of modeled possibilities (%)','variable':'Model stage'}),use_container_width=True)
                        st.caption(f'The model starts the shared effect around zero with a scale of {5*relative_goal:g}% of the overall reference, and the spread with a half-normal scale of {2*relative_goal:g}%. These are disclosed assumptions tied to your useful-effect goal, not verified business facts. Exact hyperprior settings are in the detailed export.')
            if omitted:st.caption('Groups omitted for inadequate overlap: '+', '.join(omitted))
        else:st.info('Add/select pre-treatment categorical splits to unlock this view, after resolving any validity blockers.')
        st.warning('All subgroup exploration is post hoc unless independently pre-specified. Corrections and shrinkage reduce different risks; neither erases selection, establishes a causal explanation, or validates a targeted launch.')
    with time_tab:
        st.subheader('How did the overall effect vary from week to week?')
        st.write('Each row includes all assigned users from that calendar week, across every platform and country, using the same completed outcome definition. The first week is at the top; the latest is at the bottom. This compares assignment cohorts, not cumulative users or daily event totals.')
        if valid and 'assigned_at' in df:
            cohorts=cohort_effects(df,metric,kind,denominator)
            if cohorts:
                st.plotly_chart(weekly_effect_chart(cohorts,kind,metric),use_container_width=True)
                st.caption('Dates are UTC Monday–Sunday ranges. The first/last week can contain only part of that range in this export; hover for arm sizes and observed outcomes. Lines are marginal 95% effect intervals, not an estimated decay curve. Cohort tests retain Holm adjustment in the audit.')
                st.write('**Decision implication:** investigate strong calendar/cohort changes before extrapolating. These patterns can reflect audience mix, calendar events or measurement changes; they do not prove novelty or persistence after launch.')
            else:st.info('No week has enough usable users in both arms for the supported comparison. A minimum of 30 per arm per displayed week is required.')
        else:st.info('This view requires valid outcomes and assignment timestamps. Dates are optional for the global analysis; a general daily-panel evolution is not implemented.')
        with st.expander('Separate question: does the effect change early versus late for the same users?'):
            st.write('This uses two equal-duration windows on the same assigned users. It is different from the weekly calendar view above. Purchase acceleration can resemble novelty; short-window changes do not establish long-term durability.')
            if valid and kind!='ratio' and {'assigned_at','window_end_at','observed_through'}<=set(df):
                possibilities=[c for c in numeric if c!=denominator and pd.api.types.is_numeric_dtype(df[c])]
                early=st.selectbox('Early-window outcome',possibilities,index=possibilities.index('converted_early') if 'converted_early' in possibilities else 0,help=input_help('Early-window outcome'))
                late=st.selectbox('Late-window outcome',possibilities,index=possibilities.index('converted_late') if 'converted_late' in possibilities else min(1,len(possibilities)-1),help=input_help('Late-window outcome'))
                compatible=st.checkbox('These are equal-duration, complete windows of the same outcome for every assigned user.',value=source=='Preloaded example' and metric=='converted_7d',help=input_help('These are equal-duration, complete windows of the same outcome for every assigned user.'))
                if compatible and early!=late and all(np.isfinite(df[c]).all() for c in [early,late]):
                    time_profile=temporal(df,early,late,metric,kind)
                    time_kind='binary' if detect_type(df[early])=='binary' and detect_type(df[late])=='binary' else 'numeric'
                    st.plotly_chart(observed_effect_chart([{**time_profile[k],'level':label} for k,label in [('early','Early window'),('late','Late window')]],time_kind,metric_name='per-user window outcome'),use_container_width=True)
                    t=time_profile['change_in_treatment_effect'];st.write(f'The late-minus-early contrast has p={t["p_value"]:.3g}. The calculation retains within-user covariance; this is exploratory timing evidence, not a long-run forecast.')
            else:st.info('Needs comparable early/late outcomes and maturity information. Time-specific ratios require additional modeling and are not supported here.')
        with st.expander('Optional: a different future population mix'):
            splits=[c for c in covariates if c in catcols]
            if valid and splits and kind!='ratio':
                attribute=st.selectbox('Future population attribute',splits, help=input_help('Future population attribute'))
                shares=df[attribute].astype(str).value_counts(normalize=True).sort_index()
                table=st.data_editor(pd.DataFrame({'Group':shares.index,'Future share (%)':shares.values*100}),hide_index=True,disabled=['Group'],column_config={'Future share (%)':st.column_config.NumberColumn(min_value=0.,max_value=100.,format='%.2f',help='Enter the forecast percentage for each group; all rows must add to 100%. Use an external population forecast or clearly label a hypothetical scenario.')},key='target_'+attribute+'_'+data_hash)
                try:
                    target_draws,target=transport(df,attribute,dict(zip(table.Group,table['Future share (%)']/100)),metric,kind,prior_kwargs=prior,beneficial_direction=sign)
                    use_target=st.checkbox('Use this hypothetical future mix for the economic scenarios',value=False, help=input_help('Use this hypothetical future mix for the economic scenarios'))
                    st.caption('Requires overlap and within-group transportability; it does not fix unmeasured differences. EVSI needs a corresponding future stratified sampling design and is disabled for this optional path.')
                except ValueError as error:st.error(str(error))
            else:st.info('Future user-share transport needs valid outcomes and categorical splits. Ratios need denominator weights instead and are not transported here.')

    next_step(3,'Next: model the value and downside under explicit business assumptions. Safety and validity conditions remain in force.',disabled=not valid)

# Business state is distinct from protected statistical evidence.
workspace=data_hash+'_'+metric+'_'+kind+'_'+str(denominator)
if st.session_state.get('scenario_workspace')!=workspace:
    st.session_state.scenario_workspace=workspace;st.session_state.scenario=starter(df,metric,kind,denominator)
    st.session_state.scenario_epoch=0;st.session_state.business_events=[];st.session_state.pending_proposal=None
scenario=st.session_state.scenario
result=None;worlds=None;extra=None;tornado=[];voi=None
drivers=None
with tabs[3]:
    st.header('What should we do under the current evidence?')
    st.write('A practical starting case is already filled in. The CSV informs observed traffic and outcomes; it cannot tell us engineering cost, contribution margin or long-term persistence. Those starting values are labeled assumptions—not a universal “best case”. Adjust what you know, then ask AI which changes matter.')
    include_gradual=st.checkbox('Also compare a gradual launch to part of the audience',value=scenario['options'].get('include_gradual',False),key='include_gradual_'+fingerprint(workspace)[:12], help=input_help('Also compare a gradual launch to part of the audience'))
    st.caption('Off by default: compare launching to the intended audience, keeping control, and the budget for iteration. A gradual launch is a smaller first deployment—not a new treatment effect and not automatically a new experiment.')
    if include_gradual!=scenario['options'].get('include_gradual',False):
        before=fingerprint(scenario);old=json.loads(json.dumps(scenario));scenario=json.loads(json.dumps(scenario))
        scenario['options']['include_gradual']=include_gradual
        scenario['provenance']['include_gradual']={'source':'Expert judgment','quality':'Low','reason':'Alternatives explicitly selected by the user.'}
        st.session_state.scenario=scenario
        st.session_state.business_events.append({'type':'manual_alternative_selection','before':before,'after':fingerprint(scenario),'before_scenario':old,'at':datetime.now(timezone.utc).isoformat()})
    st.info('Population and coverage work together: **total eligible people × affected share = people expected to receive the change**. For example, 100,000 eligible people and 40% coverage means 40,000 receiving it—not 40,000 conversions. If your population estimate already includes only recipients, use 100% coverage.')
    prefix=fingerprint(workspace)[:12]+str(st.session_state.scenario_epoch)
    with st.form('business_'+prefix):
        b=scenario['business'];o=scenario['options'];u=scenario['uncertainty'];p=scenario['preferences']
        st.markdown('**1. Define the central business case**')
        c1,c2,c3=st.columns(3)
        with c1:
            users=st.number_input('Total eligible people per month' if kind!='ratio' else 'Total eligible denominator units per month',min_value=1,value=int(b['monthly_users']),key=prefix+'users', help=input_help('Total eligible people per month' if kind!='ratio' else 'Total eligible denominator units per month'))
            margin=st.number_input('Contribution per extra purchase (€)' if kind=='binary' else 'Contribution per extra outcome unit (€)',min_value=.000001,value=float(b['margin']),key=prefix+'margin', help=input_help('Contribution per extra purchase (€)' if kind=='binary' else 'Contribution per extra outcome unit (€)'))
            cost=st.number_input('One-off global launch budget (€)',min_value=0.,value=float(b['implementation']),key=prefix+'cost', help=input_help('One-off global launch budget (€)'))
        with c2:
            annual=st.number_input('Ongoing maintenance per year (€)',min_value=0.,value=float(b['maintenance']*12),key=prefix+'annual', help=input_help('Ongoing maintenance per year (€)'))
            iteration=st.number_input('Budget to iterate and retest (€)',min_value=0.,value=float(o['iteration_cost']),key=prefix+'iterate', help=input_help('Budget to iterate and retest (€)'))
            opportunity=st.number_input('Cost of displacing another project (€)',min_value=0.,value=float(o['opportunity_cost']),key=prefix+'opportunity',help='If a delay is not priced, leave zero and describe it to AI as an unpriced constraint—not as evidence of no cost.')
        with c3:
            reach=st.slider('Expected affected share (%)',1,100,int(b['reach']*100),key=prefix+'reach', help=input_help('Expected affected share (%)'))/100
            months=st.slider('How many months of impact should we compare?',1,36,int(b['months']),key=prefix+'months', help=input_help('How many months of impact should we compare?'))
            retention=b['persistence']
            styles={'Balanced':1.,'More cautious about losses':2.,'Focus on expected money':0.}
            default_style=next((s for s,v in styles.items() if v==p['loss_weight']),'Balanced')
            style=st.selectbox('How should we weigh downside?',list(styles),index=list(styles).index(default_style),key=prefix+'style', help=input_help('How should we weigh downside?'))
        st.caption('For numeric revenue, use a contribution-margin factor (e.g. €0.30 per €1 of extra spend), not gross revenue twice. For ratios, use future denominator volume. Costs and the duration of the improvement are assumptions to validate.')
        st.caption('The comparison period is the number of months over which we add benefits and maintenance costs. It is not the duration of the experiment. Use the same period for every choice and one your team can plan with reasonable confidence.')
        with st.expander('Rollback constraints and optional gradual-launch budget'):
            reversible=st.checkbox('A global launch can be rolled back',value=o['global_reversible'],key=prefix+'reversible', help=input_help('A global launch can be rolled back'))
            require_reversible=st.checkbox('Only consider reversible rollouts',value=p['require_reversible'],key=prefix+'require', help=input_help('Only consider reversible rollouts'))
            if include_gradual:
                st.write('A global launch reaches your configured affected share. A gradual launch initially reaches only the fraction below **of those recipients**; it has a separate technical budget. It produces new causal evidence only if you design and retain a randomized holdout.')
                pilot_fraction=st.slider('Start with this share of the intended recipients (%)',1,90,int(o['pilot_fraction']*100),key=prefix+'fraction', help=input_help('Start with this share of the intended recipients (%)'))/100
                pilot_cost=st.number_input('Gradual rollout budget (€)',min_value=0.,value=float(o['pilot_cost']),key=prefix+'pilot', help=input_help('Gradual rollout budget (€)'))
            else:
                pilot_fraction=o['pilot_fraction'];pilot_cost=o['pilot_cost']
                st.caption('Gradual launch is excluded from both the decision comparison and the follow-up information model.')
        with st.expander('Projection assumptions kept in the background'):
            retention=st.slider('How much of the improvement remains each month? (%)',10,100,int(round(b['persistence']*100)),key=prefix+'durability',help='This is benefit durability, not user retention. 90 means the product improvement keeps 90% of its previous month’s size: 100% in month 1, 90% in month 2, 81% in month 3. Use longer follow-up or comparable historical experiments. The short test does not identify this input; do not increase it just to favor launch.')/100
            st.write(f'All financial results cover the same {months}-month comparison period. The central scenario assumes the product improvement fades: about {retention**(months-1):.0%} of the first-month benefit remains in the final month under the central assumption. This concerns the size of the product improvement over time, not customer retention. It is not estimated by the seven-day experiment. The sensitivity chart checks whether faster fading would change the economics; AI can preview a different assumption for explicit confirmation.')
        with st.expander('2. Review evidence sources and advanced assumptions—and how much to trust them'):
            st.write('Every assumption has a source, evidence-quality label and rationale. Confidence here is a human judgment label, not a calibrated probability or statistical confidence level. “Observed experiment” can support traffic pace, not a validated future forecast. AI suggestions remain unverified until you obtain better evidence.')
            keys=list(scenario['provenance'])
            provenance_rows=pd.DataFrame([{'Parameter':LABELS.get(k,k),'Source':scenario['provenance'][k]['source'],'Confidence':scenario['provenance'][k]['quality'],'Why / evidence':scenario['provenance'][k]['reason']} for k in keys])
            edited=st.data_editor(provenance_rows,disabled=['Parameter'],hide_index=True,column_config={'Source':st.column_config.SelectboxColumn(options=SOURCES,help='Choose where this estimate came from. A hypothetical or AI suggestion is not an observed financial fact.'),'Why / evidence':st.column_config.TextColumn(help='Record a source, period, estimate or explicit reasoning supporting the value. Revise it when the input changes.'),'Confidence':st.column_config.SelectboxColumn(options=['Low','Medium','High'],help='High should mean strong comparable evidence or an approved estimate; Low means an assumption needing validation. This label does not silently narrow the model’s distributions.')},key=prefix+'provenance')
        with st.expander('Advanced: uncertainty around those estimates'):
            margin_cv=st.slider('Relative uncertainty in contribution (%)',0,100,int(b['margin_uncertainty']*100),key=prefix+'mcv', help=input_help('Relative uncertainty in contribution (%)'))/100
            cost_cv=st.slider('Relative uncertainty in implementation cost (%)',0,50,int(u['cost_cv']*100),key=prefix+'ccv', help=input_help('Relative uncertainty in implementation cost (%)'))/100
            st.caption('The other starting spreads remain explicit in the evidence export: population CV 10%, reach SD 5 percentage points, monthly improvement-carryover SD 5 percentage points, unmeasured transport CV 15%; independent inputs by default. Beta keeps reach and improvement carryover bounded; lognormal inputs remain positive. Boundary means are fixed. These spreads are hypothetical judgments, not fitted facts.')
        submitted=st.form_submit_button('Update the business scenario')
    if submitted:
        before=fingerprint(scenario)
        new=json.loads(json.dumps(scenario))
        new['business'].update(monthly_users=users,margin=margin,implementation=cost,maintenance=annual/12,months=months,reach=reach,persistence=retention,margin_uncertainty=margin_cv)
        new['options'].update(pilot_fraction=pilot_fraction,pilot_cost=pilot_cost,iteration_cost=iteration,opportunity_cost=opportunity,global_reversible=reversible,include_gradual=include_gradual)
        new['preferences'].update(loss_weight=styles[style],require_reversible=require_reversible)
        new['uncertainty']['cost_cv']=cost_cv
        for i,k in enumerate(keys):new['provenance'][k]={'source':edited.iloc[i]['Source'],'quality':edited.iloc[i]['Confidence'],'reason':edited.iloc[i]['Why / evidence']}
        try:
            validate_patch({**new['business'],**new['options']})
            if any(v['source'] not in SOURCES or v['quality'] not in ['Low','Medium','High'] for v in new['provenance'].values()):raise ValueError('Choose a source and confidence for every assumption.')
        except ValueError as error:
            st.error('Scenario unchanged: '+str(error));new=scenario
        previous_values={**scenario['business'],**scenario['options'],**scenario['uncertainty']}
        new_values={**new['business'],**new['options'],**new['uncertainty']}
        for k,value in new_values.items():
            if value!=previous_values.get(k) and new['provenance'].get(k)==scenario['provenance'].get(k):
                new['provenance'][k]={'source':'Expert judgment','quality':'Low','reason':'Manually changed in the workspace; confirm the source and rationale for this revised value.'}
        prior_scenario=json.loads(json.dumps(scenario))
        st.session_state.scenario=new;scenario=new
        if fingerprint(new)!=before:st.session_state.business_events.append({'type':'manual_scenario_update','before':before,'after':fingerprint(new),'before_scenario':prior_scenario,
            'at':datetime.now(timezone.utc).isoformat()})
    st.write(f'**Active scenario:** {scenario["business"]["monthly_users"]:,} eligible units/month × {scenario["business"]["reach"]:.0%} expected to receive the change = **{scenario["business"]["monthly_users"]*scenario["business"]["reach"]:,.0f} recipients/month**, compared over **{scenario["business"]["months"]} months**. These are not expected conversions.')
    st.caption('Coverage means future delivery of the changed experience. Do not reduce it again for non-conversion or behavior already averaged into the experiment’s intention-to-treat effect.')
    if valid:
        active_effects=sign*(target_draws if use_target else draws)
        result,worlds,extra,tornado=scenario_results(active_effects,scenario,quality,guards)
        assessment=result['assessment']
        drivers=decision_drivers(clean({'assessment':assessment,'scenario':scenario,'quality':quality,'guardrails':guards,'actions':result['actions'],'sensitivity':tornado}),active_effects,DRIVER_VERSION)
        delta=scenario_change(st.session_state.get('last_decision_view'),scenario,result,workspace)
        if delta:st.session_state.last_business_change=delta
        st.session_state.last_decision_view={'workspace':workspace,'scenario':json.loads(json.dumps(scenario)),'result':result}
        if delta:
            st.subheader('What changed in this scenario?')
            st.dataframe(pd.DataFrame(delta['changes']),hide_index=True,width='stretch')
            st.write(f'Global-launch expected net value: €{delta["previous_value"]:,.0f} → €{delta["current_value"]:,.0f}; modeled loss risk: {delta["previous_loss_risk"]:.1%} → {delta["current_loss_risk"]:.1%}. Modeled preference: {delta["previous_preference"]} → {delta["current_preference"]}. The experimental evidence was not changed by these business edits.')
        full=next(r for r in result['actions'] if r['action']=='Global launch')
        c1,c2,c3=st.columns(3);c1.metric('Expected value · global launch',f'€{full["expected_net_value"]:,.0f}');c2.metric('Chance of losing money',f'{full["loss_probability"]:.1%}');c3.metric('Preferred modeled option',assessment['preferred_action'])
        st.subheader('Alternatives side by side')
        st.dataframe(pd.DataFrame([{'Alternative':r['action'],'Technical cost (€)':r['technical_cost_eur'],'Reversible':'Yes (assumed)' if r['reversible'] and r['action']!='Keep control' else 'Yes' if r['action']=='Keep control' else 'No / unconfirmed','New evidence':r['new_evidence'],
            'Expected net value':f'€{r["expected_net_value"]:,.0f}' if r['expected_net_value'] is not None else 'Not estimated: new effect unknown','Chance of loss':f'{r["loss_probability"]:.1%}' if r['loss_probability'] is not None else 'Not estimated'} for r in result['actions']]),hide_index=True,width='stretch')
        st.caption('Reversible means product rollback is possible, not that technical spending can be recovered. Iteration has a real planning budget but no estimated return until a new effect model is justified. Gradual rollout does not automatically generate causal evidence: retain a randomized holdout if you want that. The decision score gives downside extra weight; it is not a profit forecast.')
        st.subheader('What might this launch be worth?')
        st.write('These are hypothetical financial outcomes, not observed revenue. Repeated simulation combines effect uncertainty with the declared business distributions; the zero line is the point where benefits repay all modeled costs.')
        st.plotly_chart(financial_outcome_chart(worlds['Global launch'],scenario['business']['months']),use_container_width=True)
        st.write(f'The middle 90% of simulated global-launch outcomes ranges from €{full["p05"]:,.0f} to €{full["p95"]:,.0f}. These 20,000 worlds vary experimental effect, future reach/population, margin, cost, benefit durability and an unmeasured transport multiplier. Uncertainty does not disappear because expected value is positive.')
        st.write('**Why can launch lead while losing money remains plausible?** '+risk_explanation(drivers))
        st.subheader('Which assumptions move the economics across break-even?')
        st.write('We change one assumption at a time while retaining uncertainty in the others. Longer lines mean that the chosen range moves expected net value more. Points on opposite sides of the orange break-even line show a range that could turn an expected loss into an expected gain. The dark dotted line is the current estimate; coverage is highlighted in teal.')
        b=Business(**scenario['business']);opp=scenario['options']['opportunity_cost']
        chart,ranges=sensitivity_view(tornado,b,opp,full['expected_net_value'])
        st.plotly_chart(chart,use_container_width=True)
        top=tornado[0]
        top_label='how long the improvement lasts' if top['assumption']=='Effect retention' else top['assumption'].lower()
        st.write(f'**Largest declared swing: {top_label}** (about €{top["swing"]:,.0f}). This ranks the ranges we chose—not how much a research project is worth or which change is operationally possible. Hover over each endpoint to see its input and resulting value.')
        st.write('**What currently has priority:** '+drivers['next_action'])
        needed_reach=reach_threshold(float(active_effects.mean()),b,opp)
        if needed_reach is None:st.warning('At the modeled average effect, increasing coverage alone cannot produce a positive expected benefit. More recipients do not fix an unfavorable effect.')
        elif needed_reach>1:st.warning(f'Central break-even would require about {needed_reach:.0%} of eligible people to receive the change. That is above 100%: coverage alone cannot repay the current costs. Consider lower cost, a stronger effect or a different initiative.')
        else:st.info(f'**Coverage needed for central break-even: about {needed_reach:.1%}**, or {needed_reach*b.monthly_users:,.0f} recipients/month. You currently assume {b.reach:.0%} ({b.reach*b.monthly_users:,.0f} recipients). This is an expected-value threshold under central inputs, not a safety guarantee or a low-loss-risk threshold.')
        with st.expander('See the input ranges behind these lines'):
            st.write('Read each row as a what-if: changing only that assumption from the lower to the higher setting gives the two expected net values shown. The ranges are illustrative—not confidence limits or an instruction that the higher setting is achievable.')
            st.dataframe(ranges,hide_index=True,width='stretch')
        with st.expander('When might the launch repay its upfront cost?'):
            st.write('We pay the launch budget at the start, then add modeled benefits and maintenance each month. The line is expected cumulative net value. The shaded band contains the middle 90% of simulated outcomes at each month. Crossing zero on the mean line does not imply every scenario has paid back.')
            path=pd.DataFrame(extra['trajectory'])
            fig=go.Figure([go.Scatter(x=path.month,y=path.p95,mode='lines',line={'width':0},name='Upper end of modeled range'),go.Scatter(x=path.month,y=path.p05,mode='lines',line={'width':0},fill='tonexty',fillcolor='rgba(13,148,136,0.15)',name='Middle 90% of simulated outcomes'),go.Scatter(x=path.month,y=path['mean'],name='Expected net value',line={'color':'#0D9488'})])
            fig.add_hline(y=0,line_dash='dash');fig.update_layout(xaxis_title='Months after launch',yaxis_title='Cumulative net value after costs (€)');st.plotly_chart(fig,use_container_width=True)
            st.caption('This is a projection, not observed revenue. The amount of benefit that lasts is assumed; changing a time window cannot establish long-term persistence.')
        st.subheader('Should we run the experiment longer?')
        st.write('The current uncertainty may lead us to choose the wrong option. A follow-up can be valuable if it helps us avoid a loss or recognize a worthwhile launch—but only if that improvement in choices repays the cost of collecting and waiting for the data.')
        st.markdown('**What the calculation does:**\n1. Simulate possible results from the additional users you specify.\n2. Update the effect model and choose again from the same feasible options.\n3. Compare the improved choices with today’s choice, including study and delay costs.')
        st.caption('This is the expected value of sample information (EVSI). It does not predict that the next experiment will be significant. It learns about the primary effect only—not missing safety evidence, engineering costs or the effect of a redesigned feature.')
        n=st.number_input('Additional users in each arm',min_value=30,max_value=100000,value=2000,step=1000, help=input_help('Additional users in each arm'))
        study_cost=st.number_input('Cost of collecting the new sample (€)',min_value=0.,value=3000., help=input_help('Cost of collecting the new sample (€)'))
        delay_cost=st.number_input('Cost of waiting for this evidence (€)',min_value=0.,value=1000., help=input_help('Cost of waiting for this evidence (€)'))
        st.caption(f'Value of perfect information over the same feasible choices: {assessment["perfect_information_value"]:,.0f} decision-score units. This is an upper bound on information value, not an experiment budget.')
        if not use_target:
            voi=information(posterior,primary,scenario,assessment['allow_rollout'],sign,n,study_cost,delay_cost)
            a,b=st.columns(2);a.metric('Value of better-informed choices',f'{voi["evsi"]:,.0f}');b.metric('Value after study and waiting costs',f'{voi["net_information_value"]:,.0f}')
            st.caption('These use the same decision-value scale as the choice model: losses can count more heavily than gains. They are not promised extra euros. Study costs enter future outcomes before that downside preference, so their decision-value cost can exceed the cash bill.')
            if abs(voi['net_information_value'])<2*voi['mc_outer_se']:st.info('The net estimate is too close to simulation noise to support a confident study decision. Treat it as inconclusive.')
            else:st.write('Under this planning model, '+('collecting the sample may improve the choice enough to justify its costs.' if voi['net_information_value']>0 else 'collecting this sample does not appear to repay its costs. Consider cheaper evidence or revising the initiative instead of extending a test automatically.'))
            if not assessment['allow_rollout']:st.caption('Because current safety/readiness restricts the feasible actions to control, primary-only information cannot unlock rollout in this model. A study designed to resolve guardrail uncertainty is a different question.')
            with st.expander('Information-model scope and numerical precision'):
                st.write(voi['limitations']);st.write(voi['cost_handling']);st.write(f'Outer-simulation SE: {voi["mc_outer_se"]:,.0f} score units; typical inner-score MC SE: {voi["typical_inner_mc_se"]:,.0f}. Scores are money-equivalent preferences, not realized euros.')
        else:st.info('For a transported target population, specify a future stratified sampling plan before estimating EVSI.')
    else:
        active_effects=None;assessment={'status':'More evidence required','preferred_action':'Keep control','allow_rollout':False,'reasons':['Resolve the data/design blockers.'],'conditions':['Resolve all blocking checks'],'objective':'Unavailable pending valid evidence'}
        st.warning('Financial simulation is blocked by data/design issues. The starting assumptions remain available to review.')

    next_step(4,'Next: interpret the evidence with company context, or go directly to the final brief. AI is optional.')

snapshot=clean({'engine_version':'0.7','experiment_name':experiment_name,'experiment_id':experiment_id,'data_kind':'synthetic example' if source=='Preloaded example' else 'uploaded experiment',
    'data_hash':data_hash,'schema_mapping':configured['mapping'],'rows':len(df),'metric':{'name':metric,'kind':kind,'denominator':denominator,'business_role':role,'definition':definition,'direction':direction},
    'relative_goal_percent':relative_goal,'goal_reference':ref,'quality':quality,'primary':primary,'posterior':posterior,'prior_origin':prior_origin,'prior_sensitivity':prior_sensitivity,'winsorized_sensitivity':winsor,'guardrails':guards,
    'segments':segments,'heterogeneity_tests':interactions,'partial_pooling':pooled,'time_profile':time_profile,'cohorts':cohorts,'target_population':target,'using_target_effects':use_target,
    'scenario':scenario,'actions':result['actions'] if result else [],'assessment':assessment,'sensitivity':tornado,'evsi':voi,
    'business_events':st.session_state.business_events,'methods':{'confidence':'95% marginal primary intervals; Bonferroni guardrails','segments':'Holm/BH subgroup family and separate Holm omnibus family','pooling':'Normal-normal hierarchy with hyperprior quadrature'},
    'limitations':['No CSV can certify causal validity','No CUPED, cluster or sequential inference','Normal approximations for numeric/ratio and hierarchical effect likelihoods','Short-run time patterns do not prove novelty/persistence','Iteration return is not identifiable without a new effect model','Nested EVSI maximization has finite-inner-sampling bias'],
    'seeds':{'posterior':17,'economic_worlds':23,'hierarchy':71,'evsi':91}})

if drivers is None:drivers=decision_drivers(snapshot,active_effects,DRIVER_VERSION)
drivers['evsi_scope']['result']=voi
snapshot['decision_drivers']=clean(drivers)
snapshot['decision_conditions']=ranked_conditions(snapshot)
snapshot['months_of_impact_condition']=impact_months_condition(snapshot)

with tabs[4]:
    st.header('Analyze what is driving this decision')
    if st.session_state.get('confirmed_scenario_notice'):
        st.success(st.session_state.pop('confirmed_scenario_notice')+' '+assessment['preferred_action']+' · '+assessment['status']+'.')
    st.write('The Strategic Review challenges the active business case: what actually constrains the choice, which calculated what-ifs could change it, and which missing fact deserves attention first. The Decision Brief in the final step records the evidence and owner’s decision; this review helps you think through it.')
    st.markdown('**Understand** the evidence · **Challenge** assumptions · **Explore** calculated what-ifs · **Decide** with an accountable owner. AI reasons from statistical outputs and controlled calculations, not individual rows from the CSV.')
    st.subheader('Get a first AI reading—no extra input needed')
    st.write('One click uses the experiment results and active business scenario to explain the real decision, challenge its assumptions, and identify what could change the answer. You can add company context now or enrich the conversation afterward.')
    with st.form('review_context_'+source_hash,enter_to_submit=False,border=False):
        review_action=st.container()
        with st.expander('Optional: add company context to personalize the review',expanded=False):
            st.caption('You can leave this empty. Useful context includes user pain, economic priorities, engineering constraints or outside evidence. Notes inform the review; changing numeric assumptions still requires a separate preview and confirmation.')
            c1,c2=st.columns(2)
            with c1:show_cards([('User pain and initiative intent','“We simplified checkout to help first-time mobile buyers who abandon payment.”'),('Business priorities','“Contribution matters more than gross revenue; we cannot accept a material crash increase.”')])
            with c2:show_cards([('Metric definitions and outside evidence','“Revenue includes refunds. A comparable past launch lost its conversion benefit after a month.”'),('Capacity, costs and constraints','“Maintenance would cost €15,000 a year and delay another project by two weeks. We need a reversible option.”')])
            context_key='context_'+source_hash
            if context_key not in st.session_state and 'context_'+data_hash in st.session_state:st.session_state[context_key]=st.session_state['context_'+data_hash]
            context=st.text_area('Optional initiative/company context',max_chars=6000,key=context_key,placeholder='Engineering effort, roadmap priorities, customer impact, metric limitations, business value, deployment constraints, or lessons from similar experiments…', help=input_help('Optional initiative/company context'))
            st.caption('No Enter needed: click Save context, or click Get an AI strategic review to save and use your text immediately. Enter adds a new line. Save before continuing the conversation if you edited these notes.')
            save_context=st.form_submit_button('Save context',key='save_company_context',help='Save these notes for the next review, conversation and decision brief. This does not contact the AI or change business assumptions.')
        with review_action:
            request_reading=st.form_submit_button('Get an AI strategic review',disabled=not ai_active,key='request_initial_reading',type='primary',help='One click submits any text you just wrote and generates the review. No separate Save or Enter is required. Context can stay empty. This review cannot change assumptions.')
            if not ai_enabled:st.caption('Enable AI assistance in Welcome to use this button. No extra context is required.')
    if save_context:st.success('Context saved. The next review and conversation will use these notes; numeric assumptions have not changed.')
    reviews=st.session_state.setdefault('initial_reviews',{});review_key=review_fingerprint(snapshot,context)
    last_review=st.session_state.get('latest_review_key')
    if last_review and last_review!=review_key and not reviews.get(review_key):
        st.info('The previous Strategic Review is out of date because evidence, confirmed assumptions, policy or business context changed. Request a new review with the button above; the old reading is not shown as current.')
    if request_reading:
        tools=ToolSession(snapshot,active_effects,read_only=True)
        try:
            with st.spinner('Checking constraints, calculated trade-offs and decision-changing assumptions…'):
                reviews[review_key]=ask(snapshot,STRATEGIC_REVIEW,key,model_name,base_url,context=context,tool_session=tools)
            st.session_state.latest_review_key=review_key
            if tools.events:st.session_state.last_tool_events=tools.events
        except Exception as error:api_failure(error)
    if reviews.get(review_key):
        st.subheader('Your AI strategic review')
        st.markdown(reviews[review_key]);st.caption('AI reasoning about the current evidence and context—not a new statistical result or approval to launch.')
        st.markdown('**Months of impact · calculated driver check**\n\n'+snapshot['months_of_impact_condition']['text'])
    else:
        st.caption('Your AI-generated reading will appear here after you click the button. It is separate from the calculated summary and final Decision Brief.')
        with st.expander('Calculated starting point (not AI-generated)',expanded=False):
            st.write('**Current policy:** '+assessment['preferred_action']+' · '+assessment['status']+'.')
            st.write('**Next decision-relevant action:** '+drivers['next_action'])
            st.write('**Calculated condition:** '+counterfactual_readout(drivers))
            st.caption('This deterministic summary remains available without AI. Opening this step or adding context does not send a provider request automatically.')
    st.subheader('Questions you can start with')
    st.markdown('- “What can we learn despite the non-significant primary result?”\n- “Maintenance would cost €15,000 annually and we would have to delay another project by two weeks. Would you still recommend launching?”\n- “We can only afford €4,000 for a reversible rollout. Compare that with keeping control.”\n- “Could the early effect be faster purchases rather than a lasting benefit?”\n- “Which missing fact would be most likely to change your recommendation?”')
    if api_problem and ai_enabled:st.info(api_problem)
    current_fp=fingerprint({'evidence':snapshot,'context':context})
    if st.session_state.get('discussion_fingerprint')!=current_fp:
        if st.session_state.get('discussion'):st.info('Evidence or context changed. The conversation now starts from the updated facts; saved decision versions remain available.')
        st.session_state.discussion=[];st.session_state.discussion_fingerprint=current_fp
    if st.button('Clear conversation'):st.session_state.discussion=[]
    for message in st.session_state.discussion:
        with st.chat_message(message['role']):st.markdown(message['content'])
    suggested_question=None
    a,b,c=st.columns(3)
    with a:
        if st.button('Explain the main result',disabled=not ai_active,key='prompt_evidence'):suggested_question='What does this result actually establish, what is still uncertain, and which finding matters most for the launch decision?'
    with b:
        if st.button('Challenge the recommendation',disabled=not ai_active,key='prompt_change'):suggested_question='What could change the modeled preference? Which missing fact should we investigate before committing, and why?'
    with c:
        if st.button('Explore higher maintenance',disabled=not ai_active,key='prompt_maintenance'):suggested_question='Maintenance could be €15,000 annually and another project could be delayed two weeks. Use the tools to preview the priced maintenance change; ask for a monetary estimate of the project delay rather than inventing one.'
    question=st.chat_input('Ask about results or request a calculated business scenario…',disabled=not ai_active,max_chars=4000) or suggested_question
    if question:
        with st.chat_message('user'):st.markdown(question)
        tools=ToolSession(snapshot,active_effects)
        try:
            with st.chat_message('assistant'):
                with st.spinner('Reviewing evidence and running permitted calculations…'):answer=ask(snapshot,question,key,model_name,base_url,history=list(st.session_state.discussion),context=context,tool_session=tools)
                st.markdown(answer)
            st.session_state.discussion.extend([{'role':'user','content':question},{'role':'assistant','content':answer}])
            if tools.proposal:st.session_state.pending_proposal=tools.proposal
            if tools.events:st.session_state.last_tool_events=tools.events
        except Exception as error:api_failure(error)
    proposal=st.session_state.get('pending_proposal')
    if proposal and (proposal['evidence_hash']!=fingerprint(snapshot) or proposal['base_fingerprint']!=fingerprint(scenario)):
        st.info('That business preview is out of date because evidence or the active scenario changed. Request a fresh comparison before confirming.')
        st.session_state.pending_proposal=None;proposal=None
    if proposal:
        action=render_proposal(snapshot,proposal)
        if action=='confirm':
            try:
                updated,event=confirm_proposal(scenario,proposal,snapshot)
                previous_owner=st.session_state.get('owner_'+experiment_id,'')
                previous_ai=cached_explanation(snapshot,context,previous_owner,ai_enabled,st.session_state)
                event['before_version']=save_version(DB,snapshot,context,previous_owner,ai_summary=previous_ai)
                event['at']=datetime.now(timezone.utc).isoformat();st.session_state.business_events.append(event)
                st.session_state.scenario=updated;st.session_state.scenario_epoch+=1;st.session_state.pending_proposal=None
                st.session_state.confirmed_scenario_notice='The confirmed business assumptions are now active. The same evidence has been recalculated; the updated preference and status are shown below.'
                st.session_state.auto_save_version=True;st.rerun()
            except ValueError as error:st.error(str(error))
        if action=='discard':st.session_state.pending_proposal=None;st.rerun()
    with st.expander('What the AI can see and do—in plain language'):
        st.write('Read aggregate evidence, inspect a check, calculate a business preview, and propose an allowlisted business update. Proposals require your confirmation. No raw records, secrets, arbitrary code or silent statistical changes are exposed. Provider calls may cost money. Chat itself is session-only.')
        st.markdown('- **Read:** computed results, assumptions, group summaries and context you enter—not individual user rows.\n- **Calculate a preview:** compare a permitted business scenario without changing the active one.\n- **Propose:** show the exact business values and outcomes that would change; you decide whether to confirm.\n- **Cannot change:** original data, priors, significance rules, metric direction, safety limits or which alternatives you have chosen to consider.')
        if st.session_state.get('last_tool_events'):
            names={'get_decision_drivers':'Read computed decision drivers and counterfactuals','get_decision_summary':'Read the current decision','inspect_quality_check':'Read a computed check','compare_scenario':'Calculated a hypothetical comparison','propose_scenario_update':'Prepared a change for your confirmation'}
            st.dataframe(pd.DataFrame([{'Recent calculation':names.get(e['tool'],'Tool request'),'Outcome':'Rejected; active scenario unchanged' if e['result'].get('error') else 'Completed; active scenario unchanged until confirmation'} for e in st.session_state.last_tool_events]),hide_index=True,width='stretch')
        st.caption('The full machine-readable record can be downloaded separately from Decision brief. You do not need to read JSON to use this conversation.')

    next_step(5,'Next: record the supported conclusion, assumptions and accountable owner’s choice. The brief also works without AI.')

with tabs[5]:
    st.info(BRIEF_DISCLAIMER,icon='ℹ️')
    st.header('A decision document you can use and revisit')
    st.write('Separate observations from model-based inference, assumptions and owner judgment. Save a version when you commit to a set of assumptions; compare it with the next version rather than overwriting the history.')
    st.subheader(assessment['status']+' · '+assessment['preferred_action'])
    owner_note=st.text_area('Owner’s decision / remaining context (optional)',max_chars=4000,placeholder='Who will confirm maintenance cost? What evidence is needed before rollout? What action do you commit to?',key='owner_'+experiment_id, help=input_help('Owner’s decision / remaining context (optional)'))
    brief_explanation=render_ai_option(snapshot,context,owner_note,ai_enabled,key,model_name,base_url)
    report=render_brief(snapshot,context,owner_note,ai_summary=brief_explanation)
    detailed_report=render_detailed_brief(snapshot,context,owner_note,ai_summary=brief_explanation)
    if st.session_state.pop('auto_save_version',False):
        v=save_version(DB,snapshot,context,owner_note,ai_summary=brief_explanation);st.success(f'Confirmed scenario saved as decision version {v}.')
    if st.button('Save this decision version',key='save_version'):
        v=save_version(DB,snapshot,context,owner_note,ai_summary=brief_explanation);st.success(f'Saved version {v} locally.')
    history=versions(DB,experiment_id)
    if history:
        st.subheader('Decision history')
        st.dataframe(pd.DataFrame([{'Version':v['version'],'Saved (UTC)':v['created_at'],'Status':v['evidence']['assessment']['status'],'Modeled preference':v['evidence']['assessment']['preferred_action']} for v in history]),hide_index=True,width='stretch')
        selected=st.selectbox('Compare current draft with saved version',[v['version'] for v in history], help=input_help('Compare current draft with saved version'))
        previous=next(v for v in history if v['version']==selected)
        diff=compare_versions(previous['evidence'],snapshot)
        if diff:st.dataframe(pd.DataFrame(diff).astype({'before':str,'after':str}),hide_index=True,width='stretch')
        else:st.caption('No compared business-policy fields changed; inspect the full exports for evidence/context detail.')
        st.download_button('Download selected saved decision document',render_brief(previous['evidence'],previous['context'],previous['owner_note'],selected,ai_summary=previous.get('ai_summary')),f'{experiment_id}_v{selected}.md','text/markdown')
    st.download_button('Download concise decision brief (Markdown)',report,'experiment_decision_brief.md','text/markdown')
    with st.expander('Detailed appendix and audit files'):
        st.write('Use this separate appendix when you need every economic assumption, source, statistical method, subgroup result or audit detail. The default brief below focuses on the decision, supporting evidence and next action.')
        st.download_button('Download detailed appendix (Markdown)',detailed_report,'experiment_decision_appendix.md','text/markdown')
        st.download_button('Download machine-readable audit (JSON)',dumps({'evidence':snapshot,'reported_context':context,'owner_note':owner_note,'ai_summary':brief_explanation}),'experiment_decision_audit.json','application/json')
    st.caption('Versions are stored only in .local/decisions.sqlite3, ignored by Git. They contain aggregate evidence, design/assumption notes and entered report context—not raw rows or the conversation. Back up/export the file if needed; hosted ephemeral storage would need a durable database.')
    st.markdown(report)
