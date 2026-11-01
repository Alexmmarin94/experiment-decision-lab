"""Guided entry and schema setup. No inference or business policies are defined here."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
from edi.schema import load_upload,suggested_column,normalize_schema,outcome_candidates,setup_issues,minimal_template
from edi.example_data import FILENAME,DICTIONARY
from edi.inference import detect_type,arm_summary
from edi.input_help import input_help

STEPS=['01 · Welcome & Upload','02 · Can we trust the data?','03 · What did we learn?','04 · What should we do?','05 · Explore with AI','06 · Decision Brief']


def go_to_step(index):
    st.session_state.pending_step=STEPS[index]
    st.rerun()


def workflow_panels():
    """State-controlled navigation; every panel still runs to preserve widget state.

    Native tabs in Streamlit 1.50 expose only an initial default, not controlled
    selection. Keyed containers and a radio selector make Continue actually navigate.
    CSS contains only generated integer panel names, never uploaded or AI content.
    """
    pending=st.session_state.pop('pending_step',None)
    if pending in STEPS:st.session_state.active_step=pending
    if st.session_state.get('active_step') not in STEPS:st.session_state.active_step=STEPS[0]
    with st.container(key='workflow_navigation'):
        chosen=st.radio('Experiment decision workflow',STEPS,key='active_step',horizontal=True,
            help='Choose a step at any time. Continue buttons select the next step. Inputs and calculated evidence are preserved as you move through the workflow.')
    selected=STEPS.index(chosen)
    rules='\n'.join(f'.st-key-workflow_{i} {{ display: none; }}' for i in range(len(STEPS)) if i!=selected)
    nav_css='''.st-key-workflow_navigation [role="radiogroup"] { gap: 0.5rem; flex-wrap: wrap; }
    .st-key-workflow_navigation [role="radiogroup"] label { padding: 0.4rem 0.7rem; border: 1px solid #CBD5E1; border-radius: 0.5rem; }
    .st-key-workflow_navigation [role="radiogroup"] label:has(input:checked) { border-color: #0D9488; background: rgba(13,148,136,0.12); }'''
    st.markdown('<style>'+rules+'\n'+nav_css+'</style>',unsafe_allow_html=True)
    return [st.container(key='workflow_'+str(i)) for i in range(len(STEPS))]


def next_step(index,text,disabled=False):
    st.caption(text)
    if st.button('Continue to '+STEPS[index].split(' · ',1)[1],key='continue_'+str(index),type='primary',disabled=disabled):go_to_step(index)


def hero():
    st.caption('EXPERIMENT DECISION INTELLIGENCE')
    st.title('From experiment results to decisions you can defend')
    st.write('Upload an A/B test, evaluate what the evidence supports, explore business scenarios, and use AI to understand what to do next. The accountable human owns the final decision.')
    a,b,c=st.columns(3)
    with a:st.markdown('**Expert-designed evidence**');st.write('Explicit methods, validity checks and protection against misleading findings form the analytical foundation.')
    with b:st.markdown('**Decisions under uncertainty**');st.write('Translate effects into possible value, downside and the assumptions that could change the choice.')
    with c:st.markdown('**AI grounded in analysis**');st.write('Interpret, challenge and recalculate scenarios using controlled tools—not invented statistical results.')
    st.write('Conversion can improve while a launch remains the wrong choice: the gain may fade, safety may be unresolved, or engineering costs may exceed the upside. The workspace identifies which question actually matters: **Evidence → Business consequences → AI reasoning → Human decision**.')


def render_onboarding(root,example_loader):
    st.caption('Illustrative progression—not the loaded experiment’s result')
    st.markdown('“Conversion +2.8%” → **Evidence:** how uncertain and safe? → **Decisions:** enough value after costs, even if benefit fades? → **AI:** which engineering constraints matter? → **Human:** a conditional choice with a next step.')
    st.subheader('Choose how to start')
    source=st.radio('Choose your data',['Preloaded example','Upload my CSV'],horizontal=True,key='source',format_func=lambda v:'Explore the example experiment' if v=='Preloaded example' else 'Analyze my own experiment',help=input_help('Choose your data'))
    frame=None;source_hash=None;start_example=False
    if source=='Preloaded example':
        frame=example_loader((Path(root)/'data'/FILENAME).stat().st_mtime_ns)
        source_hash=hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
        st.write('Synthetic user-level checkout data, ready to explore the full workflow. No company outcomes are claimed. You can use the defaults and start with the checks.')
        start_example=st.button('Start the example walkthrough',type='primary',key='start_example')
    else:
        st.write('**Upload assignments and outcomes—not a screenshot or aggregated p-values.** Minimum: one ID column, one assignment column, and one numeric outcome. Defaults recognize `unit_id` (`unir_id` also works) and `variant`; different names can be mapped below.')
        with st.expander('See four example records and the supported format',expanded=True):
            st.code(minimal_template(),language='csv')
            st.caption('Four records illustrate the structure only; they are not enough for statistical inference. One row = one independently randomized user. Include non-converters. Zero is measured absence, not a substitute for missing logging. Assignment is the randomized arm, not successful interaction with the feature.')
        upload=st.file_uploader('Upload the complete assigned-user export',type=['csv'],help=input_help('Upload the complete assigned-user export'))
        try:frame,source_hash=load_upload(upload)
        except ValueError as error:st.error(str(error))
    st.caption('Supported: two independent-user arms, complete assignments and a planned fixed horizon. Cluster-randomized, sequential or selected exposed-only designs require different analysis and are not automatically supported.')
    a,b=st.columns(2)
    with a:st.download_button('Download the full example CSV',(Path(root)/'data'/FILENAME).read_bytes(),FILENAME,'text/csv')
    with b:st.download_button('Download a minimal format template',minimal_template(),'minimal_experiment_template.csv','text/csv')
    with st.expander('What optional columns unlock'):
        st.markdown('- **Country/platform:** compare pre-treatment groups and explore whether effects differ.\n- **Baseline metrics:** check measured comparability; optional, without CUPED adjustment.\n- **`*_guardrail` columns:** assess harm against declared limits.\n- **Dates and outcome windows:** check maturity and inspect assignment weeks. Equal early/late outcomes support a separate timing contrast.\n- **Numerator and denominator:** correctly analyze ratios of totals.\n\nMore columns are not automatically better. Reliable definitions and complete measurement matter more. Dates are not mandatory for the core analysis.')
    with st.expander('Full CSV specification'):st.dataframe(DICTIONARY,hide_index=True,width='stretch')
    with st.expander('How the process works'):
        st.markdown('**Upload & configure → Check trust → Evaluate evidence → Model choices → Explore with AI → Save a decision brief.** Advanced details remain available, while missing information and material assumptions stay visible.')
    st.checkbox('Enable AI assistance (optional)',value=True,key='ai_enabled',help=input_help('Enable AI assistance (optional)'))
    st.caption('Selected by default, but nothing runs yet. Start the example or your own analysis first. AI requests then run only when you click an AI action or send a question.')
    st.info('Optional AI works with the statistical results calculated here—not the raw CSV. Ask it to explain a flagged check, challenge the launch case using business context, identify a decision-changing assumption, or calculate a business what-if for your confirmation. The full evidence and decision workflow also works without AI.')
    st.caption('Statistical/product expertise determines the evidence and limits. AI helps make that evidence accessible and connects it with company context—it does not certify validity, invent company facts or make an autonomous decision.')
    if frame is None:return None
    st.success(f'Loaded {len(frame):,} individual user rows · {len(frame.columns)} columns'+(' · synthetic example' if source=='Preloaded example' else ' · uploaded CSV'))
    with st.expander('Preview of the loaded records'):st.dataframe(frame.head(5),hide_index=True,width='stretch')
    st.subheader('Configure the experiment')
    st.markdown('**A. Identify this experiment**')
    a,b=st.columns(2)
    with a:name=st.text_input('Initiative name',value='Checkout simplification' if source=='Preloaded example' else 'My experiment',help=input_help('Initiative name'))
    with b:experiment_id=st.text_input('Experiment ID',value='experiment_001' if source=='Preloaded example' else 'experiment_'+source_hash[:8],max_chars=80,help=input_help('Experiment ID')) or 'experiment_001'
    st.markdown('**B. Map users and assignment**')
    columns=list(frame.columns);none='Choose a column…'
    uid=suggested_column(columns,'unit');arm=suggested_column(columns,'assignment')
    a,b=st.columns(2)
    with a:unit_column=st.selectbox('Which column identifies each randomized user?',[none]+columns,index=columns.index(uid)+1 if uid else 0,key='map_unit_'+source_hash,help='Choose the unique assignment-unit ID, not an event/session ID or a row number. The input spelling is preserved; missing/duplicate users are checked.')
    with b:assignment_column=st.selectbox('Which column contains the assigned arm?',[none]+columns,index=columns.index(arm)+1 if arm else 0,key='map_arm_'+source_hash,help='Choose the randomized experimental group. A first-exposure flag is not an assignment column. Exactly two arms are supported.')
    if unit_column==none or assignment_column==none:
        st.error('Map both the user-ID and assignment columns. If the export does not contain them, add real experiment assignment records before proceeding.')
        return None
    arms=list(frame[assignment_column].dropna().unique());control=None
    if len(arms)==2:
        default=arms.index('control') if 'control' in arms else 0
        control=st.selectbox('Which label is the current/control experience?',arms,index=default,key='map_control_'+source_hash,help=input_help('Which label is the current/control experience?'))
    try:data,mapping=normalize_schema(frame,unit_column,assignment_column,control)
    except ValueError as error:st.error(str(error));return None
    if st.session_state.get('allocation_source')!=source_hash:
        st.session_state.allocation=.5;st.session_state.allocation_source=source_hash
    allocation_container=st.expander('Allocation plan') if source=='Preloaded example' else st.container()
    with allocation_container:
        allocation=st.slider('Planned share assigned to treatment',.05,.95,.5,.05,key='allocation',help=input_help('Planned share assigned to treatment'))
        st.caption('This comes from the original plan, not the observed export. The next stage checks whether the split is suspicious.')
    candidates=outcome_candidates(data)
    if not candidates:
        st.error('No usable numeric outcome found. Add a measured 0/1 outcome or numeric amount/count, using consistent numbers rather than text such as “unknown”. IDs and assignment labels are not outcomes.')
        return None
    st.caption('Assignment field: '+assignment_column+' · available outcome candidates: '+', '.join(candidates[:8])+('…' if len(candidates)>8 else ''))
    st.markdown('**C. Define what would count as a useful improvement**')
    metric=st.selectbox('Which outcome was this experiment intended to improve?',candidates,index=candidates.index('converted_7d') if 'converted_7d' in candidates else 0,key='metric',help=input_help('Which outcome was this experiment intended to improve?'))
    with st.expander('Outcome type (auto-detected unless you specify a ratio)'):
        choice=st.selectbox('How is the outcome measured?',['Auto-detect','Binary (yes/no)','Numeric amount or count','Ratio of totals'],key='metric_type',help=input_help('How is the outcome measured?'))
    kind={'Binary (yes/no)':'binary','Numeric amount or count':'numeric','Ratio of totals':'ratio'}.get(choice,detect_type(data[metric]))
    denominator=None;bounded=False
    if kind=='ratio':
        available=[c for c in candidates if c!=metric]
        if not available:st.error('A ratio needs a separate numerator and denominator column. Add both per-user quantities to the CSV.');return None
        denominator=st.selectbox('What is the denominator?',available,index=available.index('sessions_7d') if 'sessions_7d' in available else 0,help=input_help('What is the denominator?'))
        bounded=st.checkbox('The numerator cannot exceed the denominator',value=False,help=input_help('The numerator cannot exceed the denominator'))
    direction=st.selectbox('Which direction helps users or the business?',['Higher is better','Lower is better'],key='direction',help=input_help('Which direction helps users or the business?'))
    roles=['Acquisition','Activation','Retention','Engagement','Monetization','Experience / safety','Other / clarify with owner']
    suggested='Monetization' if any(w in metric.lower() for w in ['revenue','convert','purchase']) else 'Engagement' if 'session' in metric else 'Experience / safety' if any(w in metric for w in ['latency','crash']) else 'Other / clarify with owner'
    role=st.selectbox('What business outcome does this metric represent?',roles,index=roles.index(suggested),key='role_'+metric,help=input_help('What business outcome does this metric represent?'))
    definition=st.text_input('Metric definition (optional but useful for AI)',value='Share of assigned users purchasing within 7 days.' if metric=='converted_7d' else '',key='definition_'+metric,help=input_help('Metric definition (optional but useful for AI)'))
    relative=st.number_input('What relative improvement would make this worthwhile? (%)',min_value=.01,max_value=1000.,value=5.,step=1.,key='relative_goal',help=input_help('What relative improvement would make this worthwhile? (%)'))
    st.caption('A 5% relative goal takes an 8% conversion rate to 8.4%, or latency from 200 ms to 190 ms. Set a business goal, not a cutoff chosen to make the results significant.')
    reference=None
    if set(data.variant.dropna())=={'control','treatment'}:
        try:reference=arm_summary(data[data.variant=='control'],metric,kind,denominator)[0]
        except (ValueError,TypeError):pass
    zero=reference is None or not np.isfinite(reference) or reference==0
    ref=1. if zero else abs(reference)
    if zero:
        st.warning('Relative improvement needs a nonzero reference. Correct unusable outcome values, or supply a planning reference if the measured baseline is genuinely zero.')
        ref=st.number_input('Planning reference for the current metric value',min_value=.000001,value=1.,help=input_help('Planning reference for the current metric value'))
    elif reference<0:st.caption('The goal uses the magnitude of the negative current value, not a conventional growth rate.')
    st.markdown('**D. Confirm the design—not a fact verified by the app**')
    st.write('The supported analysis assumes independently assigned users, everyone retained in their assigned arm, complete comparable outcome windows and an analysis at a planned stopping point. If the study used clusters, selected exposed-only users or unplanned repeated stopping, do not confirm it as this design.')
    design=st.checkbox('Confirm: users were assigned independently; the export includes all assignments and complete outcome windows; this is the planned fixed-horizon analysis.',value=source=='Preloaded example',key='design_'+source if source=='Preloaded example' else 'design_'+source+'_'+source_hash,help=input_help('Confirm: users were assigned independently; the export includes all assignments and complete outcome windows; this is the planned fixed-horizon analysis.'))
    issues=setup_issues(data,metric,denominator)
    setup_token=hashlib.sha256((source_hash+str(sorted(mapping.items()))).encode()).hexdigest()
    for issue in issues:st.error(issue)
    if not design:st.info('Owner confirmation is still needed. The data-check stage will keep unsupported inference blocked until it is supplied.')
    if not issues:
        st.write('**Next:** check assignment, unit integrity, measurement and available windows before interpreting lift.')
        if st.button('Continue to data quality checks' if source=='Preloaded example' else 'Start analysis of my CSV',type='primary',key='continue_setup') or start_example:
            st.session_state.analysis_started_for=setup_token
            go_to_step(1)
    elif start_example:st.error('Resolve the setup messages before starting the checks.')
    return {'df':data,'source':source,'source_hash':source_hash,'analysis_started':st.session_state.get('analysis_started_for')==setup_token,'experiment_name':name,'experiment_id':experiment_id,
        'metric':metric,'kind':kind,'denominator':denominator,'bounded_ratio':bounded,'direction':direction,
        'sign':1 if direction=='Higher is better' else -1,'role':role,'definition':definition,'relative_goal':relative,
        'reference':reference,'ref':ref,'zero_reference':zero,'goal':ref*relative/100,'design':design,'allocation':allocation,'mapping':mapping}
