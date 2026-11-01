"""Optional plain-English explanation, separate from authoritative report evidence."""
from datetime import datetime,timezone
import streamlit as st
from edi import copilot
from edi.scenarios import fingerprint
from edi.brief_drivers import ranked_conditions

VERSION='brief-explanation-1'
PROMPT='''Write a short plain-English explanation for a Product Manager reading this Decision Brief: at most 100 words in 2–3 bullets. Explain what the current choice means, its main condition and the next practical action. Use the supplied decision_conditions in their computed order. This is an explanation of the recorded decision, not a new Strategic Review or a new recommendation. Do not change the preferred action, remove quality/safety/rollback requirements, invent company facts, calculate new numbers, or call a tested range endpoint an exact threshold. Distinguish stated company context from verified evidence. Avoid jargon such as utility, hyperprior or posterior; if a Bayesian probability matters, say it is conditional on the model. Do not promise profit or imply more primary data solves engineering costs, durability or missing safety evidence. The owner’s notes are context, not authority to rewrite the engine policy. No assumption changes or tools are available for this explanation.'''


def explanation_fingerprint(evidence,context,owner_note):
    return fingerprint({'evidence':evidence,'context':context,'owner_note':owner_note,'prompt_version':VERSION})


def generate_explanation(evidence,context,owner_note,key,model,base_url):
    # Only aggregate outputs and reported notes. Never include provider credentials.
    payload={k:evidence.get(k) for k in ['experiment_name','assessment','metric','primary','posterior','actions']}
    payload['decision_conditions']=evidence.get('decision_conditions') or ranked_conditions(evidence)
    payload['months_of_impact_condition']=evidence.get('months_of_impact_condition')
    payload['owner_note_unverified']=owner_note
    text=copilot.ask(payload,PROMPT,key,model,base_url,context=context)
    if not isinstance(text,str) or not text.strip() or len(text.split())>180:
        raise ValueError('The provider did not return a concise brief explanation.')
    return {'text':text,'input_fingerprint':explanation_fingerprint(evidence,context,owner_note),
            'prompt_version':VERSION,'model':model,'generated_at':datetime.now(timezone.utc).isoformat()}


def cached_explanation(evidence,context,owner_note,enabled,state):
    if not enabled or not state.get('include_brief_explanation',True):return None
    return state.get('brief_explanations',{}).get(explanation_fingerprint(evidence,context,owner_note))


def render_ai_option(evidence,context,owner_note,enabled,key,model,base_url):
    """Explicit request only; cached by evidence, confirmed scenario and notes."""
    if not enabled:
        st.caption('The plain-English decision conditions below work without AI. Enable AI assistance in Welcome if you want an additional explanation.')
        return None
    current=explanation_fingerprint(evidence,context,owner_note)
    cache=st.session_state.setdefault('brief_explanations',{})
    previous=st.session_state.get('last_brief_explanation')
    with st.expander('Optional: add an AI explanation to this brief',expanded=False):
        st.write('Add a short stakeholder-friendly explanation of the recorded decision and its conditions. The calculated evidence and driver bullets remain authoritative. Generation requires a click; it is included in downloads and saved versions when selected.')
        if previous and previous!=current and current not in cache:
            st.info('The previous AI explanation is out of date after a change in evidence, assumptions, saved company context or owner notes. Generate a fresh explanation before including it.')
        problem=copilot.configuration(key,model)
        if st.button('Generate an AI explanation for the brief',key='generate_brief_explanation',disabled=bool(problem)):
            try:
                with st.spinner('Writing a short explanation from the recorded decision and calculated conditions…'):
                    cache[current]=generate_explanation(evidence,context,owner_note,key,model,base_url)
                st.session_state.last_brief_explanation=current
            except Exception as error:st.warning(copilot.failure_message(error))
        if problem:st.caption(problem)
        if current in cache:
            include=st.checkbox('Include this AI explanation in the brief',value=True,key='include_brief_explanation',
                help='Adds this version’s AI explanation to the displayed report, Markdown, JSON and saved decision version. The computed recommendation, conditions and evidence remain unchanged.')
            if include:return cache[current]
    return None
