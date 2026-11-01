import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from edi.scenarios import starter,run,fingerprint
from edi.decision_drivers import build_decision_drivers
from edi.brief_drivers import ranked_conditions,render_conditions
from edi.brief_ai import generate_explanation,explanation_fingerprint,cached_explanation
from edi.reporting import render_brief,save_version,versions
from edi.copilot import ProviderFailure
from edi.quality import audit as original_audit


def example(safe=True):
    df=pd.DataFrame({'unit_id':range(300),'variant':['control']*150+['treatment']*150,'y':[0,1]*150})
    s=starter(df,'y','binary')
    s['business'].update(monthly_users=100000,reach=.8,margin=10.,margin_uncertainty=0.)
    s['uncertainty']={k:0. for k in s['uncertainty']}
    q={'errors':[],'warnings':[],'unavailable':[],'verdict':'Usable under the declared scope'}
    g=[{'metric':'crashes_guardrail','safe':safe}]
    effects=np.full(300,.004)
    result,_,_=run(effects,s,q,g)
    e={'experiment_id':'brief-test','experiment_name':'Checkout','data_hash':'synthetic','data_kind':'synthetic example',
       'engine_version':'test','rows':300,'metric':{'name':'y'},'primary':None,'posterior':None,'relative_goal_percent':5.,
       'scenario':s,'quality':q,'guardrails':g,'assessment':result['assessment'],'actions':result['actions'],'evsi':None,
       'sensitivity':[{'assumption':'Contribution margin','low_input':7.,'high_input':13.},
                      {'assumption':'Eligible population','low_input':70000,'high_input':130000},
                      {'assumption':'Effect retention','low_input':.65,'high_input':1.}]}
    e['decision_drivers']=build_decision_drivers(e,effects)
    return e


class BriefConditionTests(unittest.TestCase):
    def test_numbered_conditions_use_real_switches_and_rank_by_computed_impact(self):
        e=example();before=fingerprint(e)
        rows=ranked_conditions(e)
        self.assertEqual(len(rows),3)
        self.assertTrue(all(r['changes_preference'] for r in rows))
        swings=[r['score_swing'] for r in rows]
        self.assertEqual(swings,sorted(swings,reverse=True))
        text=render_conditions(e)
        for n in (1,2,3):self.assertIn(str(n)+'. **',text)
        self.assertIn('**Keep control**',text)
        self.assertIn('not an exact cutoff',text)
        self.assertIn('low confidence',text)
        self.assertIn('not customer retention',text)
        self.assertNotIn('specifically below',text)
        self.assertEqual(before,fingerprint(e))

    def test_eligibility_blockers_take_priority_over_large_economic_swings(self):
        e=example(safe=False)
        rows=ranked_conditions(e)
        self.assertEqual(rows[0]['id'],'Safety')
        self.assertTrue(all(r['kind']=='constraint' for r in rows))
        self.assertIn('cannot remove this condition',render_conditions(e))

    def test_numerical_ambiguity_is_not_reported_as_a_precise_switch(self):
        e=example()
        for c in e['decision_drivers']['counterfactuals']:
            for b in c['feasible_preference_switch_brackets']:b['endpoint_scores_separated_from_mc_noise']=False
        rows=ranked_conditions(e)
        self.assertFalse(any(r['changes_preference'] for r in rows))
        self.assertIn('simulation noise',render_conditions(e))
        self.assertNotIn('not an exact cutoff',render_conditions(e))

    def test_older_comparison_records_and_ratio_units_remain_readable(self):
        e=example();e['metric']['kind']='ratio'
        for c in e['decision_drivers']['counterfactuals']:
            for b in c['feasible_preference_switch_brackets']:b.pop('endpoint_scores_separated_from_mc_noise',None)
        text=render_conditions(e)
        self.assertIn('denominator units/month',text)
        self.assertNotIn('eligible people/month',text)
        self.assertIn('simulation noise',text)

    def test_positive_average_profit_can_still_favor_control_after_allowing_for_losses(self):
        e=example();e['scenario']['business']['margin']=20.
        effects=np.tile([-.002,.01],2000)
        result,_,_=run(effects,e['scenario'],e['quality'],e['guardrails'])
        e.update(assessment=result['assessment'],actions=result['actions'])
        e['sensitivity']=[{'assumption':'Contribution margin','low_input':10.,'high_input':26.}]
        e['decision_drivers']=build_decision_drivers(e,effects)
        self.assertEqual(e['assessment']['preferred_action'],'Global launch')
        text=render_conditions(e)
        self.assertIn('**Keep control**',text)
        self.assertIn('Average profit remains positive here',text)
        self.assertIn('extra weight on losses',text)

    def test_ai_addition_is_separate_immutable_and_saved_with_the_same_report(self):
        e=example();e['decision_conditions']=ranked_conditions(e);before=fingerprint(e)
        with patch('edi.copilot.ask',return_value='- Confirm ongoing costs before committing; the current evidence does not establish durability.') as ask:
            ai=generate_explanation(e,'Finance estimate pending.','Owner to validate costs.','fake-key','test-model','https://example.invalid')
        self.assertNotIn('fake-key',str(ask.call_args.args[0]))
        self.assertNotIn('unit_id',str(ask.call_args.args[0]))
        self.assertNotIn('tool_session',ask.call_args.kwargs)
        report=render_brief(e,'Finance estimate pending.','Owner to validate costs.',ai_summary=ai)
        self.assertIn(ai['text'],report)
        self.assertIn('1. **',report)
        self.assertEqual(before,fingerprint(e))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'history.sqlite3'
            self.assertEqual(save_version(path,e,'Finance estimate pending.','Owner to validate costs.',ai_summary=ai),1)
            stored=versions(path,e['experiment_id'])[0]
            self.assertEqual(stored['ai_summary'],ai)
            self.assertEqual(report,render_brief(stored['evidence'],stored['context'],stored['owner_note'],ai_summary=stored['ai_summary']))
        state={'include_brief_explanation':True,'brief_explanations':{ai['input_fingerprint']:ai}}
        self.assertEqual(cached_explanation(e,'Finance estimate pending.','Owner to validate costs.',True,state),ai)
        self.assertIsNone(cached_explanation(e,'Changed context','Owner to validate costs.',True,state))
        self.assertIsNone(cached_explanation(e,'Finance estimate pending.','Owner to validate costs.',False,state))


class StartAndContextJourneys(unittest.TestCase):
    def landing(self):return AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'streamlit_app.py'),default_timeout=60).run()
    def started(self):return self.landing().button(key='start_example').click().run()

    def test_default_ai_checked_but_no_analysis_or_provider_request_before_start(self):
        with patch('edi.copilot.ask') as ask,patch('edi.copilot.explain_quality') as explain,patch('edi.quality.audit',wraps=original_audit) as audit:
            app=self.landing()
            self.assertTrue(app.checkbox(key='ai_enabled').value)
            self.assertEqual(len(app.metric),0)
            self.assertEqual(audit.call_count,0)
            self.assertEqual(ask.call_count,0);self.assertEqual(explain.call_count,0)
            app.button(key='start_example').click().run()
            self.assertEqual(len(app.exception),0)
            self.assertGreater(audit.call_count,0)
            self.assertTrue(any(m.label=='Chance of benefit · model' for m in app.metric))
            self.assertEqual(ask.call_count,0);self.assertEqual(explain.call_count,0)

    def test_each_uploaded_dataset_needs_its_own_start_and_ai_can_stay_off(self):
        df=pd.DataFrame({'unit_id':range(100),'variant':['control']*50+['treatment']*50,'purchase':[0,1]*50})
        with patch('edi.onboarding.load_upload',return_value=(df,'upload-first')) as loader:
            app=self.started();app.checkbox(key='ai_enabled').uncheck().run()
            app.radio(key='source').set_value('Upload my CSV').run()
            self.assertEqual(len(app.metric),0)
            self.assertEqual(app.button(key='continue_setup').label,'Start analysis of my CSV')
            next(c for c in app.checkbox if c.label.startswith('Confirm: users')).check().run()
            app.button(key='continue_setup').click().run()
            self.assertTrue(any(m.label=='Chance of benefit · model' for m in app.metric))
            self.assertTrue(app.button(key='request_initial_reading').disabled)
            loader.return_value=(df,'upload-second')
            app.run()
            self.assertEqual(len(app.metric),0)
            self.assertFalse(app.checkbox(key='ai_enabled').value)

    def test_save_context_makes_no_ai_call_and_generate_submits_fresh_text(self):
        with patch('edi.copilot.configuration',return_value=None),patch('edi.copilot.ask',return_value='Review uses the latest maintenance estimate.') as ask:
            app=self.started()
            context=next(t for t in app.text_area if t.label=='Optional initiative/company context')
            context.set_value('Maintenance is €15,000 annually.')
            app.button(key='save_company_context').click().run()
            self.assertEqual(ask.call_count,0)
            self.assertTrue(any('Context saved' in m.value for m in app.success))
            context=next(t for t in app.text_area if t.label=='Optional initiative/company context')
            context.set_value('Updated maintenance estimate is €18,000 annually.')
            app.button(key='request_initial_reading').click().run()
            self.assertEqual(ask.call_count,1)
            self.assertIn('18,000',ask.call_args.kwargs['context'])
            self.assertEqual(len(app.exception),0)

    def test_brief_ai_is_explicit_cached_and_invalidated_by_saved_context(self):
        with patch('edi.copilot.configuration',return_value=None),patch('edi.copilot.ask',return_value='UNIQUE BRIEF EXPLANATION: confirm the cost estimate before committing.') as ask:
            app=self.started();before=copy.deepcopy(app.session_state['scenario'])
            self.assertEqual(ask.call_count,0)
            app.button(key='generate_brief_explanation').click().run()
            self.assertEqual(len(app.exception),0);self.assertEqual(ask.call_count,1)
            self.assertTrue(any('UNIQUE BRIEF EXPLANATION' in m.value for m in app.markdown))
            app.run();self.assertEqual(ask.call_count,1)
            self.assertEqual(before,app.session_state['scenario'])
            next(t for t in app.text_area if t.label=='Optional initiative/company context').set_value('Engineering estimate changed.')
            app.button(key='save_company_context').click().run()
            self.assertEqual(ask.call_count,1)
            self.assertTrue(any('previous AI explanation is out of date' in m.value for m in app.info))
            self.assertFalse(any('UNIQUE BRIEF EXPLANATION' in m.value for m in app.markdown))

    def test_brief_provider_failure_preserves_the_plain_english_report(self):
        with patch('edi.copilot.configuration',return_value=None),patch('edi.copilot.ask',side_effect=ProviderFailure(402)):
            app=self.started();before=copy.deepcopy(app.session_state['scenario'])
            app.button(key='generate_brief_explanation').click().run()
            self.assertEqual(len(app.exception),0)
            self.assertTrue(any('insufficient credits' in m.value for m in app.warning))
            self.assertTrue(any('Here are the key drivers' in m.value for m in app.markdown))
            self.assertEqual(before,app.session_state['scenario'])

    def test_confirmed_business_change_preserves_before_explanation_without_reusing_it_after(self):
        def respond(evidence,question,*args,**kwargs):
            if question.startswith('Maintenance'):
                kwargs['tool_session'].execute('propose_scenario_update',{'changes':{'annual_maintenance_eur':15000},'reason':'Updated annual estimate.'})
                return 'Review the new maintenance scenario before confirming.'
            return '- Confirm ongoing costs before committing to this decision.'
        with tempfile.TemporaryDirectory() as directory:
            db=Path(directory)/'versions.sqlite3'
            with patch('edi.copilot.configuration',return_value=None),patch('edi.copilot.ask',side_effect=respond), \
                 patch('edi.reporting.save_version',side_effect=lambda path,*a,**k:save_version(db,*a,**k)), \
                 patch('edi.reporting.versions',side_effect=lambda path,experiment_id:versions(db,experiment_id)):
                app=self.started()
                app.button(key='generate_brief_explanation').click().run()
                app.chat_input[0].set_value('Maintenance would cost €15,000 annually.').run()
                app.button(key='confirm_proposal').click().run()
                self.assertEqual(len(app.exception),0)
                history=versions(db,'experiment_001')
                self.assertEqual(len(history),2)
                self.assertIsNotNone(history[0]['ai_summary'])
                self.assertIsNone(history[1]['ai_summary'])
                self.assertEqual(history[0]['evidence']['posterior'],history[1]['evidence']['posterior'])
