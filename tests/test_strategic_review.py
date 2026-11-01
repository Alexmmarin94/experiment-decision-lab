from pathlib import Path
import copy
import unittest
from unittest.mock import Mock,patch
import numpy as np
import pandas as pd
import requests
from streamlit.testing.v1 import AppTest

from edi.copilot import failure_message,post,ProviderFailure,review_fingerprint,STRATEGIC_REVIEW
from edi.decision_drivers import build_decision_drivers,counterfactual_readout,risk_explanation
from edi.scenarios import starter,run,fingerprint,proposed_scenario
from edi.tools import ToolSession
from edi.proposal_ui import assumption_display,preview_description,preview_interpretation
from edi.presentation import observed_effect_chart
from edi.onboarding import STEPS


def workspace(effect=.003,safe=True):
    df=pd.DataFrame({'unit_id':range(100),'variant':['control']*50+['treatment']*50,'outcome':[0,1]*50})
    scenario=starter(df,'outcome','binary')
    scenario['business'].update(monthly_users=100000,reach=.8,margin=10.,margin_uncertainty=0.)
    for k in scenario['uncertainty']:scenario['uncertainty'][k]=0.
    q={'errors':[],'warnings':[],'unavailable':[]}
    guards=[{'metric':'crash_guardrail','safe':safe}]
    effects=np.full(300,effect)
    result,_,_=run(effects,scenario,q,guards)
    evidence={'assessment':result['assessment'],'scenario':scenario,'quality':q,'guardrails':guards,
              'actions':result['actions'],'sensitivity':[{'assumption':'Maintenance per month','low_input':250.,'high_input':1000.}]}
    return evidence,effects


class DriverTests(unittest.TestCase):
    def test_economic_control_is_distinct_from_safety_block_and_switch_is_computed(self):
        e,effects=workspace();before=fingerprint(e)
        d=build_decision_drivers(e,effects)
        self.assertEqual(d['current_decision']['preferred_action'],'Keep control')
        self.assertTrue(d['current_decision']['allow_rollout'])
        self.assertFalse(any(c['binding'] for c in d['constraints']))
        test=d['counterfactuals'][0]
        self.assertTrue(test['feasible_preference_switch_brackets'])
        points=test['tested_points']
        self.assertEqual(points[0]['preferred_feasible_action'],'Global launch')
        self.assertEqual(points[-1]['preferred_feasible_action'],'Keep control')
        # Maintenance is a fixed monthly cost in exactly the same payoff function.
        self.assertAlmostEqual(points[0]['global_launch_expected_value']-points[-1]['global_launch_expected_value'],750*6)
        self.assertTrue(all(250<=p['input']<=1000 for p in points))
        self.assertEqual(before,fingerprint(e))
        self.assertEqual(d,build_decision_drivers(e,effects))
        self.assertIn('not an exact threshold',counterfactual_readout(d))

    def test_positive_economics_cannot_override_guardrail(self):
        e,effects=workspace(effect=.01,safe=False)
        d=build_decision_drivers(e,effects)
        self.assertEqual(d['eligible_actions'],['Keep control'])
        self.assertGreater(d['comparison_baseline']['global_launch_expected_value'],0)
        self.assertEqual(d['comparison_baseline']['economic_leader_ignoring_safety_gate'],'Global launch')
        self.assertTrue(any(c['kind']=='Safety' and c['binding'] for c in d['constraints']))
        self.assertFalse(d['counterfactuals'][0]['feasible_preference_switch_brackets'])
        self.assertIn('safety',d['next_action'])
        self.assertIn('does not remove',counterfactual_readout(d))

    def test_rollback_and_iteration_constraints_are_not_economic_scores(self):
        e,effects=workspace(effect=.01)
        e['scenario']['options']['global_reversible']=False
        e['scenario']['preferences']['require_reversible']=True
        r,_,_=run(effects,e['scenario'],e['quality'],e['guardrails']);e.update(assessment=r['assessment'],actions=r['actions'])
        d=build_decision_drivers(e,effects)
        self.assertEqual(d['eligible_actions'],['Keep control'])
        excluded={r['action']:r['reason'] for r in d['excluded_actions']}
        self.assertIn('reversibility',excluded['Global launch'])
        self.assertIn('future effect',excluded['Iterate the feature'])

    def test_validity_blocks_comparison_and_primary_information_does_not_repair_it(self):
        e,_=workspace();e['quality']['errors']=['Duplicate assigned users']
        d=build_decision_drivers(e,None)
        self.assertEqual(d['counterfactuals'],[])
        self.assertIn('invalid export',d['next_action'])
        self.assertIn('durability',d['evsi_scope']['cannot_learn'])

    def test_near_tie_precision_and_gain_size_are_not_win_probability(self):
        e,_=workspace()
        s=e['scenario'];s['business'].update(implementation=0.,maintenance=0.)
        # Half lose, half win much more: risk-adjusted launch still beats control.
        effects=np.r_[np.full(150,-.001),np.full(150,.01)]
        r,_,_=run(effects,s,e['quality'],e['guardrails']);e.update(assessment=r['assessment'],actions=r['actions'])
        d=build_decision_drivers(e,effects)
        self.assertEqual(d['current_decision']['preferred_action'],'Global launch')
        self.assertEqual(d['launch_payoff_balance']['loss_probability'],.5)
        self.assertGreater(d['launch_payoff_balance']['expected_gain_component'],2*d['launch_payoff_balance']['expected_loss_component'])
        self.assertIn('not near-certain profit',risk_explanation(d))
        tie=np.r_[np.full(150,-.001),np.full(150,.002)]
        r,_,_=run(tie,s,e['quality'],e['guardrails']);e.update(assessment=r['assessment'],actions=r['actions'])
        d=build_decision_drivers(e,tie)
        self.assertFalse(d['leading_gap']['separation_clear_of_mc_noise'])

    def test_read_only_driver_tool_cannot_prepare_updates(self):
        e,effects=workspace();session=ToolSession(e,effects,read_only=True)
        self.assertNotIn('propose_scenario_update',[x['function']['name'] for x in session.definitions])
        before=fingerprint(e)
        result=session.execute('get_decision_drivers',{})
        self.assertIn('counterfactuals',result)
        self.assertIn('error',session.execute('propose_scenario_update',{'changes':{'maintenance':900},'reason':'Invent a new baseline.'}))
        self.assertIsNone(session.proposal);self.assertEqual(before,fingerprint(e))
        self.assertIn('error',session.execute('get_decision_drivers',{'code':'anything'}))


class ProviderErrorTests(unittest.TestCase):
    def error(self,status,source=None):
        response=requests.Response();response.status_code=status
        response._content=b'{"error":{"message":"sensitive provider response must not be echoed"}}'
        if source:return ProviderFailure(status,source)
        return requests.HTTPError(response=response)

    def test_credit_auth_rate_timeout_and_connection_messages_are_distinct_and_safe(self):
        self.assertIn('insufficient credits',failure_message(self.error(402)))
        self.assertIn('spending limit',failure_message(self.error(402)))
        self.assertIn('concurrent spending budget',failure_message(self.error(402,'openrouter_in_flight_budget')))
        self.assertIn('authenticate',failure_message(self.error(401)))
        self.assertIn('rate-limiting',failure_message(self.error(429)))
        self.assertIn('timed out',failure_message(requests.Timeout()))
        self.assertIn('connection',failure_message(requests.ConnectionError()))
        for code in (400,401,402,403,404,429,500):
            self.assertNotIn('sensitive provider',failure_message(self.error(code)))

    def test_error_in_success_http_envelope_is_also_classified(self):
        response=Mock();response.json.return_value={'error':{'code':402,'message':'Not enough credits'}}
        with patch('edi.copilot.requests.post',return_value=response):
            with self.assertRaises(ProviderFailure) as caught:post([],'test-key','test-model','https://example.invalid')
        self.assertIn('insufficient credits',failure_message(caught.exception))


class PreviewAndPlotTests(unittest.TestCase):
    def test_annual_cost_and_duration_are_in_business_units(self):
        e,effects=workspace(effect=.004)
        changed,patch=proposed_scenario(e['scenario'],{'annual_maintenance_eur':15000},'User estimate')
        r,_,_=run(effects,changed,e['quality'],e['guardrails'])
        proposal={'scenario':changed,'changes':patch,'preview':r}
        rows=preview_description(e['scenario'],proposal)
        self.assertEqual(rows[0]['before'],'€6,000/year')
        self.assertEqual(rows[0]['after'],'€15,000/year')
        self.assertIn('not a change in the experimental effect',preview_interpretation(e,r))
        self.assertIn('previous month',assumption_display('persistence',.9)[1])
        other,other_effects=workspace()
        no_switch,_,_=run(other_effects,changed,other['quality'],other['guardrails'])
        self.assertIn('remains preferred',preview_interpretation(other,no_switch))

    def test_increased_cost_does_not_hide_a_binding_safety_restriction(self):
        e,effects=workspace(effect=.01,safe=False)
        new,_=proposed_scenario(e['scenario'],{'maintenance':1000.},'User estimate')
        r,_,_=run(effects,new,e['quality'],e['guardrails'])
        self.assertIn('still unavailable',preview_interpretation(e,r))
        self.assertIn('does not resolve that blocker',preview_interpretation(e,r))

    def test_confidence_interval_has_clear_marker_caps_legend_and_exact_limits(self):
        row={'level':'Overall','effect':.003,'ci_low':-.001,'ci_high':.007,'control':.08,'treatment':.083,'n_control':1000,'n_treatment':1000}
        plot=observed_effect_chart([row],'binary')
        t=plot.data[0]
        self.assertEqual(t.marker.symbol,'diamond')
        self.assertGreaterEqual(t.error_x.thickness,3)
        self.assertIn('95% confidence interval',t.name)
        self.assertIn('95% confidence interval',t.hovertemplate)
        self.assertAlmostEqual(t.x[0]-t.error_x.arrayminus[0],-.1)
        self.assertAlmostEqual(t.x[0]+t.error_x.array[0],.7)


class StrategicAppJourneys(unittest.TestCase):
    def app(self):
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'streamlit_app.py'),default_timeout=60).run()
        return app.button(key='start_example').click().run()

    def test_one_click_review_requires_no_company_context(self):
        with patch('edi.copilot.configuration',return_value=None),patch('edi.copilot.ask',return_value='The key tension is short-run upside versus uncertain ongoing costs.') as ask:
            app=self.app();app.checkbox(key='ai_enabled').check().run()
            self.assertEqual(ask.call_count,0)
            context=next(t for t in app.text_area if t.label=='Optional initiative/company context')
            self.assertEqual(context.value,'')
            optional=next(e for e in app.expander if e.label=='Optional: add company context to personalize the review')
            self.assertFalse(optional.proto.expanded)
            self.assertEqual(app.button(key='request_initial_reading').label,'Get an AI strategic review')
            app.button(key='request_initial_reading').click().run()
            self.assertEqual(len(app.exception),0)
            self.assertEqual(ask.call_count,1)
            self.assertEqual(ask.call_args.kwargs['context'],'')
            self.assertIn('decision_drivers',ask.call_args.args[0])
            self.assertTrue(any(h.value=='Your AI strategic review' for h in app.subheader))
            self.assertTrue(any('The key tension is short-run upside' in m.value for m in app.markdown))

    def test_continue_selects_the_visible_panel_and_preserves_inputs(self):
        app=self.app()
        app.number_input(key='relative_goal').set_value(6.).run()
        for index,key in enumerate(['continue_setup','continue_2','continue_3','continue_4','continue_5'],start=1):
            app.button(key=key).click().run()
            self.assertEqual(len(app.exception),0)
            self.assertEqual(app.radio(key='active_step').value,STEPS[index])
            styles=next(m.value for m in app.markdown if m.value.startswith('<style>.st-key-workflow_'))
            self.assertNotIn(f'.st-key-workflow_{index} ',styles)
            self.assertEqual(styles.count('display: none'),5)
            self.assertEqual(app.number_input(key='relative_goal').value,6.)
        app.radio(key='active_step').set_value(STEPS[0]).run()
        self.assertEqual(app.session_state['active_step'],STEPS[0])

    def test_review_uses_latest_context_and_becomes_stale_without_an_automatic_call(self):
        with patch('edi.copilot.configuration',return_value=None),patch('edi.copilot.ask',return_value='UNIQUE strategic reading of the current context.') as ask:
            app=self.app();app.checkbox(key='ai_enabled').check().run()
            context=next(t for t in app.text_area if t.label=='Optional initiative/company context')
            context.set_value('Maintenance is €15,000 annually; a roadmap delay is not yet priced.')
            self.assertEqual(ask.call_count,0)
            app.button(key='request_initial_reading').click().run()
            self.assertIn('15,000',ask.call_args.kwargs['context'])
            self.assertEqual(ask.call_args.args[1],STRATEGIC_REVIEW)
            self.assertTrue(ask.call_args.kwargs['tool_session'].read_only)
            self.assertIn('decision_drivers',ask.call_args.args[0])
            self.assertIsNone(app.session_state['pending_proposal'])
            context=next(t for t in app.text_area if t.label=='Optional initiative/company context')
            context.set_value('Maintenance is now confirmed as €20,000 annually.')
            app.button(key='save_company_context').click().run()
            self.assertEqual(ask.call_count,1)
            self.assertTrue(any('out of date' in m.value for m in app.info))
            self.assertFalse(any('UNIQUE strategic reading' in m.value for m in app.markdown))
            app.button(key='request_initial_reading').click().run()
            self.assertEqual(ask.call_count,2)
            self.assertIn('20,000',ask.call_args.kwargs['context'])
            app.slider(key='allocation').set_value(.6).run()
            self.assertEqual(ask.call_count,2)
            self.assertTrue(any('out of date' in m.value for m in app.info))

    def test_credit_failure_keeps_the_active_scenario_and_results(self):
        with patch('edi.copilot.configuration',return_value=None),patch('edi.copilot.ask',side_effect=ProviderFailure(402)):
            app=self.app();app.checkbox(key='ai_enabled').check().run()
            before=copy.deepcopy(app.session_state['scenario'])
            probability=next(m.value for m in app.metric if m.label=='Chance of benefit · model')
            app.button(key='request_initial_reading').click().run()
            self.assertEqual(len(app.exception),0)
            self.assertTrue(any('insufficient credits' in m.value for m in app.warning))
            self.assertEqual(before,app.session_state['scenario'])
            self.assertEqual(probability,next(m.value for m in app.metric if m.label=='Chance of benefit · model'))

    def test_review_fingerprint_covers_context_scenario_policy_and_version(self):
        e,_=workspace();initial=review_fingerprint(e,'context')
        self.assertNotEqual(initial,review_fingerprint(e,'new context'))
        for group,key,value in [('business','maintenance',1500),('preferences','loss_weight',2)]:
            changed=copy.deepcopy(e);changed['scenario'][group][key]=value
            self.assertNotEqual(initial,review_fingerprint(changed,'context'))
        with patch('edi.copilot.REVIEW_VERSION','next-version'):
            self.assertNotEqual(initial,review_fingerprint(e,'context'))
