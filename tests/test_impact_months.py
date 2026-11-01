from pathlib import Path
import copy
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from edi.scenarios import starter,run,fingerprint
from edi.decision_drivers import build_decision_drivers
from edi.brief_drivers import ranked_conditions,render_conditions,impact_months_condition
from edi.reporting import render_brief,render_detailed_brief
from edi.explanations import BRIEF_DISCLAIMER
from edi.copilot import STRATEGIC_REVIEW


def workspace(implementation=9000.,safe=True,reversible_only=False):
    df=pd.DataFrame({'unit_id':range(100),'variant':['control']*50+['treatment']*50,'y':[0,1]*50})
    s=starter(df,'y','binary')
    s['business'].update(monthly_users=100000,reach=.8,margin=10.,implementation=implementation,margin_uncertainty=0.)
    s['uncertainty']={k:0. for k in s['uncertainty']}
    if reversible_only:
        s['business']['maintenance']=0.
        s['options'].update(include_gradual=True,global_reversible=False)
        s['preferences']['require_reversible']=True
    q={'errors':[],'warnings':[],'unavailable':[],'verdict':'Usable under declared scope'}
    guards=[{'metric':'crash_guardrail','safe':safe,'harm_direction':'Increase','maximum_harm':.001,'harm_upper_bound':.0001}]
    effects=np.full(300,.003)
    result,_,_=run(effects,s,q,guards)
    e={'experiment_id':'impact','experiment_name':'Checkout','data_hash':'synthetic','data_kind':'synthetic example',
       'engine_version':'test','rows':100,'metric':{'name':'y'},'primary':None,'posterior':None,'relative_goal_percent':5.,
       'scenario':s,'quality':q,'guardrails':guards,'assessment':result['assessment'],'actions':result['actions'],'evsi':None,'sensitivity':[]}
    e['decision_drivers']=build_decision_drivers(e,effects)
    return e,effects


class ImpactMonthTests(unittest.TestCase):
    def test_every_integer_month_is_computed_with_same_payoff_and_multiple_switches(self):
        e,effects=workspace();before=fingerprint(e)
        m=e['decision_drivers']['months_of_impact'];points={p['input']:p for p in m['tested_points']}
        self.assertEqual(list(points),list(range(1,37)))
        self.assertEqual(points[6]['preferred_feasible_action'],'Keep control')
        self.assertEqual(points[7]['preferred_feasible_action'],'Global launch')
        expected=100000*.8*.003*10*sum(.9**i for i in range(7))-9000-500*7
        self.assertAlmostEqual(points[7]['global_launch_expected_value'],expected)
        self.assertEqual(points[36]['preferred_feasible_action'],'Keep control')
        self.assertGreaterEqual(len(m['clear_switch_windows']),2)
        self.assertEqual(before,fingerprint(e))
        self.assertTrue(np.array_equal(effects,np.full(300,.003)))
        text=impact_months_condition(e)['text']
        self.assertIn('6 months → Keep control',text)
        self.assertIn('7 months → Global launch',text)
        self.assertIn('more months are not always better',text)
        self.assertIn('not experiment length',text)

    def test_no_switch_and_safety_blocks_are_not_invented_month_thresholds(self):
        expensive,_=workspace(implementation=1e8)
        self.assertIn('No change from **Keep control**',impact_months_condition(expensive)['text'])
        self.assertIn('1–36',impact_months_condition(expensive)['text'])
        blocked,_=workspace(safe=False)
        self.assertIn('No period alone',impact_months_condition(blocked)['text'])
        self.assertFalse(impact_months_condition(blocked)['changes_preference'])

    def test_month_changes_can_select_an_eligible_gradual_rollout(self):
        e,_=workspace(reversible_only=True)
        condition=impact_months_condition(e)
        self.assertTrue(condition['changes_preference'])
        self.assertIn('Gradual rollout',condition['text'])
        self.assertIn('gradual-rollout expected net value',condition['text'])

    def test_months_remain_in_brief_but_interface_disclaimer_is_not_exported(self):
        e,_=workspace()
        e['decision_conditions']=ranked_conditions(e)
        self.assertNotIn('months',[r['id'] for r in e['decision_conditions']])
        self.assertIn('**Months of impact:**',render_conditions(e))
        short=render_brief(e);long=render_detailed_brief(e)
        for report in [short,long]:
            self.assertNotIn(BRIEF_DISCLAIMER,report)
            self.assertNotIn("**Decision context:**",report)
            self.assertIn('**Months of impact:**',report)
        for phrase in ['secondary outcome','interaction','iterative release','strategic bets','acquisition or margin','05 · Explore with AI','data/safety constraints']:
            self.assertIn(phrase,BRIEF_DISCLAIMER)

    def test_missing_or_numerically_inconclusive_comparisons_remain_explicit(self):
        e,_=workspace();missing=copy.deepcopy(e)
        missing['decision_drivers'].pop('months_of_impact')
        self.assertIn('not available',impact_months_condition(missing)['text'])
        e['decision_drivers']['simulation']['comparison_action_matches_full_run']=False
        self.assertIn('improve simulation precision',impact_months_condition(e)['text'])


class ReviewMonthJourney(unittest.TestCase):
    def test_generated_review_always_displays_computed_months_even_if_provider_omits_it(self):
        with patch('edi.copilot.configuration',return_value=None),patch('edi.copilot.ask',return_value='Review: validate engineering costs.') as ask:
            app=AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'streamlit_app.py'),default_timeout=60).run()
            app.button(key='start_example').click().run()
            panel=next(c for c in app.main.children.values() if c.type=='flex_container' and c.proto.id.endswith('-workflow_5'))
            self.assertEqual(panel.children[0].value,BRIEF_DISCLAIMER)
            self.assertEqual(panel.children[0].type,'info')
            app.button(key='request_initial_reading').click().run()
            self.assertEqual(len(app.exception),0)
            self.assertEqual(ask.call_count,1)
            evidence=ask.call_args.args[0]
            self.assertIn('months_of_impact',evidence['decision_drivers'])
            self.assertIn('months_of_impact_condition',evidence)
            self.assertIn('Always include a distinct **Months of impact** bullet',STRATEGIC_REVIEW)
            self.assertTrue(any('Months of impact · calculated driver check' in m.value for m in app.markdown))
