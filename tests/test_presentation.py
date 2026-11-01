import unittest
import numpy as np
from edi.presentation import posterior_chart,reach_threshold,sensitivity_view
from edi.engine import Business
from edi.scenarios import starter,run,shared_evsi
from edi.inference import infer,effect_draws
from edi.tools import ToolSession
import pandas as pd


class PresentationTests(unittest.TestCase):
    def test_colored_posterior_bars_share_one_probability_scale(self):
        values=np.linspace(-.1,.2,1000)
        chart=posterior_chart(values,5)
        self.assertAlmostEqual(sum(float(np.sum(trace.y)) for trace in chart.data),100.)
        self.assertEqual(sum(len(trace.x) for trace in chart.data),len(values))
        self.assertIn('modeled',chart.layout.yaxis.title.text)

    def test_reach_threshold_matches_central_business_equation(self):
        b=Business(margin_uncertainty=0)
        threshold=reach_threshold(.003,b,0)
        self.assertAlmostEqual(threshold,25000/(200000*.003*12*6))
        self.assertIsNone(reach_threshold(-.003,b,0))
        expensive=Business(implementation=1e9)
        self.assertGreater(reach_threshold(.003,expensive,0),1)

    def test_excluded_gradual_action_is_absent_from_policy_and_evsi(self):
        rng=np.random.default_rng(5)
        df=pd.DataFrame({'unit_id':np.arange(400),'variant':['control']*200+['treatment']*200,'y':rng.binomial(1,.2,400)})
        r=infer(df,'y','binary');d,m=effect_draws(df,r)
        scenario=starter(df,'y','binary')
        q={'errors':[],'warnings':[],'unavailable':[]};guards=[{'safe':True}]
        result,_,_=run(d,scenario,q,guards)
        self.assertNotIn('Gradual rollout',[a['action'] for a in result['actions']])
        info=shared_evsi(m,r,scenario,True,n_per_arm=100,outer=10,inner=50)
        self.assertNotIn('Gradual rollout',info['alternatives'])
        scenario['options']['include_gradual']=True
        result,_,_=run(d,scenario,q,guards)
        self.assertIn('Gradual rollout',[a['action'] for a in result['actions']])
        info=shared_evsi(m,r,scenario,True,n_per_arm=100,outer=10,inner=50)
        self.assertIn('Gradual rollout',info['alternatives'])

    def test_ai_cannot_silently_enable_an_excluded_alternative(self):
        scenario={'options':{'include_gradual':False}}
        tools=ToolSession({'scenario':scenario},np.zeros(100))
        result=tools.execute('propose_scenario_update',{'changes':{'include_gradual':True},'reason':'Try an additional choice.'})
        self.assertIn('error',result)
        self.assertIsNone(tools.proposal)
        self.assertFalse(scenario['options']['include_gradual'])
