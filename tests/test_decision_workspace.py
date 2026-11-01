import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock,patch
import numpy as np
import pandas as pd
from edi.inference import infer,effect_draws,relative_effect_draws
from edi.hierarchy import partial_pool
from edi.scenarios import starter,run,proposed_scenario,validate_patch,shared_evsi,fingerprint
from edi.decision_policy import Preferences,utility
from edi.tools import ToolSession,confirm_proposal
from edi.reporting import save_version,versions,compare_versions,render_brief,render_detailed_brief
from edi.copilot import ask,explain_quality


def workspace():
    rng=np.random.default_rng(2)
    df=pd.DataFrame({'unit_id':np.arange(1000),'variant':['control']*500+['treatment']*500,
                     'latency':np.r_[rng.normal(100,10,500),rng.normal(120,10,500)]})
    r=infer(df,'latency','numeric',sesoi=5)
    d,m=effect_draws(df,r,prior_scale=50,beneficial_direction=-1,relative_goal_pct=5)
    q={'errors':[],'warnings':[],'unavailable':[],'verdict':'Conditionally usable','checks':[{'check':'Units','status':'pass','detail':'Unique IDs.'}]}
    g=[{'metric':'latency_guardrail','safe':True,'maximum_harm':10,'harm_upper_bound':5,'harm_direction':'Increase'}]
    s=starter(df,'latency','numeric');out,worlds,extra=run(-d,s,q,g)
    evidence={'engine_version':'test','experiment_id':'exp1','experiment_name':'Latency initiative','data_kind':'synthetic example','data_hash':'test-hash','rows':len(df),
              'quality':q,'primary':r,'posterior':m,'relative_goal_percent':5.,'goal_reference':100.,'metric':{'name':'latency'},
              'scenario':s,'guardrails':g,'assessment':out['assessment'],'actions':out['actions'],'evsi':None,'sensitivity':[]}
    return df,r,d,m,evidence


class DirectionTests(unittest.TestCase):
    def test_increase_is_harm_for_lower_is_better_metric(self):
        _,r,d,m,_=workspace()
        self.assertGreater(m['raw_probabilities']['effect_above_zero'],.99)
        self.assertLess(m['directional_probabilities']['probability_benefit'],.01)
        self.assertGreater(m['directional_probabilities']['probability_harm'],.99)

    def test_binary_relative_goal_retains_control_uncertainty(self):
        rng=np.random.default_rng(6)
        df=pd.DataFrame({'variant':['control']*500+['treatment']*500,'y':rng.binomial(1,.2,1000)})
        r=infer(df,'y','binary',sesoi=.01)
        d,m=effect_draws(df,r,relative_goal_pct=5,seed=17)
        rng=np.random.default_rng(17)
        pc,pt=m['arm_posterior_beta'];c=rng.beta(*pc,20000);t=rng.beta(*pt,20000)
        self.assertEqual(m['directional_probabilities']['probability_useful_benefit'],float(((t-c)>.05*c).mean()))
        self.assertEqual(float((relative_effect_draws(m,d,r['control'])>.05).mean()),m['directional_probabilities']['probability_useful_benefit'])


class HierarchyTests(unittest.TestCase):
    def test_noisy_segment_shrinks_toward_shared_pattern(self):
        rows=[{'level':str(i),'effect':y,'se':se,'ci_low':y-1.96*se,'ci_high':y+1.96*se} for i,(y,se) in enumerate([(.01,.001),(.012,.001),(.09,.05)])]
        m=partial_pool(rows,.05,.01,.005)
        self.assertLess(abs(m['groups'][2]['pooled_effect']-.011),abs(.09-.011))
        self.assertLess(m['upper_grid_mass'],.01)
        self.assertTrue(all(r['credible_low']<r['credible_high'] for r in m['groups']))
        self.assertEqual(m,partial_pool(rows,.05,.01,.005))

    def test_prior_truncation_not_hidden_for_large_common_effect(self):
        rows=[{'level':str(i),'effect':10.,'se':.01,'ci_low':9.9804,'ci_high':10.0196} for i in range(3)]
        m=partial_pool(rows,.1,.04,.01)
        self.assertLess(m['upper_grid_mass'],.01)


class ToolTests(unittest.TestCase):
    def test_proposal_is_read_only_then_confirmed_and_audited(self):
        _,_,d,_,e=workspace();before=fingerprint(e)
        session=ToolSession(e,-d)
        response=session.execute('propose_scenario_update',{'changes':{'annual_maintenance_eur':15000},'reason':'User-reported annual engineering cost.'})
        self.assertTrue(response['active_scenario_unchanged'])
        self.assertEqual(fingerprint(e),before)
        new,event=confirm_proposal(e['scenario'],session.proposal,e)
        self.assertEqual(new['business']['maintenance'],1250)
        self.assertTrue(event['protected_evidence_unchanged'])

    def test_unknown_statistical_and_guardrail_changes_are_rejected(self):
        for changes in [{'prior_scale':1.},{'guardrail_limit':.2},{'data':[]},{'loss_weight':0},{'monthly_users':2.5},{'maintenance':float('inf')}]:
            with self.assertRaises(ValueError):validate_patch(changes)

    def test_stale_proposal_rejected(self):
        _,_,d,_,e=workspace();tools=ToolSession(e,-d)
        tools.execute('propose_scenario_update',{'changes':{'implementation':10000},'reason':'New estimate'})
        changed=json.loads(json.dumps(e));changed['relative_goal_percent']=8
        with self.assertRaises(ValueError):confirm_proposal(e['scenario'],tools.proposal,changed)

    def test_read_only_comparison_does_not_create_pending_confirmation(self):
        _,_,d,_,e=workspace();tools=ToolSession(e,-d)
        result=tools.execute('compare_scenario',{'changes':{'opportunity_cost':50000},'reason':'Priced displaced project.'})
        self.assertIsNone(tools.proposal)
        self.assertTrue(result['active_scenario_unchanged'])
        before=next(r for r in e['actions'] if r['action']=='Global launch')['expected_net_value']
        after=next(r for r in result['preview']['actions'] if r['action']=='Global launch')['expected_net_value']
        self.assertAlmostEqual(before-after,50000)

    def test_provider_tool_loop_returns_engine_result(self):
        _,_,d,_,e=workspace();tools=ToolSession(e,-d)
        first=Mock();first.json.return_value={'choices':[{'message':{'content':None,'tool_calls':[{'id':'call_1','type':'function','function':{'name':'get_decision_summary','arguments':'{}'}}]}}]}
        second=Mock();second.json.return_value={'choices':[{'message':{'content':'Keeping control remains the modeled preference.'}}]}
        with patch('edi.copilot.requests.post',side_effect=[first,second]) as call:
            answer=ask(e,'Compare options','test-key','test-model',tool_session=tools)
            self.assertIn('Keeping control',answer)
            self.assertEqual(call.call_count,2)
            self.assertTrue(any(m['role']=='tool' for m in call.call_args.kwargs['json']['messages']))


class PolicyAndInformationTests(unittest.TestCase):
    def test_common_objective_and_reversibility_constraint(self):
        _,r,d,m,e=workspace();s=e['scenario'];s['options']['global_reversible']=False;s['preferences']['require_reversible']=True
        out,_,_=run(-d,s,e['quality'],e['guardrails'])
        full=next(a for a in out['actions'] if a['action']=='Global launch')
        self.assertFalse(full['eligible'])
        info=shared_evsi(m,r,s,True,-1,n_per_arm=100,outer=20,inner=100)
        self.assertNotIn('Global launch',info['alternatives'])
        self.assertEqual(out['assessment']['objective'],info['objective'])

    def test_costs_enter_shared_utility_before_optimization(self):
        _,r,d,m,e=workspace()
        info=shared_evsi(m,r,e['scenario'],False,-1,n_per_arm=100,study_cost=1000,delay_cost=500,outer=10,inner=30)
        self.assertEqual(info['evsi'],0)
        self.assertEqual(info['net_information_value'],-3000)  # costs + equal downside penalty

    def test_ratio_starter_uses_denominator_volume_not_user_count(self):
        df,_,_,_,_=workspace()
        df['assigned_at']=pd.date_range('2026-09-01',periods=len(df),freq='h')
        df['sessions']=4
        numeric=starter(df,'latency','numeric')
        ratio=starter(df,'latency','ratio','sessions')
        self.assertAlmostEqual(ratio['business']['monthly_users']/numeric['business']['monthly_users'],4,places=2)
        self.assertIn('denominator',ratio['provenance']['monthly_users']['reason'])

    def test_no_invented_iteration_return(self):
        *_,e=workspace()
        iteration=next(r for r in e['actions'] if r['action']=='Iterate the feature')
        self.assertGreater(iteration['technical_cost_eur'],0)
        self.assertIsNone(iteration['expected_net_value']);self.assertFalse(iteration['eligible'])


class ReportTests(unittest.TestCase):
    def test_versioned_storage_is_append_only_and_repeat_save_idempotent(self):
        *_,e=workspace()
        with tempfile.TemporaryDirectory() as directory:
            db=Path(directory)/'versions.sqlite3'
            self.assertEqual(save_version(db,e),1);self.assertEqual(save_version(db,e),1)
            new=json.loads(json.dumps(e));new['scenario']['business']['maintenance']=1250
            self.assertEqual(save_version(db,new),2)
            history=versions(db,'exp1');self.assertEqual(len(history),2)
            self.assertTrue(compare_versions(history[0]['evidence'],history[1]['evidence']))
            self.assertNotEqual(history[0]['evidence']['scenario'],history[1]['evidence']['scenario'])

    def test_pm_document_distinguishes_evidence_assumptions_and_limits(self):
        *_,e=workspace();text=render_detailed_brief(e,context='Maintenance estimate from finance.',owner_note='Owner to confirm rollout feasibility.',version=1)
        for phrase in ['Observed evidence','model-conditioned','Economic assumptions','Risks and constraints','What is not justified','Decision conditions','Missing evidence','Decision version: 1']:
            self.assertIn(phrase,text)

    def test_default_document_is_concise_and_appendix_retains_assumptions(self):
        *_,e=workspace()
        short=render_brief(e);detail=render_detailed_brief(e)
        self.assertLess(len(short.split()),500)
        self.assertIn('Why this decision',short)
        self.assertIn('Evidence that matters',short)
        self.assertIn('Next step',short)
        self.assertNotIn('| Parameter |',short)
        self.assertIn('| Parameter |',detail)
        self.assertLess(len(short),len(detail))

    def test_quality_ai_cannot_modify_computed_status(self):
        quality={'verdict':'Blocked','checks':[{'check':'Units','status':'block','detail':'Duplicate IDs.'}]}
        response=Mock();response.json.return_value={'choices':[{'message':{'content':json.dumps({'details':[{'check':'Units','explanation':'Resolve duplicate records.'},{'check':'invented','explanation':'Fake check.'}],'trust_comment':'Resolve the blocker first.'})}}]}
        with patch('edi.copilot.requests.post',return_value=response):result=explain_quality(quality,'test-key','test-model')
        self.assertEqual(quality['checks'][0]['status'],'block')
        self.assertNotIn('invented',result['details'])
