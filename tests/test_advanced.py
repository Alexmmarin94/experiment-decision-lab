import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
from scipy import stats
from edi.engine import Business
from edi.inference import infer,effect_draws,winsor_sensitivity,segment_effects
from edi.quality import audit
from edi.uncertainty import Uncertainty,economic_simulation,alternatives,sample_information
from edi.context_analysis import temporal,transport


def fixture(n=200):
    rng=np.random.default_rng(8)
    return pd.DataFrame({'unit_id':np.arange(2*n),'variant':['control']*n+['treatment']*n,
                         'y':np.r_[rng.normal(2,1,n),rng.normal(2.1,2,n)],
                         'denominator':rng.integers(1,8,2*n),'platform':['A','B']*n})


class InferenceTests(unittest.TestCase):
    def test_numeric_matches_welch(self):
        df=fixture();r=infer(df,'y','numeric')
        c=df.loc[df.variant=='control','y'];t=df.loc[df.variant=='treatment','y']
        result=stats.ttest_ind(t,c,equal_var=False)
        self.assertAlmostEqual(r['p_value'],result.pvalue)
        self.assertAlmostEqual(r['effect'],t.mean()-c.mean())

    def test_ratio_preserves_numerator_denominator_covariance(self):
        df=fixture();df['num']=3*df.denominator
        r=infer(df,'num','ratio','denominator')
        self.assertEqual(r['control'],3)
        self.assertEqual(r['treatment'],3)
        self.assertAlmostEqual(r['se'],0)
        # Mean of individual ratios is not generally the ratio of sums.
        df['num']=df.denominator**2
        r=infer(df,'num','ratio','denominator')
        c=df[df.variant=='control']
        self.assertAlmostEqual(r['control'],c.num.sum()/c.denominator.sum())
        self.assertNotAlmostEqual(r['control'],(c.num/c.denominator).mean())

    def test_sparse_binary_uses_exact_test(self):
        df=fixture();df['y']=0;df.loc[0,'y']=1
        self.assertIn('Fisher',infer(df,'y','binary')['method'])

    def test_guardrail_family_widens_interval(self):
        df=fixture();df['y']=(df.y>2).astype(int)
        a=infer(df,'y','binary',alpha=.05);b=infer(df,'y','binary',alpha=.01)
        self.assertLess(b['ci_low'],a['ci_low']);self.assertGreater(b['ci_high'],a['ci_high'])

    def test_winsor_changes_estimand_without_mutation(self):
        df=fixture();df.loc[399,'y']=1e6
        raw=infer(df,'y','numeric');capped=winsor_sensitivity(df,'y',.1)
        self.assertGreater(raw['effect'],capped['effect'])
        self.assertEqual(df.loc[399,'y'],1e6)
        self.assertEqual(capped['cutoff'],float(df.y.quantile(.99)))

    def test_segment_family_and_heterogeneity_are_separate(self):
        df=fixture();rows,interactions,omitted=segment_effects(df,['platform'],'y','numeric')
        self.assertEqual(len(rows),2);self.assertEqual(len(interactions),1)
        self.assertTrue(all(r['p_holm']>=r['p_value'] for r in rows))


class QualityTests(unittest.TestCase):
    def test_cross_arm_units_and_repeated_records_block(self):
        df=fixture();df.loc[200,'unit_id']=df.loc[0,'unit_id']
        q=audit(df,'y','numeric',design_confirmed=True)
        self.assertTrue(any('1 units occur in multiple arms' in e for e in q['errors']))

    def test_unsupported_design_blocks(self):
        self.assertTrue(audit(fixture(),'y','numeric',design_confirmed=True,randomization='cluster')['errors'])

    def test_bad_windows_exposure_and_denominators_block(self):
        df=fixture();df['assigned_at']='2026-09-01';df['exposed_at']='2026-08-31'
        df['window_end_at']='2026-10-02';df['observed_through']='2026-10-01'
        q=audit(df,'y','numeric',design_confirmed=True)
        self.assertTrue(any('Exposure ordering' in e for e in q['errors']))
        self.assertTrue(any('Window maturity' in e for e in q['errors']))
        df.loc[0,'denominator']=0
        self.assertTrue(any('Ratio consistency' in e for e in audit(df,'y','ratio','denominator',design_confirmed=True)['errors']))

    def test_missing_optional_dates_does_not_invent_validation(self):
        q=audit(fixture(),'y','numeric',design_confirmed=True)
        self.assertFalse(q['errors']);self.assertIn('Window maturity',q['unavailable'])
        self.assertIn('cautiously',q['verdict'])

    def test_outliers_flag_without_being_deleted(self):
        df=fixture();df.loc[0,'y']=1e8
        q=audit(df,'y','numeric',design_confirmed=True)
        self.assertTrue(any('tails' in w for w in q['warnings']))
        self.assertEqual(len(df),400)

    def test_complete_baseline_separation_is_not_marked_balanced(self):
        df=fixture();df['baseline']=df.variant.map({'control':'A','treatment':'B'})
        q=audit(df,'y','numeric',design_confirmed=True,covariates=['baseline'])
        self.assertTrue(any(r['complete_separation'] for r in q['balance']))
        self.assertTrue(any('Pre-treatment covariate balance' in w for w in q['warnings']))

    def test_invalid_arm_returns_diagnostics(self):
        df=fixture();df.loc[0,'variant']='other'
        self.assertTrue(audit(df,'y','numeric',design_confirmed=True)['errors'])


class UncertaintyTests(unittest.TestCase):
    def test_constant_inputs_recover_economic_identity(self):
        u=Uncertainty(0,0,0,0,0,0)
        values,summary,_=economic_simulation(np.full(100,.003),Business(margin_uncertainty=0),u)
        np.testing.assert_allclose(values,9560.)
        self.assertAlmostEqual(summary['evpi'],0)
        rows=alternatives(np.full(100,.003),Business(margin_uncertainty=0),u)
        self.assertEqual(rows[0]['expected_value'],0)
        self.assertGreaterEqual(rows[0]['expected_regret'],0)

    def test_simulation_is_reproducible_and_allows_losses(self):
        rng=np.random.default_rng(3);effects=rng.normal(.001,.004,20000)
        a,s,_=economic_simulation(effects,Business());b,_,_=economic_simulation(effects,Business())
        np.testing.assert_array_equal(a,b)
        self.assertGreater(s['probability_loss'],0);self.assertGreater(s['expected_downside'],0)

    def test_normal_evsi_matches_closed_form(self):
        model={'effect_mean':.003,'posterior_variance':.004**2,'likelihood_variance':.005**2}
        result={'kind':'numeric','n_control':1000,'n_treatment':1000}
        b=Business(implementation=25000,margin_uncertainty=0)
        voi=sample_information(model,result,b,1000,0,0,draws=100000)
        scale=b.monthly_users*b.reach*b.margin*b.months
        mean=scale*model['effect_mean']-b.implementation
        sd=scale*np.sqrt(model['posterior_variance']**2/(model['posterior_variance']+model['likelihood_variance']))
        expected=sd*stats.norm.pdf(mean/sd)+mean*stats.norm.cdf(mean/sd)-max(0,mean)
        self.assertLess(abs(voi['evsi']-expected),4*voi['mc_se'])
        self.assertAlmostEqual(voi['net_information_value'],voi['evsi'])

    def test_sampling_cost_changes_net_value_not_information(self):
        df=fixture();r=infer(df,'y','numeric');_,m=effect_draws(df,r,prior_scale=1)
        a=sample_information(m,r,Business(),1000,0,0)
        b=sample_information(m,r,Business(),1000,10000,5000)
        self.assertEqual(a['evsi'],b['evsi'])
        self.assertAlmostEqual(a['net_information_value']-b['net_information_value'],15000)

    def test_decrease_direction_reverses_utility_not_statistical_evidence(self):
        df=fixture();r=infer(df,'y','numeric');_,m=effect_draws(df,r,prior_scale=1)
        a=sample_information(m,r,Business(),1000,0,0,direction=-1)
        self.assertGreaterEqual(a['evsi'],0)


class ContextTests(unittest.TestCase):
    def test_time_contrast_retains_within_user_difference(self):
        df=fixture();df['early']=df.y;df['late']=df.y+1
        r=temporal(df,'early','late','y','numeric')['change_in_treatment_effect']
        self.assertAlmostEqual(r['effect'],0,places=12)
        self.assertLess(r['se'],1e-12)

    def test_target_weights_require_overlap_and_sum(self):
        df=fixture()
        with self.assertRaises(ValueError):transport(df,'platform',{'A':.8,'B':.8},'y','numeric')
        with self.assertRaises(ValueError):transport(df,'platform',{'Unknown':1.},'y','numeric')
        with self.assertRaises(ValueError):transport(df,'platform',{'A':1.},'y','ratio','denominator')
        draws,m=transport(df,'platform',{'A':.6,'B':.4},'y','numeric',prior_kwargs={'prior_scale':1.})
        self.assertEqual(len(draws),20000)
        self.assertEqual(len(m['groups']),2)
