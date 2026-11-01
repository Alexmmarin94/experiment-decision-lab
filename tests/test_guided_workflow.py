from io import BytesIO
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from edi.schema import parse_csv,normalize_schema,suggested_column,outcome_candidates,setup_issues,load_upload,minimal_template
from edi.context_analysis import cohort_effects
from edi.presentation import weekly_effect_chart,observed_effect_chart
from edi.interpretation import segment_readout
from edi.reporting import render_brief


def own_frame():
    rng=np.random.default_rng(9)
    return pd.DataFrame({'customer_id':[f'{i:05d}' for i in range(1,401)],'group':['A']*200+['B']*200,
        'purchase':rng.binomial(1,.12,400).astype(str)})


class SchemaTests(unittest.TestCase):
    def test_custom_mapping_retains_leading_zero_ids_and_raw_input(self):
        raw=parse_csv(own_frame().to_csv(index=False).encode());saved=raw.copy()
        self.assertEqual(suggested_column(raw.columns,'unit'),'customer_id')
        self.assertEqual(suggested_column(raw.columns,'assignment'),'group')
        normalized,mapping=normalize_schema(raw,'customer_id','group','A')
        self.assertEqual(normalized.unit_id.iloc[0],'00001')
        self.assertEqual(set(normalized.variant),{'control','treatment'})
        self.assertIn('purchase',outcome_candidates(normalized))
        self.assertEqual(setup_issues(normalized,'purchase'),[])
        pd.testing.assert_frame_equal(raw,saved)
        self.assertEqual(mapping['arm_mapping'],{'A':'control','B':'treatment'})

    def test_canonical_and_spelling_alias_are_accepted(self):
        raw=parse_csv(minimal_template().encode())
        raw=raw.rename(columns={'unit_id':'unir_id'})
        self.assertEqual(suggested_column(raw.columns,'unit'),'unir_id')
        normalized,_=normalize_schema(raw,'unir_id','variant','control')
        self.assertEqual(setup_issues(normalized,'converted_7d'),[])

    def test_specific_file_errors(self):
        for data in [b'',b'unit_id,variant,y\n',b'id;variant;y\nu1;control;1',b'id,id,y\n1,A,0',b'\xff\xfe']:
            with self.subTest(data=data),self.assertRaises(ValueError):parse_csv(data)

    def test_bad_required_values_are_not_imputed(self):
        raw=own_frame();raw.loc[0,'customer_id']=None;raw.loc[1,'purchase']='unknown'
        normalized,_=normalize_schema(raw,'customer_id','group','A')
        issues=setup_issues(normalized,'purchase')
        self.assertTrue(any('IDs are missing' in i for i in issues))
        self.assertTrue(any('nonnumeric' in i for i in issues))
        self.assertEqual(normalized.purchase.iloc[1],'unknown')

    def test_map_conflicts_and_missing_arms_are_actionable(self):
        raw=own_frame()
        with self.assertRaises(ValueError):normalize_schema(raw,'customer_id','customer_id')
        raw['variant']='another field'
        with self.assertRaises(ValueError):normalize_schema(raw,'customer_id','group','A')
        self.assertTrue(setup_issues(pd.DataFrame({'y':[1]}),'y'))

    def test_upload_helper_matches_csv_parser(self):
        f=BytesIO(own_frame().to_csv(index=False).encode())
        frame,digest=load_upload(f)
        self.assertEqual(len(frame),400);self.assertEqual(len(digest),64)
        self.assertEqual(load_upload(None),(None,None))


class StatisticalRegressionTests(unittest.TestCase):
    def test_example_primary_estimator_is_unchanged(self):
        from edi.inference import infer
        df=pd.read_csv('data/experiment_users.csv')
        r=infer(df,'converted_7d','binary')
        self.assertAlmostEqual(r['effect'],.004579534897924023,places=14)
        self.assertAlmostEqual(r['p_value'],.09807377484391533,places=14)
        self.assertAlmostEqual(r['ci_low'],-.0008468851515316244,places=14)
        self.assertAlmostEqual(r['ci_high'],.010010014754609529,places=14)


class WeeklyPresentationTests(unittest.TestCase):
    def test_all_user_weeks_show_date_ranges_in_chronological_order(self):
        df=pd.read_csv('data/experiment_users.csv')
        rows=cohort_effects(df,'converted_7d','binary')
        self.assertEqual([r['week_start'] for r in rows],sorted(r['week_start'] for r in rows))
        self.assertEqual(sum(r['n_control']+r['n_treatment'] for r in rows),len(df))
        self.assertEqual(rows[0]['week_label'],'31 Aug – 06 Sep 2026')
        self.assertEqual(rows[-1]['week_label'],'21 Sep – 27 Sep 2026')
        plot=weekly_effect_chart(rows,'binary','conversion')
        self.assertEqual(list(plot.layout.yaxis.categoryarray),[r['week_label'] for r in rows])
        self.assertEqual(plot.layout.yaxis.autorange,'reversed')
        self.assertIn('percentage points',plot.layout.xaxis.title.text)
        self.assertIn('Observed assignment dates',plot.data[0].hovertemplate)
        self.assertAlmostEqual(plot.data[0].x[0],rows[0]['effect']*100)

    def test_cross_year_week_is_one_calendar_cohort(self):
        rng=np.random.default_rng(7)
        df=pd.DataFrame({'variant':['control']*100+['treatment']*100,'y':rng.binomial(1,.1,200),
            'assigned_at':['2026-12-31T12:00:00Z']*100+['2027-01-01T12:00:00Z']*100})
        rows=cohort_effects(df,'y','binary')
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['week_label'],'28 Dec 2026 – 03 Jan 2027')


class SegmentInterpretationTests(unittest.TestCase):
    def evidence(self,effects):
        return {'segments':effects,'posterior':{'directional_probabilities':{'beneficial_direction':'increase'}},'heterogeneity_tests':[]}

    def test_raw_significance_is_not_reported_as_a_validated_segment(self):
        e=self.evidence([{'segment':'platform','level':'Android','effect':.01,'p_value':.01,'p_holm':.2}])
        text=segment_readout(e)
        self.assertIn('No subgroup passes the Holm multiple-comparison check',text)
        self.assertIn('effect estimates have not changed',text)
        self.assertIn('not proven nonresponders',text)

    def test_corrected_benefit_harm_and_heterogeneity_are_separate(self):
        e=self.evidence([{'segment':'platform','level':'Android','effect':.02,'p_holm':.01},
                         {'segment':'platform','level':'Web','effect':-.01,'p_holm':.02}])
        e['heterogeneity_tests']=[{'attribute':'platform','p_holm':.01}]
        text=segment_readout(e)
        self.assertIn('improvements survive',text);self.assertIn('deterioration survives',text)
        self.assertIn('strategically relevant follow-up',text)
        self.assertIn('not validated targeted-launch',text)
        self.assertIn('separate across-group',text)


class GuidedAppJourneys(unittest.TestCase):
    def app(self):
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'streamlit_app.py'),default_timeout=60).run()
        return app.button(key='start_example').click().run()

    def test_onboarding_value_and_next_action_without_result_spoiler(self):
        app=self.app()
        self.assertEqual(len(app.exception),0)
        self.assertEqual(app.title[0].value,'From experiment results to decisions you can defend')
        welcome=next(c for c in app.main.children.values() if c.type=='flex_container' and c.proto.id.endswith('-workflow_0'))
        welcome_text=' '.join(m.value for kind in [welcome.markdown,welcome.caption,welcome.success] for m in kind)
        self.assertNotIn('0.098',welcome_text)
        self.assertIn('No company outcomes',welcome_text)
        app.button(key='continue_setup').click().run()
        self.assertEqual(app.session_state['active_step'],'02 · Can we trust the data?')

    def test_custom_upload_maps_and_requires_its_own_design_confirmation(self):
        with patch('edi.onboarding.load_upload',return_value=(own_frame(),'custom-upload')):
            app=self.app();app.radio(key='source').set_value('Upload my CSV').run()
            self.assertEqual(len(app.exception),0)
            self.assertFalse(next(c for c in app.checkbox if c.label.startswith('Confirm: users')).value)
            app.button(key='continue_setup').click().run()
            self.assertTrue(any('More evidence required' in h.value for h in app.subheader))
            next(c for c in app.checkbox if c.label.startswith('Confirm: users')).check().run()
            self.assertEqual(len(app.exception),0)
            self.assertTrue(any(m.label=='Observed relative improvement' for m in app.metric))

    def test_new_upload_never_inherits_design_confirmation(self):
        with patch('edi.onboarding.load_upload',return_value=(own_frame(),'first-file')) as loader:
            app=self.app();app.radio(key='source').set_value('Upload my CSV').run()
            next(c for c in app.checkbox if c.label.startswith('Confirm: users')).check().run()
            self.assertTrue(next(c for c in app.checkbox if c.label.startswith('Confirm: users')).value)
            loader.return_value=(own_frame(),'second-file')
            app.run()
            self.assertEqual(len(app.exception),0)
            self.assertFalse(next(c for c in app.checkbox if c.label.startswith('Confirm: users')).value)

    def test_missing_ids_or_assignments_and_unusable_outcomes_do_not_crash(self):
        frames=[pd.DataFrame({'variant':['control','treatment'],'y':[0,1]}),
                pd.DataFrame({'unit_id':['u1','u2'],'y':[0,1]}),
                pd.DataFrame({'unit_id':['u1','u2'],'variant':['control','treatment'],'y':['bad','unknown']})]
        for i,frame in enumerate(frames):
            with self.subTest(frame=i),patch('edi.onboarding.load_upload',return_value=(frame,f'invalid-{i}')):
                app=self.app();app.radio(key='source').set_value('Upload my CSV').run()
                self.assertEqual(len(app.exception),0)
                self.assertTrue(len(app.error)>0)
                self.assertFalse(any(m.label=='Observed relative improvement' for m in app.metric))

    def test_context_survives_metric_change(self):
        app=self.app();context=next(t for t in app.text_area if t.label=='Optional initiative/company context')
        context.set_value('Finance prioritizes contribution and a reversible launch.')
        app.button(key='save_company_context').click().run()
        app.selectbox(key='metric').select('revenue_7d').run()
        self.assertEqual(len(app.exception),0)
        self.assertEqual(next(t for t in app.text_area if t.label=='Optional initiative/company context').value,'Finance prioritizes contribution and a reversible launch.')

    def test_manual_scenario_shows_changed_value_and_preserves_effect_probability(self):
        app=self.app()
        before=next(m.value for m in app.metric if m.label=='Chance of benefit · model')
        next(n for n in app.number_input if n.label=='Ongoing maintenance per year (€)').set_value(15000)
        next(b for b in app.button if b.label=='Update the business scenario').click().run()
        self.assertEqual(len(app.exception),0)
        self.assertEqual(app.session_state['scenario']['business']['maintenance'],1250)
        self.assertTrue(any(h.value=='What changed in this scenario?' for h in app.subheader))
        self.assertEqual(next(m.value for m in app.metric if m.label=='Chance of benefit · model'),before)
