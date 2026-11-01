import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from edi.reporting import save_version as real_save,versions as real_versions


class AppTests(unittest.TestCase):
    def app(self):
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'streamlit_app.py'),default_timeout=60).run()
        return app.button(key='start_example').click().run()

    def test_six_step_workflow_and_relative_goal(self):
        app=self.app()
        self.assertEqual(len(app.exception),0)
        self.assertEqual(len(app.radio(key='active_step').options),6)
        self.assertEqual(app.radio(key='active_step').value,'02 · Can we trust the data?')
        self.assertEqual(len(app.tabs),4)  # native evidence subtabs; main steps use controlled panels
        self.assertTrue(any('40,000 individual user rows' in m.value for m in app.success))
        self.assertEqual(app.number_input(key='relative_goal').value,5.)
        labels=[s.label for s in app.selectbox]
        self.assertFalse(any('Analysis population' in s or 'Stopping rule' in s for s in labels))
        self.assertTrue(any(m.label=='Chance of benefit · model' for m in app.metric))
        self.assertEqual(len(app.chat_input),1)

    def test_optional_gradual_launch_starts_off_and_is_added_explicitly(self):
        app=self.app()
        self.assertFalse(app.session_state['scenario']['options']['include_gradual'])
        tables=[d.value for d in app.dataframe if 'Alternative' in d.value.columns]
        self.assertNotIn('Gradual rollout',set(tables[0]['Alternative']))
        self.assertIn('Iterate the feature',set(tables[0]['Alternative']))
        toggle=next(c for c in app.checkbox if c.label.startswith('Also compare a gradual'))
        toggle.check().run()
        self.assertEqual(len(app.exception),0)
        tables=[d.value for d in app.dataframe if 'Alternative' in d.value.columns]
        self.assertIn('Gradual rollout',set(tables[0]['Alternative']))

    def test_readable_charts_no_json_and_help_on_editable_inputs(self):
        app=self.app()
        self.assertEqual(len(app.json),0)
        headings=[h.value for h in app.subheader]
        self.assertTrue(any('How plausible is a useful improvement?' in h for h in headings))
        self.assertTrue(any('Can related groups help us' in h for h in headings))
        for widgets in [app.number_input,app.slider,app.checkbox,app.selectbox,app.text_input,app.text_area,app.multiselect,app.radio]:
            self.assertTrue(all(bool(w.proto.help) for w in widgets))
        self.assertFalse(any('Effect retained from one month' in w.label for w in app.slider))
        self.assertTrue(any('Budget to iterate and retest' in w.label for w in app.number_input))

    def test_srm_still_blocks_inference(self):
        app=self.app();app.slider(key='allocation').set_value(.6).run()
        self.assertEqual(len(app.exception),0)
        self.assertTrue(any('More evidence required' in m.value for m in app.subheader))
        self.assertFalse(any('Expected value · global' in m.label for m in app.metric))
        self.assertEqual(len(app.chat_input),1)

    def test_pending_upload_keeps_six_step_guide(self):
        app=self.app();app.radio(key='source').set_value('Upload my CSV').run()
        self.assertEqual(len(app.exception),0);self.assertEqual(len(app.radio(key='active_step').options),6)
        self.assertEqual(len(app.tabs),0)  # evidence is not calculated for an incomplete upload

    def test_numeric_outcome_and_partial_pooling_render(self):
        app=self.app();app.selectbox(key='metric').select('revenue_7d').run()
        self.assertEqual(len(app.exception),0)
        self.assertTrue(any('p99-capped' in m.value for m in app.markdown))
        self.assertTrue(any('Can related groups help us' in m.value for m in app.subheader))

    def test_ratio_and_lower_is_better_render(self):
        app=self.app();app.selectbox(key='metric').select('revenue_7d').run()
        app.selectbox(key='metric_type').select('Ratio of totals').run()
        app.selectbox(key='direction').select('Lower is better').run()
        self.assertEqual(len(app.exception),0)
        self.assertTrue(any('ratios' in i.value.lower() or 'ratio' in i.value.lower() for i in app.info))

    def test_ai_quality_initial_reading_and_conversation_are_separate(self):
        with patch('edi.copilot.configuration',return_value=None),patch('edi.copilot.explain_quality',return_value={'details':{},'trust_comment':'Review the missing exposure evidence.'}) as quality, \
             patch('edi.copilot.ask',return_value='Keep the primary uncertainty visible and validate the engineering cost.') as ask:
            app=self.app();app.checkbox(key='ai_enabled').check().run()
            self.assertEqual(len(app.exception),0);self.assertEqual(quality.call_count,0);self.assertEqual(ask.call_count,0)
            next(b for b in app.button if b.label=='Explain these checks with AI').click().run()
            self.assertEqual(quality.call_count,1)
            app.button(key='request_initial_reading').click().run()
            self.assertEqual(ask.call_count,1)  # review is an explicit request
            app.chat_input[0].set_value('What next?').run()
            app.chat_input[0].set_value('What if maintenance costs €15,000 annually?').run()
            self.assertEqual(len(app.exception),0)
            self.assertEqual(len(ask.call_args.kwargs['history']),2)
            self.assertIn('partial_pooling',ask.call_args.args[0])
            app.slider(key='allocation').set_value(.6).run()
            self.assertEqual(len(app.session_state['discussion']),0)

    def test_confirmed_ai_change_saves_before_and_after_versions(self):
        def respond(evidence,question,*args,**kwargs):
            if question.startswith('Maintenance'):
                kwargs['tool_session'].execute('propose_scenario_update',{'changes':{'annual_maintenance_eur':15000},'reason':'Annual maintenance estimate supplied by the user.'})
                return 'A preview is ready; confirm to apply it. The unpriced project delay remains a constraint.'
            return 'Initial reading: keep assumptions explicit.'
        with tempfile.TemporaryDirectory() as directory:
            db=Path(directory)/'versions.sqlite3'
            with patch('edi.copilot.configuration',return_value=None),patch('edi.copilot.explain_quality',return_value={'details':{},'trust_comment':'Check logging.'}),patch('edi.copilot.ask',side_effect=respond),patch('edi.reporting.save_version',side_effect=lambda path,*a,**k:real_save(db,*a,**k)),patch('edi.reporting.versions',side_effect=lambda path,experiment_id:real_versions(db,experiment_id)):
                app=self.app();app.checkbox(key='ai_enabled').check().run()
                original=app.session_state['scenario']['business']['maintenance']
                app.chat_input[0].set_value('Maintenance would cost €15,000 annually.').run()
                self.assertEqual(app.session_state['scenario']['business']['maintenance'],original)
                text=' '.join(m.value for m in app.markdown)
                self.assertIn('€6,000/year → **€15,000/year**',text)
                self.assertTrue(any('What would change if we used this assumption?' in h.value for h in app.subheader))
                comparison=next(t.value for t in app.table if 'With this change' in t.value.columns)
                self.assertIn('Preferred eligible action',comparison.index)
                app.button(key='confirm_proposal').click().run()
                self.assertEqual(len(app.exception),0)
                self.assertEqual(app.session_state['scenario']['business']['maintenance'],1250)
                history=real_versions(db,'experiment_001')
                self.assertEqual(len(history),2)
                self.assertEqual(history[0]['evidence']['scenario']['business']['maintenance'],original)
                self.assertEqual(history[1]['evidence']['scenario']['business']['maintenance'],1250)
                self.assertEqual(history[0]['evidence']['posterior'],history[1]['evidence']['posterior'])


if __name__=='__main__':unittest.main()
