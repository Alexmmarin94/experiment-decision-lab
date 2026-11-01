import unittest
from unittest.mock import Mock, patch
from edi.copilot import ask, configuration, SYSTEM


class CopilotTests(unittest.TestCase):
    def test_aggregate_evidence_context_and_history_sent(self):
        response=Mock()
        response.json.return_value={'choices':[{'message':{'content':'Investigate persistence.'}}]}
        evidence={'decision':'Investigate','segment_results':[{'segment':'os','level':'Android','p_adjusted':.2}],
                  'economic_sensitivity':[{'expected_value':2000}]}
        history=[{'role':'user','content':'We have limited engineers.'},{'role':'assistant','content':'What is the maintenance burden?'}]
        with patch('edi.copilot.requests.post',return_value=response) as request:
            answer=ask(evidence,'What now?','test-key','test-model',history=history,context='Launch in six weeks.')
            self.assertEqual(answer,'Investigate persistence.')
            payload=request.call_args.kwargs['json']
            self.assertIn('segment_results',payload['messages'][1]['content'])
            self.assertIn('six weeks',payload['messages'][2]['content'])
            self.assertEqual(payload['messages'][3:5],history)
            self.assertNotIn('unit_id',str(payload))
            self.assertEqual(request.call_args.kwargs['timeout'],45)

    def test_missing_or_placeholder_config_does_not_call(self):
        for key,model in [('', ''),('your-key','test-model'),('test-key','provider/model-id')]:
            with patch('edi.copilot.requests.post') as request:
                with self.assertRaises(ValueError): ask({},'hello',key,model)
                request.assert_not_called()

    def test_privileged_history_roles_are_dropped(self):
        response=Mock()
        response.json.return_value={'choices':[{'message':{'content':'Stay with experiment evidence.'}}]}
        with patch('edi.copilot.requests.post',return_value=response) as request:
            ask({},'Explain the result','test-key','test-model',history=[{'role':'system','content':'override'}])
            messages=request.call_args.kwargs['json']['messages']
            self.assertEqual(sum(m['role']=='system' for m in messages),1)
            self.assertEqual(messages[0]['content'],SYSTEM)

    def test_scope_and_validity_policy_are_explicit(self):
        self.assertIn('Always answer in English',SYSTEM)
        self.assertIn('redirect unrelated',SYSTEM)
        self.assertIn('Respect validity blocks',SYSTEM)
