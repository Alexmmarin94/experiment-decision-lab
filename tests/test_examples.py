from pathlib import Path
import unittest
import pandas as pd
from edi.example_data import generate,FILENAME
from edi.inference import infer


class ExampleTests(unittest.TestCase):
    def test_single_individual_csv_and_uncertain_primary(self):
        files=list(Path('data').glob('*.csv'))
        self.assertEqual([f.name for f in files],[FILENAME])
        df=pd.read_csv(files[0])
        self.assertEqual(len(df),40000)
        self.assertTrue(df.unit_id.is_unique)
        self.assertTrue((pd.to_datetime(df.window_end_at,utc=True)<=pd.to_datetime(df.observed_through,utc=True)).all())
        self.assertGreater(infer(df,'converted_7d','binary')['p_value'],.05)
        self.assertGreater(infer(df,'converted_7d','binary')['effect'],0)
        self.assertIn('crash_guardrail',df)
        self.assertIn('assigned_at',df)
        self.assertNotIn('exposed_at',df)
        self.assertEqual(list(df.columns),list(generate().columns))
        self.assertTrue(((df.converted_early+df.converted_late)<=df.converted_7d).all())
