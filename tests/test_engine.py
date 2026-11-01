import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
from edi.engine import Business, adjust, decision, demo, estimate, posterior, simulate, validate


class EngineTests(unittest.TestCase):
    def test_multiplicity_known_values(self):
        np.testing.assert_allclose(adjust([.008,.041,.21,.61]), [.032,.123,.42,.61])
        np.testing.assert_allclose(adjust([.01,.04,.03], 'bh'), [.03,.04,.04])

    def test_srm_blocks(self):
        d = validate(demo('B'))
        self.assertLess(d['srm_p'], .001)
        self.assertTrue(d['errors'])
        self.assertEqual(decision(d, None, None, None)[0], 'Investigate')

    def test_missing_and_duplicate_are_not_silent(self):
        df = demo('A').iloc[:1000].copy()
        df.loc[0, 'converted_7d'] = np.nan
        df.loc[1, 'unit_id'] = df.loc[2, 'unit_id']
        self.assertGreaterEqual(len(validate(df)['errors']), 2)

    def test_boundary_interval_not_zero(self):
        df = pd.DataFrame({'variant':['control']*100+['treatment']*100, 'converted_7d':[0]*200})
        result = estimate(df)
        self.assertLess(result['ci_low'], 0)
        self.assertGreater(result['ci_high'], 0)
        self.assertEqual(result['p_value'], 1.)

    def test_posterior_reproducible_and_prior_sensitive(self):
        df = demo('A')
        np.testing.assert_array_equal(posterior(df), posterior(df))
        self.assertFalse(np.array_equal(posterior(df), posterior(df, prior=5)))

    def test_economic_identity(self):
        b = Business(margin_uncertainty=0)
        values, summary = simulate(np.full(100, .003), b)
        np.testing.assert_allclose(values, 9560.)
        self.assertAlmostEqual(summary['break_even_effect'], 25000/(200000*.8*12*6))
        self.assertAlmostEqual(summary['evpi'], 0.)
        lower, _ = simulate(np.full(100,.003), replace(b, persistence=.5))
        self.assertLess(lower.mean(), values.mean())

    def test_negative_value_prefers_wait(self):
        b = Business(implementation=1e8)
        df = demo('A')
        _, e = simulate(posterior(df), b)
        label, _ = decision(validate(df), estimate(df), estimate(df,'crashes_7d'), e)
        self.assertNotIn('Ship', label)

    def test_aa_interval_coverage(self):
        rng = np.random.default_rng(93)
        covered = 0
        for _ in range(200):
            df = pd.DataFrame({'variant':['control']*1000+['treatment']*1000,
                               'converted_7d':rng.binomial(1,.1,2000)})
            r = estimate(df)
            covered += r['ci_low'] <= 0 <= r['ci_high']
        self.assertGreater(covered/200, .90)
        self.assertLessEqual(covered/200, 1.)

    def test_invalid_business(self):
        with self.assertRaises(ValueError):
            simulate(np.zeros(100), Business(reach=0))


if __name__ == '__main__':
    unittest.main()
