import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from calibrate_population import classify_spectrum, fit_population
import star_generator as generator

DATA = Path(__file__).resolve().parents[1] / 'data'


class CalibrationTests(unittest.TestCase):
    def test_historical_and_ambiguous_spectra_remain_distinct(self):
        cases = {'gG9': ('G', 'III'), 'dF6': ('F', 'V'), 'sgG6': ('G', None),
                 'M2+III': ('M', 'III'), 'K0IIIbCN-0.5': ('K', 'III'),
                 'F2III-IV': ('F', None), 'K0II-IIICNIV': ('K', None),
                 'F9V+dM0': ('F', None), 'G8III:': ('G', None),
                 'Am(A2/A9V/F0)': ('A', None), 'C5III': (None, None), 'pec': (None, None)}
        for raw, expected in cases.items():
            with self.subTest(spectrum=raw):
                self.assertEqual(classify_spectrum(raw), expected)

    def test_profile_rebuilds_from_frozen_reference_and_pilots(self):
        reference = json.loads((DATA / 'bsc5_reference.json').read_text())
        pilots = json.loads((DATA / 'population_calibration_pilots.json').read_text())
        profile = fit_population(reference, pilots)
        stored = generator.POPULATION_PROFILE
        self.assertEqual(set(profile), set(stored))
        for key, value in profile.items():
            if key in ('components', 'calibrated_parameters', 'default_local_density_per_pc3'):
                continue
            self.assertEqual(value, stored[key], key)
        self.assertAlmostEqual(profile['default_local_density_per_pc3'],
                               stored['default_local_density_per_pc3'], places=12)
        self.assertAlmostEqual(profile['calibrated_parameters']['local_density_per_pc3'],
                               stored['calibrated_parameters']['local_density_per_pc3'], places=12)
        self.assertEqual(len(profile['components']), len(stored['components']))
        for rebuilt, original in zip(profile['components'], stored['components']):
            self.assertEqual(rebuilt.keys(), original.keys())
            for field, value in rebuilt.items():
                if isinstance(value, float):
                    self.assertAlmostEqual(value, original[field], places=12, msg=field)
                else:
                    self.assertEqual(value, original[field], field)
        self.assertEqual(reference['visible_count'], 8404)
        self.assertEqual(reference['modeled_type_count'] + reference['unmodeled_type_count'], 8404)
        self.assertTrue(all(c['fraction'] > 0 for c in profile['components']))
        self.assertAlmostEqual(sum(c['fraction'] for c in profile['components']), 1)
        # 极少量可见 M 矮星不应用来重拟合占总体多数的暗星。
        fitted = next(c for c in profile['components'] if (c['spectral_type'], c['luminosity_class']) == ('M', 'V'))
        original = next(c for c in pilots[0]['components'] if (c['spectral_type'], c['luminosity_class']) == ('M', 'V'))
        self.assertEqual(fitted['local_density_per_pc3'], original['local_density'])

    def test_incompatible_training_conditions_are_rejected(self):
        reference = json.loads((DATA / 'bsc5_reference.json').read_text())
        pilots = json.loads((DATA / 'population_calibration_pilots.json').read_text())
        pilots[1]['generation_parameters']['av_per_kpc'] = 0
        with self.assertRaises(ValueError):
            fit_population(reference, pilots)

    def test_statistics_never_become_a_visible_star_quota(self):
        metadata = {'generation_parameters': generator.POPULATION_PROFILE['calibrated_parameters'],
                    'population_profile_version': generator.POPULATION_PROFILE['profile_version']}
        catalog = {'metadata': copy.deepcopy(metadata), 'stars': []}
        comparison = generator.reference_comparison(catalog)
        self.assertTrue(comparison['applicable'])
        self.assertFalse(comparison['within_tolerance'])
        self.assertEqual(catalog['stars'], [])
        catalog['metadata']['generation_parameters']['limiting_magnitude'] = 5
        self.assertFalse(generator.reference_comparison(catalog)['applicable'])


if __name__ == '__main__':
    unittest.main()
