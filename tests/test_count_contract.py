import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import star_generator as generator
import stellar_physics as physics


class CountContractTests(unittest.TestCase):
    def setUp(self):
        self.component = dict(spectral_type='G', luminosity_class='V', local_density=.1, scale_height_pc=300)
        self.parameters = physics.sample_stellar_parameters('G', 'V', np.random.default_rng(33))

    def mock_realizations(self, counts, config):
        batches = [[(n, np.tile([10., 0., 0.], (n, 1)))] for n in counts]
        with patch.object(generator, 'population_components', return_value=[self.component]), \
             patch.object(generator, 'spatial_batches', side_effect=batches), \
             patch.object(generator, 'sample_stellar_parameters', return_value=self.parameters):
            return generator.generate_catalog(config, generation_id='count_test', progress=None)

    def test_complete_realizations_are_retried_without_changing_stars(self):
        config = generator.GenerationConfig(minimum_visible_stars=3, maximum_visible_stars=4, max_catalog_attempts=3)
        catalog = self.mock_realizations([2, 5, 3], config)
        selection = catalog['metadata']['count_selection']
        self.assertEqual([a['visible_count'] for a in selection['attempts']], [2, 5, 3])
        self.assertEqual([a['spawn_key'] for a in selection['attempts']], [[0], [1], [2]])
        self.assertEqual(selection['accepted_attempt'], 3)
        self.assertEqual(len(catalog['stars']), 3)
        self.assertTrue(generator.validate_catalog(catalog)['all_passed'])
        self.assertTrue(all(s['distance_pc'] == 10 and s['abs_mag'] == self.parameters['abs_mag']
                            for s in catalog['stars']))
        self.assertNotEqual(np.random.default_rng(np.random.SeedSequence(config.seed, spawn_key=(0,))).random(),
                            np.random.default_rng(np.random.SeedSequence(config.seed, spawn_key=(1,))).random())

    def test_exhaustion_returns_no_partial_catalog(self):
        config = generator.GenerationConfig(minimum_visible_stars=3, maximum_visible_stars=4, max_catalog_attempts=2)
        with self.assertRaises(generator.CountConstraintError) as error:
            self.mock_realizations([2, 5], config)
        self.assertEqual([a['visible_count'] for a in error.exception.attempts], [2, 5])

    def test_both_bounds_are_inclusive_and_invalid_ranges_fail(self):
        config = generator.GenerationConfig()
        self.assertEqual([generator.count_in_range(n, config) for n in (8999, 9000, 9500, 9501)],
                         [False, True, True, False])
        for args in ({'minimum_visible_stars': -1}, {'maximum_visible_stars': 8999},
                     {'max_catalog_attempts': 0}, {'minimum_visible_stars': 2.5}):
            with self.assertRaises(ValueError):
                generator.GenerationConfig(**args)

    def test_save_rechecks_quantity_even_with_forged_pass_marker(self):
        catalog = self.mock_realizations([3], generator.GenerationConfig(minimum_visible_stars=3, maximum_visible_stars=4))
        catalog['stars'].pop()
        catalog['metadata']['count'] = 2
        # 保持记账自洽，确保失败确由硬下限触发。
        stats = catalog['metadata']['generation_stats']
        stats.update(population_stars_sampled=2)
        stats['components'][0].update(sampled=2, visible=2)
        catalog['metadata']['validation_stats'] = {'all_passed': True}
        self.assertFalse(generator.validate_catalog(catalog)['all_passed'])
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                generator.save_catalog(catalog, folder, plots=False)
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_legacy_catalog_has_no_new_count_requirement(self):
        catalog = self.mock_realizations([2], generator.GenerationConfig(minimum_visible_stars=0, maximum_visible_stars=None))
        catalog['metadata']['generator_version'] = '4.2'
        for field in ('minimum_visible_stars', 'maximum_visible_stars', 'max_catalog_attempts'):
            del catalog['metadata']['generation_parameters'][field]
        catalog['metadata'].pop('count_selection')
        self.assertTrue(generator.validate_catalog(catalog)['all_passed'])
        catalog['metadata']['generator_version'] = generator.GENERATOR_VERSION
        self.assertFalse(generator.validate_catalog(catalog)['all_passed'])

    def test_inconsistent_quantity_history_is_rejected(self):
        catalog = self.mock_realizations([3], generator.GenerationConfig(minimum_visible_stars=3, maximum_visible_stars=4))
        for key, value in (('minimum', 0), ('mode', 'unconditioned'), ('accepted_attempt', 2),
                           ('attempts', []), ('attempts', None)):
            bad = copy.deepcopy(catalog)
            bad['metadata']['count_selection'][key] = value
            self.assertFalse(generator.validate_catalog(bad)['all_passed'])
        for key, value in (('visible_count', 4), ('spawn_key', [1]), ('accepted', False)):
            bad = copy.deepcopy(catalog)
            bad['metadata']['count_selection']['attempts'][0][key] = value
            self.assertFalse(generator.validate_catalog(bad)['all_passed'])


if __name__ == '__main__':
    unittest.main()
