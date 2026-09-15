import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import stellar_physics as physics
import star_generator as generator


class PhysicsTests(unittest.TestCase):
    def setUp(self):
        self.config = generator.GenerationConfig(seed=33)
        self.parameters = physics.sample_stellar_parameters('G', 'V', np.random.default_rng(33))
        self.star = generator.make_star(self.parameters, [10, 0, 0], 'G', 'V', self.config)
        self.star['id'] = 'test_000001'
        self.assertTrue(physics.validate_star(self.star)[0])

    def test_iau_solar_zero_point_and_angular_diameter(self):
        self.assertAlmostEqual(physics.SUN_M_BOL, 4.73999593391946, places=12)
        self.assertAlmostEqual(physics.SOLAR_DIAMETER_MAS_AT_PC, 9.300934521924315, places=10)
        self.assertAlmostEqual(10 ** ((physics.SUN_M_BOL - 4.73999593391946) / 2.5), 1)

    def test_reference_anchors_link_hot_stars_to_high_mass(self):
        mass, luminosity, bc = physics.main_sequence_parameters(29000, 'B')
        self.assertAlmostEqual(mass, 14.8)
        self.assertAlmostEqual(math.log10(luminosity), 4.43)
        self.assertAlmostEqual(bc, -2.83)
        self.assertAlmostEqual(physics.main_sequence_parameters(10700, 'B')[0], 2.75)
        self.assertAlmostEqual(physics.main_sequence_parameters(5770, 'G')[0], 1.0)

    def test_forward_photometry_has_no_1141_pc_ceiling(self):
        p = {**self.parameters, 'abs_mag': -6}
        result = generator.make_star(p, [1780.9897890590523, 0, 0], 'B', 'I', self.config)
        self.assertAlmostEqual(result['app_mag'], 6.5, places=10)
        self.assertEqual(result['distance_pc'], 1780.9897890590523)
        for mv, expected in [(-6, 1780.9897890590523), (-7, 2349.791073682951), (-8, 3010.117856626054)]:
            self.assertAlmostEqual(physics.solve_distance(mv, 6.5), expected, places=8)

    def test_zero_extinction_distance(self):
        self.assertAlmostEqual(physics.solve_distance(5, 5, 0), 10)
        with self.assertRaises(ValueError):
            physics.solve_distance(5, 5, -.1)

    def test_dust_integral_matches_independent_quadrature(self):
        for z0 in [-200, -20, 0, 20, 200]:
            for latitude in [-90, -10, -.0001, 0, .0001, 10, 90]:
                d = 3000.0
                path = np.linspace(0, d, 20001)
                density = np.exp((abs(z0) - np.abs(z0 + path * math.sin(math.radians(latitude)))) / 120)
                expected = .0007 * np.sum((density[:-1] + density[1:]) / 2) * d / 20000
                actual = physics.extinction_av(d, latitude, observer_height=z0)
                self.assertAlmostEqual(actual, expected, delta=3e-7)

    def test_dust_poles_plane_and_crossing(self):
        self.assertEqual(physics.extinction_av(0, 90), 0)
        self.assertAlmostEqual(physics.extinction_av(1000, 0), .7)
        self.assertAlmostEqual(physics.extinction_av(10000, 90, observer_height=0), .084)
        for d in [19.999, 20, 20.001, 1000]:
            self.assertAlmostEqual(physics.extinction_av(d, -90, observer_height=20),
                                   physics.extinction_av(d, 90, observer_height=-20))

    def test_corrupted_fields_are_rejected_without_exceptions(self):
        changes = {'mass_solar': [10000, 10**1000, float('nan'), True],
                   'radius_solar': [1e8, 0], 'luminosity_solar': [1e-20, float('inf')],
                   'temperature_K': [1, 100000], 'distance_pc': [1e-323, -1],
                   'pos_cartesian': [[0, 0, 0], [10**1000, 0, 0], [10, 0]],
                   'spectral_type': [[], {}, 'Q'], 'luminosity_class': [[], 'VI'],
                   'gal_lon': [360, float('nan')], 'gal_lat': [91], 'app_mag': [20],
                   'dist_ly': [1], 'angular_diameter_mas': [1e5]}
        for key, values in changes.items():
            for value in values:
                with self.subTest(field=key, value=str(value)[:50]):
                    damaged = {**self.star, key: value}
                    self.assertFalse(physics.validate_star(damaged)[0])

    def test_detached_mass_fails_even_when_other_equations_still_hold(self):
        p = physics.sample_stellar_parameters('B', 'V', np.random.default_rng(3))
        mass, luminosity, bc = physics.main_sequence_parameters(29000, 'B')
        p.update(mass_solar=2.515, temperature_K=29000, luminosity_solar=luminosity,
                 radius_solar=math.sqrt(luminosity)/(29000/5772)**2,
                 bolometric_mag=physics.SUN_M_BOL-2.5*math.log10(luminosity), bc_correction=bc)
        p['abs_mag'] = p['bolometric_mag']-bc
        star = generator.make_star(p, [100, 0, 0], 'B', 'V', self.config)
        self.assertIn('主序星质量与温度不符', physics.validate_star(star)[1])

    def test_evolved_outside_empirical_support_fails(self):
        temp, mv, mass = 5600, 2, 2
        bc = physics.get_bolometric_correction(temp)
        lum = 10**((physics.SUN_M_BOL-mv-bc)/2.5)
        p = dict(mass_solar=mass, temperature_K=temp, luminosity_solar=lum,
                 radius_solar=math.sqrt(lum)/(temp/5772)**2, abs_mag=mv,
                 bolometric_mag=mv+bc, bc_correction=bc)
        star = generator.make_star(p, [10, 0, 0], 'G', 'III', self.config)
        self.assertIn('演化星绝对星等超出该阶段的经验范围', physics.validate_star(star)[1])


class PopulationTests(unittest.TestCase):
    def test_density_normalization_and_disk(self):
        config = generator.GenerationConfig(observer_height_pc=0, local_density_per_pc3=.1)
        components = generator.population_components(config)
        self.assertAlmostEqual(sum(c['local_density'] for c in components), .1)
        c = next(c for c in components if c['spectral_type']=='B' and c['luminosity_class']=='V')
        self.assertAlmostEqual(generator.spatial_density([0,0,90], c, config)/c['local_density'], math.exp(-1))
        self.assertGreater(generator.spatial_density([1000,0,0],c,config), generator.spatial_density([-1000,0,0],c,config))

    def test_constant_density_poisson_shell_and_batch_tail(self):
        config = generator.GenerationConfig(min_distance_pc=1, radial_scale_pc=1e100,
                                             observer_height_pc=0, batch_size=7)
        component = {'local_density': .01, 'scale_height_pc': 1e100}
        rng = np.random.default_rng(123)
        counts = []
        sin_lat = []
        volume_radii = []
        longitudes = []
        for _ in range(1500):
            count = 0
            for candidates, positions in generator.spatial_batches(component, 10, config, rng):
                self.assertEqual(candidates, len(positions))
                count += len(positions)
                radii = np.linalg.norm(positions, axis=1)
                self.assertTrue(np.all((radii>=1)&(radii<=10)))
                sin_lat.extend(positions[:,2]/radii)
                volume_radii.extend((radii**3-1)/999)
                longitudes.extend(np.arctan2(positions[:,1], positions[:,0]))
            counts.append(count)
        expected = .01*4*math.pi/3*(1000-1)
        self.assertAlmostEqual(np.mean(counts), expected, delta=5*math.sqrt(expected/1500))
        self.assertAlmostEqual(np.var(counts), expected, delta=expected*.12)
        self.assertAlmostEqual(np.mean(sin_lat), 0, delta=.015)
        self.assertAlmostEqual(np.mean(np.square(sin_lat)), 1/3, delta=.015)
        self.assertAlmostEqual(np.mean(volume_radii), .5, delta=.01)
        self.assertAlmostEqual(np.var(volume_radii), 1/12, delta=.01)
        self.assertAlmostEqual(np.mean(np.cos(longitudes)), 0, delta=.015)
        self.assertAlmostEqual(np.mean(np.sin(longitudes)), 0, delta=.015)

    def test_density_bound_includes_towards_center_and_across_plane(self):
        rng = np.random.default_rng(614)
        for z0 in (-1000, -20, 0, 20, 1000):
            config = generator.GenerationConfig(observer_height_pc=z0)
            for horizon in (2., 30., 1000., 10000., 50000.):
                directions = rng.normal(size=(2000, 3))
                directions /= np.linalg.norm(directions, axis=1)[:, None]
                positions = directions * (horizon * rng.random(2000)**(1/3))[:, None]
                for height in (45., 90., 300., 900.):
                    component = {'local_density': .01, 'scale_height_pc': height}
                    bound = .01 * math.exp((config.observer_radius_pc-max(0,config.observer_radius_pc-horizon))/config.radial_scale_pc
                                          + (abs(z0)-max(0,abs(z0)-horizon))/height)
                    density = generator.spatial_density(positions, component, config)
                    self.assertTrue(np.all(density >= 0))
                    self.assertLessEqual(float(density.max()), bound*(1+1e-12))

    def test_visibility_horizons_are_conservative(self):
        config = generator.GenerationConfig()
        rng = np.random.default_rng(9)
        for component in generator.population_components(config):
            stype, lclass = component['spectral_type'], component['luminosity_class']
            horizon = generator.visibility_horizon(stype, lclass, 6.5, 50000)
            for _ in range(20):
                for attempt in range(1000):
                    try: p=physics.sample_stellar_parameters(stype,lclass,rng);break
                    except physics.CandidateRejected:pass
                else:self.fail('No physical support')
                if horizon < 50000:
                    self.assertGreaterEqual(p['abs_mag']+5*math.log10(horizon/10),6.5-1e-10)

    def test_visibility_is_applied_after_spatial_and_physical_generation(self):
        config=generator.GenerationConfig(minimum_visible_stars=0, maximum_visible_stars=None)
        component={'spectral_type':'G','luminosity_class':'V','local_density':.1,'scale_height_pc':300}
        mass,lum,bc=physics.main_sequence_parameters(5770,'G')
        mbol=physics.SUN_M_BOL-2.5*math.log10(lum)
        p=dict(mass_solar=mass,temperature_K=5770,luminosity_solar=lum,
               radius_solar=math.sqrt(lum)/(5770/5772)**2,abs_mag=mbol-bc,bolometric_mag=mbol,bc_correction=bc)
        with patch.object(generator,'population_components',return_value=[component]), \
             patch.object(generator,'spatial_batches',return_value=[(2,np.array([[10,0,0],[1000,0,0]]))]), \
             patch.object(generator,'sample_stellar_parameters',return_value=p):
            catalog=generator.generate_catalog(config,generation_id='test',progress=None)
        self.assertEqual(len(catalog['stars']),1)
        stats=catalog['metadata']['generation_stats']
        self.assertEqual(stats['population_stars_sampled'],2)
        self.assertEqual(stats['not_visible'],1)
        self.assertEqual(catalog['stars'][0]['distance_pc'],10)

    def test_budget_exhaustion_raises_instead_of_returning_partial_catalog(self):
        with self.assertRaises(RuntimeError):
            generator.generate_catalog(generator.GenerationConfig(max_candidate_points=1),progress=None)

    def test_import_has_no_generation_side_effects(self):
        self.assertFalse(hasattr(generator,'stars'))


class CatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=generator.GenerationConfig(seed=123,max_distance_pc=30,local_density_per_pc3=.002,
                                               minimum_visible_stars=0,maximum_visible_stars=None)
        cls.catalog=generator.generate_catalog(cls.config,generation_id='repeatable',progress=None)

    def test_reproducible_and_complete(self):
        again=generator.generate_catalog(self.config,generation_id='repeatable',progress=None)
        self.assertEqual(self.catalog,again)
        stats=again['metadata']['generation_stats']
        self.assertEqual(stats['population_stars_sampled'],stats['not_visible']+len(again['stars']))
        self.assertEqual(sum(c['visible'] for c in stats['components']),len(again['stars']))
        self.assertTrue(generator.validate_catalog(again)['all_passed'])

    def test_bad_metadata_and_ids_fail(self):
        for replacement in [[],None,'wrong_batch',self.catalog['stars'][1]['id']]:
            bad=copy.deepcopy(self.catalog);bad['stars'][0]['id']=replacement
            self.assertFalse(generator.validate_catalog(bad)['all_passed'])
        bad=copy.deepcopy(self.catalog);bad['metadata']['count']+=1
        self.assertFalse(generator.validate_catalog(bad)['all_passed'])
        bad=copy.deepcopy(self.catalog);bad['metadata']['generation_parameters']['seed']=10**1000
        self.assertFalse(generator.validate_catalog(bad)['all_passed'])

    def test_save_checks_actual_stars_and_preserves_existing_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            path=generator.save_catalog(copy.deepcopy(self.catalog),root,plots=False)
            original=path.read_bytes()
            with self.assertRaises(FileExistsError):generator.save_catalog(self.catalog,root,plots=False)
            self.assertEqual(path.read_bytes(),original)
            bad=copy.deepcopy(self.catalog);bad['stars'][0]['mass_solar']=10000
            bad['metadata']['validation_stats']={'all_passed':True}
            with self.assertRaises(ValueError):generator.save_catalog(bad,root,plots=False)
            self.assertEqual(path.read_bytes(),original)
            stored=json.loads(original)
            self.assertTrue(all(s['id'].startswith(stored['metadata']['generation_id']+'_') for s in stored['stars']))
            self.assertEqual(stored['metadata']['validation_stats']['passed'],len(stored['stars']))

    def test_manifest_excludes_incomplete_or_missing_catalogs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'output_missing').mkdir();(root/'output_complete').mkdir()
            (root/'output_complete/star_map_complete.json').write_text('{}')
            self.assertEqual(generator.generate_folders_json(root),['output_complete'])


if __name__=='__main__':unittest.main()
