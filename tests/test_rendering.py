import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import star_shader as renderer


class RenderingTests(unittest.TestCase):
    def test_visibility_threshold_is_inclusive_for_both_collections(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'input.json'
            data = {'metadata': {'generation_parameters': {'limiting_magnitude': 6.5}},
                    'stars': [{'id': str(i), 'ra': 10, 'dec': 20, 'app_mag': m}
                              for i, m in enumerate([6.4999, 6.5, 6.5001])],
                    'neighbors': [{'id': 'neighbor', 'ra': 0, 'dec': -20, 'app_mag': 20}]}
            path.write_text(json.dumps(data))
            stars, stats = renderer.load_render_catalog(path)
            self.assertEqual([s['id'] for s in stars], ['0', '1'])
            self.assertEqual(stats['excluded_ids'], ['2', 'neighbor'])
            self.assertEqual([(s['ra'], s['dec']) for s in stars], [(10, 20)] * 2)
            data['neighbors'][0]['id'] = '0'
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                renderer.load_render_catalog(path)

    def test_rotation_preserves_angles_and_galactic_pole(self):
        matrix = renderer.galactic_to_equatorial_rotation_matrix()
        np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-14)
        self.assertAlmostEqual(np.linalg.det(matrix), 1)
        ra, dec = renderer.galactic_to_equatorial(0, 90)
        self.assertAlmostEqual(ra, 192.85948)
        self.assertAlmostEqual(dec, 27.12825)
        a = renderer.ra_dec_to_cartesian(12, -40)
        b = renderer.ra_dec_to_cartesian(249, 58)
        self.assertAlmostEqual(np.dot(a, b), np.dot(matrix @ a, matrix @ b))

    def test_cardinal_directions_poles_and_seam(self):
        layout = renderer.MapLayout(4096, 40)
        center, radius = 2048, 2008
        # 独立固定方向锚点，包含旧图转角；不能只拿投影函数与自己比较。
        for dec in (90, -90):
            np.testing.assert_allclose(renderer.project_equatorial(123, dec, layout), [center, center])
        for ra, dx, dy in ((13.564125, 0, 1), (103.564125, -1, 0),
                           (193.564125, 0, -1), (283.564125, 1, 0)):
            np.testing.assert_allclose(renderer.project_equatorial(ra, 0, layout),
                                       [center + radius * dx, center + radius * dy], atol=1e-10)
            np.testing.assert_allclose(renderer.project_equatorial(ra, -45, layout),
                                       [center - radius * dx / 2, center + radius * dy / 2], atol=1e-10)
        np.testing.assert_allclose(renderer.project_equatorial(0, 45, layout),
                                   renderer.project_equatorial(360, 45, layout), atol=1e-10)

    def test_hemisphere_is_complete_and_equator_is_not_duplicated(self):
        stars = [{'id': str(i), 'ra': 0, 'dec': dec, 'app_mag': 6.5}
                 for i, dec in enumerate([-90, -1e-100, 0, 1e-100, 90])]
        layout = renderer.MapLayout.for_stars(stars, 2048)
        records = renderer.prepare_stars(stars, layout)
        self.assertEqual([r['hemisphere'] for r in records], ['south', 'south', 'north', 'north', 'north'])
        for r in records:
            self.assertAlmostEqual(r['merged_x_px'] - r['x_px'], layout.margin +
                                   (2048 if r['hemisphere'] == 'north' else 0))

    def test_monotone_symbols_and_all_subpixel_phases_have_a_pixel(self):
        magnitudes = [-5, 0, 3.999999, 4, 4.000001, 6.5, 8, 15]
        sizes = [renderer.symbol_diameter(m, 2048) for m in magnitudes]
        self.assertTrue(all(a >= b for a, b in zip(sizes, sizes[1:])))
        self.assertAlmostEqual(sizes[2], sizes[4], delta=1e-5)
        for magnitude in (6.5, 8, 15):
            for dx in np.linspace(0, .99, 11):
                for dy in np.linspace(0, .99, 11):
                    record = dict(x_px=50 + dx, y_px=50 + dy,
                                  diameter_px=renderer.symbol_diameter(magnitude, 2048),
                                  gain=10**(-.4*magnitude)+.6, color_rgb=[1, 1, 1])
                    _, _, patch = renderer.reference_symbol_patch(record, 128)
                    self.assertGreater(patch.max()*255, 15)

    def test_every_edge_symbol_fits_inside_mask(self):
        stars = [{'id': str(i), 'ra': float(ra), 'dec': 0., 'app_mag': -3.}
                 for i, ra in enumerate(np.arange(0, 360, .5))]
        layout = renderer.MapLayout.for_stars(stars, 2048)
        for r in renderer.prepare_stars(stars, layout):
            self.assertLess(math.hypot(r['x_px']-1024, r['y_px']-1024)+r['diameter_px']/2, 1023)

    def test_lines_are_clipped_before_projection(self):
        layout = renderer.MapLayout(2048, 30)
        outside = np.array([[0., -30.], [90., -45.]])
        self.assertEqual(len(renderer.hemisphere_segments(outside, True, layout, False)), 0)
        crossing = np.array([[13.564125, -30], [13.564125, 30]])
        projected = renderer.hemisphere_segments(crossing, True, layout, False)
        np.testing.assert_allclose(projected[0], [1024, 1024+layout.sky_radius], atol=1e-3)
        self.assertEqual(len(projected), 2)

    def test_final_png_cannot_pass_as_white_or_wrong_color_witness(self):
        stars = [dict(id='red', ra=23., dec=45., app_mag=6., color_hex='#ff0000')]
        layout = renderer.MapLayout.for_stars(stars, 128)
        records = renderer.prepare_stars(stars, layout, True)
        pixels = np.zeros((128,128,3), dtype='u1')
        left, top, patch = renderer.reference_symbol_patch(records[0], 128)
        pixels[top:top+patch.shape[0], left:left+patch.shape[1]] = np.rint(patch*255).astype('u1')
        renderer.verify_star_layer(records, pixels, layout)
        scene = {'north': Image.fromarray(pixels), 'south': Image.new('RGB',(128,128),'black')}
        before = renderer.create_merged_image(renderer.add_white_background(scene['north'],128),
                                              renderer.add_white_background(scene['south'],128), layout)
        self.assertTrue(renderer.verify_merged_image(before, records, layout, scene, before)['all_passed'])
        with self.assertRaises(RuntimeError):
            renderer.verify_merged_image(Image.new('RGB',layout.merged_size,'white'), records, layout, scene, before)
        damaged = before.copy()
        r = records[0]
        damaged.putpixel((r['merged_witness_x'], r['merged_witness_y']), (0,0,255))
        with self.assertRaises(RuntimeError):
            renderer.verify_merged_image(damaged, records, layout, scene, before)


if __name__ == '__main__':
    unittest.main()
