"""需要实际 OpenGL 的图片验收：独立方向锚点、边缘、亚像素、颜色及 PNG 读回。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import moderngl
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import star_shader as renderer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ctx = moderngl.create_context(standalone=True)
    texture = ctx.texture((128, 128), 3, dtype='f1')
    fbo = ctx.framebuffer(color_attachments=[texture])
    program = renderer.create_star_shader(ctx)
    errors, minimum_peak, checked = [], 255, 0
    try:
        fbo.use()
        ctx.viewport = (0, 0, 128, 128)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        # 121 个采样相位，检验最暗且最小的圆形星点不会在像素间消失。
        for dx in np.linspace(0, .99, 11):
            for dy in np.linspace(0, .99, 11):
                record = dict(id='phase', x_px=float(np.float32(64 + dx)),
                              y_px=float(np.float32(64 + dy)), diameter_px=2.5,
                              gain=float(np.float32(.600001)), color_rgb=[1, 1, 1])
                ctx.clear(0, 0, 0, 1)
                renderer.draw_stars(ctx, program, [record], 128)
                actual = renderer.read_framebuffer(fbo)
                result = renderer.verify_star_layer([record], actual, renderer.MapLayout(128, 4))
                minimum_peak = min(minimum_peak, result['minimum_witness_peak_code'])
                errors.append(result['max_channel_error'])
                checked += 1
        # 固定位置下的单星积分码值、峰值、覆盖面积必须随星等不增。
        responses = []
        for magnitude in [-3, 0, 3.99999, 4, 4.00001, 6.5, 8, 15]:
            record = dict(id='brightness', x_px=64.25, y_px=64.75,
                          diameter_px=float(np.float32(renderer.symbol_diameter(magnitude, 2048))),
                          gain=float(np.float32(10**(-.4*magnitude)+.6)), color_rgb=[1, 1, 1])
            ctx.clear(0, 0, 0, 1)
            renderer.draw_stars(ctx, program, [record], 128)
            actual = renderer.read_framebuffer(fbo)
            renderer.verify_star_layer([record], actual, renderer.MapLayout(128, 30))
            responses.append({'magnitude': magnitude, 'sum_codes': int(actual.sum()),
                              'peak_code': int(actual.max()), 'lit_pixels': int(np.any(actual > 0, axis=2).sum())})
        for a, b in zip(responses, responses[1:]):
            assert all(a[key] >= b[key] for key in ('sum_codes', 'peak_code', 'lit_pixels')), (a, b)
        gpu = {k: ctx.info[k] for k in ('GL_VENDOR', 'GL_RENDERER', 'GL_VERSION')}
    finally:
        program.release(); fbo.release(); texture.release(); ctx.release()

    # 实际 2K/4K 出图；极点、赤道两侧、经度接缝与亮暗阈值都进入实际管线。
    stars = [{'id': f'anchor_{i}_{j}', 'ra': ra, 'dec': dec, 'app_mag': 6.5,
              'color_hex': '#ffcc6f'}
             for i, ra in enumerate([0, 13.564125, 103.564125, 193.564125, 283.564125, 359.999999])
             for j, dec in enumerate([-90, -45, -1e-100, 0, 1e-100, 45, 90])]
    stars += [{'id': 'threshold_excluded', 'ra': 32, 'dec': 12, 'app_mag': 6.500001}]
    exports = []
    with tempfile.TemporaryDirectory(prefix='starmap-gpu-check-') as folder:
        source = Path(folder) / 'anchors.json'
        source.write_text(json.dumps({'stars': stars}))
        for resolution in (2048, 4096):
            report = renderer.export_map(source, Path(folder) / f'anchors_{resolution}.png',
                                         resolution=resolution, use_color=True)
            assert report['selection']['selected_count'] == 42
            assert report['selection']['excluded_ids'] == ['threshold_excluded']
            records = {r['id']: r for r in report['stars']}
            # RA=offset+90/180 的北赤道锚点：左侧/上侧；独立于生产投影函数。
            left, top = records['anchor_2_3'], records['anchor_3_3']
            center, radius = resolution / 2, report['sky_radius_px']
            assert abs(left['x_px']-(center-radius)) < .001 and abs(left['y_px']-center) < .001
            assert abs(top['x_px']-center) < .001 and abs(top['y_px']-(center-radius)) < .001
            assert records['anchor_0_2']['hemisphere'] == 'south'
            assert records['anchor_0_3']['hemisphere'] == 'north'
            exports.append({k: v for k, v in report.items() if k != 'stars'})
            # 保存失败及覆盖路径不应破坏已完成图像。
            destination = Path(folder) / f'anchors_{resolution}.png'
            original = destination.read_bytes()
            try:
                renderer.export_map(source, destination, resolution)
            except FileExistsError:
                pass
            else:
                raise AssertionError('Existing PNG was not protected')
            assert destination.read_bytes() == original
        empty = Path(folder) / 'empty.json'
        empty.write_text('{"stars": []}')
        report = renderer.export_map(empty, Path(folder) / 'empty.png', 128, lines=False)
        assert report['png_validation']['checked_stars'] == 0
    result = {'all_passed': True, 'gpu': gpu, 'subpixel_phases_checked': checked,
              'minimum_phase_peak_code': minimum_peak, 'maximum_phase_pixel_error': max(errors),
              'brightness_responses': responses, 'full_exports': exports,
              'renderer_sha256': hashlib.sha256(Path(renderer.__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(f'PASS: {checked} subpixel phases; minimum peak {minimum_peak}/255; 2K/4K coordinate and PNG checks')


if __name__ == '__main__':
    main()
