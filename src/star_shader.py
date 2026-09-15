import json
import numpy as np
import moderngl
from PIL import Image
import argparse
import hashlib
import tempfile
from dataclasses import dataclass
import math
from pathlib import Path

# 解除 PIL 图片大小限制
Image.MAX_IMAGE_PIXELS = None


# --- 辅助数学函数 ---

# 坐标系转换参数：银道坐标系 → 天球赤道坐标系
#
# 固定架空天球姿态：接近地球 J2000，黄赤交角改为 25°。
# 1. 行星自转轴倾角（黄赤交角）：25° （地球为23.44°）
# 2. 黄道面与银道面夹角：约60°
# 3. 天球赤道面与银道面夹角：约63°
#
# 银道北极在天球赤道坐标系中的位置（J2000标准历元）：
GALACTIC_POLE_RA = 192.85948  # 银道北极的赤经（度）12h 51m 26.28s
GALACTIC_POLE_DEC = 27.12825  # 银道北极的赤纬（度）27° 07' 41.7"
GALACTIC_CENTER_RA = 266.4  # 银道中心方向的赤经（度）17h 45m 36s

# 黄道北极在天球赤道坐标系中的位置：
ECLIPTIC_POLE_RA = 270.0  # 黄道北极的赤经（度）18h 00m 00s（垂直于春分点）
ECLIPTIC_POLE_DEC = 65.0  # 黄道北极的赤纬（度）= 90° - 25°（行星倾角）


def galactic_to_equatorial_rotation_matrix():
    """
    构建银道坐标系到天球赤道坐标系的旋转矩阵
    基于银道北极位置和银道中心方向
    """
    # 银道北极在天球赤道系中的单位向量
    pole_ra_rad = np.radians(GALACTIC_POLE_RA)
    pole_dec_rad = np.radians(GALACTIC_POLE_DEC)

    # 银道中心方向（l=0, b=0）在天球赤道系中的方向
    center_ra_rad = np.radians(GALACTIC_CENTER_RA)
    center_dec_rad = np.radians(-28.9)  # 沿用旧星图的近似银心朝向

    # 构建银道坐标系的三个基向量在天球赤道系中的表示
    # z_gal: 银道北极方向
    z_gal = np.array(
        [
            np.cos(pole_dec_rad) * np.cos(pole_ra_rad),
            np.cos(pole_dec_rad) * np.sin(pole_ra_rad),
            np.sin(pole_dec_rad),
        ]
    )

    # x_gal: 银道中心方向（l=0, b=0）
    x_gal = np.array(
        [
            np.cos(center_dec_rad) * np.cos(center_ra_rad),
            np.cos(center_dec_rad) * np.sin(center_ra_rad),
            np.sin(center_dec_rad),
        ]
    )

    # y_gal: z × x 确保右手系
    y_gal = np.cross(z_gal, x_gal)
    y_gal = y_gal / np.linalg.norm(y_gal)

    # 重新正交化 x
    x_gal = np.cross(y_gal, z_gal)
    x_gal = x_gal / np.linalg.norm(x_gal)

    # 旋转矩阵：列向量是银道系基向量在赤道系中的表示
    R = np.column_stack([x_gal, y_gal, z_gal])

    return R


def galactic_to_equatorial(l_deg, b_deg):
    """
    将银道坐标(l, b)转换为天球赤道坐标(RA, Dec)

    参数:
        l_deg: 银经（度）
        b_deg: 银纬（度）

    返回:
        (ra, dec): 赤经、赤纬（度）
    """
    # 银道坐标转笛卡尔（在银道坐标系中）
    l_rad = np.radians(l_deg)
    b_rad = np.radians(b_deg)

    # 银道系中的笛卡尔坐标
    x_gal = np.cos(b_rad) * np.cos(l_rad)
    y_gal = np.cos(b_rad) * np.sin(l_rad)
    z_gal = np.sin(b_rad)

    vec_gal = np.array([x_gal, y_gal, z_gal])

    # 旋转到天球赤道系
    R = galactic_to_equatorial_rotation_matrix()
    vec_eq = R @ vec_gal

    # 转换为 RA/Dec
    x, y, z = vec_eq
    dec = np.degrees(np.arcsin(np.clip(z, -1, 1)))
    ra = np.degrees(np.arctan2(y, x))
    if ra < 0:
        ra += 360

    return ra, dec


def ra_dec_to_cartesian(ra, dec):
    """赤经赤纬转单位向量"""
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    return np.array([x, y, z])


def cartesian_to_ra_dec(vec):
    """单位向量转赤经赤纬"""
    x, y, z = vec
    dec = np.degrees(np.arcsin(z))
    ra = np.degrees(np.arctan2(y, x))
    if ra < 0:
        ra += 360
    return ra, dec


def create_great_circle_points(pole_ra, pole_dec, num_points=1080):
    """
    生成大圆路径点 (RA, Dec)
    原理: 大圆是距离极点 90 度的圆
    """
    # 1. 计算极点的笛卡尔坐标作为法向量
    normal = ra_dec_to_cartesian(pole_ra, pole_dec)

    # 2. 建立圆平面的基底向量 (Tangent Space)
    # 找一个辅助向量 (0,0,1) 或 (1,0,0) 来计算叉积
    if abs(normal[2]) < 0.9:
        tangent1 = np.cross(normal, [0, 0, 1])
    else:
        tangent1 = np.cross(normal, [1, 0, 0])
    tangent1 = tangent1 / np.linalg.norm(tangent1)

    tangent2 = np.cross(normal, tangent1)
    tangent2 = tangent2 / np.linalg.norm(tangent2)

    # 3. 生成圆周点
    points = []
    for i in range(num_points):
        # 0 到 2pi
        theta = 2 * np.pi * i / num_points

        # 圆上的点 P = t1 * cos + t2 * sin
        point_cartesian = tangent1 * np.cos(theta) + tangent2 * np.sin(theta)

        ra, dec = cartesian_to_ra_dec(point_cartesian)
        points.append((ra, dec))

    return np.array(points, dtype="f4")


def create_meridian_line(ra, south_lat=-80.0, north_lat=80.0, num_points=180):
    """
    生成经线（meridian）路径点
    
    参数:
        ra: 赤经（度）
        south_lat: 南端纬度（度）
        north_lat: 北端纬度（度）
        num_points: 点数
    
    返回:
        numpy array of (RA, Dec) points
    """
    points = []
    for i in range(num_points):
        # 从南纬到北纬线性插值
        dec = south_lat + (north_lat - south_lat) * i / (num_points - 1)
        points.append((ra, dec))
    
    return np.array(points, dtype="f4")


def create_latitude_circle(dec, num_points=360):
    """
    生成纬线（latitude circle）路径点
    
    参数:
        dec: 赤纬（度）
        num_points: 点数
    
    返回:
        numpy array of (RA, Dec) points
    """
    points = []
    for i in range(num_points):
        # RA 从 0 到 360 度
        ra = 360.0 * i / num_points
        points.append((ra, dec))
    
    return np.array(points, dtype="f4")


def hex_to_rgb(hex_color):
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


RENDERER_VERSION = "4.3"
ORIENTATION_OFFSET_DEG = 13.564125  # 沿用旧成品的图面转角，与天球物理姿态分开。
PIXEL_TOLERANCE = 2  # RGB8 量化与 GPU 浮点差异；逐像素验收的最大码值误差。


def load_render_catalog(json_path, limiting_magnitude=None):
    """按声明的星等筛选；显式赤道字段不会再次进行银道旋转。"""
    from stellar_physics import is_finite_number
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    metadata = data.get("metadata", {})
    if metadata.get("schema_version") == 2:
        from star_generator import validate_catalog
        validation = validate_catalog(data)
        if not validation["all_passed"]:
            raise ValueError(f"输入星表验证失败: {validation['errors'][:3]}")
    limit = (metadata.get("generation_parameters", {}).get("limiting_magnitude", 6.5)
             if limiting_magnitude is None else limiting_magnitude)
    if not is_finite_number(limit) or not -10 <= limit <= 15:
        raise ValueError("图片极限星等须在 -10 至 15 之间")
    selected, excluded, identifiers = [], [], set()
    for collection in ("stars", "neighbors"):
        entries = data.get(collection, [])
        if not isinstance(entries, list):
            raise ValueError(f"{collection} 必须是列表")
        for index, source in enumerate(entries):
            if not isinstance(source, dict):
                raise ValueError("恒星必须为对象")
            star = dict(source)
            identifier = star.get("id", f"{collection}:{index}")
            if not isinstance(identifier, str) or not identifier or identifier in identifiers:
                raise ValueError(f"恒星 ID 缺失或重复: {identifier}")
            identifiers.add(identifier)
            star["id"] = identifier
            galactic = "gal_lon" in star or "gal_lat" in star
            lon_key, lat_key = ("gal_lon", "gal_lat") if galactic else ("ra", "dec")
            lon, lat, mag = star.get(lon_key), star.get(lat_key), star.get("app_mag")
            if not all(is_finite_number(v) for v in (lon, lat, mag)):
                raise ValueError(f"恒星 {identifier} 的坐标或星等无效")
            if not 0 <= lon < 360 or not -90 <= lat <= 90 or not -30 <= mag <= 30:
                raise ValueError(f"恒星 {identifier} 的坐标或星等越界")
            if mag > limit:
                excluded.append(identifier)
                continue
            if galactic:
                star["ra"], star["dec"] = map(float, galactic_to_equatorial(lon, lat))
            selected.append(star)
    return selected, {"input_count": len(identifiers), "limiting_magnitude": limit,
                      "selected_count": len(selected), "excluded_ids": excluded,
                      "coordinate_convention": "fixed near-J2000 orientation; obliquity 25 degrees"}


def load_star_data(json_path):
    return load_render_catalog(json_path)[0]


def symbol_diameter(magnitude, resolution):
    """符号直径随星等连续且不增；最小 2.5 px 避免暗星完全漏采样。"""
    size = 52 - 8 * magnitude if magnitude <= 4 else 20 * math.exp(-.536 * (magnitude - 4))
    return max(2.5, size * resolution / 4096)


@dataclass(frozen=True)
class MapLayout:
    resolution: int
    padding_px: int

    @classmethod
    def for_stars(cls, stars, resolution):
        if not isinstance(resolution, int) or resolution < 128:
            raise ValueError("半球分辨率至少为 128 像素")
        maximum = max((symbol_diameter(s["app_mag"], resolution) for s in stars), default=2.5)
        padding = math.ceil(maximum / 2) + 2
        if padding >= resolution / 4:
            raise ValueError("星点符号相对画面过大，无法保留完整天球和边缘余量")
        return cls(resolution, padding)

    @property
    def sky_radius(self):
        return self.resolution / 2 - self.padding_px

    @property
    def margin(self):
        return int(self.resolution * .05)

    @property
    def merged_size(self):
        return (self.resolution * 2 + self.margin * 2,
                self.resolution + self.margin * 2 + int(self.resolution * .08))


def project_equatorial(ra, dec, layout, is_north=None):
    """方位等距投影，直接返回最终半球图片坐标（左上原点、像素边界为整数）。

    北图 RA 沿顺时针增加，南图沿逆时针增加，符合从天球内部仰望的方向。
    赤道星只归北图；线条可以显式指定半球以绘制共同的赤道边界。
    """
    north = dec >= 0 if is_north is None else is_north
    radius = layout.sky_radius * (1 - abs(dec) / 90)
    angle = math.radians(ra - ORIENTATION_OFFSET_DEG)
    return (layout.resolution / 2 + (-1 if north else 1) * radius * math.sin(angle),
            layout.resolution / 2 + radius * math.cos(angle))


def prepare_stars(stars, layout, use_color=False):
    records = []
    for star in stars:
        north = star["dec"] >= 0  # 在 float64 上决定归属，避免微小负赤纬转 float32 后成为 -0。
        x, y = project_equatorial(star["ra"], star["dec"], layout)
        # GPU 只接收已经投影好的最终像素中心，不再独立换算坐标或决定半球。
        x, y, diameter = map(float, np.float32([x, y, symbol_diameter(star["app_mag"], layout.resolution)]))
        color = np.asarray(hex_to_rgb(star.get("color_hex", "#ffffff")) if use_color else (1, 1, 1))
        if not np.any(color > 0):
            raise ValueError("彩色星点不能使用纯黑色")
        color = (color / color.max()).astype("f4").tolist()  # 色相符号，不把色值误当 V 波段通量。
        records.append({"id": star["id"], "ra_deg": star["ra"], "dec_deg": star["dec"],
                        "app_mag": star["app_mag"], "hemisphere": "north" if north else "south",
                        "x_px": x, "y_px": y, "diameter_px": diameter, "color_rgb": color,
                        "gain": float(np.float32(10 ** (-.4 * star["app_mag"]) + .6)),
                        "merged_x_px": x + layout.margin + (layout.resolution if north else 0),
                        "merged_y_px": y + layout.margin})
    return records


def create_star_shader(ctx):
    """GPU 只画实例化符号；位置、归属、星等选择在 CPU 完成并记录。"""
    return ctx.program(vertex_shader="""
        #version 330
        in vec2 in_quad_pos;
        in vec2 in_center;
        in float in_diameter;
        in float in_gain;
        in vec3 in_color;
        uniform float resolution;
        flat out vec2 center;
        flat out float diameter;
        flat out float gain;
        flat out vec3 color;
        void main() {
            vec2 p = in_center + in_quad_pos * in_diameter;
            gl_Position = vec4(2.0 * p.x / resolution - 1.0,
                               1.0 - 2.0 * p.y / resolution, 0.0, 1.0);
            center = in_center;
            diameter = in_diameter;
            gain = in_gain;
            color = in_color;
        }
    """, fragment_shader="""
        #version 330
        flat in vec2 center;
        flat in float diameter;
        flat in float gain;
        flat in vec3 color;
        uniform float resolution;
        out vec4 result;
        void main() {
            vec2 pixel = vec2(gl_FragCoord.x, resolution - gl_FragCoord.y);
            float d = length((pixel - center) / diameter);
            if (d > 0.5) discard;
            float core = exp(-d * d / (2.0 * 0.08 * 0.08));
            float glow = 0.4 / (1.0 + max(0.0, d - 0.2) / 0.3 * 10.0);
            float a = max(0.0, d - 0.4);
            float alpha = exp(-a * a / (2.0 * 0.15 * 0.15));
            // 光晕连续降到零，避免硬圆边界的浮点舍入造成整像素亮/灭跳变。
            float edge = 1.0 - smoothstep(0.45, 0.5, d);
            result = vec4(clamp(color * (core + glow) * gain * alpha * edge, 0.0, 1.0), 1.0);
        }
    """)


def draw_stars(ctx, program, records, resolution):
    if not records:
        return
    packed = np.asarray([[r["x_px"], r["y_px"], r["diameter_px"], r["gain"], *r["color_rgb"]]
                         for r in records], dtype="f4")
    data = ctx.buffer(packed.tobytes())
    quad = ctx.buffer(np.asarray([[-.5, -.5], [.5, -.5], [-.5, .5], [.5, .5]], dtype="f4").tobytes())
    vao = ctx.vertex_array(program, [(quad, "2f", "in_quad_pos"),
                          (data, "2f 1f 1f 3f/i", "in_center", "in_diameter", "in_gain", "in_color")])
    try:
        program["resolution"].value = float(resolution)
        vao.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=len(records))
    finally:
        vao.release(); quad.release(); data.release()


def hemisphere_segments(points, north, layout, closed=True):
    """先在球面赤道处分段，再投影；不依赖方形视口冒充半球裁剪。"""
    segments = []
    pairs = zip(points, np.roll(points, -1, axis=0)) if closed else zip(points[:-1], points[1:])
    for left, right in pairs:
        left, right = list(left), list(right)
        inside_l = left[1] >= 0 if north else left[1] <= 0
        inside_r = right[1] >= 0 if north else right[1] <= 0
        if not inside_l and not inside_r:
            continue
        if inside_l != inside_r:
            a, b = ra_dec_to_cartesian(*left), ra_dec_to_cartesian(*right)
            crossing = a + (-a[2] / (b[2] - a[2])) * (b - a)
            ra = math.degrees(math.atan2(crossing[1], crossing[0])) % 360
            if inside_l:
                right = [ra, 0]
            else:
                left = [ra, 0]
        segments.extend([project_equatorial(*left, layout, north), project_equatorial(*right, layout, north)])
    return np.asarray(segments, dtype="f4").reshape(-1, 2)


def draw_grid(ctx, layout, north):
    program = ctx.program(vertex_shader="""
        #version 330
        in vec2 position;
        uniform float resolution;
        void main() { gl_Position = vec4(2.0 * position.x / resolution - 1.0,
                                        1.0 - 2.0 * position.y / resolution, 0.0, 1.0); }
    """, fragment_shader="""
        #version 330
        uniform vec4 line_color;
        uniform float resolution;
        uniform float sky_radius;
        out vec4 result;
        void main() {
            if (length(gl_FragCoord.xy - vec2(resolution / 2.0)) > sky_radius) discard;
            result = vec4(line_color.rgb * line_color.a, 1.0);
        }
    """)
    program["resolution"].value = float(layout.resolution)
    program["sky_radius"].value = float(layout.sky_radius)
    ctx.line_width = max(1, 6 * layout.resolution / 4096)
    lines = [(create_great_circle_points(GALACTIC_POLE_RA, GALACTIC_POLE_DEC), (.26, .53, 1, .8), True),
             (create_great_circle_points(ECLIPTIC_POLE_RA, ECLIPTIC_POLE_DEC), (1, .66, 0, .9), True),
             (create_latitude_circle(0, 1080), (.7, .7, .7, .4), True)]
    lines += [(create_meridian_line(i * 360 / 26, -88, 88), (.7, .7, .7, .4), False) for i in range(26)]
    lines += [(create_latitude_circle(dec), (.7, .7, .7, .4), True) for dec in (-88, -74, -58, -34, 34, 58, 74, 88)]
    try:
        for points, color, closed in lines:
            projected = hemisphere_segments(points, north, layout, closed)
            if not len(projected):
                continue
            buffer = ctx.buffer(projected.tobytes())
            vao = ctx.vertex_array(program, [(buffer, "2f", "position")])
            try:
                program["line_color"].value = color
                vao.render(moderngl.LINES)
            finally:
                vao.release(); buffer.release()
    finally:
        program.release()


def reference_symbol_patch(record, resolution):
    """独立 CPU 光斑栅格计算，供实际 GPU 读回核验；不是第二次调用 shader。"""
    x, y, diameter = (record[k] for k in ("x_px", "y_px", "diameter_px"))
    radius = diameter / 2
    left, top = max(0, math.floor(x - radius)), max(0, math.floor(y - radius))
    right, bottom = min(resolution, math.ceil(x + radius)), min(resolution, math.ceil(y + radius))
    dx = np.arange(left, right, dtype=float) + .5 - x
    dy = np.arange(top, bottom, dtype=float) + .5 - y
    d = np.hypot(dx[None, :], dy[:, None]) / diameter
    core = np.exp(-d**2 / (2 * .08**2))
    glow = .4 / (1 + np.maximum(0, d - .2) / .3 * 10)
    alpha = np.exp(-np.maximum(0, d - .4)**2 / (2 * .15**2))
    t = np.clip((d - .45) / .05, 0, 1)
    edge = 1 - t*t*(3-2*t)
    value = np.clip(((core + glow) * record["gain"] * alpha * edge)[..., None] * record["color_rgb"], 0, 1)
    value[d > .5] = 0
    return left, top, value


def verify_star_layer(records, actual, layout):
    """每颗星必须有非零孤立贡献，完整 GPU 星点层必须符合独立 CPU 像素计算。"""
    expected = np.zeros_like(actual)
    min_peak = 255
    for record in records:
        left, top, patch = reference_symbol_patch(record, layout.resolution)
        height, width = patch.shape[:2]
        peak = float(patch.max()) * 255
        if peak <= PIXEL_TOLERANCE:
            raise RuntimeError(f"恒星 {record['id']} 在该分辨率没有可靠像素贡献")
        # 为整个星盘保留余量，而不只保证中心未被遮罩裁去。
        edge = math.hypot(record['x_px'] - layout.resolution / 2,
                          record['y_px'] - layout.resolution / 2) + record['diameter_px'] / 2
        if edge > layout.resolution / 2 - 1:
            raise RuntimeError(f"恒星 {record['id']} 的符号会被边缘裁切")
        py, px, _ = np.unravel_index(patch.argmax(), patch.shape)
        witness_x, witness_y = left + int(px), top + int(py)
        actual_peak = int(actual[witness_y, witness_x].max())
        if actual_peak < peak - PIXEL_TOLERANCE:
            raise RuntimeError(f"恒星 {record['id']} 的实际图像像素未通过")
        record.update(witness_x=witness_x, witness_y=witness_y,
                      isolated_peak_code=round(peak, 3), star_layer_peak_code=actual_peak,
                      isolated_lit_pixels=int(np.count_nonzero(patch.max(axis=2) * 255 >= .5)))
        min_peak = min(min_peak, actual_peak)
        region = expected[top:top + height, left:left + width]
        region[:] = np.minimum(255, np.rint(region.astype(float) + patch * 255)).astype('u1')
    max_error = bad_pixels = 0
    examples = []
    for row in range(0, layout.resolution, 128):
        delta = np.abs(expected[row:row + 128].astype('i2') - actual[row:row + 128].astype('i2'))
        max_error = max(max_error, int(delta.max()))
        bad_pixels += int(np.count_nonzero(np.any(delta > PIXEL_TOLERANCE, axis=2)))
        for yy, xx in np.argwhere(np.any(delta > PIXEL_TOLERANCE, axis=2))[:3]:
            examples.append({'x': int(xx), 'y': int(row + yy),
                             'expected': expected[row + yy, xx].tolist(),
                             'actual': actual[row + yy, xx].tolist()})
    if bad_pixels:
        raise RuntimeError(f"GPU 星点层与 CPU 参考不符: {bad_pixels} 个像素超差，最大 {max_error}；{examples[:3]}")
    return {"checked_stars": len(records), "all_passed": True, "max_channel_error": max_error,
            "tolerance_codes": PIXEL_TOLERANCE, "pixels_outside_tolerance": bad_pixels,
            "minimum_witness_peak_code": min_peak if records else None}


def read_framebuffer(fbo):
    return np.frombuffer(fbo.read(components=3, alignment=1), dtype='u1').reshape(fbo.height, fbo.width, 3)[::-1].copy()


def add_white_background(image, resolution):
    pixels = np.array(image)
    dx = np.arange(resolution, dtype=float) + .5 - resolution / 2
    for top in range(0, resolution, 128):
        dy = np.arange(top, min(top + 128, resolution), dtype=float) + .5 - resolution / 2
        outside = dx[None, :]**2 + dy[:, None]**2 > (resolution / 2)**2
        pixels[top:top + 128][outside] = 255
    return Image.fromarray(pixels)


def render_hemisphere(ctx, program, fbo, records, layout, north, lines=True):
    subset = [r for r in records if r['hemisphere'] == ('north' if north else 'south')]
    fbo.use()
    ctx.viewport = (0, 0, layout.resolution, layout.resolution)
    ctx.clear(0, 0, 0, 1)
    ctx.enable(moderngl.BLEND)
    ctx.disable_direct(0x0BD0)  # GL_DITHER：关闭每次混合时的颜色抖动，使 RGB8 验收可重现。
    ctx.blend_func = moderngl.ONE, moderngl.ONE
    draw_stars(ctx, program, subset, layout.resolution)
    pixels = read_framebuffer(fbo)
    validation = verify_star_layer(subset, pixels, layout)
    if lines:
        draw_grid(ctx, layout, north)
    scene = read_framebuffer(fbo)
    if np.any(scene < pixels):
        raise RuntimeError("参考线覆盖了已验证的星点层")
    reference = Image.fromarray(scene)
    image = add_white_background(reference, layout.resolution)
    return image, validation, reference


def create_merged_image(north_image, south_image, layout):
    from PIL import ImageDraw, ImageFont
    n, margin = layout.resolution, layout.margin
    merged = Image.new("RGB", layout.merged_size, "white")
    merged.paste(south_image, (margin, margin))
    merged.paste(north_image, (n + margin, margin))
    draw = ImageDraw.Draw(merged)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", max(10, int(n * .05)))
    except OSError:
        font = ImageFont.load_default()
    for label, x in (("SOUTH", margin + n / 2), ("NORTH", margin + n * 1.5)):
        box = draw.textbbox((0, 0), label, font=font)
        draw.text((x - (box[2] - box[0]) / 2, margin + n + int(n * .024)), label, font=font, fill="black")
    return merged


def verify_merged_image(image, records, layout, scene_references, before_save):
    """核对全部半球 RGB、几何遮罩和无损往返；白图或单点伪像不能通过。"""
    if image.size != layout.merged_size or image.mode != 'RGB':
        raise RuntimeError("合并图片尺寸或颜色模式错误")
    n, margin = layout.resolution, layout.margin
    for hemisphere in ('south', 'north'):
        reference = scene_references[hemisphere]
        if reference.size != (n, n) or reference.mode != 'RGB':
            raise RuntimeError("GPU 场景参考尺寸或颜色模式错误")
        offset = margin + (n if hemisphere == 'north' else 0)
        for row in range(0, n, 128):
            end = min(row + 128, n)
            expected = np.array(reference.crop((0, row, n, end)))
            # 直接在最终半球像素坐标中检验圆盘；不调用待验收的背景合成函数。
            xx = np.arange(n) + .5
            yy = np.arange(row, end) + .5
            outside = np.hypot(xx[None, :] - n/2, yy[:, None] - n/2) > n/2
            expected[outside] = 255
            actual = np.asarray(image.crop((offset, margin + row, offset + n, margin + end)))
            if not np.array_equal(actual, expected):
                raise RuntimeError(f"最终 PNG 的 {hemisphere} 半球像素、颜色或遮罩不符")
    # 包括标签和外边距在内的整张PNG必须与保存前逐码值相同。
    width, height = layout.merged_size
    for top in range(0, height, 128):
        box = (0, top, width, min(top + 128, height))
        if not np.array_equal(np.asarray(image.crop(box)), np.asarray(before_save.crop(box))):
            raise RuntimeError("PNG 保存/读取改变了图像像素")
    for record in records:
        x = record['witness_x'] + layout.margin + (layout.resolution if record['hemisphere'] == 'north' else 0)
        y = record['witness_y'] + layout.margin
        pixel = image.getpixel((x, y))
        if max(pixel) < record['star_layer_peak_code'] - 1:
            raise RuntimeError(f"最终 PNG 中恒星 {record['id']} 的像素丢失")
        record.update(merged_witness_x=x, merged_witness_y=y, png_witness_rgb=list(pixel))
    return {"checked_stars": len(records), "all_passed": True,
            "hemisphere_pixels_checked": 2*n*n, "roundtrip_pixels_checked": width*height,
            "differing_pixels": 0}


def export_map(input_path, output_path, resolution=4096, use_color=False, limiting_magnitude=None, lines=True):
    input_path, output_path = Path(input_path), Path(output_path)
    report_path = output_path.with_suffix('.render.json')
    if output_path.suffix.lower() != '.png':
        raise ValueError("输出图片必须为 PNG")
    for path in (output_path, report_path):
        if path.exists():
            raise FileExistsError(f"已有成品，拒绝覆盖: {path}")
    stars, selection = load_render_catalog(input_path, limiting_magnitude)
    layout = MapLayout.for_stars(stars, resolution)
    records = prepare_stars(stars, layout, use_color)
    ctx = moderngl.create_context(standalone=True)
    try:
        maximum = min(*ctx.info['GL_MAX_VIEWPORT_DIMS'], ctx.info['GL_MAX_TEXTURE_SIZE'])
        if resolution > maximum:
            raise ValueError(f"分辨率 {resolution} 超过当前显卡限制 {maximum}")
        texture = ctx.texture((resolution, resolution), 3, dtype='f1')
        fbo = ctx.framebuffer(color_attachments=[texture])
        program = create_star_shader(ctx)
        try:
            north, north_check, north_reference = render_hemisphere(ctx, program, fbo, records, layout, True, lines)
            south, south_check, south_reference = render_hemisphere(ctx, program, fbo, records, layout, False, lines)
            merged = create_merged_image(north, south, layout)
            gpu = {k: ctx.info[k] for k in ('GL_VENDOR', 'GL_RENDERER', 'GL_VERSION')}
        finally:
            program.release(); fbo.release(); texture.release()
    finally:
        ctx.release()
    if north_check['checked_stars'] + south_check['checked_stars'] != len(stars):
        raise RuntimeError("半球归属数量不完整")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.render-', dir=output_path.parent) as temporary:
        staged = Path(temporary) / output_path.name
        merged.save(staged, dpi=(300, 300))
        with Image.open(staged) as reopened:
            final_check = verify_merged_image(reopened, records, layout,
                                              {'north': north_reference, 'south': south_reference}, merged)
        report = {"renderer_version": RENDERER_VERSION,
                  "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "input_file": input_path.name, "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                  "output_file": output_path.name, "output_sha256": hashlib.sha256(staged.read_bytes()).hexdigest(),
                  "selection": selection, "projection": "polar azimuthal equidistant; south left, north right",
                  "pixel_coordinates": "top-left origin, pixel centers at (column+0.5, row+0.5)",
                  "equator_ownership": "north", "resolution_per_hemisphere": resolution,
                  "sky_radius_px": layout.sky_radius, "symbol_guard_px": layout.padding_px,
                  "orientation_offset_degrees": ORIENTATION_OFFSET_DEG, "use_color": use_color,
                  "grid_lines": lines, "mapping": "monotone size and nondecreasing intensity with brightness; minimum diameter 2.5 px",
                  "limitations": "symbol chart, not photometry; overlapping stars may merge and RGB8 addition may saturate",
                  "gpu": gpu, "north_validation": north_check, "south_validation": south_check,
                  "png_validation": final_check, "stars": records}
        staged_report = Path(temporary) / report_path.name
        staged_report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        # 'xb' 使保存瞬间也不能覆盖已有成品；验证失败时不会写出正式结果。
        import shutil
        with output_path.open('xb') as destination, staged.open('rb') as source:
            shutil.copyfileobj(source, destination)
        with report_path.open('x', encoding='utf-8') as destination:
            destination.write(staged_report.read_text(encoding='utf-8'))
    return report


def main():
    parser = argparse.ArgumentParser(description="导出并逐星验证南北天球符号星图")
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, help='另存新 PNG；已有图片和报告均不覆盖')
    parser.add_argument('--color', action='store_true', help='用归一化光谱色表示色相；默认白色')
    parser.add_argument('--limit-mag', type=float, help='默认使用星表的极限星等，旧格式默认 6.5')
    parser.add_argument('--no-lines', action='store_true', help='省略网格、银道线与黄道线')
    parser.add_argument('--res', choices=['2k', '4k', '8k', '16k', '32k'], default='4k')
    args = parser.parse_args()
    output = args.output or args.input.with_name(f'{args.input.stem}_Merged_{args.res}.png')
    try:
        report = export_map(args.input, output, int(args.res[:-1]) * 1024,
                            args.color, args.limit_mag, not args.no_lines)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))
    print(f"已验证并保存 {report['selection']['selected_count']} 颗可见星: {output}")
    print(f"逐星坐标和像素验收报告: {output.with_suffix('.render.json')}")


if __name__ == '__main__':
    main()
