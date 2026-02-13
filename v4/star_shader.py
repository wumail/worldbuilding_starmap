import json
import numpy as np
import moderngl
from PIL import Image
import argparse
import os
import sys

# 解除 PIL 图片大小限制
Image.MAX_IMAGE_PIXELS = None


# --- 辅助数学函数 ---

# 坐标系转换参数：银道坐标系 → 天球赤道坐标系
#
# 天文学标准参数（与真实地球-太阳-银河系统一致）：
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
    center_dec_rad = np.radians(-28.9)  # 标准值

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


def hex_to_rgb(hex_color):
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


# --- 数据加载 ---
def load_star_data(json_path):
    print(f"正在读取数据: {json_path} ...")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"错误: 无法读取 JSON 文件. {e}")
        sys.exit(1)

    stars = data.get("stars", [])
    neighbors = data.get("neighbors", [])
    all_stars = stars + neighbors

    # 坐标转换：银道坐标 → 天球赤道坐标
    print("正在转换坐标系：银道坐标 → 天球赤道坐标...")
    for star in all_stars:
        # 读取银道坐标 (新数据格式使用 gal_lon/gal_lat，旧数据使用 ra/dec)
        l_gal = star.get("gal_lon")  # 银经
        b_gal = star.get("gal_lat")  # 银纬

        # 转换为天球赤道坐标
        ra_eq, dec_eq = galactic_to_equatorial(l_gal, b_gal)

        # 保存为天球赤道坐标
        star["ra"] = ra_eq
        star["dec"] = dec_eq
        # 保留原始银道坐标
        star["gal_lon"] = l_gal
        star["gal_lat"] = b_gal

    print(f"共加载 {len(all_stars)} 颗星星（坐标已转换）")
    return all_stars


# --- Shader 定义 ---
def create_star_shader(ctx):
    """星星 Shader: 使用实例化四边形渲染，绕过 macOS gl_PointSize 64px 硬件限制"""
    vertex_shader = """
        #version 330

        const float BASE_GRID = 32768.0;  // 固定参考分辨率，用于位置量化，确保不同分辨率一致

        // 四边形顶点属性（每个四边形4个顶点）
        in vec2 in_quad_pos;  // 四边形局部坐标 (-0.5,-0.5) 到 (0.5,0.5)

        // 星星实例属性（每颗星一份）
        in float in_ra;
        in float in_dec;
        in float in_mag;
        in vec3 in_color;

        uniform float scale_factor;
        uniform float resolution;  // 视口分辨率（像素）
        uniform int is_north;

        out vec3 vColor;
        out float vIntensity;
        out vec2 vUV;  // 模拟 gl_PointCoord

        void main() {
            vColor = in_color;
            float intensity = pow(10.0, -0.4 * in_mag);
            vIntensity = intensity;

            // 1. 半球筛选
            bool visible = (is_north == 1 && in_dec >= 0.0) || (is_north == 0 && in_dec < 0.0);
            if (!visible) {
                gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
                return;
            }

            // 2. 投影逻辑 (真实观星视角的极方位投影)
            float r;
            if (is_north == 1) {
                r = (90.0 - in_dec) / 90.0;
            } else {
                r = (in_dec + 90.0) / 90.0;
            }

            float theta;
            if (is_north == 1) {
                theta = radians(-in_ra + 27.12825 / 2.0 + 180.0);
            } else {
                theta = radians(in_ra - 27.12825 / 2.0);
            }

            float x = r * cos(theta);
            float y = r * sin(theta);

            // 将 NDC 位置量化到固定网格
            float px = (x * 0.5 + 0.5) * BASE_GRID;
            float py = (y * 0.5 + 0.5) * BASE_GRID;
            px = floor(px) + 0.5;
            py = floor(py) + 0.5;
            float xq = (px / BASE_GRID - 0.5) * 2.0;
            float yq = (py / BASE_GRID - 0.5) * 2.0;

            // 分段拟合: 4K下 mag -1.5→64px, mag 0→52px, mag 4→20px, mag 7→4px
            // 亮星区 (mag ≤ 4): 线性，更大更明显
            // 暗星区 (mag > 4): 指数衰减
            float pointSize;
            if (in_mag <= 4.0) {
                pointSize = scale_factor * (52.0 - 8.0 * in_mag);
            } else {
                pointSize = scale_factor * max(1.0, 163.0 * exp(-0.536 * in_mag));
            }

            // 将像素大小转换为 NDC 偏移量
            // NDC 范围是 -1 到 1（宽度为2），对应 resolution 像素
            float halfSizeNDC = pointSize / resolution;

            // 四边形顶点 = 星星中心 + 局部偏移（缩放到NDC）
            vec2 offset = in_quad_pos * 2.0 * halfSizeNDC;
            gl_Position = vec4(xq + offset.x, yq + offset.y, 0.0, 1.0);

            // UV 坐标：模拟 gl_PointCoord（0,0 在左上，1,1 在右下）
            vUV = in_quad_pos + vec2(0.5);
        }
    """

    fragment_shader = """
        #version 330
        in vec3 vColor;
        in float vIntensity;
        in vec2 vUV;  // 模拟 gl_PointCoord (0,0)-(1,1)
        out vec4 fColor;

        void main() {
            vec2 coord = vUV - vec2(0.5);
            float dist = length(coord);  // 0.0 到 0.5 的归一化距离
            if (dist > 0.5) discard;

            // 固定核心和光晕的比例
            float coreRadius = 0.2;
            float glowStart = 0.2;
            float glowEnd = 0.5;
            
            // 核心：高斯分布
            float coreSigma = coreRadius / 2.5;
            float core = exp(-dist * dist / (2.0 * coreSigma * coreSigma));
            
            // 光晕：反距离衰减
            float glowDist = max(0.0, dist - glowStart);
            float glowRange = glowEnd - glowStart;
            float normalizedGlowDist = glowDist / glowRange;
            float glow = (1.0 / (1.0 + normalizedGlowDist * 10.0)) * 0.4;

            float intensity = (core + glow) * (vIntensity + 0.6);
            
            // Alpha 高斯衰减
            float alphaSigma = 0.15;
            float alphaCenter = 0.4;
            float alphaDist = max(0.0, dist - alphaCenter);
            float alpha = exp(-alphaDist * alphaDist / (2.0 * alphaSigma * alphaSigma));
            
            vec3 finalColor = vec3(1.0);
            fColor = vec4(finalColor * intensity, alpha);
        }
    """
    return ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)


def create_line_shader(ctx):
    """线条 Shader: 用于绘制银道线和黄道线"""
    vertex_shader = """
        #version 330

        const float BASE_GRID = 32768.0;  // 与星点一致的量化网格

        in float in_ra;
        in float in_dec;

        uniform int is_north;
        uniform float scale_factor; // 这里的 scale_factor 主要用于可能的线宽调整逻辑(目前未用)

        void main() {
            // 投影逻辑必须与星星完全一致
            float r;
            if (is_north == 1) {
                r = (90.0 - in_dec) / 90.0;
            } else {
                r = (in_dec + 90.0) / 90.0;
            }

            // 使用与星星相同的 theta 计算逻辑
            float theta;
            if (is_north == 1) {
                // 北半球：RA逆时针增加，需要取负并加180度偏移使RA=0在下方
                theta = radians(-in_ra + 27.12825 / 2.0 + 180.0);
            } else {
                // 南半球：RA顺时针增加，加90度偏移使RA=0在下方
                theta = radians(in_ra - 27.12825 / 2.0);
            }

            float x = r * cos(theta);
            float y = r * sin(theta);

            // 与星点同样的量化，确保位置一致
            float px = (x * 0.5 + 0.5) * BASE_GRID;
            float py = (y * 0.5 + 0.5) * BASE_GRID;
            px = floor(px) + 0.5;
            py = floor(py) + 0.5;
            float xq = (px / BASE_GRID - 0.5) * 2.0;
            float yq = (py / BASE_GRID - 0.5) * 2.0;

            // 如果 r > 1.0，点会落在视口外，ModernGL/OpenGL 会自动裁剪连线
            // 只要我们画的是连续的 LINE_LOOP，它会正确处理边界
            gl_Position = vec4(xq, yq, 0.0, 1.0);
        }
    """

    fragment_shader = """
        #version 330
        uniform vec4 line_color;
        out vec4 fColor;

        void main() {
            fColor = line_color;
        }
    """
    return ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)


# --- 渲染逻辑 ---
def render_hemisphere(
    ctx,
    star_prog,
    line_prog,
    vao_stars,
    vao_galactic,
    vao_ecliptic,
    fbo,
    resolution,
    is_north,
    star_count,
):
    """渲染半球，返回图像但不保存"""
    name = "北半球" if is_north else "南半球"
    print(f"正在渲染 {name} ({resolution}x{resolution})...")

    fbo.use()
    ctx.viewport = (0, 0, resolution, resolution)  # 确保视口与FBO分辨率一致
    ctx.clear(0.0, 0.0, 0.0, 1.0)

    # 全局设置
    # 开启混合模式 (对星星至关重要，对半透明线条也有用)
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
    # 设置线宽 (随分辨率缩放，使其更清晰)
    # 基础线宽 6.0px，按分辨率倍数缩放
    scale_factor = resolution / 4096.0
    line_width = max(6.0, 6.0 * scale_factor)
    ctx.line_width = line_width

    # 1. 渲染星星（实例化四边形渲染）
    star_prog["scale_factor"].value = scale_factor
    star_prog["resolution"].value = float(resolution)
    star_prog["is_north"].value = 1 if is_north else 0
    vao_stars.render(moderngl.TRIANGLE_STRIP, instances=star_count)

    # 2. 渲染线条
    line_prog["is_north"].value = 1 if is_north else 0

    # 绘制银道线 (蓝色 #4488ff, alpha 提高到 0.8)
    line_prog["line_color"].value = (0.26, 0.53, 1.0, 0.8)
    vao_galactic.render(moderngl.LINE_LOOP)

    # 绘制黄道线 (橙色 #ffaa00, alpha 提高到 0.9)
    line_prog["line_color"].value = (1.0, 0.66, 0.0, 0.9)
    vao_ecliptic.render(moderngl.LINE_LOOP)

    # 读取图像数据
    print("正在读取图像数据...")
    data = fbo.read(components=3)
    image = Image.frombytes("RGB", fbo.size, data)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)

    # 添加白色背景，保持星空区域为黑色
    image = add_white_background(image, resolution)

    # 在函数层面进行旋转：
    # 北半球：逆时针旋转90度
    # 南半球：顺时针旋转90度
    if is_north:
        image = image.rotate(90, expand=False)
    else:
        image = image.rotate(-90, expand=False)

    return image


def add_white_background(star_image, resolution):
    """
    在黑色星空图周围添加白色背景
    星空图是圆形区域，外部填充白色
    """
    # 创建白色背景
    background = Image.new("RGB", (resolution, resolution), (255, 255, 255))

    # 创建圆形遮罩
    mask = Image.new("L", (resolution, resolution), 0)
    from PIL import ImageDraw

    draw = ImageDraw.Draw(mask)

    # 绘制圆形遮罩 (星空区域)
    center = resolution // 2
    radius = resolution // 2
    draw.ellipse(
        [center - radius, center - radius, center + radius, center + radius], fill=255
    )

    # 将星空图粘贴到白色背景上
    background.paste(star_image, (0, 0), mask)

    return background


def create_merged_image(north_image, south_image, resolution, output_path):
    """
    创建南北半球相切的合并图像
    北半球在右侧，南半球在左侧，两个圆相切
    添加边距使星图与边缘保持距离
    """
    print("正在创建合并图像...")

    # 计算边距（分辨率的5%）
    margin = int(resolution * 0.05)

    # 为文字标签预留额外空间
    label_height = int(resolution * 0.08)  # 文字区域高度

    # 计算合并图像尺寸：两个圆相切 + 边距 + 底部标签区域
    merged_width = resolution * 2 + margin * 2
    merged_height = resolution + margin * 2 + label_height

    # 创建白色背景
    merged = Image.new("RGB", (merged_width, merged_height), (255, 255, 255))

    # 粘贴南半球（左侧，添加边距偏移）
    merged.paste(south_image, (margin, margin))

    # 粘贴北半球（右侧，添加边距偏移）
    merged.paste(north_image, (resolution + margin, margin))

    # 添加文字标签
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(merged)

    # 计算字体大小（根据分辨率自适应）
    font_size = int(resolution * 0.05)
    try:
        # 尝试使用系统字体
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except:  # noqa: E722
        # 如果失败，使用默认字体
        font = ImageFont.load_default()

    # 计算文字位置
    south_text = "SOUTH"
    north_text = "NORTH"

    # 获取文字边界框来计算居中位置
    south_bbox = draw.textbbox((0, 0), south_text, font=font)
    north_bbox = draw.textbbox((0, 0), north_text, font=font)

    south_text_width = south_bbox[2] - south_bbox[0]
    north_text_width = north_bbox[2] - north_bbox[0]

    # 南半球标签位置（左侧圆的中心下方）
    south_x = margin + resolution // 2 - south_text_width // 2
    south_y = margin + resolution + int(label_height * 0.3)

    # 北半球标签位置（右侧圆的中心下方）
    north_x = resolution + margin + resolution // 2 - north_text_width // 2
    north_y = margin + resolution + int(label_height * 0.3)

    # 绘制文字（黑色）
    draw.text((south_x, south_y), south_text, fill=(0, 0, 0), font=font)
    draw.text((north_x, north_y), north_text, fill=(0, 0, 0), font=font)

    print("正在保存合并图像...")
    merged.save(output_path, dpi=(300, 300))
    print(f"合并图像已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="生成全天星图 (含银道/黄道线)")
    parser.add_argument("input", help="输入JSON文件路径")
    parser.add_argument(
        "--res",
        choices=["2k", "4k", "8k", "16k", "32k"],
        default="4k",
        help="输出分辨率",
    )

    args = parser.parse_args()
    res_map = {"2k": 2048, "4k": 4096, "8k": 8192, "16k": 16384, "32k": 32768}
    resolution = res_map[args.res]

    if not os.path.exists(args.input):
        print(f"错误: 文件不存在 {args.input}")
        sys.exit(1)

    try:
        ctx = moderngl.create_context(standalone=True)
    except Exception:
        print("错误: 无法创建 OpenGL 上下文。")
        sys.exit(1)

    # 显存检查
    max_size = ctx.info["GL_MAX_VIEWPORT_DIMS"][0]
    if resolution > max_size:
        print(f"警告: 分辨率 {resolution} 超过显卡限制 {max_size}")

    # 创建 FBO
    try:
        texture = ctx.texture((resolution, resolution), 3, dtype="f1")
        depth_attachment = ctx.depth_renderbuffer((resolution, resolution))
        fbo = ctx.framebuffer(
            color_attachments=[texture], depth_attachment=depth_attachment
        )
    except Exception as e:
        print(f"显存分配失败: {e}")
        sys.exit(1)

    # --- 1. 准备星星数据 ---
    all_stars = load_star_data(args.input)
    star_count = len(all_stars)
    star_data = np.zeros(
        star_count,
        dtype=[
            ("in_ra", "f4", 1),
            ("in_dec", "f4", 1),
            ("in_mag", "f4", 1),
            ("in_color", "f4", 3),
        ],
    )
    for i, star in enumerate(all_stars):
        star_data["in_ra"][i] = star["ra"]
        star_data["in_dec"][i] = star["dec"]
        star_data["in_mag"][i] = star["app_mag"]
        star_data["in_color"][i] = hex_to_rgb(star.get("color_hex", "#ffffff"))

    vbo_stars = ctx.buffer(star_data.tobytes())
    prog_stars = create_star_shader(ctx)

    # 创建四边形顶点缓冲 (triangle strip: 4个顶点)
    # 局部坐标从 (-0.5, -0.5) 到 (0.5, 0.5)
    quad_vertices = np.array(
        [
            [-0.5, -0.5],  # 左下
            [0.5, -0.5],  # 右下
            [-0.5, 0.5],  # 左上
            [0.5, 0.5],  # 右上
        ],
        dtype="f4",
    )
    vbo_quad = ctx.buffer(quad_vertices.tobytes())

    # 实例化渲染 VAO：
    # - vbo_quad 是每个顶点的数据（4个顶点/实例）
    # - vbo_stars 是每个实例的数据（每颗星一份，divisor=1）
    vao_stars = ctx.vertex_array(
        prog_stars,
        [
            (vbo_quad, "2f", "in_quad_pos"),
            (vbo_stars, "1f 1f 1f 3f/i", "in_ra", "in_dec", "in_mag", "in_color"),
        ],
    )

    # --- 2. 准备线条数据 ---
    # 定义极点在天球赤道坐标系中的位置
    galactic_pole = (
        GALACTIC_POLE_RA,
        GALACTIC_POLE_DEC,
    )  # 银道北极（已在天球赤道系中定义）
    ecliptic_pole = (ECLIPTIC_POLE_RA, ECLIPTIC_POLE_DEC)  # 黄道北极（天球赤道系）

    # 生成点集
    galactic_points = create_great_circle_points(*galactic_pole)
    ecliptic_points = create_great_circle_points(*ecliptic_pole)

    # 转换为 VBO 格式 (只需要 RA, Dec)
    # numpy array 已经是 Nx2，可以直接用
    vbo_galactic = ctx.buffer(galactic_points.tobytes())
    vbo_ecliptic = ctx.buffer(ecliptic_points.tobytes())

    prog_lines = create_line_shader(ctx)

    # 绑定 VAO
    vao_galactic = ctx.vertex_array(
        prog_lines, [(vbo_galactic, "1f 1f", "in_ra", "in_dec")]
    )
    vao_ecliptic = ctx.vertex_array(
        prog_lines, [(vbo_ecliptic, "1f 1f", "in_ra", "in_dec")]
    )

    # --- 3. 渲染输出 ---
    input_dir = os.path.dirname(args.input) or "."
    input_base = os.path.basename(args.input).replace(".json", "")

    out_merged = os.path.join(input_dir, f"{input_base}_Merged_{args.res}.png")

    # 渲染北半球
    north_image = render_hemisphere(
        ctx,
        prog_stars,
        prog_lines,
        vao_stars,
        vao_galactic,
        vao_ecliptic,
        fbo,
        resolution,
        True,
        star_count,
    )

    # 渲染南半球
    south_image = render_hemisphere(
        ctx,
        prog_stars,
        prog_lines,
        vao_stars,
        vao_galactic,
        vao_ecliptic,
        fbo,
        resolution,
        False,
        star_count,
    )

    # 创建并保存合并图像
    create_merged_image(north_image, south_image, resolution, out_merged)

    print("渲染完成！")


if __name__ == "__main__":
    main()
