import json
import numpy as np
import math


def load_stars(filename="star_map.json"):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data["stars"]
    except FileNotFoundError:
        print(f"Error: 文件 {filename} 未找到。请先运行生成脚本。")
        return []


def analyze_disk_concentration(stars):
    """
    统计验证：计算不同距离区间内，星星落在银盘平面(±10度)内的比例。
    """
    print("\n" + "=" * 60)
    print(f"{'【科学验证】银盘面集中度分析':^50}")
    print("=" * 60)
    print(
        f"{'距离区间 (pc)':<20} | {'总数':<8} | {'盘面占比 (±10°)':<18} | {'集中倍率':<10}"
    )
    print("-" * 65)

    # 理论上的各向同性比例 (sin(10°) - sin(-10°)) / 2 ≈ 0.1736
    ISOTROPIC_REF = math.sin(math.radians(10))

    # 定义距离分桶
    bins = {
        "近邻 (0-300pc)": {"count": 0, "in_disk": 0},
        "中距 (300-1k pc)": {"count": 0, "in_disk": 0},
        "远景 (>1000pc)": {"count": 0, "in_disk": 0},
    }

    for star in stars:
        # 提取数据
        dist = star.get("distance_pc", 0)
        pos = star.get("pos_cartesian", [0, 0, 0])
        x, y, z = pos

        # 计算银纬的正弦值 sin(b) = z / dist
        # 绝对值越小，说明越靠近赤道面（银盘）
        if dist > 0:
            sin_b = abs(z / dist)
        else:
            sin_b = 0

        # 判断归属哪个桶
        key = ""
        if dist < 300:
            key = "近邻 (0-300pc)"
        elif dist < 1000:
            key = "中距 (300-1k pc)"
        else:
            key = "远景 (>1000pc)"

        bins[key]["count"] += 1

        # 判断是否在盘面内 (sin(b) < sin(10°))
        if sin_b < ISOTROPIC_REF:
            bins[key]["in_disk"] += 1

    # 输出结果
    for label, data in bins.items():
        count = data["count"]
        if count == 0:
            print(f"{label:<20} | {0:<8} | {'N/A':<18} | N/A")
            continue

        ratio = data["in_disk"] / count
        factor = ratio / ISOTROPIC_REF

        # 集中倍率评价
        eval_str = ""
        if factor < 1.3:
            eval_str = "(球状分布)"
        elif factor > 2.0:
            eval_str = "(显著盘状!)"

        print(
            f"{label:<20} | {count:<8} | {ratio:.1%}             | {factor:.1f}x {eval_str}"
        )

    print("-" * 65)
    print(f"* 理论各向同性占比: {ISOTROPIC_REF:.1%}")
    print("* 解释: 远景恒星的集中倍率越高，说明银河带越明显。")


def draw_ascii_map(stars, width=80, height=24):
    """
    可视化：生成 ASCII 密度图
    """
    print("\n" + "=" * 60)
    print(f"{'【可视化】全天星图密度投影 (Mollweide-like)':^50}")
    print("=" * 60)

    # 初始化网格
    grid = [[0 for _ in range(width)] for _ in range(height)]
    max_count = 0

    # 筛选：只绘制较远的星星 (>500pc) 以凸显银河结构，
    # 否则近处的随机分布会掩盖银河
    render_stars = [s for s in stars if s.get("distance_pc", 0) > 500]
    print(f"正在绘制 {len(render_stars)} 颗远距离恒星 (>500pc)...")

    for star in render_stars:
        pos = star.get("pos_cartesian", [0, 0, 0])
        x, y, z = pos
        dist = star.get("distance_pc", 1)

        # 计算球面坐标 (RA, Dec) 但这里我们将其视为银经(l)和银纬(b)
        # 银纬 (b): -90 ~ 90
        lat = math.degrees(math.asin(z / dist)) if dist > 0 else 0
        # 银经 (l): 0 ~ 360
        lon = math.degrees(math.atan2(y, x))
        if lon < 0:
            lon += 360

        # 映射到网格坐标
        # Y轴: 0 (top/90deg) -> height-1 (bottom/-90deg)
        r = int((1 - (lat + 90) / 180) * (height - 1))
        # X轴: 0 (0deg) -> width-1 (360deg)
        c = int((lon / 360) * (width - 1))

        # 边界保护
        r = max(0, min(r, height - 1))
        c = max(0, min(c, width - 1))

        grid[r][c] += 1
        max_count = max(max_count, grid[r][c])

    # 字符映射表 (从稀疏到密集)
    chars = " .-:;+=*#%@"

    # 打印网格
    print("   " + "-" * width)
    for r in range(height):
        line = ""
        for c in range(width):
            count = grid[r][c]
            if count == 0:
                char = " "
            else:
                # 归一化强度
                intensity = count / max_count
                idx = int(intensity * (len(chars) - 1))
                char = chars[idx]
            line += char

        # 标注纬度
        lat_label = 90 - (r / (height - 1)) * 180
        print(f"{int(lat_label):3d}|{line}|")
    print("   " + "-" * width)
    print(f"{'Density Map: Blank=0, @=Max Density':^{width}}")


# =========================
# 执行
# =========================
if __name__ == "__main__":
    stars = load_stars()
    if stars:
        print(f"成功加载 {len(stars)} 颗恒星数据。")
        analyze_disk_concentration(stars)
        draw_ascii_map(stars)
