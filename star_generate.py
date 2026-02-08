import numpy as np
import json
from collections import Counter
import random

# =========================
# 基本参数
# =========================

TARGET_COUNT = 9000

# IMF 参数
M_MIN = 0.08
M_MAX = 20.0

# 银河盘参数
DISK_SCALE = 300.0       # pc
MAX_DISTANCE = 10_000.0 # pc
AV_PER_KPC = 0.7         # mag / kpc

# =========================
# 目标光谱类型分布
# =========================
# 基于肉眼可见恒星的光谱类型分布（总和 = 100%）
TARGET_DISTRIBUTION = {
    "O": 0.01,    # 1%  - 极其罕见的蓝超巨星
    "B": 0.10,    # 10% - 蓝白色亮星，肉眼星空中最重要的成员之一
    "A": 0.22,    # 22% - 白色恒星，距离较近的高光度主序星
    "F": 0.19,    # 19% - 黄白色恒星，比太阳略大略亮
    "G": 0.14,    # 14% - 黄色恒星，类似太阳或演化中的巨星
    "K": 0.31,    # 31% - 橙色巨星，数量最多（很多演化到了巨星阶段）
    "M": 0.03,    # 3%  - 红色巨星，只有巨星能被看见
}

# =========================
# 24 个邻居恒星数据
# =========================
FIXED_NEIGHBORS = [
    {"id": "Sys_1",  "dist": 4.79,  "type": "M", "pos": [0.03, -4.78, -0.34],  "app_mag": 7.83,  "ang_size": 0.7829, },
    {"id": "Sys_2",  "dist": 11.57, "type": "M", "pos": [-6.23, -5.99, -7.7],  "app_mag": 9.75,  "ang_size": 0.3241, },
    {"id": "Sys_3",  "dist": 10.24, "type": "K", "pos": [-0.83, 10.09, 1.51],  "app_mag": 4.48,  "ang_size": 1.0986, },
    {"id": "Sys_4",  "dist": 10.79, "type": "M", "pos": [-3.12, 7.93, 6.62],   "app_mag": 9.6,   "ang_size": 0.3475, },
    {"id": "Sys_5",  "dist": 10.75, "type": "M", "pos": [4.34, 9.82, 0.53],    "app_mag": 9.59,  "ang_size": 0.3488, },
    {"id": "Sys_6",  "dist": 5.64,  "type": "G", "pos": [-5.12, 0.79, 2.22],   "app_mag": 0.99,  "ang_size": 2.6596, },
    {"id": "Sys_7",  "dist": 8.65,  "type": "M", "pos": [-3.6, -0.32, -7.86],  "app_mag": 9.12,  "ang_size": 0.4335, },
    {"id": "Sys_8",  "dist": 7.09,  "type": "F", "pos": [-1.36, 0.2, 6.95],    "app_mag": -0.31, "ang_size": 2.7504, },
    {"id": "Sys_9",  "dist": 9.91,  "type": "M", "pos": [2.58, 9.56, 0.36],    "app_mag": 9.41,  "ang_size": 0.3784, },
    {"id": "Sys_10", "dist": 10.73, "type": "M", "pos": [3.25, 7.71, -6.72],   "app_mag": 9.59,  "ang_size": 0.3495, },
    {"id": "Sys_11", "dist": 10.61, "type": "M", "pos": [-6.66, 4.21, 7.1],    "app_mag": 9.56,  "ang_size": 0.3534, },
    {"id": "Sys_12", "dist": 7.97,  "type": "K", "pos": [-4.93, -5.26, -3.39], "app_mag": 3.94,  "ang_size": 1.4115, },
    {"id": "Sys_13", "dist": 11.68, "type": "G", "pos": [10.67, 4.28, -2.09],  "app_mag": 2.57,  "ang_size": 1.2842, },
    {"id": "Sys_14", "dist": 5.53,  "type": "M", "pos": [-0.35, -2.12, -5.1],  "app_mag": 8.15,  "ang_size": 0.6781, },
    {"id": "Sys_15", "dist": 5.53,  "type": "K", "pos": [-0.65, -2.47, 4.91],  "app_mag": 3.15,  "ang_size": 2.0344, },
    {"id": "Sys_16", "dist": 11.73, "type": "M", "pos": [8.97, -4.35, -6.18],  "app_mag": 9.78,  "ang_size": 0.3197, },
    {"id": "Sys_17", "dist": 4.77,  "type": "M", "pos": [-2.09, 4.28, 0.35],   "app_mag": 7.83,  "ang_size": 0.7862, },
    {"id": "Sys_18", "dist": 11.33, "type": "M", "pos": [6.96, -0.86, -8.9],   "app_mag": 9.7,   "ang_size": 0.331,  },
    {"id": "Sys_19", "dist": 9.82,  "type": "M", "pos": [4.42, 6.91, -5.39],   "app_mag": 9.39,  "ang_size": 0.3819, },
    {"id": "Sys_20", "dist": 9.56,  "type": "M", "pos": [4.45, 7.92, -2.98],   "app_mag": 9.34,  "ang_size": 0.3923, },
    {"id": "Sys_21", "dist": 10.65, "type": "M", "pos": [3.06, 5.59, -8.53],   "app_mag": 9.57,  "ang_size": 0.3521, },
    {"id": "Sys_22", "dist": 11.87, "type": "M", "pos": [-3.23, -10.5, 4.49],  "app_mag": 9.81,  "ang_size": 0.3159, },
    {"id": "Sys_23", "dist": 10.99, "type": "M", "pos": [0.29, -2.82, -10.62], "app_mag": 9.64,  "ang_size": 0.3412, },
    {"id": "Sys_24", "dist": 11.14, "type": "M", "pos": [-10.4, 0.59, -3.94],  "app_mag": 9.67,  "ang_size": 0.3366, },
]

# =========================
# 颜色映射 (光谱类型 -> RGB Hex)
# =========================
COLOR_MAP = {
    "O": "#9bb0ff",  # 蓝
    "B": "#aabfff",  # 蓝白
    "A": "#cad7ff",  # 白
    "F": "#f8f7ff",  # 黄白
    "G": "#fff4ea",  # 黄 (太阳)
    "K": "#ffd2a1",  # 橙
    "M": "#ffcc6f",  # 红
}

# =========================
# 坐标工具函数
# =========================

def cartesian_to_spherical(x, y, z):
    """将笛卡尔坐标转换为球面坐标 (RA, Dec)"""
    import math
    dist = math.sqrt(x**2 + y**2 + z**2)
    if dist == 0:
        return 0.0, 0.0
    # Dec (Declination): -90 to 90
    dec = math.degrees(math.asin(z / dist))
    # RA (Right Ascension): 0 to 360
    ra = math.degrees(math.atan2(y, x))
    if ra < 0:
        ra += 360
    return ra, dec


def uniform_sphere_sample():
    """生成均匀分布的球面方向向量"""
    theta = np.random.uniform(0, 2 * np.pi)  # 方位角
    cos_phi = np.random.uniform(-1, 1)        # cos(天顶角)
    sin_phi = np.sqrt(1 - cos_phi * cos_phi)
    
    x = sin_phi * np.cos(theta)
    y = sin_phi * np.sin(theta)
    z = cos_phi
    
    return np.array([x, y, z])


def calculate_angular_diameter_mas(radius_solar, dist_ly):
    """计算角直径 (毫角秒)"""
    # 太阳在1光年处的视直径约为 15 mas
    if dist_ly <= 0:
        return 0.0
    return (radius_solar / dist_ly) * 15.0


def estimate_stellar_radius(mass, giant=False):
    """估算恒星半径（太阳半径）"""
    if giant:
        return mass ** 0.8 * 10.0  # 巨星半径更大
    return mass ** 0.8


# =========================
# 工具函数
# =========================

def sample_apparent_magnitude(size, m_min=-1.5, m_max=6.5, b=0.51):
    """
    根据恒星计数规律 log10(N(m)) = a + b*m 采样视星等
    使用逆变换采样方法 (Inverse Transform Sampling)
    
    核心公式：
    m = log10(r * (10^(b*m_max) - 10^(b*m_min)) + 10^(b*m_min)) / b
    
    参数:
        size: 需要生成的星等数量
        m_min: 最亮极限（默认 -1.5，天狼星级别）
        m_max: 最暗极限（默认 6.5，肉眼可见极限）
        b: 分布斜率（默认 0.51，根据实际数据拟合的最优值）
    
    返回:
        符合恒星计数规律分布的视星等数组
    """
    r = np.random.rand(size)
    
    # 计算 10^(b*m_min) 和 10^(b*m_max)
    term_min = 10 ** (b * m_min)
    term_max = 10 ** (b * m_max)
    
    # 逆变换采样
    m = np.log10(r * (term_max - term_min) + term_min) / b
    
    return m

def sample_kroupa_mass(size):
    u = np.random.rand(size)
    masses = np.zeros(size)

    p_break = (0.5**(-0.3) - M_MIN**(-0.3)) / (
        (0.5**(-0.3) - M_MIN**(-0.3)) +
        (M_MAX**(-1.3) - 0.5**(-1.3))
    )

    mask_low = u < p_break
    mask_high = ~mask_low

    r = np.random.rand(mask_low.sum())
    masses[mask_low] = (
        (r * (0.5**(-0.3) - M_MIN**(-0.3)) + M_MIN**(-0.3))
    ) ** (-1 / 0.3)

    r = np.random.rand(mask_high.sum())
    masses[mask_high] = (
        (r * (M_MAX**(-1.3) - 0.5**(-1.3)) + 0.5**(-1.3))
    ) ** (-1 / 1.3)

    return masses


def main_sequence_lifetime(mass):
    return 10.0 * mass ** (-2.5)


def stellar_properties(mass, giant=False):
    if not giant:
        L = mass ** 3.5
        R = mass ** 0.8
    else:
        if mass < 2.2:
            L = mass ** 3.5 * np.random.uniform(100, 1000)
        else:
            L = mass ** 3.5 * np.random.uniform(1e4, 1e5)
        R = np.sqrt(L) * 5

    T = (L / (R ** 2)) ** 0.25 * 5778
    return L, R, T

def visibility_weight(M_abs, m_lim=6.0, d_ref=2000.0):
    d_max = 10 ** ((m_lim - M_abs + 5) / 5)
    w = (d_max / d_ref) ** 3
    return min(1.0, w)



def spectral_type(T):
    if T > 30000:
        return "O"
    elif T > 10000:
        return "B"
    elif T > 7500:
        return "A"
    elif T > 6000:
        return "F"
    elif T > 5200:
        return "G"
    elif T > 3700:
        return "K"
    else:
        return "M"


def sample_distance(size):
    u = np.random.rand(size)
    d = -DISK_SCALE * np.log(1 - u)
    return np.clip(d, 1.0, MAX_DISTANCE)


def absolute_magnitude(L):
    return 4.83 - 2.5 * np.log10(L)


# =========================
# 主生成循环
# =========================

# 先生成足够多的星星，按类型分组
stars_by_type = {t: [] for t in ["O", "B", "A", "F", "G", "K", "M"]}

# 计算每种类型需要的最小数量
MIN_PER_TYPE = {t: int(TARGET_COUNT * p * 1.5) + 100 for t, p in TARGET_DISTRIBUTION.items()}

def generate_synthetic_star(stype, distance_range=(10, 500)):
    """为稀有类型生成合成星星"""
    if stype == "O":
        T = np.random.uniform(30000, 50000)
        m = np.random.uniform(16, 50)
    elif stype == "B":
        T = np.random.uniform(10000, 30000)
        m = np.random.uniform(2.1, 16)
    else:
        raise ValueError(f"Synthetic generation not supported for type {stype}")
    
    L = m ** 3.5
    R = m ** 0.8
    d = np.random.uniform(*distance_range)
    av = AV_PER_KPC * (d / 1000.0)
    M_abs = 4.83 - 2.5 * np.log10(L)
    m_app = M_abs + 5 * np.log10(d / 10.0) + av
    
    # 生成均匀球面位置
    direction = uniform_sphere_sample()
    pos_cartesian = direction * d
    ra, dec = cartesian_to_spherical(pos_cartesian[0], pos_cartesian[1], pos_cartesian[2])
    
    # 计算距离和角直径
    dist_ly = d * 3.2616
    radius = estimate_stellar_radius(m, False)
    ang_diam = calculate_angular_diameter_mas(radius, dist_ly)
    
    return {
        "mass_solar": round(float(m), 4),
        "giant": False,
        "temperature_K": round(float(T), 1),
        "luminosity_solar": round(float(L), 4),
        "distance_pc": round(float(d), 2),
        "dist_ly": round(float(dist_ly), 2),
        "extinction_Av": round(float(av), 3),
        "abs_mag": round(float(M_abs), 2),
        "app_mag": round(float(m_app), 3),
        "spectral_type": stype,
        "color_hex": COLOR_MAP.get(stype, "#ffffff"),
        "ra": round(float(ra), 4),
        "dec": round(float(dec), 4),
        "pos_cartesian": [round(float(v), 2) for v in pos_cartesian],
        "angular_diameter_mas": round(float(ang_diam), 4),
    }

print("Generating stars with p(m) ∝ 10^(0.5m) distribution...")

# 直接生成目标数量的星星
generated_stars = []
batch_size = 50000

# 光谱类型的典型绝对星等范围
SPECTRAL_ABS_MAG = {
    "O": (-6.5, -4.0),
    "B": (-3.5, 0.5),
    "A": (0.5, 2.5),
    "F": (2.5, 4.0),
    "G": (4.0, 5.5),
    "K": (5.5, 8.0),
    "M": (8.0, 12.0),
}

# 光谱类型的典型温度范围
SPECTRAL_TEMP = {
    "O": (30000, 50000),
    "B": (10000, 30000),
    "A": (7500, 10000),
    "F": (6000, 7500),
    "G": (5200, 6000),
    "K": (3700, 5200),
    "M": (2400, 3700),
}

while len(generated_stars) < TARGET_COUNT * 1.2:  # 多生成一些以便筛选
    # 1. 采样视星等 (使用恒星计数规律 log10(N) = a + 0.51*m)
    app_mags = sample_apparent_magnitude(batch_size, m_min=-2.0, m_max=6.5)
    
    # 2. 按目标分布采样光谱类型
    type_probs = list(TARGET_DISTRIBUTION.values())
    type_names = list(TARGET_DISTRIBUTION.keys())
    spectral_types = np.random.choice(type_names, size=batch_size, p=type_probs)
    
    for app_mag, stype in zip(app_mags, spectral_types):
        # 3. 根据光谱类型采样绝对星等
        abs_mag_range = SPECTRAL_ABS_MAG[stype]
        abs_mag = np.random.uniform(*abs_mag_range)
        
        # 4. 根据视星等和绝对星等计算距离
        # m - M = 5 * log10(d/10) + Av
        # 先忽略消光，粗略估计距离
        distance_modulus = app_mag - abs_mag
        d_raw = 10 ** (distance_modulus / 5 + 1)  # 距离（秒差距）
        
        # 加入消光修正（迭代一次）
        av = AV_PER_KPC * (d_raw / 1000.0)
        distance_modulus_corrected = app_mag - abs_mag - av
        d = 10 ** (distance_modulus_corrected / 5 + 1)
        
        # 确保距离合理
        if d < 1 or d > 50000:
            continue
        
        # 重新计算消光
        av = AV_PER_KPC * (d / 1000.0)
        
        # 5. 生成位置
        direction = uniform_sphere_sample()
        pos_cartesian = direction * d
        ra, dec = cartesian_to_spherical(pos_cartesian[0], pos_cartesian[1], pos_cartesian[2])
        
        # 6. 采样温度
        temp_range = SPECTRAL_TEMP[stype]
        T = np.random.uniform(*temp_range)
        
        # 7. 计算其他物理参数
        L = 10 ** ((4.83 - abs_mag) / 2.5)  # 从绝对星等反推光度
        
        # 估算质量和半径
        if L > 1:
            mass = L ** (1/3.5)  # 主序星质量-光度关系
        else:
            mass = L ** (1/2.3)
        
        radius = mass ** 0.8
        dist_ly = d * 3.2616
        ang_diam = calculate_angular_diameter_mas(radius, dist_ly)
        
        star = {
            "mass_solar": round(float(mass), 4),
            "giant": False,
            "temperature_K": round(float(T), 1),
            "luminosity_solar": round(float(L), 4),
            "distance_pc": round(float(d), 2),
            "dist_ly": round(float(dist_ly), 2),
            "extinction_Av": round(float(av), 3),
            "abs_mag": round(float(abs_mag), 2),
            "app_mag": round(float(app_mag), 3),
            "spectral_type": stype,
            "color_hex": COLOR_MAP.get(stype, "#ffffff"),
            "ra": round(float(ra), 4),
            "dec": round(float(dec), 4),
            "pos_cartesian": [round(float(v), 2) for v in pos_cartesian],
            "angular_diameter_mas": round(float(ang_diam), 4),
        }
        generated_stars.append(star)
    
    print(f"  Generated {len(generated_stars)} stars so far...")

# 按类型分组
stars_by_type = {t: [] for t in ["O", "B", "A", "F", "G", "K", "M"]}
for star in generated_stars:
    stars_by_type[star["spectral_type"]].append(star)

print("Resampling to match target distribution...")

# 根据目标分布重采样
# 使用 round() 避免取整误差，并确保总数精确
target_counts = {}
total_allocated = 0
for stype in ["O", "B", "A", "F", "G", "K", "M"]:
    target_counts[stype] = round(TARGET_COUNT * TARGET_DISTRIBUTION[stype])
    total_allocated += target_counts[stype]

# 调整以确保总数准确（差额加到 A 型）
diff = TARGET_COUNT - total_allocated
target_counts["A"] += diff

stars = []
for stype in ["O", "B", "A", "F", "G", "K", "M"]:
    target_count = target_counts[stype]
    available = stars_by_type[stype]
    
    if len(available) >= target_count:
        # 随机采样
        selected = random.sample(available, target_count)
    else:
        # 如果不够，使用所有可用的
        selected = available
        print(f"  Warning: {stype} only has {len(available)}, need {target_count}")
    
    stars.extend(selected)

# 如果总数不够，从剩余池中补充
if len(stars) < TARGET_COUNT:
    remaining = TARGET_COUNT - len(stars)
    print(f"  Need to add {remaining} more stars...")
    # 按目标比例从剩余星星中补充
    for stype in ["A", "F", "G", "K", "B"]:
        if remaining <= 0:
            break
        used_ids = {id(s) for s in stars}
        available_extra = [s for s in stars_by_type[stype] if id(s) not in used_ids]
        if len(available_extra) > 0:
            to_add = min(remaining, len(available_extra))
            stars.extend(random.sample(available_extra, to_add))
            remaining = TARGET_COUNT - len(stars)

# 打乱顺序
random.shuffle(stars)

# 给星星添加 ID
for i, star in enumerate(stars, 1):
    star["id"] = f"Bg_{i}"

print(f"Final star count: {len(stars)}")

# =========================
# 将邻居恒星作为预生成背景星加入 stars
# =========================

for n in FIXED_NEIGHBORS:
    d = n["dist"]
    
    # 直接使用表格中的视星等（如果有）
    app_mag = n.get("app_mag", 10.0)
    
    # 计算坐标
    pos = n["pos"]
    ra, dec = cartesian_to_spherical(pos[0], pos[1], pos[2])
    
    # 距离转换
    dist_ly = d * 3.2616
    
    # 直接使用表格中的角直径（如果有）
    ang_diam = n.get("ang_size", 0.5)
    
    stars.append({
        "id": n["id"],
        "spectral_type": n["type"],
        "distance_pc": d,
        "dist_ly": round(float(dist_ly), 2),
        "app_mag": round(float(app_mag), 3),
        "color_hex": COLOR_MAP.get(n["type"], "#ffffff"),
        "ra": round(float(ra), 4),
        "dec": round(float(dec), 4),
        "pos_cartesian": pos,
        "angular_diameter_mas": round(float(ang_diam), 4),
    })

print(f"Added {len(FIXED_NEIGHBORS)} fixed neighbor stars to background stars.")

# =========================
# 保存为 JSON
# =========================

output = {
    "metadata": {
        "planet": "Terrax",
        "visible_magnitude_limit": 7.0,
        "star_count": len(stars),
        "notes": "Physically motivated stellar population with observational selection"
    },
    "stars": stars
}

with open("star_map.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"Saved to star_map.json ({len(stars)} stars total)")

# =========================
# 类型统计（校验用）
# =========================

types = Counter(s["spectral_type"] for s in stars)
total = len(stars)

print("\nSpectral type distribution:")
for t in ["O", "B", "A", "F", "G", "K", "M"]:
    print(f"{t}: {types[t] / total * 100:.2f}%")
