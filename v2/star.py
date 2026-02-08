import numpy as np
import json
import math

# =========================
# 基本参数
# =========================

TARGET_COUNT = 9000
DISK_SCALE = 300.0  # pc
MAX_DISTANCE = 10_000.0  # pc
AV_PER_KPC = 0.7  # mag / kpc

# 视星等限制
APP_MAG_MIN = -1.5
APP_MAG_MAX = 6.5

# 目标分布
TARGET_DISTRIBUTION = {
    "O": 0.001,
    "B": 0.05,
    "A": 0.10,
    "F": 0.15,
    "G": 0.20,
    "K": 0.30,
    "M": 0.199,
}

# =========================
# 物理核心：质量映射表 (主序星 Class V)
# =========================
TYPE_TO_MASS_RANGE = {
    "O": (16.0, 100.0),
    "B": (2.1, 16.0),
    "A": (1.4, 2.1),
    "F": (1.04, 1.4),
    "G": (0.8, 1.04),
    "K": (0.45, 0.8),
    "M": (0.08, 0.45),
}

# =========================
# 颜色映射
# =========================
COLOR_MAP = {
    "O": "#9bb0ff",
    "B": "#aabfff",
    "A": "#cad7ff",
    "F": "#f8f7ff",
    "G": "#fff4ea",
    "K": "#ffd2a1",
    "M": "#ffcc6f",
}

# =========================
# 工具函数
# =========================


def cartesian_to_spherical(x, y, z):
    dist = math.sqrt(x**2 + y**2 + z**2)
    if dist == 0:
        return 0.0, 0.0
    dec = math.degrees(math.asin(z / dist))
    ra = math.degrees(math.atan2(y, x))
    if ra < 0:
        ra += 360
    return ra, dec


def uniform_sphere_sample():
    theta = np.random.uniform(0, 2 * np.pi)
    cos_phi = np.random.uniform(-1, 1)
    sin_phi = np.sqrt(1 - cos_phi * cos_phi)
    return np.array([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi])


def sample_apparent_magnitude(size, m_min=APP_MAG_MIN, m_max=APP_MAG_MAX):
    """p(m) ∝ 10^(0.502m) 逆变换采样 - 匹配天文观测分布 log10(N) = 0.769 + 0.502*m"""
    u = np.random.rand(size)
    a = 10 ** (0.502 * m_min)
    b = 10 ** (0.502 * m_max)
    return (1 / 0.502) * np.log10(u * (b - a) + a)


# =========================
# 【新增】热光校正逻辑
# =========================
def get_bolometric_correction(teff):
    """
    根据有效温度 (Teff) 获取热光校正值 (BC)。
    BC = M_bol - M_V (通常为负值，表示目视星等比热光星等暗/数值更大)
    数据来源近似值：Flower (1996) / Allen's Astrophysical Quantities
    """
    # (温度 K, BC 值)
    bc_table = [
        (50000, -4.60),  # O3
        (40000, -3.90),  # O5
        (33000, -3.15),  # B0
        (22000, -2.25),  # B2
        (15000, -1.30),  # B5
        (10000, -0.40),  # A0
        (8500, -0.15),  # A5
        (7500, -0.05),  # F0
        (6500, 0.00),  # F5 (峰值接近 V 波段)
        (6000, -0.02),  # G0
        (5778, -0.07),  # Sun
        (5200, -0.20),  # K0
        (4500, -0.65),  # K5
        (3800, -1.40),  # M0
        (3000, -2.70),  # M5
        (2500, -4.30),  # M8
    ]

    # 提取列以便插值
    temps = [p[0] for p in bc_table]
    bcs = [p[1] for p in bc_table]

    # 线性插值
    return np.interp(teff, temps, bcs)


# =========================
# 严格校验函数 (更新版)
# =========================
def validate_star_strict(star: dict) -> tuple[bool, list[str]]:
    errors = []
    SUN_MV = 4.83
    SUN_TEFF = 5778

    TEFF_RANGE = {
        "O": (30000, 60000),
        "B": (10000, 30000),
        "A": (7500, 10000),
        "F": (6000, 7500),
        "G": (5200, 6000),
        "K": (3700, 5200),
        "M": (2000, 3700),
    }
    RADIUS_RANGE = {
        "V": (0.08, 25),
        "IV": (2, 6),
        "III": (5, 100),
        "II": (30, 300),
        "I": (100, 1500),
    }

    stype = star.get("spectral_type")
    lclass = star.get("luminosity_class")
    m = star.get("app_mag")  # 目视视星等
    M_v = star.get("abs_mag")  # 目视绝对星等
    M_bol = star.get("bolometric_mag")  # 热光绝对星等 (物理核心)
    d_pc = star.get("distance_pc")
    theta = star.get("angular_diameter_mas")
    Av = star.get("extinction_Av", 0.0)

    # 0. 基础检查
    if not lclass:
        return False, ["缺失 Luminosity Class"]

    # 1. 距离模数 (使用目视星等 M_v)
    m_calc = M_v + 5 * math.log10(d_pc) - 5 + Av
    if abs(m - m_calc) > 0.5:
        errors.append(f"距离模数不一致: m={m:.2f}, calc={m_calc:.2f}")

    # 2. 角直径 -> 半径 (物理检查)
    # R_sol = (theta_mas * d_pc) / 9.305
    R_est = (theta * d_pc) / 9.305
    r_min, r_max = RADIUS_RANGE.get(lclass, (0, 0))
    if not (r_min <= R_est <= r_max * 1.5):
        errors.append(f"半径不合理 R={R_est:.2f} (范围 {r_min}-{r_max})")

    # 3. 绝对星等 -> 光度 (使用热光星等 M_bol)
    # 只有使用 M_bol 才能还原真实的物理光度 L
    # 如果字典里没有 M_bol (旧数据)，回退到 abs_mag 并警告
    if M_bol is None:
        M_bol = M_v

    L_est = 10 ** ((SUN_MV - M_bol) / 2.5)  # 注意：这里假设 SUN_MV 约为 4.83 作为参考点

    # 4. 光度 + 半径 -> 温度 (L = R^2 T^4)
    if R_est > 0:
        T_est = SUN_TEFF * (L_est / (R_est**2)) ** 0.25
        t_min, t_max = TEFF_RANGE.get(stype, (0, 0))
        if not (t_min * 0.9 <= T_est <= t_max * 1.1):
            errors.append(f"温度不匹配 T_calc={T_est:.0f} (类型 {stype})")

    return len(errors) == 0, errors


# =========================
# 核心生成逻辑 (修复版)
# =========================


def generate_main_sequence_star(stype, app_mag):
    """
    基于物理链生成恒星：
    Type -> Mass -> (L, R) -> (T, M_bol) -> (BC, M_v) -> Distance
    """
    # 1. 采样质量
    m_min, m_max = TYPE_TO_MASS_RANGE[stype]
    mass = np.random.uniform(m_min, m_max)

    # 2. 物理关系 (主序星近似)
    # 光度 L (Solar)
    if mass < 0.43:
        L = 0.23 * (mass**2.3)
    elif mass < 2:
        L = mass**4
    elif mass < 20:
        L = 1.4 * (mass**3.5)
    else:
        L = 32000 * mass

    # 半径 R (Solar)
    if mass < 1:
        R = mass**0.8
    else:
        R = mass**0.57

    # 3. 反推温度
    T = 5778 * (L / (R**2)) ** 0.25

    # 温度钳位
    t_range = {
        "O": (30000, 50000),
        "B": (10000, 30000),
        "A": (7500, 10000),
        "F": (6000, 7500),
        "G": (5200, 6000),
        "K": (3700, 5200),
        "M": (2400, 3700),
    }
    t_min, t_max = t_range[stype]
    T = np.clip(T, t_min, t_max)

    # 微调 R 以匹配 Stefan-Boltzmann
    R = math.sqrt(L) / ((T / 5778) ** 2)

    # 4. 计算绝对星等 (关键修复步骤)
    # M_bol: 热光绝对星等 (总能量)
    M_bol = 4.83 - 2.5 * math.log10(L)

    # BC: 热光校正
    bc = get_bolometric_correction(T)

    # M_v: 目视绝对星等 (人眼亮度) = M_bol - BC
    # 注意：BC通常是负值，例如 O型星 BC=-4，M_v = -10 - (-4) = -6 (变暗)
    M_v = M_bol - bc

    # 5. 根据目标【目视视星等】推算距离
    # 使用 M_v 而不是 M_bol 进行距离计算
    dist_mod = app_mag - M_v
    dist_pc = 10 ** ((dist_mod + 5) / 5)

    # 加上消光修正
    av = AV_PER_KPC * (dist_pc / 1000.0)
    dist_mod_corr = app_mag - M_v - av
    dist_pc = 10 ** ((dist_mod_corr + 5) / 5)

    # 距离截断
    if dist_pc > MAX_DISTANCE:
        dist_pc = MAX_DISTANCE
    if dist_pc < 1.0:
        dist_pc = 1.0

    # 重新计算最终的 App Mag (保证数据一致性)
    final_av = AV_PER_KPC * (dist_pc / 1000.0)
    final_app_mag = M_v + 5 * math.log10(dist_pc / 10) + final_av

    # 【关键修复】确保最终视星等不超过上限
    # 如果超过，需要调整距离使其符合限制
    if final_app_mag > APP_MAG_MAX:
        # 反推符合 APP_MAG_MAX 的最大距离
        # APP_MAG_MAX = M_v + 5*log10(d/10) + Av
        # 简化：忽略消光的迭代影响，直接用当前 Av 估算
        dist_mod_max = APP_MAG_MAX - M_v - final_av
        dist_pc_max = 10 ** ((dist_mod_max + 5) / 5)
        dist_pc = min(dist_pc, dist_pc_max)

        # 重新计算消光和视星等
        final_av = AV_PER_KPC * (dist_pc / 1000.0)
        final_app_mag = M_v + 5 * math.log10(dist_pc / 10) + final_av

        # 最终保险：强制截断到上限
        final_app_mag = min(final_app_mag, APP_MAG_MAX)

    # 6. 坐标
    pos = uniform_sphere_sample() * dist_pc
    ra, dec = cartesian_to_spherical(pos[0], pos[1], pos[2])

    # 7. 角直径
    theta = 9.305 * R / dist_pc

    return {
        "id": "",
        "spectral_type": stype,
        "luminosity_class": "V",
        "mass_solar": round(float(mass), 4),
        "temperature_K": round(float(T), 1),
        "luminosity_solar": round(float(L), 4),
        "radius_solar": round(float(R), 4),
        "distance_pc": round(float(dist_pc), 2),
        "dist_ly": round(float(dist_pc * 3.2616), 2),
        "extinction_Av": round(float(final_av), 3),
        # 星等数据区分
        "abs_mag": round(float(M_v), 2),  # 目视绝对星等 (用于显示)
        "bolometric_mag": round(float(M_bol), 2),  # 热光绝对星等 (物理真实)
        "bc_correction": round(float(bc), 2),  # 校正值
        "app_mag": round(float(final_app_mag), 3),  # 目视视星等
        "color_hex": COLOR_MAP.get(stype, "#ffffff"),
        "ra": round(float(ra), 4),
        "dec": round(float(dec), 4),
        "pos_cartesian": [round(float(x), 2) for x in pos],
        "angular_diameter_mas": round(float(theta), 5),
    }


# =========================
# 主程序 (修复死锁 Bug 版)
# =========================

stars = []
validated_count = 0
rejected_count = 0

# 1. 直接按比例生成，不再预先生成 total_mags 数组
for stype, percentage in TARGET_DISTRIBUTION.items():
    count = int(TARGET_COUNT * percentage)

    generated_for_type = 0
    # 动态生成，直到该光谱型的星星数量达标
    while generated_for_type < count:
        # 【关键修复】实时生成一个视星等目标
        m_target = sample_apparent_magnitude(1, APP_MAG_MIN, APP_MAG_MAX)[0]

        # 尝试基于物理规律生成
        star = generate_main_sequence_star(stype, m_target)

        # 校验
        is_valid, msgs = validate_star_strict(star)

        if is_valid:
            star["id"] = f"Star_{len(stars) + 1}"
            stars.append(star)
            validated_count += 1
            generated_for_type += 1  # 成功生成一颗，计数器+1
        else:
            rejected_count += 1
            # 失败后会自动进入下一次 while 循环
            # 重新抽取全新的 m_target 和恒星参数，永远不会卡死！

# 保存代码 (与之前相同)
output = {
    "metadata": {
        "count": len(stars),
        "validated_rate": f"{validated_count / (validated_count + rejected_count + 1) * 100:.1f}%",
        "note": "Visual magnitudes corrected, deadlock bug fixed",
    },
    "stars": stars,
}

with open("star_map_corrected.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"Done. Generated {len(stars)} stars. (Rejections: {rejected_count})")
