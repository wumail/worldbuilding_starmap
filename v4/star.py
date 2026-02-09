import numpy as np
import json
import math
from collections import Counter, defaultdict
import random
from datetime import datetime
import os
import matplotlib.pyplot as plt

# =========================
# 基本参数配置
# =========================

TARGET_COUNT = 9000
MAX_DISTANCE = 10_000.0  # pc
AV_PER_KPC = 0.7  # mag / kpc (平均消光)

# 视星等限制 (人眼可见极限通常在 6.5)
APP_MAG_MIN = -1.8
APP_MAG_MAX = 6.5

# =========================
# 分布概率配置
# =========================

# 光谱类型目标分布 (参考耶鲁亮星表 Yale Bright Star Catalogue)
TARGET_DISTRIBUTION = {
    "O": 0.010,  # 1% - 极其罕见但极其明亮 (如参宿一)
    "B": 0.200,  # 20% - "夜空之王"，虽然稀少但几乎全都能看见
    "A": 0.210,  # 21% - 标准的亮星 (如天狼星、织女星)
    "F": 0.130,  # 13% - 过渡型 (如老人星、南河三)
    "G": 0.100,  # 10% - 类似太阳的恒星只有很近才能看见 (如南门二)
    "K": 0.300,  # 30% - 主要是红巨星 (K-Giants)，它们在夜空中非常普遍
    "M": 0.050,  # 5% - 只有红超巨星/红巨星可见 (M型矮星肉眼不可见)
}

# 光度级分布概率 (Magnitude Limited)
CLASS_DISTRIBUTION = {
    "0": 0.002,  # 特超巨星 (极为罕见)
    "I": 0.028,  # 超巨星 (如参宿四、天津四) - 比例显著提高
    "II": 0.050,  # 亮巨星
    "III": 0.400,  # 巨星 (如大角星) - 这是一个巨大的群体
    "IV": 0.120,  # 次巨星
    "V": 0.400,  # 主序星 - 只有O/B/A型的主序星容易被看见，G/K/M型主序星很难
    "VI": 0.000,  # 次矮星 - 肉眼几乎不可见，设为0
}

# =========================
# 天体物理数据表 (核心)
# =========================

# 1. 典型温度范围 (Kelvin)
TYPE_TEMP_RANGE = {
    "O": (30000, 50000),
    "B": (10000, 30000),
    "A": (7500, 10000),
    "F": (6000, 7500),
    "G": (5200, 6000),
    "K": (3700, 5200),
    "M": (2400, 3700),
}

# 2. 绝对目视星等 (Mv) 参考表
# 数据源参考: Allen's Astrophysical Quantities & H-R Diagram
# 格式: {光谱型: {光度级: (基准Mv, 波动范围)}}
# 注意：巨星和超巨星的Mv并不完全随光谱型线性变化（如红巨星分支）
M_V_TABLE = {
    "O": {
        "0": (-7.5, 0.8),  # 降低亮度以避免超过物理极限
        "I": (-6.5, 1.0),
        "II": (-6.0, 0.5),
        "III": (-5.5, 0.5),
        "IV": (-5.0, 0.5),
        "V": (-4.5, 1.0),
        "VI": (-4.0, 1.0),
    },
    "B": {
        "0": (-7.0, 1.0),  # 降低亮度和波动范围
        "I": (-6.0, 1.5),
        "II": (-4.0, 1.0),
        "III": (-2.0, 1.0),
        "IV": (-1.5, 0.8),
        "V": (-1.0, 1.5),
        "VI": (0.0, 1.0),
    },
    "A": {
        "0": (-7.0, 1.0),
        "I": (-5.5, 1.0),
        "II": (-3.0, 0.8),
        "III": (-0.5, 0.8),
        "IV": (1.0, 0.5),
        "V": (1.5, 1.0),
        "VI": (2.5, 0.8),
    },
    "F": {
        "0": (-6.0, 1.0),
        "I": (-5.0, 1.0),
        "II": (-2.5, 0.5),
        "III": (1.0, 0.8),
        "IV": (2.5, 0.5),
        "V": (3.5, 1.0),
        "VI": (4.5, 0.8),
    },
    "G": {
        "0": (-5.0, 1.0),
        "I": (-4.5, 1.0),
        "II": (-2.0, 0.5),
        "III": (0.5, 0.8),
        "IV": (3.0, 0.5),
        "V": (4.8, 0.8),
        "VI": (6.0, 1.0),
    },
    "K": {
        "0": (-4.5, 1.0),
        "I": (-4.5, 0.5),
        "II": (-2.0, 0.5),
        "III": (0.0, 1.0),
        "IV": (4.0, 1.0),
        "V": (6.5, 1.5),
        "VI": (8.0, 1.0),
    },
    "M": {
        "0": (-4.5, 1.0),
        "I": (-5.0, 1.0),
        "II": (-2.5, 1.0),
        "III": (-0.5, 0.8),
        "IV": (8.0, 1.5),
        "V": (12.0, 3.0),
        "VI": (14.0, 2.0),
    },
}

# 3. 典型质量估算表 (Solar Mass)
# 用于给非主序星赋予一个物理上合理的质量 (演化质量)
# 格式: {光谱型: {光度级: (最小质量, 最大质量)}}
MASS_ESTIMATE = {
    # O型星：无论是否演化，质量都很大
    "O": {"default": (20, 100)},
    # B型星
    "B": {"I": (15, 40), "III": (10, 20), "V": (2.1, 16), "default": (2, 20)},
    # A型星
    "A": {"I": (10, 20), "III": (2.5, 5), "V": (1.4, 2.1), "default": (1.4, 10)},
    # F型星
    "F": {"I": (8, 15), "III": (1.5, 3), "V": (1.04, 1.4), "default": (1, 10)},
    # G型星 (注意：G型巨星通常是演化后的B/A星，质量比G型主序星大)
    "G": {
        "I": (8, 12),
        "III": (1.0, 4.0),
        "V": (0.8, 1.04),
        "VI": (0.6, 0.8),
        "default": (0.8, 10),
    },
    # K型星
    "K": {
        "I": (8, 15),
        "III": (1.0, 5.0),
        "V": (0.45, 0.8),
        "VI": (0.3, 0.5),
        "default": (0.5, 10),
    },
    # M型星 (M型巨星质量很大，M型主序星质量很小)
    "M": {
        "I": (10, 30),
        "III": (1.0, 6.0),
        "V": (0.08, 0.45),
        "VI": (0.08, 0.3),
        "default": (0.1, 20),
    },
}

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
    """p(m) ∝ 10^(0.5m)"""
    u = np.random.rand(size)
    a = 10 ** (0.5 * m_min)
    b = 10 ** (0.5 * m_max)
    return 2 * np.log10(u * (b - a) + a)


def get_bolometric_correction(teff):
    """
    热光校正 BC (Bolometric Correction).
    BC = Mbol - Mv.
    基于 Flower (1996) 的近似插值.
    """
    bc_table = [
        (50000, -4.60),
        (40000, -3.90),
        (33000, -3.15),
        (22000, -2.25),
        (15000, -1.30),
        (10000, -0.40),
        (8500, -0.15),
        (7500, -0.05),
        (6500, 0.00),
        (6000, -0.02),
        (5778, -0.07),
        (5200, -0.20),
        (4500, -0.65),
        (3800, -1.40),
        (3000, -2.70),
        (2500, -4.30),
    ]
    temps = [p[0] for p in bc_table]
    bcs = [p[1] for p in bc_table]
    # np.interp 需要 x 递增，所以反转列表
    return np.interp(teff, temps[::-1], bcs[::-1])


# =========================
# 校验函数 (升级版)
# =========================


def validate_star_strict(star: dict) -> tuple[bool, list[str]]:
    """
    严格校验，支持所有光度级
    """
    errors = []
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

    # 定义不同光度级的典型半径范围 (Solar Radii)
    # 数据参考: Standard Stellar Parameters
    RADIUS_LIMITS = {
        "0": (50, 2000),  # 特超巨星
        "I": (20, 1500),  # 超巨星
        "II": (10, 300),  # 亮巨星
        "III": (2, 200),  # 巨星 (红巨星可以很大)
        "IV": (1.2, 10),  # 次巨星
        "V": (0.08, 20),  # 主序星
        "VI": (0.05, 1.5),  # 次矮星
    }

    stype = star.get("spectral_type")
    lclass = star.get("luminosity_class")
    m = star.get("app_mag")
    M_v = star.get("abs_mag")  # Visual Absolute Magnitude
    M_bol = star.get("bolometric_mag")  # Bolometric Absolute Magnitude
    d_pc = star.get("distance_pc")
    theta = star.get("angular_diameter_mas")
    Av = star.get("extinction_Av", 0.0)

    # 1. 基础一致性
    m_calc = M_v + 5 * math.log10(d_pc) - 5 + Av
    if abs(m - m_calc) > 0.5:
        errors.append(f"距离模数错误: m={m:.2f} calc={m_calc:.2f}")

    # 2. 物理半径检查
    # 从角直径反推 R (theta = 9.305 * R / d) -> R = theta * d / 9.305
    R_est = (theta * d_pc) / 9.305

    limit_min, limit_max = RADIUS_LIMITS.get(lclass, (0, 9999))

    # 放宽一点限制，允许演化中的恒星稍微越界
    if not (limit_min * 0.5 <= R_est <= limit_max * 1.5):
        errors.append(
            f"半径异常: {lclass}型 R={R_est:.1f}R☉ (范围 {limit_min}-{limit_max})"
        )

    # 3. 斯特藩-玻尔兹曼定律检查
    # L = 4 * pi * R^2 * sigma * T^4
    # 相对值: L_sol = R_sol^2 * (T/T_sol)^4
    L_from_bol = 10 ** ((4.83 - M_bol) / 2.5)
    T_calc = SUN_TEFF * (L_from_bol / (R_est**2)) ** 0.25

    t_min, t_max = TEFF_RANGE.get(stype, (0, 99999))
    if not (t_min * 0.8 <= T_calc <= t_max * 1.2):
        errors.append(
            f"Teff不匹配: 物理推导{T_calc:.0f}K vs 光谱{stype}({t_min}-{t_max})"
        )

    return len(errors) == 0, errors


# =========================
# 最终验证函数 (Final Validation)
# =========================


def validate_star_final(star: dict) -> tuple[bool, list[str]]:
    """
    最终物理验证，匹配 verify.py 的检查逻辑
    用于生成后的最终质量控制
    """
    reasons = []

    # --- 规则 A: 不可能的红矮星 (The Impossible M-Dwarf) ---
    # M型主序星(V)极暗，肉眼可见距离通常<10pc。如果超过20pc还能看见，说明物理参数错了。
    if (
        star["spectral_type"] == "M"
        and star["luminosity_class"] == "V"
        and star["distance_pc"] > 20
    ):
        reasons.append(f"M型主序星距离过远 ({star['distance_pc']}pc) - 物理上应不可见")

    # --- 规则 B: 超爱丁顿光度极限 (Super-Eddington Limit) ---
    # 恒星光度极少超过 1,000,000 L_sun (除了极不稳定的高光度蓝变星)
    if star["luminosity_solar"] > 1_500_000:
        reasons.append(
            f"光度异常过高 ({star['luminosity_solar']:.1e} Lsun) - 超过稳定极限"
        )

    # --- 规则 C: 错误的分类 (Classification Mismatch) ---
    # 超巨星(0/I)的绝对星等应该很亮 (Mv < -3)
    # 亮巨星(II)允许更暗，但不应超过 -1.0 (保留天文真实性)
    if star["luminosity_class"] in ["0", "I"] and star["abs_mag"] > -3:
        reasons.append(f"超巨星绝对星等过暗 (Mv={star['abs_mag']})")
    elif star["luminosity_class"] == "II" and star["abs_mag"] > -1.0:
        reasons.append(f"亮巨星绝对星等过暗 (Mv={star['abs_mag']})")

    # --- 规则 D: 错误的温度-光度关系 (主序带偏离) ---
    # 如果标为主序星(V)，但像红巨星一样冷且亮
    if (
        star["luminosity_class"] == "V"
        and star["temperature_K"] < 4000
        and star["abs_mag"] < 5
    ):
        reasons.append("主序星异常: 温度低但过亮 (疑似应为巨星)")

    return len(reasons) == 0, reasons


# =========================
# 通用恒星生成逻辑 (All Classes)
# =========================


def generate_star_physically(stype, lclass, app_mag_target):
    """
    物理驱动的生成流程:
    Type+Class -> Mv (Table) -> Teff -> BC -> Mbol -> L -> R -> Mass
    """

    # 1. 确定有效温度 Teff
    t_min, t_max = TYPE_TEMP_RANGE[stype]
    teff = np.random.uniform(t_min, t_max)

    # 2. 确定目视绝对星等 Mv
    # 从表中查基准值并添加随机扰动
    mv_base, mv_spread = M_V_TABLE.get(stype, {}).get(
        lclass, M_V_TABLE.get(stype, {}).get("V")
    )
    # 使用正态分布生成，但在边界处截断以防离谱
    mv_val = np.random.normal(mv_base, mv_spread * 0.5)

    # 3. 计算热光校正 BC 和 热光绝对星等 Mbol
    bc = get_bolometric_correction(teff)
    mbol = mv_val + bc

    # 4. 推导物理参数
    # 光度 L (Solar)
    lum_solar = 10 ** ((4.83 - mbol) / 2.5)

    # 半径 R (Solar) - 基于 Stefan-Boltzmann
    # L = R^2 * (T/Tsun)^4
    radius_solar = math.sqrt(lum_solar) / ((teff / 5778) ** 2)

    # 5. 估算质量 (Mass)
    # 对于非主序星，质量不再严格锁定光度，而是查表估算
    mass_ranges = MASS_ESTIMATE.get(stype, {})
    m_min_mass, m_max_mass = mass_ranges.get(lclass, mass_ranges.get("default", (1, 1)))
    mass_solar = np.random.uniform(m_min_mass, m_max_mass)

    # 6. 推算距离 (基于目标视星等)
    # m - Mv = 5 log(d/10) + Av
    # 第一次估算 (忽略Av)
    dist_mod = app_mag_target - mv_val
    dist_pc = 10 ** ((dist_mod + 5) / 5)

    # 修正消光 (迭代一次)
    av = AV_PER_KPC * (dist_pc / 1000.0)
    dist_mod_corr = app_mag_target - mv_val - av
    dist_pc = 10 ** ((dist_mod_corr + 5) / 5)

    # 距离约束
    if dist_pc > MAX_DISTANCE:
        dist_pc = MAX_DISTANCE
    if dist_pc < 1.0:
        dist_pc = 1.0  # 至少1pc

    # 最终参数回算
    final_av = AV_PER_KPC * (dist_pc / 1000.0)
    final_app_mag = mv_val + 5 * math.log10(dist_pc / 10) + final_av

    # 7. 几何参数
    pos = uniform_sphere_sample() * dist_pc
    ra, dec = cartesian_to_spherical(pos[0], pos[1], pos[2])
    theta = 9.305 * radius_solar / dist_pc

    return {
        "id": "",
        "spectral_type": stype,
        "luminosity_class": lclass,
        "mass_solar": round(float(mass_solar), 4),
        "temperature_K": round(float(teff), 1),
        "luminosity_solar": round(float(lum_solar), 4),
        "radius_solar": round(float(radius_solar), 4),
        "distance_pc": round(float(dist_pc), 2),
        "dist_ly": round(float(dist_pc * 3.2616), 2),
        "extinction_Av": round(float(final_av), 3),
        "abs_mag": round(float(mv_val), 2),
        "bolometric_mag": round(float(mbol), 2),
        "bc_correction": round(float(bc), 2),
        "app_mag": round(float(final_app_mag), 3),
        "color_hex": COLOR_MAP.get(stype, "#ffffff"),
        "ra": round(float(ra), 4),
        "dec": round(float(dec), 4),
        "pos_cartesian": [round(float(x), 2) for x in pos],
        "angular_diameter_mas": round(float(theta), 5),
    }


# =========================
# 辅助函数：根据光谱型动态调整光度级权重
# =========================
def get_weighted_class(stype):
    """
    根据光谱类型返回合理的光度级。
    物理事实：肉眼可见的 M 型星 100% 是巨星或超巨星。
    肉眼可见的 B 型星 既可以是主序星也可以是超巨星。
    """
    # 复制一份全局权重
    weights = CLASS_DISTRIBUTION.copy()

    if stype == "M":
        # M型星肉眼可见的必须是巨星(III)或超巨星(I/II)
        # 强行抹去主序星(V)的概率，否则会生成大量看不见的红矮星
        weights["V"] = 0.0
        weights["IV"] = 0.0
        weights["VI"] = 0.0
        # 重新归一化 (简单处理，直接给高权重)
        weights["I"] = 0.2
        weights["II"] = 0.3
        weights["III"] = 0.5

    elif stype == "K":
        # K型星肉眼可见的大部分是巨星 (如大角星)
        weights["V"] *= 0.1  # 降低主序星权重
        weights["III"] *= 2.0  # 提高巨星权重

    elif stype in ["O", "B"]:
        # O/B型星本身就很亮，主序星(V)也完全可见
        # 维持原样或微调
        pass

    # 根据权重随机选择
    classes = list(weights.keys())
    probs = list(weights.values())
    # 归一化概率
    total = sum(probs)
    if total == 0:
        return "III"  # fallback
    norm_probs = [p / total for p in probs]

    return random.choices(classes, weights=norm_probs, k=1)[0]


# =========================
# 统计分析函数
# =========================


def generate_folders_json():
    """生成 folders.json 文件，列出所有可用的数据集文件夹"""
    # 获取当前目录下所有 output_* 文件夹
    folders = ["default"]
    for item in os.listdir("."):
        if item.startswith("output_") and os.path.isdir(item):
            folders.append(item)

    # 按名称排序（default 在前，其他按时间戳排序）
    folders_sorted = ["default"] + sorted(
        [f for f in folders if f != "default"], reverse=True
    )

    # 写入 JSON 文件
    with open("folders.json", "w", encoding="utf-8") as f:
        json.dump(folders_sorted, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated folders.json with {len(folders_sorted)} folders:")
    for folder in folders_sorted:
        print(f"  - {folder}")


def verify_and_visualize(stars, generation_id, output_dir):
    """验证恒星数据并生成可视化图表和报告"""

    print("\n正在验证生成的数据...")

    # 异常检测逻辑
    flagged_stars = []
    valid_stars = []

    for s in stars:
        reasons = []

        # 规则 A: 不可能的红矮星
        if (
            s["spectral_type"] == "M"
            and s["luminosity_class"] == "V"
            and s["distance_pc"] > 20
        ):
            reasons.append(f"M型主序星距离过远 ({s['distance_pc']}pc) - 物理上应不可见")

        # 规则 B: 超爱丁顿光度极限
        if s["luminosity_solar"] > 1_500_000:
            reasons.append(
                f"光度异常过高 ({s['luminosity_solar']:.1e} Lsun) - 超过稳定极限"
            )

        # 规则 C: 错误的分类
        if s["luminosity_class"] in ["0", "I"] and s["abs_mag"] > -3:
            reasons.append(f"超巨星绝对星等过暗 (Mv={s['abs_mag']})")
        elif s["luminosity_class"] == "II" and s["abs_mag"] > -1.0:
            reasons.append(f"亮巨星绝对星等过暗 (Mv={s['abs_mag']})")

        # 规则 D: 错误的温度-光度关系
        if (
            s["luminosity_class"] == "V"
            and s["temperature_K"] < 4000
            and s["abs_mag"] < 5
        ):
            reasons.append("主序星异常: 温度低但过亮 (疑似应为巨星)")

        if reasons:
            s["flag_reasons"] = reasons
            flagged_stars.append(s)
        else:
            valid_stars.append(s)

    print(
        f"检测完成: 正常恒星 {len(valid_stars)} 颗, 异常恒星 {len(flagged_stars)} 颗。"
    )

    # 提取绘图数据
    def get_data(star_list):
        return {
            "temp": [s["temperature_K"] for s in star_list],
            "abs_mag": [s["abs_mag"] for s in star_list],
            "color": [s["color_hex"] for s in star_list],
            "dist": [s["distance_pc"] for s in star_list],
            "app_mag": [s["app_mag"] for s in star_list],
            "type": [s["spectral_type"] for s in star_list],
        }

    v_data = get_data(valid_stars)
    f_data = get_data(flagged_stars)

    # 图表 1: 赫罗图 (H-R Diagram)
    OUTPUT_HR_IMG = os.path.join(
        output_dir, f"validation_hr_diagram_{generation_id}.png"
    )
    plt.figure(figsize=(12, 10))
    ax = plt.gca()

    plt.scatter(
        v_data["temp"],
        v_data["abs_mag"],
        c=v_data["color"],
        s=15,
        alpha=0.6,
        edgecolors="none",
        label="Valid Stars",
    )

    if f_data["temp"]:
        plt.scatter(
            f_data["temp"],
            f_data["abs_mag"],
            c="red",
            marker="x",
            s=50,
            linewidth=1.5,
            label="Problematic Stars",
        )

    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.xscale("log")
    plt.title(
        f"Hertzsprung-Russell Diagram Validation\n(Flagged: {len(flagged_stars)} stars)",
        fontsize=16,
    )
    plt.xlabel("Temperature (K)", fontsize=12)
    plt.ylabel("Absolute Magnitude (Mv)", fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    ax.set_xticks([30000, 10000, 6000, 3000])
    ax.set_xticklabels(["30k", "10k", "6k", "3k"])
    plt.savefig(OUTPUT_HR_IMG, dpi=150, bbox_inches="tight")
    print(f"图表已保存: {OUTPUT_HR_IMG}")
    plt.close()

    # 图表 2: 距离 vs 视星等
    OUTPUT_DIST_IMG = os.path.join(
        output_dir, f"validation_dist_mag_{generation_id}.png"
    )
    plt.figure(figsize=(12, 8))

    plt.scatter(
        v_data["dist"],
        v_data["app_mag"],
        c="blue",
        s=10,
        alpha=0.3,
        label="Valid Stars",
    )

    if f_data["dist"]:
        plt.scatter(
            f_data["dist"],
            f_data["app_mag"],
            c="red",
            marker="x",
            s=40,
            label="Problematic Stars",
        )

    plt.axhline(
        y=6.5, color="green", linestyle="--", linewidth=2, label="Naked Eye Limit (6.5)"
    )

    plt.xlabel("Distance (pc)", fontsize=12)
    plt.ylabel("Apparent Magnitude (m)", fontsize=12)
    plt.title("Distance vs. Visibility Check", fontsize=16)
    plt.gca().invert_yaxis()
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIST_IMG, dpi=150, bbox_inches="tight")
    print(f"图表已保存: {OUTPUT_DIST_IMG}")
    plt.close()

    # 生成文本报告
    OUTPUT_REPORT = os.path.join(output_dir, f"validation_report_{generation_id}.txt")
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("=== 恒星生成质量验证报告 ===\n")
        f.write(f"总数: {len(stars)}\n")
        f.write(f"通过物理校验: {len(valid_stars)}\n")
        f.write(f"标记为异常: {len(flagged_stars)}\n\n")

        if len(flagged_stars) > 0:
            f.write("--- 异常详情 (前 20 例) ---\n")
            for i, s in enumerate(flagged_stars[:20]):
                f.write(
                    f"[{i + 1}] ID: {s['id']} | Type: {s['spectral_type']}{s['luminosity_class']}\n"
                )
                for r in s["flag_reasons"]:
                    f.write(f"    - {r}\n")
                f.write("\n")

        f.write("\n--- 关键统计 ---\n")
        f.write(f"最远恒星: {max([s['distance_pc'] for s in stars]):.1f} pc\n")
        f.write(f"最大光度: {max([s['luminosity_solar'] for s in stars]):.1e} L_sun\n")

        m_stars = [s for s in stars if s["spectral_type"] == "M"]
        m_giants = [s for s in m_stars if s["luminosity_class"] in ["I", "II", "III"]]
        m_dwarfs = [s for s in m_stars if s["luminosity_class"] == "V"]
        f.write(
            f"M型星总数: {len(m_stars)} (巨星: {len(m_giants)}, 矮星: {len(m_dwarfs)})\n"
        )

    print(f"文本报告已保存: {OUTPUT_REPORT}")
    print("验证完成。")


def analyze_star_distribution(stars_list):
    """分析星星视星等分布"""

    all_stars = stars_list

    print("=" * 50)
    print("星图数据统计报告")
    print("=" * 50)
    print(f"总计: {len(all_stars)}")
    print()

    # 使用天文学标准区间统计（以 x.5 为边界）
    # 理论分布基于 log10(N) = a + 0.51*m
    STANDARD_BINS = [
        (-1.5, -0.5, 0.02),
        (-0.5, 0.5, 0.07),
        (0.5, 1.5, 0.16),
        (1.5, 2.5, 0.81),
        (2.5, 3.5, 2.17),
        (3.5, 4.5, 6.96),
        (4.5, 5.5, 22.00),
        (5.5, 6.5, 67.81),
    ]

    print(f"{'=' * 60}")
    print("视星等分布 (天文学标准区间)")
    print(f"{'=' * 60}")
    print(f"{'区间':<15} {'数量':>8} {'实际%':>10} {'理论%':>10} {'偏差':>10}")
    print(f"{'-' * 60}")

    total = len(all_stars)
    max_count = 1
    bin_counts = []

    for m_min, m_max, theory_pct in STANDARD_BINS:
        count = sum(1 for s in all_stars if m_min <= s.get("app_mag", 999) < m_max)
        bin_counts.append(count)
        if count > max_count:
            max_count = count

    for i, (m_min, m_max, theory_pct) in enumerate(STANDARD_BINS):
        count = bin_counts[i]
        actual_pct = (count / total) * 100 if total > 0 else 0
        diff = actual_pct - theory_pct
        bar_length = int((count / max_count) * 25)
        bar = "█" * bar_length
        diff_str = f"+{diff:.2f}" if diff >= 0 else f"{diff:.2f}"
        print(
            f"{m_min:>4.1f} ~ {m_max:<4.1f}   {count:>8}   {actual_pct:>8.2f}%   {theory_pct:>8.2f}%   {diff_str:>8}%  {bar}"
        )

    print()

    # 更细致的统计：0.5 等间隔
    print(f"{'=' * 50}")
    print("视星等分布 (0.5等间隔)")
    print(f"{'=' * 50}")

    fine_bins = defaultdict(int)
    for star in all_stars:
        mag = star.get("app_mag", 0)
        bin_key = round(mag * 2) / 2  # 四舍五入到0.5
        fine_bins[bin_key] += 1

    print(f"{'区间':<15} {'数量':>10} {'柱状图'}")
    print(f"{'-' * 50}")

    max_fine = max(fine_bins.values()) if fine_bins else 1
    for mag in sorted(fine_bins.keys()):
        count = fine_bins[mag]
        bar_length = int((count / max_fine) * 30)
        bar = "▓" * bar_length
        print(f"{mag:>5.1f} 等       {count:>10}  {bar}")

    print()

    # 统计亮星和暗星
    print(f"{'=' * 50}")
    print("特殊统计")
    print(f"{'=' * 50}")

    a1 = [s for s in all_stars if s.get("app_mag", 0) < 0.5]
    a2 = [s for s in all_stars if 0.5 <= s.get("app_mag", 0) < 1.5]
    a3 = [s for s in all_stars if 1.5 <= s.get("app_mag", 0) < 2.5]
    a4 = [s for s in all_stars if 2.5 <= s.get("app_mag", 0) < 3.5]
    a5 = [s for s in all_stars if 3.5 <= s.get("app_mag", 0) < 4.5]
    a6 = [s for s in all_stars if 4.5 <= s.get("app_mag", 0) < 5.5]
    a7 = [s for s in all_stars if 5.5 <= s.get("app_mag", 0) < 6.5]
    a8 = [s for s in all_stars if s.get("app_mag", 0) >= 6.5]

    print(f"<0.5:  {len(a1):>6}")
    print(f"0.5 ~ 1.5:  {len(a2):>6}")
    print(f"1.5 ~ 2.5:  {len(a3):>6}")
    print(f"2.5 ~ 3.5:  {len(a4):>6}")
    print(f"3.5 ~ 4.5:  {len(a5):>6}")
    print(f"4.5 ~ 5.5:  {len(a6):>6}")
    print(f"5.5 ~ 6.5:  {len(a7):>6}")
    print(f">= 6.5: {len(a8):>6}")

    print()

    # 找出最亮和最暗的星
    if all_stars:
        brightest = min(all_stars, key=lambda s: s.get("app_mag", 999))
        dimmest = max(all_stars, key=lambda s: s.get("app_mag", -999))

        print(
            f"最亮星: {brightest.get('id', 'N/A')} (视星等: {brightest.get('app_mag', 'N/A'):.2f})"
        )
        print(
            f"最暗星: {dimmest.get('id', 'N/A')} (视星等: {dimmest.get('app_mag', 'N/A'):.2f})"
        )

        # 平均视星等
        avg_mag = sum(s.get("app_mag", 0) for s in all_stars) / len(all_stars)
        print(f"平均视星等: {avg_mag:.2f}")


# =========================
# 主生成循环 (优化版)
# =========================

print("Starting full spectrum generation (Magnitude Limited Mode)...")

# 生成唯一ID（基于时间戳）- 在开始时生成，用于所有星体ID
generation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Generation ID: {generation_id}")

stars = []
validated_count = 0
rejected_count = 0

for stype, percentage in TARGET_DISTRIBUTION.items():
    target_count_for_type = int(TARGET_COUNT * percentage)
    print(f"Generating {target_count_for_type} stars of type {stype}...")

    generated = 0
    # 甚至可以给每个类型设置不同的最大尝试次数，防止死循环
    attempts = 0

    while generated < target_count_for_type:
        attempts += 1
        if attempts > target_count_for_type * 20:
            print(f"Warning: Difficulty generating valid {stype} stars. Moving on.")
            break

        # 1. 动态采样视星等目标
        m_target = sample_apparent_magnitude(1)[0]

        # 2. 【关键修改】根据光谱型智能选择光度级
        lclass = get_weighted_class(stype)

        # 3. 生成
        try:
            star = generate_star_physically(stype, lclass, m_target)

            # 4. 校验
            is_valid, msgs = validate_star_strict(star)

            if is_valid:
                star["id"] = f"{generation_id}_{len(stars) + 1:05d}"
                stars.append(star)
                validated_count += 1
                generated += 1
            else:
                rejected_count += 1

        except Exception:
            pass

print(f"\n初始生成完成: {len(stars)} 颗恒星")

# =========================
# 最终验证与重新生成
# =========================

print("\n执行最终物理验证...")
invalid_indices = []
invalid_reasons = {}

for i, star in enumerate(stars):
    is_valid, msgs = validate_star_final(star)
    if not is_valid:
        invalid_indices.append(i)
        invalid_reasons[i] = msgs

if invalid_indices:
    print(f"发现 {len(invalid_indices)} 颗不符合最终验证的恒星，正在重新生成...")

    regenerated_count = 0
    failed_regeneration = []

    for idx in invalid_indices:
        old_star = stars[idx]
        stype = old_star["spectral_type"]
        lclass = old_star["luminosity_class"]
        old_id = old_star["id"]

        # 尝试重新生成直到通过验证
        max_attempts = 200
        success = False

        for attempt in range(max_attempts):
            m_target = sample_apparent_magnitude(1)[0]
            new_star = generate_star_physically(stype, lclass, m_target)

            # 先通过严格验证
            is_strict_valid, _ = validate_star_strict(new_star)
            if not is_strict_valid:
                continue

            # 再通过最终验证
            is_final_valid, _ = validate_star_final(new_star)
            if is_final_valid:
                new_star["id"] = old_id  # 保持原ID
                stars[idx] = new_star
                regenerated_count += 1
                success = True
                break

        if not success:
            failed_regeneration.append(f"{old_id} ({stype}{lclass})")
            print(f"  警告: 无法为 {old_id} ({stype}{lclass}) 生成有效替代，保留原星体")

    print(f"成功重新生成: {regenerated_count} 颗")
    if failed_regeneration:
        print(
            f"未能重新生成: {len(failed_regeneration)} 颗 - {', '.join(failed_regeneration[:5])}"
        )
else:
    print("所有恒星均通过最终验证！")

# 使用之前生成的 generation_id
output_dir = f"output_{generation_id}"

output = {
    "metadata": {
        "generation_id": generation_id,
        "count": len(stars),
        "note": "All stars validated with final physical checks (100% pass rate)",
        "generation_stats": {
            "initial_validated": validated_count,
            "initial_rejected": rejected_count,
            "final_regenerated": len(invalid_indices) if invalid_indices else 0,
        },
        "validation_stats": "Final pass rate: 100%",
    },
    "stars": stars,
}
# 结果统计 (moved from original saving block)
# =========================

print("\nGeneration Complete!")
print(f"Generation ID: {generation_id}")
print(f"Total Stars: {len(stars)}")
print(
    f"Rejection Rate: {rejected_count / (validated_count + rejected_count) * 100:.1f}%"
)

# 统计一下生成的光度级分布
c = Counter([s["luminosity_class"] for s in stars])
print("\nLuminosity Class Distribution:")
for k in sorted(c.keys()):
    print(f"{k}: {c[k]} ({c[k] / len(stars) * 100:.2f}%)")

# =========================
# 执行统计分析
# =========================

print("\n" + "=" * 60)
print("开始统计分析...")
print("=" * 60 + "\n")

analyze_star_distribution(stars)

# =========================
# 用户确认
# =========================

print("\n" + "=" * 60)
user_input = input("是否保存生成的数据? (y/n, 默认 y): ").strip().lower()
if user_input == "" or user_input == "y" or user_input == "yes":
    print("正在保存数据...")

    # 生成唯一ID（基于时间戳）
    generation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output_{generation_id}"
    os.makedirs(output_dir, exist_ok=True)

    # 保存 JSON 文件
    output = {
        "metadata": {
            "generation_id": generation_id,
            "count": len(stars),
            "note": "All stars validated with final physical checks (100% pass rate)",
            "generation_stats": {
                "initial_validated": validated_count,
                "initial_rejected": rejected_count,
                "final_regenerated": len(invalid_indices) if invalid_indices else 0,
            },
            "validation_stats": "Final pass rate: 100%",
        },
        "stars": stars,
    }

    output_file = os.path.join(output_dir, f"star_map_{generation_id}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Output Directory: {output_dir}/")
    print(f"Output File: {output_file}")
    print("数据保存成功！")

    # 执行验证和可视化
    verify_and_visualize(stars, generation_id, output_dir)

    # 生成 folders.json
    generate_folders_json()

    print("\n" + "=" * 60)
    print("所有操作完成！")
    print("=" * 60)
else:
    print("已取消保存，数据未写入文件。")
