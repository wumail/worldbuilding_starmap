import numpy as np
import json
import math
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

def sample_apparent_magnitude(size, m_min=-1.5, m_max=6.5):
    """
    根据 p(m) ∝ 10^(0.5m) 分布采样视星等
    使用逆变换采样方法
    
    CDF: F(m) = (10^(0.5m) - 10^(0.5*m_min)) / (10^(0.5*m_max) - 10^(0.5*m_min))
    逆CDF: m = 2 * log10(u * (10^(0.5*m_max) - 10^(0.5*m_min)) + 10^(0.5*m_min))
    """
    u = np.random.rand(size)
    
    a = 10 ** (0.5 * m_min)
    b = 10 ** (0.5 * m_max)
    
    # 逆变换采样
    m = 2 * np.log10(u * (b - a) + a)
    
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
# 恒星参数验证函数
# =========================

# def validate_star(star_data: dict) -> tuple[bool, list[str]]:
#     """
#     验证恒星参数是否符合物理和天文学规律。
    
#     应用的物理/天文定律：
    
#     1. 【距离模数公式】 (Distance Modulus)
#        m - M = 5 * log10(d/10) + A_v
#        其中 m=视星等, M=绝对星等, d=距离(pc), A_v=消光
#        来源：天文学基本定义，基于平方反比定律
    
#     2. 【斯特藩-玻尔兹曼定律】 (Stefan-Boltzmann Law)
#        L = 4πR²σT⁴
#        恒星光度与半径和温度的关系
#        用于验证：光度、半径、温度的自洽性
    
#     3. 【质光关系】 (Mass-Luminosity Relation) 
#        L ∝ M^α，其中 α ≈ 3.5 (主序星)
#        来源：恒星物理学的经验关系
    
#     4. 【角直径公式】
#        θ = 2 * R / d (弧度) = 2 * R_solar * R_sun_rad / (d_ly * ly_to_m) * 206265 * 1000 (mas)
#        简化为：θ_mas ≈ 15 * R_solar / d_ly
    
#     5. 【光谱类型-温度对应关系】 (MK Classification)
#        O: >30000K, B: 10000-30000K, A: 7500-10000K
#        F: 6000-7500K, G: 5200-6000K, K: 3700-5200K, M: <3700K
    
#     6. 【恒星计数定律】 (Star Counts)
#        log N(m) = a + b*m，其中 b ≈ 0.5
#        亮星比暗星少，每增加1等星等，数量增加约3倍
    
#     7. 【坐标系有效性】
#        RA: 0° - 360°, Dec: -90° - 90°
    
#     返回: (is_valid, error_messages)
#     """
#     errors = []
#     warnings = []
    
#     # 提取数据
#     app_mag = star_data.get("app_mag")
#     abs_mag = star_data.get("abs_mag")
#     dist_pc = star_data.get("distance_pc")
#     dist_ly = star_data.get("dist_ly")
#     av = star_data.get("extinction_Av", 0)
#     ra = star_data.get("ra")
#     dec = star_data.get("dec")
#     stype = star_data.get("spectral_type")
#     temp = star_data.get("temperature_K")
#     lum = star_data.get("luminosity_solar")
#     ang_diam = star_data.get("angular_diameter_mas")
#     mass = star_data.get("mass_solar")
    
#     star_id = star_data.get("id", "Unknown")
    
#     # ===== 1. 距离模数验证 =====
#     if all(v is not None for v in [app_mag, abs_mag, dist_pc]):
#         # m - M = 5 * log10(d) - 5 + A_v
#         expected_app_mag = abs_mag + 5 * math.log10(dist_pc) - 5 + av
#         diff = abs(app_mag - expected_app_mag)
#         if diff > 0.5:  # 允许0.5等误差
#             errors.append(f"距离模数不一致: m={app_mag:.2f}, 计算值={expected_app_mag:.2f}, 差={diff:.2f}")
    
#     # ===== 2. 距离单位一致性 =====
#     if dist_pc is not None and dist_ly is not None:
#         expected_ly = dist_pc * 3.2616
#         diff_pct = abs(dist_ly - expected_ly) / max(expected_ly, 0.01) * 100
#         if diff_pct > 1:  # 允许1%误差
#             errors.append(f"距离单位不一致: {dist_pc}pc vs {dist_ly}ly (差{diff_pct:.1f}%)")
    
#     # ===== 3. 坐标有效性验证 =====
#     if ra is not None and (ra < 0 or ra > 360):
#         errors.append(f"赤经超出范围: RA={ra}° (应为0-360)")
#     if dec is not None and (dec < -90 or dec > 90):
#         errors.append(f"赤纬超出范围: Dec={dec}° (应为-90到90)")
    
#     # ===== 4. 光谱类型-温度对应验证 =====
#     SPECTRAL_TEMP_RANGES = {
#         "O": (30000, 100000),
#         "B": (10000, 30000),
#         "A": (7500, 10000),
#         "F": (6000, 7500),
#         "G": (5200, 6000),
#         "K": (3700, 5200),
#         "M": (2400, 3700),
#     }
#     if stype is not None and temp is not None and stype in SPECTRAL_TEMP_RANGES:
#         t_min, t_max = SPECTRAL_TEMP_RANGES[stype]
#         if temp < t_min * 0.8 or temp > t_max * 1.2:  # 允许20%容差
#             warnings.append(f"温度{temp:.0f}K与光谱类型{stype}不匹配 (期望{t_min}-{t_max}K)")
    
#     # ===== 5. 绝对星等范围验证 =====
#     SPECTRAL_ABS_MAG_RANGES = {
#         "O": (-10, -3),   # 超高光度
#         "B": (-8, 1),     # 高光度
#         "A": (-2, 3),     # 中等
#         "F": (1, 5),      # 较暗
#         "G": (2, 7),      # 太阳级别
#         "K": (3, 10),     # 较暗
#         "M": (5, 16),     # 最暗(主序)或很亮(巨星)
#     }
#     if stype is not None and abs_mag is not None and stype in SPECTRAL_ABS_MAG_RANGES:
#         m_min, m_max = SPECTRAL_ABS_MAG_RANGES[stype]
#         if abs_mag < m_min - 2 or abs_mag > m_max + 2:  # 宽松容差，允许巨星等
#             warnings.append(f"绝对星等M={abs_mag:.1f}与光谱类型{stype}不匹配 (典型值{m_min}到{m_max})")
    
#     # ===== 6. 角直径验证 =====
#     if ang_diam is not None and dist_ly is not None and mass is not None:
#         # 估算半径: R ≈ M^0.8 (主序星)
#         est_radius = mass ** 0.8 if mass > 0 else 1
#         expected_ang = est_radius / dist_ly * 15.0  # mas
#         ratio = ang_diam / max(expected_ang, 0.001)
#         if ratio < 0.1 or ratio > 10:  # 允许一个数量级误差（考虑巨星）
#             warnings.append(f"角直径{ang_diam:.3f}mas偏离估算{expected_ang:.3f}mas (比例{ratio:.1f}x)")
    
#     # ===== 7. 视星等合理性 =====
#     if app_mag is not None:
#         if app_mag < -2:
#             warnings.append(f"视星等{app_mag:.2f}极亮（比天狼星还亮）")
#         if app_mag > 8:
#             warnings.append(f"视星等{app_mag:.2f}极暗（肉眼不可见）")
    
#     # ===== 8. 质光关系验证（主序星）=====
#     if mass is not None and lum is not None and mass > 0:
#         # 理论上 L ∝ M^3.5
#         expected_lum = mass ** 3.5
#         ratio = lum / max(expected_lum, 0.001)
#         # 巨星光度会比主序星高很多，所以只检查下限
#         if ratio < 0.1:
#             warnings.append(f"光度{lum:.1f}L☉对于{mass:.2f}M☉的恒星偏低")
    
#     is_valid = len(errors) == 0
#     all_messages = errors + warnings
    
#     return is_valid, all_messages
def validate_star_strict(star: dict) -> tuple[bool, list[str]]:
    """
    严格恒星物理一致性验证
    返回: (is_valid, errors)
    """

    errors = []

    # =========================
    # 常数
    # =========================
    SUN_MV = 4.83
    SUN_TEFF = 5778

    # 光谱型 → 温度
    TEFF_RANGE = {
        "O": (30000, 50000),
        "B": (10000, 30000),
        "A": (7500, 10000),
        "F": (6000, 7500),
        "G": (5200, 6000),
        "K": (3700, 5200),
        "M": (2400, 3700),
        "WD": (5000, 100000),
    }

    # 光度级 → 半径 (R☉)
    RADIUS_RANGE = {
        "V":   (0.1, 20),
        "IV":  (2, 6),
        "III": (5, 100),
        "II":  (30, 300),
        "I":   (100, 1500),
        "WD":  (0.008, 0.02),
    }

    # 光度级 → 绝对星等
    ABSMAG_RANGE = {
        "V":   (-2, 12),
        "IV":  (-3, 4),
        "III": (-5, 1),
        "II":  (-7, -3),
        "I":   (-10, -6),
        "WD":  (10, 16),
    }

    # =========================
    # 读取字段
    # =========================
    stype = star.get("spectral_type")
    lclass = star.get("luminosity_class")
    m = star.get("app_mag")
    M = star.get("abs_mag")
    d_pc = star.get("distance_pc")
    theta = star.get("angular_diameter_mas")
    ra = star.get("ra")
    dec = star.get("dec")
    Av = star.get("extinction_Av", 0.0)

    # =========================
    # 0. 基本完整性
    # =========================
    required = [stype, lclass, m, M, d_pc, theta, ra, dec]
    if any(v is None for v in required):
        return False, ["缺失必要字段"]

    if stype not in TEFF_RANGE:
        errors.append(f"未知光谱型: {stype}")
    if lclass not in RADIUS_RANGE:
        errors.append(f"未知光度级: {lclass}")
    if not (0 <= ra <= 360):
        errors.append("RA 超出范围")
    if not (-90 <= dec <= 90):
        errors.append("Dec 超出范围")

    if errors:
        return False, errors

    # =========================
    # 1. 距离模数（严格）
    # =========================
    m_calc = M + 5 * math.log10(d_pc) - 5 + Av
    if abs(m - m_calc) > 0.2:
        errors.append(
            f"距离模数不一致: m={m:.2f}, 计算={m_calc:.2f}"
        )

    # =========================
    # 2. 角直径 → 半径（严格公式）
    # =========================
    R_est = 107.5 * theta * d_pc  # R☉

    r_min, r_max = RADIUS_RANGE[lclass]
    if not (r_min <= R_est <= r_max):
        errors.append(
            f"半径不合理: R≈{R_est:.1f}R☉ 不在 {lclass} 范围 {r_min}-{r_max}"
        )

    # =========================
    # 3. 绝对星等 → 光度
    # =========================
    L_est = 10 ** ((SUN_MV - M) / 2.5)

    # =========================
    # 4. 光度 + 半径 → 温度
    # =========================
    T_est = SUN_TEFF * (L_est / (R_est ** 2)) ** 0.25

    t_min, t_max = TEFF_RANGE[stype]
    if not (t_min <= T_est <= t_max):
        errors.append(
            f"温度不匹配: 推导T≈{T_est:.0f}K 不符合 {stype} ({t_min}-{t_max})"
        )

    # =========================
    # 5. 光度级 ↔ 绝对星等
    # =========================
    M_min, M_max = ABSMAG_RANGE[lclass]
    if not (M_min <= M <= M_max):
        errors.append(
            f"绝对星等 {M:.2f} 不符合光度级 {lclass} ({M_min}~{M_max})"
        )

    return len(errors) == 0, errors

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
    # 1. 采样视星等 (使用 p(m) ∝ 10^(0.5m) 分布)
    app_mags = sample_apparent_magnitude(batch_size, m_min=-1.8, m_max=6.5)
    
    # 2. 按目标分布采样光谱类型
    type_probs = list(TARGET_DISTRIBUTION.values())
    type_names = list(TARGET_DISTRIBUTION.keys())
    spectral_types = np.random.choice(type_names, size=batch_size, p=type_probs)
    
    # 统计验证结果
    validated_count = 0
    rejected_count = 0
    
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
            rejected_count += 1
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
        
        # ===== 验证恒星参数 =====
        is_valid, messages = validate_star_strict(star)
        
        if is_valid:
            generated_stars.append(star)
            validated_count += 1
        else:
            rejected_count += 1
            # 可选：记录第一个错误用于调试
            # if rejected_count == 1:
            #     print(f"  First rejection: {messages[0]}")
    
    if validated_count > 0 or rejected_count > 0:
        acceptance_rate = validated_count / (validated_count + rejected_count) * 100
        print(f"  Generated {len(generated_stars)} stars (✓{validated_count} ✗{rejected_count}, 通过率{acceptance_rate:.1f}%)")

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
