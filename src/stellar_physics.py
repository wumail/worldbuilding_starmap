"""恒星参数与独立一致性检查。主序星使用观测均值序列；演化星为受约束近似。"""
from __future__ import annotations

import csv
import math
import re

import numpy as np

from paths import DATA_DIR

SUN_TEFF = 5772.0
SUN_M_BOL = -2.5 * math.log10(3.828e26 / 3.0128e28)  # IAU 2015 B2/B3
PC_TO_LY = 3.2615637771674333
SOLAR_DIAMETER_MAS_AT_PC = 2 * 6.957e8 / 1.495978707e11 * 1000
LOG_G_SUN = 4.438
EDDINGTON_PER_MASS = 32000.0  # 电子散射近似，作为本模型的筛选上限
REFERENCE_PATH = DATA_DIR / "main_sequence.csv"

# 以下演化星范围保留原有经验模型；不代表恒星演化轨道或等龄线。
TYPE_TEMP_RANGE = {'O': (30000, 50000),
 'B': (10000, 30000),
 'A': (7500, 10000),
 'F': (6000, 7500),
 'G': (5200, 6000),
 'K': (3700, 5200),
 'M': (2400, 3700)}

COLOR_MAP = {'O': '#9bb0ff',
 'B': '#aabfff',
 'A': '#cad7ff',
 'F': '#f8f7ff',
 'G': '#fff4ea',
 'K': '#ffd2a1',
 'M': '#ffcc6f'}

EVOLVED_MV = {'O': {'0': (-7.5, 0.8),
       'I': (-6.5, 1.0),
       'II': (-6.0, 0.5),
       'III': (-5.5, 0.5),
       'IV': (-5.0, 0.5)},
 'B': {'0': (-7.0, 1.0),
       'I': (-6.0, 1.5),
       'II': (-4.0, 1.0),
       'III': (-2.0, 1.0),
       'IV': (-1.5, 0.8)},
 'A': {'0': (-7.0, 1.0), 'I': (-5.5, 1.0), 'II': (-3.0, 0.8), 'III': (-0.5, 0.8), 'IV': (1.0, 0.5)},
 'F': {'0': (-6.0, 1.0), 'I': (-5.0, 1.0), 'II': (-2.5, 0.5), 'III': (1.0, 0.8), 'IV': (2.5, 0.5)},
 'G': {'0': (-5.0, 1.0), 'I': (-4.5, 1.0), 'II': (-2.0, 0.5), 'III': (0.5, 0.8), 'IV': (3.0, 0.5)},
 'K': {'0': (-4.5, 1.0), 'I': (-4.5, 0.5), 'II': (-2.0, 0.5), 'III': (0.0, 1.0), 'IV': (4.0, 1.0)},
 'M': {'0': (-4.5, 1.0), 'I': (-5.0, 1.0), 'II': (-2.5, 1.0), 'III': (-0.5, 0.8), 'IV': (8.0, 1.5)}}

EVOLVED_MASS = {'O': {'default': (20, 100)},
 'B': {'I': (15, 40), 'III': (10, 20), 'default': (2, 20)},
 'A': {'I': (10, 20), 'III': (2.5, 5), 'default': (1.4, 10)},
 'F': {'I': (8, 15), 'III': (1.5, 3), 'default': (1, 10)},
 'G': {'I': (8, 12), 'III': (1.0, 4.0), 'default': (0.8, 10)},
 'K': {'I': (8, 15), 'III': (1.0, 5.0), 'default': (0.5, 10)},
 'M': {'I': (10, 30), 'III': (1.0, 6.0), 'default': (0.1, 20)}}

RADIUS_LIMITS = {"0": (25, 3000), "I": (10, 2250), "II": (5, 450),
                 "III": (1, 300), "IV": (0.6, 15)}
LOG_G_LIMITS = {"0": (-1.5, 3.5), "I": (-1, 3.8), "II": (0, 4),
                "III": (0, 4.2), "IV": (2.5, 4.5)}

with REFERENCE_PATH.open(newline="", encoding="utf-8") as reference_file:
    MAIN_SEQUENCE = tuple(sorted(
        ({"subtype": row["subtype"], **{key: float(value) for key, value in row.items()
           if key != "subtype"}} for row in csv.DictReader(reference_file)),
        key=lambda row: row["temperature_K"],
    ))


class CandidateRejected(ValueError):
    """参数落在模型或观测范围外，需要重新抽样。程序错误不应被吞掉。"""


def is_finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def main_sequence_parameters(temperature_K, spectral_type):
    """以细分光谱序列为共同参数，同时插值质量、光度与热光校正。

    原始经验表的 A 型质量估计并非严格单调，故以单调的温度轴插值，
    避免对质量排序后连接不相邻的光谱亚型。
    """
    rows = [r for r in MAIN_SEQUENCE if r["subtype"].startswith(spectral_type)]
    if not rows or not rows[0]["temperature_K"] <= temperature_K <= rows[-1]["temperature_K"]:
        raise CandidateRejected("温度超出该光谱型主序参考序列")
    x = np.log([r["temperature_K"] for r in rows])
    t = math.log(temperature_K)
    mass = math.exp(float(np.interp(t, x, np.log([r["mass_solar"] for r in rows]))))
    log_lum = float(np.interp(t, x, [r["log_luminosity_solar"] for r in rows]))
    bc = float(np.interp(t, x, [r["bc_correction"] for r in rows]))
    return mass, 10 ** log_lum, bc


def get_bolometric_correction(teff):
    """演化星沿用粗略温度插值；其大气重力和金属丰度尚未建模。"""
    table = [(2400, -4.6), (2500, -4.30), (3000, -2.70), (3800, -1.40),
             (4500, -0.65), (5200, -0.20), (5772, -0.085), (6000, -0.02),
             (6500, 0.00), (7500, -0.05), (8500, -0.15), (10000, -0.40),
             (15000, -1.30), (22000, -2.25), (33000, -3.15), (40000, -3.90),
             (50000, -4.60)]
    return float(np.interp(teff, [t for t, _ in table], [bc for _, bc in table]))


def evolved_mass_range(stype, lclass):
    if lclass == "0":
        return {"O": (40, 120), "B": (20, 100), "A": (15, 60), "F": (15, 60),
                "G": (15, 60), "K": (15, 40), "M": (15, 40)}[stype]
    if lclass == "IV" and stype in "FGK":
        return {"F": (1, 3), "G": (0.8, 3), "K": (0.8, 3)}[stype]
    table = EVOLVED_MASS[stype]
    return table.get(lclass, table["default"])


def sample_stellar_parameters(stype, lclass, rng):
    if stype not in TYPE_TEMP_RANGE or lclass not in (*RADIUS_LIMITS, "V"):
        raise ValueError("不支持的光谱型或光度级")
    if lclass == "V":
        rows = [r for r in MAIN_SEQUENCE if r["subtype"].startswith(stype)]
        # Kroupa 型质量先验加恒定形成率下的主序存活时间近似。
        # 每段质量跨度是积分的 Jacobian，避免把各温度或亚型当作等概率。
        weights = []
        for left, right in zip(rows, rows[1:]):
            mean_mass = math.sqrt(left["mass_solar"] * right["mass_solar"])
            imf = mean_mass ** -1.3 if mean_mass < .5 else .5 * mean_mass ** -2.3
            lifetime_fraction = min(1.0, max(.0003, mean_mass ** -2.5))
            weights.append(abs(right["mass_solar"] - left["mass_solar"]) * imf * lifetime_fraction)
        segment = int(rng.choice(len(weights), p=np.asarray(weights) / sum(weights)))
        teff = math.exp(rng.uniform(math.log(rows[segment]["temperature_K"]),
                                    math.log(rows[segment + 1]["temperature_K"])))
        mass, luminosity, bc = main_sequence_parameters(teff, stype)
        mbol = SUN_M_BOL - 2.5 * math.log10(luminosity)
        mv = mbol - bc
    else:
        if stype == "M" and lclass == "IV":
            raise CandidateRejected("本模型未标定 M 型次巨星")
        teff = float(rng.uniform(*TYPE_TEMP_RANGE[stype]))
        base, spread = EVOLVED_MV[stype][lclass]
        mv = float(rng.normal(base, spread / 2))
        if abs(mv - base) > spread:
            raise CandidateRejected("演化星绝对星等超出经验范围")
        if (lclass in ("0", "I") and mv > -3) or (lclass == "II" and mv > -1):
            raise CandidateRejected("光度级与绝对星等不符")
        bc = get_bolometric_correction(teff)
        mbol = mv + bc
        luminosity = 10 ** ((SUN_M_BOL - mbol) / 2.5)
        radius = math.sqrt(luminosity) / (teff / SUN_TEFF) ** 2
        rmin, rmax = RADIUS_LIMITS[lclass]
        if not rmin <= radius <= rmax:
            raise CandidateRejected("演化星半径超出经验范围")
        low, high = evolved_mass_range(stype, lclass)
        gmin, gmax = LOG_G_LIMITS[lclass]
        # 质量必须同时满足阶段先验、表面重力与光度约束。
        low = max(low, luminosity / EDDINGTON_PER_MASS,
                  radius ** 2 * 10 ** (gmin - LOG_G_SUN))
        high = min(high, radius ** 2 * 10 ** (gmax - LOG_G_SUN))
        if low > high:
            raise CandidateRejected("演化星的质量、光度与重力范围无交集")
        mass = float(rng.uniform(low, high))
    radius = math.sqrt(luminosity) / (teff / SUN_TEFF) ** 2
    return {"mass_solar": mass, "temperature_K": teff, "luminosity_solar": luminosity,
            "radius_solar": radius, "abs_mag": mv, "bolometric_mag": mbol,
            "bc_correction": bc}


def solve_distance(abs_mag, app_mag, av_per_kpc=0.7):
    """求解 m = M + 5 log10(d/10) + k d/1000，不截断距离或改变目标星等。"""
    if not all(math.isfinite(v) for v in (abs_mag, app_mag, av_per_kpc)) or av_per_kpc < 0:
        raise ValueError("星等必须有限，消光率必须非负")
    distance_without_extinction = 10 ** ((app_mag - abs_mag + 5) / 5)
    if av_per_kpc == 0:
        return distance_without_extinction
    low, high = 0.0, distance_without_extinction
    for _ in range(80):
        mid = (low + high) / 2
        calculated = abs_mag + 5 * math.log10(mid / 10) + av_per_kpc * mid / 1000
        if calculated < app_mag:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def cartesian_to_galactic(x, y, z):
    distance = math.hypot(x, y, z)
    if not math.isfinite(distance) or distance == 0:
        raise ValueError("方向向量必须有限且非零")
    return math.degrees(math.atan2(y, x)) % 360, math.degrees(math.asin(max(-1, min(1, z / distance))))


def uniform_sphere_sample(rng):
    longitude = float(rng.uniform(0, 2 * math.pi))
    sin_lat = float(rng.uniform(-1, 1))
    cos_lat = math.sqrt(max(0, 1 - sin_lat * sin_lat))
    return np.array([cos_lat * math.cos(longitude), cos_lat * math.sin(longitude), sin_lat])


def validate_star(star, *, app_mag_range=(-math.inf, 6.5), distance_range=(1.0, 10000.0),
                  av_per_kpc=0.7, dust_scale_height=120.0, observer_height=20.0,
                  expected_extinction=None, reference_parameters=None):
    """检查实际存储字段及它们的交叉关系；不依赖生成时的通过标记。"""
    errors = []
    if not isinstance(star, dict):
        return False, ["恒星必须是对象"]
    numeric = ("mass_solar", "temperature_K", "luminosity_solar", "radius_solar",
               "distance_pc", "dist_ly", "extinction_Av", "abs_mag", "bolometric_mag",
               "bc_correction", "app_mag", "gal_lon", "gal_lat",
               "angular_diameter_mas")
    for key in numeric:
        value = star.get(key)
        if not is_finite_number(value):
            errors.append(f"{key} 缺失或不是有限数值")
    stype, lclass = star.get("spectral_type"), star.get("luminosity_class")
    if not isinstance(stype, str) or stype not in TYPE_TEMP_RANGE or not isinstance(lclass, str) or lclass not in (*RADIUS_LIMITS, "V"):
        errors.append("未知光谱型或光度级")
    position = star.get("pos_cartesian")
    if not isinstance(position, (list, tuple)) or len(position) != 3 or any(
        not is_finite_number(v) for v in position
    ):
        errors.append("pos_cartesian 必须包含三个有限数值")
    if not isinstance(star.get("color_hex"), str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", star["color_hex"]):
        errors.append("颜色格式错误")
    if errors:
        return False, errors
    for key in ("mass_solar", "temperature_K", "luminosity_solar", "radius_solar",
                "distance_pc", "dist_ly", "angular_diameter_mas"):
        if star[key] <= 0:
            errors.append(f"{key} 必须为正数")
    if errors:
        return False, errors
    mass, temp, lum, radius = (star[k] for k in
        ("mass_solar", "temperature_K", "luminosity_solar", "radius_solar"))
    d, av, mv, mbol = (star[k] for k in ("distance_pc", "extinction_Av", "abs_mag", "bolometric_mag"))
    if not 0.07 <= mass <= 150 or not 2000 <= temp <= (300000 if reference_parameters else 60000):
        errors.append("质量或温度超出模型范围")
    if not 1e-6 <= lum <= 1e8 or not 0.01 <= radius <= 5000:
        errors.append("光度或半径超出模型范围")
    if not -50 <= mbol <= 50 or not -50 <= mv <= 50:
        errors.append("绝对星等超出模型范围")
    if errors:
        return False, errors
    if not distance_range[0] <= d <= distance_range[1]:
        errors.append("距离超出配置范围")
    if not app_mag_range[0] <= star["app_mag"] <= app_mag_range[1]:
        errors.append("视星等超出配置范围")
    if not 0 <= star["gal_lon"] < 360 or not -90 <= star["gal_lat"] <= 90:
        errors.append("银道坐标越界")
    if errors:
        return False, errors

    def check(actual, expected, label):
        if not math.isclose(actual, expected, rel_tol=2e-7, abs_tol=2e-7):
            errors.append(label)

    check(star["app_mag"], mv + 5 * math.log10(d / 10) + av, "距离模数不一致")
    check(av, expected_extinction if expected_extinction is not None else
          extinction_av(d, star["gal_lat"], av_per_kpc, dust_scale_height, observer_height),
          "消光与距离及视线方向不一致")
    check(mbol, mv + star["bc_correction"], "热光校正不一致")
    check(lum, 10 ** ((SUN_M_BOL - mbol) / 2.5), "热光星等与光度不一致")
    check(lum, radius ** 2 * (temp / SUN_TEFF) ** 4, "温度、半径与光度不一致")
    check(star["angular_diameter_mas"], SOLAR_DIAMETER_MAS_AT_PC * radius / d, "角直径不一致")
    check(star["dist_ly"], d * PC_TO_LY, "光年与秒差距不一致")
    norm = math.hypot(*position)
    check(norm, d, "三维坐标与距离不一致")
    if norm == 0:
        errors.append("位置不能为零向量")
    else:
        longitude, latitude = map(math.radians, (star["gal_lon"], star["gal_lat"]))
        expected = (math.cos(latitude) * math.cos(longitude),
                    math.cos(latitude) * math.sin(longitude), math.sin(latitude))
        if math.dist([v / norm for v in position], expected) > 1e-7:
            errors.append("三维坐标与银经银纬不一致")
    if reference_parameters is not None:
        for key,value in reference_parameters.items():
            check(star[key],value,f"等龄线参数不一致: {key}")
        return not errors, errors
    if lum > EDDINGTON_PER_MASS * mass * (1 + 1e-7):
        errors.append("光度超过该质量的电子散射筛选上限")
    if lclass == "V":
        try:
            expected_mass, expected_lum, expected_bc = main_sequence_parameters(temp, stype)
            if not math.isclose(mass, expected_mass, rel_tol=0.05):
                errors.append("主序星质量与温度不符")
            if abs(math.log10(lum / expected_lum)) > 0.08:
                errors.append("主序星光度与温度不符")
            if abs(star["bc_correction"] - expected_bc) > 0.02:
                errors.append("主序星热光校正与温度不符")
        except CandidateRejected as exc:
            errors.append(str(exc))
    else:
        tmin, tmax = TYPE_TEMP_RANGE[stype]
        rmin, rmax = RADIUS_LIMITS[lclass]
        mmin, mmax = evolved_mass_range(stype, lclass)
        gmin, gmax = LOG_G_LIMITS[lclass]
        log_g = LOG_G_SUN + math.log10(mass / radius ** 2)
        if not tmin <= temp <= tmax:
            errors.append("光谱型与温度不符")
        base, spread = EVOLVED_MV[stype][lclass]
        if not base - spread <= mv <= base + spread:
            errors.append("演化星绝对星等超出该阶段的经验范围")
        if not rmin <= radius <= rmax or not mmin <= mass <= mmax or not gmin <= log_g <= gmax:
            errors.append("演化星半径、质量或重力超出该阶段的范围")
        check(star["bc_correction"], get_bolometric_correction(temp), "演化星热光校正与温度不符")
        if (lclass in ("0", "I") and mv > -3) or (lclass == "II" and mv > -1):
            errors.append("光度级与绝对星等不符")
        if stype == "M" and lclass == "IV":
            errors.append("本模型未标定 M 型次巨星")
    return not errors, errors


def extinction_av(distance_pc, gal_lat, av_per_kpc=0.7, dust_scale_height=120.0,
                  observer_height=20.0):
    """沿视线积分指数尘埃盘；离开盘面后消光趋于饱和。

    av_per_kpc 是观测者位置的局部消光率；尘埃仅有垂直结构，无径向变化。
    分段解析积分处理穿越盘面的视线，expm1 保证小倾角精度。
    """
    if not all(math.isfinite(v) for v in (distance_pc, gal_lat, av_per_kpc,
                                          dust_scale_height, observer_height)):
        raise ValueError("消光参数必须有限")
    if distance_pc < 0 or av_per_kpc < 0 or dust_scale_height <= 0 or abs(gal_lat) > 90:
        raise ValueError("消光参数越界")
    slope = math.sin(math.radians(gal_lat))
    if abs(slope) < 1e-14:
        return av_per_kpc * distance_pc / 1000

    def segment(length, rate):
        if abs(rate * length) < 1e-8:
            return length * (1 - rate * length / 2 + (rate * length) ** 2 / 6)
        return -math.expm1(-rate * length) / rate

    crossing = -observer_height / slope
    if 0 < crossing < distance_pc:
        towards_plane = -abs(slope) / dust_scale_height
        column = segment(crossing, towards_plane)
        column += math.exp(abs(observer_height) / dust_scale_height) * segment(
            distance_pc - crossing, abs(slope) / dust_scale_height)
    else:
        side = math.copysign(1, observer_height if observer_height else slope)
        column = segment(distance_pc, side * slope / dust_scale_height)
    return av_per_kpc * column / 1000
