#!/usr/bin/env python3
"""从盘状恒星总体生成背景恒星，再按观测亮度筛选可见星表。"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import secrets
import tempfile

import numpy as np

from paths import DATA_DIR, OUTPUT_DIR, SRC_DIR
from stellar_physics import (
    CandidateRejected, COLOR_MAP, EVOLVED_MV, MAIN_SEQUENCE, PC_TO_LY,
    REFERENCE_PATH, SOLAR_DIAMETER_MAS_AT_PC, cartesian_to_galactic,
    extinction_av, is_finite_number, sample_stellar_parameters, validate_star,
)

BASE_DIR = SRC_DIR
GENERATOR_VERSION = "4.3"
LUMINOSITY_CLASSES = ("V", "IV", "III", "II", "I", "0")
POPULATION_PATH = DATA_DIR / "population_profile.json"
POPULATION_PROFILE = json.loads(POPULATION_PATH.read_text(encoding="utf-8"))
CALIBRATED_LOCAL_DENSITY = POPULATION_PROFILE["default_local_density_per_pc3"]
# 保留 BSC5 的相对群体丰度；将无条件期望移到所需数量区间附近。
DEFAULT_COUNT_TARGET = 9250
DEFAULT_LOCAL_DENSITY = CALIBRATED_LOCAL_DENSITY * DEFAULT_COUNT_TARGET / POPULATION_PROFILE["reference_count"]

# 这些是离线拟合得到的总体丰度；可见星数量仍由 Poisson 空间抽样与亮度筛选产生。
_components = POPULATION_PROFILE["components"]
if (not math.isclose(sum(c["fraction"] for c in _components), 1.0, abs_tol=1e-12)
        or any(not is_finite_number(c["fraction"]) or c["fraction"] <= 0
               or not is_finite_number(c["scale_height_pc"]) or c["scale_height_pc"] <= 0
               for c in _components)):
    raise ValueError("恒星总体校准文件中的丰度或尺度高度无效")


@dataclass(frozen=True)
class GenerationConfig:
    seed: int = 20260913
    limiting_magnitude: float = 6.5
    min_distance_pc: float = 1.0
    max_distance_pc: float = 10000.0
    local_density_per_pc3: float = DEFAULT_LOCAL_DENSITY
    observer_radius_pc: float = 30712.0 / PC_TO_LY
    observer_height_pc: float = 20.0
    radial_scale_pc: float = 2600.0
    av_per_kpc: float = 0.7
    dust_scale_height_pc: float = 120.0
    batch_size: int = 20000
    max_candidate_points: int = 20000000
    minimum_visible_stars: int = 9000
    maximum_visible_stars: int | None = 9500
    max_catalog_attempts: int = 8

    def __post_init__(self):
        values = asdict(self)
        if any(not is_finite_number(v) for k, v in values.items()
               if not (k == "maximum_visible_stars" and v is None)):
            raise ValueError("生成参数必须为有限数值")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("随机种子必须是非负整数")
        if not -10 <= self.limiting_magnitude <= 15:
            raise ValueError("极限视星等范围为 -10 到 15")
        for name in ("min_distance_pc", "max_distance_pc", "local_density_per_pc3",
                     "radial_scale_pc", "dust_scale_height_pc", "batch_size", "max_candidate_points"):
            if values[name] <= 0:
                raise ValueError(f"{name} 必须大于零")
        if self.min_distance_pc >= self.max_distance_pc or self.av_per_kpc < 0 or self.observer_radius_pc < 0:
            raise ValueError("距离或消光参数越界")
        if not isinstance(self.batch_size, int) or not isinstance(self.max_candidate_points, int):
            raise ValueError("批量大小与候选点预算必须为整数")
        if abs(self.observer_height_pc) > 1000 or self.max_distance_pc > 50000:
            raise ValueError("当前简化模型支持高度 ±1000 pc、距离不超过 50000 pc")
        for key in ("minimum_visible_stars", "max_catalog_attempts"):
            if not isinstance(values[key], int) or isinstance(values[key], bool):
                raise ValueError(f"{key} 必须为整数")
        if self.minimum_visible_stars < 0 or not 1 <= self.max_catalog_attempts <= 100:
            raise ValueError("最低数量必须非负；整表尝试上限为 1–100")
        if self.maximum_visible_stars is not None and (
                not isinstance(self.maximum_visible_stars, int)
                or self.maximum_visible_stars < self.minimum_visible_stars):
            raise ValueError("最高数量必须为不低于最低数量的整数，或 None")

    def validation_options(self):
        return {"app_mag_range": (-math.inf, self.limiting_magnitude),
                "distance_range": (self.min_distance_pc, self.max_distance_pc),
                "av_per_kpc": self.av_per_kpc, "dust_scale_height": self.dust_scale_height_pc,
                "observer_height": self.observer_height_pc}


def population_components(config):
    return [{"spectral_type": c["spectral_type"], "luminosity_class": c["luminosity_class"],
             "local_density": config.local_density_per_pc3 * c["fraction"],
             "scale_height_pc": c["scale_height_pc"]} for c in POPULATION_PROFILE["components"]]


def reference_comparison(catalog):
    """同星等的地球参照比较；与用户要求的数量条件分别记录。"""
    metadata = catalog["metadata"]
    actual = metadata["generation_parameters"]
    calibrated = POPULATION_PROFILE["calibrated_parameters"]
    calibrated_match = (metadata.get("population_profile_version") == POPULATION_PROFILE["profile_version"]
                  and all(math.isclose(actual[key], value, rel_tol=1e-10, abs_tol=1e-12)
                          for key, value in calibrated.items()))
    if not math.isclose(actual["limiting_magnitude"], 6.5, abs_tol=1e-12):
        return {"applicable": False, "reason": "极限星等与 BSC5 V <= 6.5 统计不同"}
    target = POPULATION_PROFILE["reference_count"]
    difference = (len(catalog["stars"]) - target) / target
    tolerance = POPULATION_PROFILE["count_tolerance_fraction"]
    return {"applicable": True, "reference": "BSC5 V <= 6.5", "reference_count": target,
            "relative_difference": difference, "tolerance_fraction": tolerance,
            "calibration_parameters_match": calibrated_match,
            "within_tolerance": abs(difference) <= tolerance}


def count_in_range(count, config):
    return (count >= config.minimum_visible_stars
            and (config.maximum_visible_stars is None or count <= config.maximum_visible_stars))


class CountConstraintError(RuntimeError):
    def __init__(self, attempts, config):
        self.attempts = attempts
        super().__init__(f"{len(attempts)} 次完整生成均未满足可见数量 "
                         f"[{config.minimum_visible_stars}, {config.maximum_visible_stars}]；"
                         f"实际数量 {[a['visible_count'] for a in attempts]}，未保存星表")


def visibility_horizon(stype, lclass, limiting_magnitude, maximum_distance):
    """只裁去即使无消光、取该类最亮值也绝不可能看见的体积。"""
    if lclass == "V":
        from stellar_physics import SUN_M_BOL
        # Mv 在各段 log(Teff) 插值上为线性，最小值必在参考节点。
        brightest_mv = min(SUN_M_BOL - 2.5 * row["log_luminosity_solar"] - row["bc_correction"]
                           for row in MAIN_SEQUENCE if row["subtype"].startswith(stype))
    else:
        mean, spread = EVOLVED_MV[stype][lclass]
        brightest_mv = mean - spread
    return min(maximum_distance, 10 ** ((limiting_magnitude - brightest_mv + 5) / 5))


def spatial_density(position, component, config):
    x, y, z = np.moveaxis(np.asarray(position), -1, 0)
    # 观测者原点，+x 指向银心，+z 指向银北极。
    radius = np.hypot(config.observer_radius_pc - x, y)
    height = config.observer_height_pc + z
    return component["local_density"] * np.exp(
        (config.observer_radius_pc - radius) / config.radial_scale_pc
        + (abs(config.observer_height_pc) - np.abs(height)) / component["scale_height_pc"])


def spatial_batches(component, horizon, config, rng):
    if horizon <= config.min_distance_pc:
        return
    nearest_radius = max(0.0, config.observer_radius_pc - horizon)
    nearest_height = max(0.0, abs(config.observer_height_pc) - horizon)
    bound = component["local_density"] * math.exp(
        (config.observer_radius_pc - nearest_radius) / config.radial_scale_pc
        + (abs(config.observer_height_pc) - nearest_height) / component["scale_height_pc"])
    volume = 4 * math.pi / 3 * (horizon ** 3 - config.min_distance_pc ** 3)
    mean_candidates = bound * volume
    if mean_candidates > config.max_candidate_points:
        raise RuntimeError("候选恒星数量超过预算；请缩小体积或提高 --max-candidates，未保存不完整星表")
    candidate_count = int(rng.poisson(mean_candidates))
    if candidate_count > config.max_candidate_points:
        raise RuntimeError("候选点数超过预算，未保存不完整星表")
    for start in range(0, candidate_count, config.batch_size):
        n = min(config.batch_size, candidate_count - start)
        distances = np.cbrt(rng.uniform(config.min_distance_pc ** 3, horizon ** 3, n))
        longitude = rng.uniform(0, 2 * math.pi, n)
        sin_lat = rng.uniform(-1, 1, n)
        radial = distances * np.sqrt(1 - sin_lat ** 2)
        positions = np.column_stack((radial * np.cos(longitude), radial * np.sin(longitude), distances * sin_lat))
        acceptance = spatial_density(positions, component, config) / bound
        if np.any(acceptance > 1 + 1e-10):
            raise RuntimeError("空间密度上界错误")
        yield n, positions[rng.random(n) < acceptance]


def make_star(parameters, position, stype, lclass, config):
    distance = math.hypot(*position)
    longitude, latitude = cartesian_to_galactic(*position)
    av = extinction_av(distance, latitude, config.av_per_kpc,
                       config.dust_scale_height_pc, config.observer_height_pc)
    star = {"id": "", "spectral_type": stype, "luminosity_class": lclass, **parameters,
            "distance_pc": distance, "dist_ly": distance * PC_TO_LY, "extinction_Av": av,
            "app_mag": parameters["abs_mag"] + 5 * math.log10(distance / 10) + av,
            "color_hex": COLOR_MAP[stype], "gal_lon": longitude, "gal_lat": latitude,
            "pos_cartesian": [float(v) for v in position],
            "angular_diameter_mas": SOLAR_DIAMETER_MAS_AT_PC * parameters["radius_solar"] / distance}
    return star


def validate_catalog(catalog):
    errors = []
    try:
        metadata = catalog["metadata"]
        parameters = dict(metadata["generation_parameters"])
        count_fields = ("minimum_visible_stars", "maximum_visible_stars", "max_catalog_attempts")
        if metadata.get("generator_version") in (GENERATOR_VERSION, "4.4"):
            if any(key not in parameters for key in count_fields):
                raise ValueError("新星表缺少数量约束参数")
        else:
            # 历史星表没有新数量合同，不能套用新默认值而误判旧成品。
            parameters.setdefault("minimum_visible_stars", 0)
            parameters.setdefault("maximum_visible_stars", None)
            parameters.setdefault("max_catalog_attempts", 1)
        config = GenerationConfig(**parameters)
        stars = catalog["stars"]
        generation_id = metadata["generation_id"]
        if not isinstance(stars, list) or not isinstance(generation_id, str) or not generation_id:
            raise ValueError("无效星表结构")
    except (KeyError, TypeError, ValueError) as exc:
        return {"checked": 0, "passed": 0, "failed": 0, "all_passed": False,
                "errors": [{"id": None, "reasons": [f"元数据错误: {exc}"]}]}
    identifiers = set()
    failed = 0
    dust = grid = None
    if metadata.get('generator_version') == '4.4':
        try:
            from galaxy_environment import GalacticDust
            from cluster_population import Isochrones
            dust=GalacticDust.from_dict(catalog['galaxy'])
            grid=Isochrones()
            if (grid.sha256 != metadata['isochrone_sha256'] or
                    not math.isclose(dust.parameters.observer_radius_pc,config.observer_radius_pc,abs_tol=1e-10)):
                raise ValueError('银河坐标或等龄线来源不一致')
            expected_av=dust.extinction([s['pos_cartesian'] for s in stars])
        except (KeyError, TypeError, ValueError) as exc:
            return {'checked':len(stars),'passed':0,'failed':len(stars),'all_passed':False,
                    'errors':[{'id':None,'reasons':[f'银河模型无效: {exc}']}]}
    for index,star in enumerate(stars):
        options=config.validation_options()
        if dust is not None:
            options['expected_extinction']=float(expected_av[index])
            if star.get('cluster_id'):
                try:
                    options['reference_parameters']=grid.parameters(star['metallicity_mh'],star['log_age'],star['initial_mass_solar'])
                except (KeyError, TypeError, ValueError) as exc:
                    errors.append({'id':star.get('id'),'reasons':[f'等龄线成员无效: {exc}']})
                    failed+=1
                    continue
        valid, reasons = validate_star(star, **options)
        identifier = star.get("id") if isinstance(star, dict) else None
        if not isinstance(identifier, str) or not identifier.startswith(generation_id + "_"):
            reasons.append("恒星 ID 与批次不一致")
        if isinstance(identifier, str) and identifier in identifiers:
            reasons.append("恒星 ID 重复")
        if isinstance(identifier, str):
            identifiers.add(identifier)
        if reasons:
            failed += 1
            errors.append({"id": identifier, "reasons": reasons})
    if metadata.get("count") != len(stars):
        errors.append({"id": None, "reasons": ["元数据数量与实际数量不一致"]})
    if not count_in_range(len(stars), config):
        errors.append({"id": None, "reasons": ["可见恒星数量未满足配置的硬性范围"]})
    if metadata.get("generator_version") in (GENERATOR_VERSION, "4.4"):
        try:
            selection = metadata["count_selection"]
            mode = "whole_catalog_rejection" if (config.minimum_visible_stars
                    or config.maximum_visible_stars is not None) else "unconditioned"
            if any(selection[key] != expected for key, expected in (
                    ("minimum", config.minimum_visible_stars), ("maximum", config.maximum_visible_stars),
                    ("max_attempts", config.max_catalog_attempts), ("mode", mode))):
                raise ValueError("数量记录与配置不符")
            attempts = selection["attempts"]
            if (not isinstance(attempts, list) or not 1 <= len(attempts) <= config.max_catalog_attempts
                    or selection["accepted_attempt"] != len(attempts)):
                raise ValueError("尝试次数无效")
            for index, attempt in enumerate(attempts):
                count = attempt["visible_count"]
                if (not isinstance(count, int) or isinstance(count, bool) or count < 0
                        or attempt["attempt"] != index + 1 or attempt["spawn_key"] != [index]
                        or attempt["accepted"] is not (index == len(attempts) - 1)
                        or count_in_range(count, config) != attempt["accepted"]):
                    raise ValueError("尝试记录或随机子流不一致")
            if attempts[-1]["visible_count"] != len(stars):
                raise ValueError("接受数量与实际星表不符")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append({"id": None, "reasons": [f"数量条件抽样记录无效: {exc}"]})
    stats = metadata.get("generation_stats", {})
    if (stats.get("population_stars_sampled") != stats.get("not_visible", -1) + len(stars)
            or sum(c.get("visible", 0) for c in stats.get("components", [])) != len(stars)
            or sum(c.get("sampled", 0) for c in stats.get("components", [])) != stats.get("population_stars_sampled")):
        errors.append({"id": None, "reasons": ["总体、不可见、可见数量记账不一致"]})
    if dust is not None:
        from deep_sky import validate_deep_sky
        errors.extend({'id':None,'reasons':[e]} for e in validate_deep_sky(catalog,grid))
    return {"checked": len(stars), "passed": len(stars) - failed, "failed": failed,
            "all_passed": not errors, "errors": errors}


def _sample_population(config, rng, generation_id, progress):
    """一次完整无条件总体实现；任何资源或物理错误均直接失败。"""
    components = population_components(config)
    stars = []
    stats = {"candidate_points": 0, "population_stars_sampled": 0, "not_visible": 0,
             "physical_draws_rejected": 0, "components": []}
    for component in components:
        stype, lclass = component["spectral_type"], component["luminosity_class"]
        horizon = visibility_horizon(stype, lclass, config.limiting_magnitude, config.max_distance_pc)
        sampled, visible = 0, 0
        for candidates, positions in spatial_batches(component, horizon, config, rng):
            stats["candidate_points"] += candidates
            if stats["candidate_points"] > config.max_candidate_points:
                raise RuntimeError("候选点总数超过预算，未保存不完整星表")
            for position in positions:
                for _ in range(1000):
                    try:
                        parameters = sample_stellar_parameters(stype, lclass, rng)
                        break
                    except CandidateRejected:
                        stats["physical_draws_rejected"] += 1
                else:
                    raise RuntimeError(f"{stype}{lclass} 无法生成有效物理参数，未保存不完整星表")
                star = make_star(parameters, position, stype, lclass, config)
                sampled += 1
                if star["app_mag"] > config.limiting_magnitude:
                    stats["not_visible"] += 1
                    continue
                star["id"] = f"{generation_id}_{len(stars) + 1:06d}"
                valid, reasons = validate_star(star, **config.validation_options())
                if not valid:
                    raise RuntimeError(f"生成器产生不一致的数据: {reasons}")
                stars.append(star)
                visible += 1
        stats["population_stars_sampled"] += sampled
        stats["components"].append({**component, "horizon_pc": horizon, "sampled": sampled, "visible": visible})
        if progress:
            progress(f"{stype}{lclass}: 总体样本 {sampled}，可见 {visible}")
    return stars, stats


def generate_catalog(config, *, generation_id=None, progress=print):
    generation_id = generation_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    attempts = []
    for index in range(config.max_catalog_attempts):
        sequence = np.random.SeedSequence(config.seed, spawn_key=(index,))
        stars, stats = _sample_population(config, np.random.default_rng(sequence), generation_id, progress)
        accepted = count_in_range(len(stars), config)
        attempts.append({"attempt": index + 1, "spawn_key": list(sequence.spawn_key),
                         "visible_count": len(stars), "accepted": accepted})
        if accepted:
            break
        if progress:
            progress(f"完整尝试 {index + 1}: {len(stars)} 颗不在数量范围内，重新生成整个总体。")
    else:
        raise CountConstraintError(attempts, config)
    components = population_components(config)
    catalog = {"metadata": {"schema_version": 2, "generator_version": GENERATOR_VERSION,
                "generation_id": generation_id, "count": len(stars), "coordinate_system": "galactic",
                "generation_parameters": asdict(config), "random_generator": "numpy.PCG64",
                "numpy_version": np.__version__, "population_model": "bsc5_calibrated_exponential_disk_v1",
                "population_profile_version": POPULATION_PROFILE["profile_version"],
                "population_profile_sha256": hashlib.sha256(POPULATION_PATH.read_bytes()).hexdigest(),
                "brightness_reference_sha256": POPULATION_PROFILE["reference_sha256"],
                "population_components": components, "extinction_model": "exponential_vertical_dust_v1",
                "stellar_models": {"V": "Mamajek mean dwarf sequence 2022.04.16",
                                   "evolved": "feasible temperature/Mv draws, then uniform mass within their allowed interval"},
                "main_sequence_sampling": "piecewise Kroupa IMF with approximate main-sequence survival weighting",
                "sampling_scope": "Per-component visibility horizons; unobservable outer volumes are omitted analytically.",
                "main_sequence_reference_sha256": hashlib.sha256(REFERENCE_PATH.read_bytes()).hexdigest(),
                "source_sha256": {name: hashlib.sha256((SRC_DIR / name).read_bytes()).hexdigest()
                                  for name in ("star_generator.py", "stellar_physics.py")},
                "count_selection": {"mode": "whole_catalog_rejection" if
                                    config.minimum_visible_stars or config.maximum_visible_stars is not None
                                    else "unconditioned",
                                    "minimum": config.minimum_visible_stars,
                                    "maximum": config.maximum_visible_stars,
                                    "max_attempts": config.max_catalog_attempts,
                                    "accepted_attempt": len(attempts), "attempts": attempts},
                "generation_stats": stats,
                "note": "Visibility-selected synthetic catalog; population priors and evolved-star parameters are approximate."},
               "stars": stars}
    validation = validate_catalog(catalog)
    if not validation["all_passed"]:
        raise RuntimeError(f"最终验证失败: {validation['errors'][:3]}")
    catalog["metadata"]["validation_stats"] = {k: v for k, v in validation.items() if k != "errors"}
    catalog["metadata"]["reference_comparison"] = reference_comparison(catalog)
    return catalog


def _catalog_present(folder):
    gid = folder.name.removeprefix("output_")
    return (folder / f"star_map_{gid}.json").is_file() or (folder / f"sky_view_{gid}.json").is_file()


def generate_folders_json(root=OUTPUT_DIR):
    root = Path(root)
    folders = sorted((p.name for p in root.glob("output_*") if p.is_dir() and _catalog_present(p)), reverse=True)
    temporary = root / ".folders.json.tmp"
    temporary.write_text(json.dumps(folders, indent=2) + "\n", encoding="utf-8")
    temporary.replace(root / "folders.json")
    return folders


def write_report(catalog, output_dir, plots=True):
    metadata, stars = catalog["metadata"], catalog["stars"]
    gid = metadata["generation_id"]
    validation = validate_catalog(catalog)
    lines = ["恒星生成验证报告", f"批次: {gid}", f"可见恒星: {len(stars)}",
             f"通过: {validation['passed']}，失败: {validation['failed']}",
             f"各类可见地平线内的总体样本: {metadata['generation_stats']['population_stars_sampled']}",
             "地平线外必不可见的体积已解析裁除；样本数不等于整个星系的恒星总数。",
             "说明: 通过表示满足已实现的数值与经验约束，不代表完整恒星演化验证。"]
    if stars:
        lines += [f"最远恒星: {max(s['distance_pc'] for s in stars):.3f} pc",
                  f"视星等范围: {min(s['app_mag'] for s in stars):.3f} 至 {max(s['app_mag'] for s in stars):.3f}"]
    comparison = reference_comparison(catalog)
    if comparison["applicable"]:
        lines += [f"BSC5 同星等参照: {comparison['reference_count']} 颗",
                  f"相对地球参照的数量偏差: {comparison['relative_difference']:+.2%}"]
    else:
        lines.append("极限星等不同，不应用地球 6.5 等数量参照。")
    selection = metadata.get("count_selection")
    if selection:
        lines += [f"数量合同: [{selection['minimum']}, {selection['maximum']}]；模式: {selection['mode']}",
                  f"整表尝试数量: {[a['visible_count'] for a in selection['attempts']]}",
                  "接受整份实现，因此输出服从数量条件下的模型；没有补星、删星或修改视星等。"]
    (output_dir / f"validation_report_{gid}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not plots or not stars:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 8))
    for lclass in LUMINOSITY_CLASSES:
        sample = [s for s in stars if s["luminosity_class"] == lclass]
        if sample:
            ax.scatter([s["temperature_K"] for s in sample], [s["abs_mag"] for s in sample],
                       c=[s["color_hex"] for s in sample], edgecolors="#444444", linewidths=.2,
                       s=8, alpha=.6, label=lclass)
    ax.set(xscale="log", xlabel="Temperature (K)", ylabel="Absolute visual magnitude",
           title="Visible stars: empirical stellar sequences")
    ax.invert_xaxis(); ax.invert_yaxis(); ax.legend(title="Luminosity class"); ax.grid(alpha=.2)
    fig.savefig(output_dir / f"validation_hr_diagram_{gid}.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter([s["distance_pc"] for s in stars], [s["app_mag"] for s in stars], s=3, alpha=.3)
    ax.axhline(metadata["generation_parameters"]["limiting_magnitude"], color="green", label="Visibility limit")
    ax.set(xlabel="Distance (pc)", ylabel="Apparent visual magnitude", title="Visibility selection")
    ax.invert_yaxis(); ax.legend(); ax.grid(alpha=.2)
    fig.savefig(output_dir / f"validation_dist_mag_{gid}.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def save_catalog(catalog, output_root=OUTPUT_DIR, *, plots=True):
    validation = validate_catalog(catalog)
    if not validation["all_passed"]:
        raise ValueError(f"拒绝保存无效星表: {validation['errors'][:3]}")
    # 重新计算报告，不能信任调用者传入的通过标记。
    catalog["metadata"]["validation_stats"] = {k: v for k, v in validation.items() if k != "errors"}
    catalog["metadata"]["reference_comparison"] = reference_comparison(catalog)
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    gid = catalog["metadata"]["generation_id"]
    if not gid.replace("_", "").isalnum():
        raise ValueError("批次 ID 只能包含字母、数字和下划线")
    destination = root / f"output_{gid}"
    if destination.exists():
        raise FileExistsError(f"已有数据集，拒绝覆盖: {destination}")
    with tempfile.TemporaryDirectory(prefix=".starmap-", dir=root) as temporary:
        folder = Path(temporary) / destination.name
        folder.mkdir()
        (folder / f"star_map_{gid}.json").write_text(json.dumps(catalog, indent=2, ensure_ascii=False,
                                                               allow_nan=False) + "\n", encoding="utf-8")
        write_report(catalog, folder, plots)
        folder.rename(destination)
    generate_folders_json(root)
    return destination / f"star_map_{gid}.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, help="可复现的随机种子；省略时生成并记录新种子")
    parser.add_argument("--limit-mag", type=float, default=6.5, help="可见极限星等；不限制更亮的恒星")
    parser.add_argument("--max-distance", type=float, default=10000, help="背景总体边界，单位 pc")
    parser.add_argument("--min-distance", type=float, default=1, help="宿主恒星系统之外的内边界，单位 pc")
    parser.add_argument("--density", type=float, default=DEFAULT_LOCAL_DENSITY,
                        help=f"观测者附近的恒星密度，颗/pc³；校准默认值 {DEFAULT_LOCAL_DENSITY:.6f}")
    parser.add_argument("--av-per-kpc", type=float, default=.7)
    parser.add_argument("--observer-radius", type=float, default=30712.0 / PC_TO_LY,
                        help="距银心的距离，pc；默认采用设定中的 30712 光年")
    parser.add_argument("--max-candidates", type=int, default=20000000)
    parser.add_argument("--min-visible", type=int, default=9000, help="成功输出的最低可见数量")
    parser.add_argument("--max-visible", type=int, default=9500, help="成功输出的最高可见数量")
    parser.add_argument("--max-attempts", type=int, default=8, help="整份总体重抽上限；失败不保存")
    parser.add_argument("--unconditioned", action="store_true", help="研究用：显式关闭可见数量条件")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--validate", type=Path, help="仅检查已有星表，不生成或修改数据")
    args = parser.parse_args()
    if args.validate:
        result = validate_catalog(json.loads(args.validate.read_text(encoding="utf-8")))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["all_passed"] else 1
    config = GenerationConfig(seed=args.seed if args.seed is not None else secrets.randbits(64),
                              limiting_magnitude=args.limit_mag, min_distance_pc=args.min_distance,
                              max_distance_pc=args.max_distance, local_density_per_pc3=args.density,
                              observer_radius_pc=args.observer_radius,
                              av_per_kpc=args.av_per_kpc, max_candidate_points=args.max_candidates,
                              minimum_visible_stars=0 if args.unconditioned else args.min_visible,
                              maximum_visible_stars=None if args.unconditioned else args.max_visible,
                              max_catalog_attempts=1 if args.unconditioned else args.max_attempts)
    print(f"随机种子: {config.seed}；从盘状恒星总体筛选 m ≤ {config.limiting_magnitude} 的恒星")
    catalog = generate_catalog(config)
    path = save_catalog(catalog, args.output_root, plots=not args.no_plots)
    print(f"保存 {len(catalog['stars'])} 颗可见恒星: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
