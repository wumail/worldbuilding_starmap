"""用公开 BSC5 统计及独立生成样本，离线校准恒星总体的空间丰度。

此工具只调整总体密度。运行时生成器不读取观测恒星的位置，不按可见数量补星或删星。
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re

TYPES = "OBAFGKM"
CLASSES = ("V", "IV", "III", "II", "I", "0")
REFERENCE_URL = "https://cdsarc.cds.unistra.fr/ftp/cats/V/50/catalog.gz"
PREFIX = re.compile(r"^(sg|sd|g|d|c):?")
COMPOSITE = re.compile(r"\+\s*(?:(?:sg|sd|g|d|c):?)?[OBAFGKMCSWN]")
LCLASS = re.compile(r"(?<![A-Z])(VII|III|VI|IV|II|V|I)(?:ab|a|b)?")
LCLASS_RANGE = re.compile(r"\s*[-/]\s*(?:VII|III|VI|IV|II|V|I)")


def classify_spectrum(spectrum):
    """粗光谱取首分量；仅用明确光度级及历史 g/d 前缀估计阶段比例。

    复合谱、级别范围、问号、冒号和括号中的特殊分类不强行指定光度级。
    M2+III 的 + 是亚型记号；K0IIIbCN-0.5 的负号不代表光度级范围。
    """
    prefix = PREFIX.match(spectrum)
    primary = spectrum[prefix.end():] if prefix else spectrum
    stype = primary[0] if primary and primary[0] in TYPES else None
    if stype is None or COMPOSITE.search(spectrum) or any(mark in spectrum for mark in ":?()"):
        return stype, None
    match = LCLASS.search(primary)
    if match:
        if LCLASS_RANGE.match(primary[match.end():]) or match[1] not in CLASSES:
            return stype, None
        return stype, match[1]
    # sg/sd/c 的历史语义不作为本模型精确演化阶段；仍计入光谱总数。
    legacy_class = {"g": "III", "d": "V"}.get(prefix[1]) if prefix else None
    return stype, legacy_class


def summarize_reference(path):
    compressed = Path(path).read_bytes()
    raw = gzip.decompress(compressed)
    rows = raw.decode("ascii").splitlines()
    stars = []
    identifiers = set()
    for row in rows:
        identifier = int(row[:4])
        if identifier in identifiers:
            raise ValueError("参考目录 HR 重复")
        identifiers.add(identifier)
        if not row[102:107].strip():
            continue
        stars.append((float(row[102:107]), row[127:147].strip(), float(row[96:102])))
    if len(rows) != 9110 or len(stars) != 9096:
        raise ValueError("输入与 BSC5 目录的版本或字段布局不符")
    visible = [row for row in stars if row[0] <= 6.5]
    types = Counter()
    known = {stype: Counter() for stype in TYPES}
    unknown = Counter()
    for _, spectrum, _ in visible:
        stype, lclass = classify_spectrum(spectrum)
        if stype is None:
            continue
        types[stype] += 1
        if lclass is None:
            unknown[stype] += 1
        else:
            known[stype][lclass] += 1
    return {
        "catalog": "Bright Star Catalogue, 5th Revised Edition (Hoffleit & Warren, 1991)",
        "source_url": REFERENCE_URL,
        "source_sha256": hashlib.sha256(compressed).hexdigest(),
        "decompressed_sha256": hashlib.sha256(raw).hexdigest(),
        "catalog_entries": len(rows), "stellar_entries": len(stars),
        "limiting_magnitude": 6.5, "visible_count": len(visible),
        "modeled_type_count": sum(types.values()),
        "unmodeled_type_count": len(visible) - sum(types.values()),
        "spectral_counts": dict(types),
        "classified_luminosity_counts": {k: dict(v) for k, v in known.items()},
        "unclassified_luminosity_counts": dict(unknown),
        "cumulative_magnitude_counts": {str(m): sum(row[0] <= m for row in stars)
                                         for m in (0, 1, 2, 3, 4, 5, 6, 6.5)},
        "within_galactic_latitude_10_deg": sum(abs(row[2]) <= 10 for row in visible),
        "classification_method": "Primary broad spectral type; unambiguous Roman class or legacy g/d prefix. Composite, range and uncertain classes remain unclassified.",
        "scope": "Catalog entries, including unresolved systems; historical photometry, variability and completeness limit a literal naked-eye count.",
    }


def fit_population(reference, pilots, prior_strength=5.0):
    """N_visible 对总体密度线性；用训练样本估计有效可见体积，再求密度。

    缺失光度级按同光谱型中已分类样本的比例估计；5 颗的弱先验避免小样本零值
    永久删除一个物理群体。M 矮星太少，保留其近邻密度先验；I/0 合并拟合。
    """
    if not pilots or prior_strength <= 0:
        raise ValueError("需要训练样本及正的先验强度")
    expected_parameters = {"limiting_magnitude": 6.5, "min_distance_pc": 1.0,
        "max_distance_pc": 10000.0, "local_density_per_pc3": 0.1,
        "observer_radius_pc": 8200.0, "observer_height_pc": 20.0,
        "radial_scale_pc": 2600.0, "av_per_kpc": 0.7, "dust_scale_height_pc": 120.0}
    for pilot in pilots:
        if any(pilot["generation_parameters"].get(k) != v for k, v in expected_parameters.items()):
            raise ValueError("训练样本不对应本次校准的观测参数")
        if pilot["stellar_physics_sha256"] != pilots[0]["stellar_physics_sha256"]:
            raise ValueError("训练样本的恒星物理模型不一致")
        if sum(c["visible"] for c in pilot["components"]) != pilot["count"]:
            raise ValueError("训练样本的可见数不一致")
    baseline = pilots[0]["components"]
    means = {}
    for component in baseline:
        key = component["spectral_type"], component["luminosity_class"]
        samples = []
        for pilot in pilots:
            matches = [c for c in pilot["components"]
                       if (c["spectral_type"], c["luminosity_class"]) == key]
            if len(matches) != 1:
                raise ValueError("训练样本的恒星群体不一致")
            item = matches[0]
            for field in ("local_density", "scale_height_pc", "horizon_pc"):
                if item[field] != component[field]:
                    raise ValueError("训练样本的物理或空间参数不一致")
            samples.append(item["visible"])
        mean = sum(samples) / len(samples)
        if mean <= 0:
            raise ValueError(f"{key} 的训练可见样本不足")
        means[key] = mean
    fitted = []
    for component in baseline:
        stype, lclass = component["spectral_type"], component["luminosity_class"]
        group = "I" if lclass == "0" else lclass
        prior = Counter()
        for (st, lc), mean in means.items():
            if st == stype:
                prior["I" if lc == "0" else lc] += mean
        classified = reference["classified_luminosity_counts"][stype]
        known_count = sum(classified.values())
        probability = (classified.get(group, 0) + prior_strength * prior[group] / sum(prior.values())) / (known_count + prior_strength)
        expected = reference["spectral_counts"][stype] * probability
        if group == "I":
            expected *= means[stype, lclass] / prior["I"]
        if (stype, lclass) == ("M", "V"):
            expected = means[stype, lclass]
        density = component["local_density"] * expected / means[stype, lclass]
        fitted.append({"spectral_type": stype, "luminosity_class": lclass,
                       "scale_height_pc": component["scale_height_pc"],
                       "local_density_per_pc3": density,
                       "pilot_mean_visible": means[stype, lclass],
                       "estimated_visible": expected})
    total_density = sum(c["local_density_per_pc3"] for c in fitted)
    for c in fitted:
        c["fraction"] = c["local_density_per_pc3"] / total_density
    return {
        "profile_version": "bsc5_v1", "reference_file": "bsc5_reference.json",
        "reference_sha256": reference["source_sha256"],
        "default_local_density_per_pc3": total_density,
        "calibrated_parameters": {"limiting_magnitude": 6.5, "min_distance_pc": 1.0,
            "max_distance_pc": 10000.0, "local_density_per_pc3": total_density,
            "observer_radius_pc": 8200.0, "observer_height_pc": 20.0,
            "radial_scale_pc": 2600.0, "av_per_kpc": 0.7, "dust_scale_height_pc": 120.0},
        "reference_count": reference["visible_count"], "count_tolerance_fraction": 0.05,
        "pilot_seeds": [p["seed"] for p in pilots], "prior_strength": prior_strength,
        "pilot_stellar_physics_sha256": pilots[0]["stellar_physics_sha256"],
        "estimated_visible_count": sum(c["estimated_visible"] for c in fitted),
        "fit_method": "Per-component density scaled by observed counts / pilot effective visible volume. Unclassified luminosity classes use the classified distribution within the same spectral type with a weak prior. M dwarfs retain their local-density prior; I and 0 share calibration.",
        "components": fitted,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path, help="原始 BSC5 catalog.gz")
    parser.add_argument("pilots", type=Path, help="相同旧模型下的训练样本统计 JSON")
    parser.add_argument("--output", type=Path, required=True, help="校准文件输出目录")
    args = parser.parse_args()
    reference = summarize_reference(args.catalog)
    profile = fit_population(reference, json.loads(args.pilots.read_text()))
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in (("bsc5_reference.json", reference), ("population_profile.json", profile)):
        (args.output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"BSC5 V ≤ 6.5: {reference['visible_count']}; fitted expectation: {profile['estimated_visible_count']:.1f}; local density: {profile['default_local_density_per_pc3']:.6f}/pc³")


if __name__ == "__main__":
    main()
