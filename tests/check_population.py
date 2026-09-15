"""默认星空的多种子验收；每个根种子都记录完整数量条件抽样的尝试。"""
import argparse
from collections import Counter
import hashlib
from itertools import count, islice
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import star_generator as generator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=int, default=12)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 2:
        parser.error('至少需要两个随机种子')
    reference_path = generator.DATA_DIR / 'bsc5_reference.json'
    reference = json.loads(reference_path.read_text())
    results = []
    training_seeds = set(generator.POPULATION_PROFILE['pilot_seeds'])
    heldout_seeds = (seed for seed in count() if seed not in training_seeds)
    for seed in islice(heldout_seeds, args.runs):
        catalog = generator.generate_catalog(generator.GenerationConfig(seed=seed),
                                               generation_id=f'acceptance_{seed}', progress=None)
        stars = catalog['stars']
        row = {'seed': seed, 'count': len(stars),
               'count_selection': catalog['metadata']['count_selection'],
               'population_sampled': catalog['metadata']['generation_stats']['population_stars_sampled'],
               'validation': catalog['metadata']['validation_stats'],
               'comparison': catalog['metadata']['reference_comparison'],
               'cumulative_magnitude_counts': {m: sum(s['app_mag'] <= float(m) for s in stars)
                                               for m in reference['cumulative_magnitude_counts']},
               'spectral_counts': dict(Counter(s['spectral_type'] for s in stars)),
               'within_galactic_latitude_10_deg': sum(abs(s['gal_lat']) <= 10 for s in stars)}
        results.append(row)
        print(f"seed {seed}: {len(stars)} ({row['comparison']['relative_difference']:+.2%})", flush=True)
    counts = [row['count'] for row in results]
    config = generator.GenerationConfig()
    passed = all(row['validation']['all_passed'] and generator.count_in_range(row['count'], config) for row in results)
    report = {'generator_version': generator.GENERATOR_VERSION,
              'population_profile_sha256': hashlib.sha256(generator.POPULATION_PATH.read_bytes()).hexdigest(),
              'generation_parameters': catalog['metadata']['generation_parameters'],
              'source_sha256': catalog['metadata']['source_sha256'],
              'reference_sha256': reference['source_sha256'], 'reference_count': reference['visible_count'],
              'runs': len(results), 'minimum': min(counts), 'maximum': max(counts),
              'mean': statistics.mean(counts), 'sample_stddev': statistics.stdev(counts),
              'all_passed': passed, 'results': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f"{len(results)} runs: {min(counts)}–{max(counts)}, mean {statistics.mean(counts):.1f}; {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
