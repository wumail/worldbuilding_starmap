#!/usr/bin/env python3
"""
统计 star_map.json 中各个视星等区间星星的分布情况
"""

import json
from collections import defaultdict

def analyze_star_distribution(json_path: str = "star_map.json"):
    """分析星星视星等分布"""
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 合并 neighbors 和 stars
    neighbors = data.get('neighbors', [])
    stars = data.get('stars', [])
    all_stars = neighbors + stars
    
    print(f"{'='*50}")
    print(f"星图数据统计报告")
    print(f"{'='*50}")
    print(f"邻居星 (neighbors): {len(neighbors)}")
    print(f"背景星 (stars): {len(stars)}")
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
    
    print(f"{'='*60}")
    print(f"视星等分布 (天文学标准区间)")
    print(f"{'='*60}")
    print(f"{'区间':<15} {'数量':>8} {'实际%':>10} {'理论%':>10} {'偏差':>10}")
    print(f"{'-'*60}")
    
    total = len(all_stars)
    max_count = 1
    bin_counts = []
    
    for m_min, m_max, theory_pct in STANDARD_BINS:
        count = sum(1 for s in all_stars if m_min <= s.get('app_mag', 999) < m_max)
        bin_counts.append(count)
        if count > max_count:
            max_count = count
    
    for i, (m_min, m_max, theory_pct) in enumerate(STANDARD_BINS):
        count = bin_counts[i]
        actual_pct = (count / total) * 100 if total > 0 else 0
        diff = actual_pct - theory_pct
        bar_length = int((count / max_count) * 25)
        bar = '█' * bar_length
        diff_str = f"+{diff:.2f}" if diff >= 0 else f"{diff:.2f}"
        print(f"{m_min:>4.1f} ~ {m_max:<4.1f}   {count:>8}   {actual_pct:>8.2f}%   {theory_pct:>8.2f}%   {diff_str:>8}%  {bar}")
    
    print()
    
    # 更细致的统计：0.5 等间隔
    print(f"{'='*50}")
    print(f"视星等分布 (0.5等间隔)")
    print(f"{'='*50}")
    
    fine_bins = defaultdict(int)
    for star in all_stars:
        mag = star.get('app_mag', 0)
        bin_key = round(mag * 2) / 2  # 四舍五入到0.5
        fine_bins[bin_key] += 1
    
    print(f"{'区间':<15} {'数量':>10} {'柱状图'}")
    print(f"{'-'*50}")
    
    max_fine = max(fine_bins.values()) if fine_bins else 1
    for mag in sorted(fine_bins.keys()):
        count = fine_bins[mag]
        bar_length = int((count / max_fine) * 30)
        bar = '▓' * bar_length
        print(f"{mag:>5.1f} 等       {count:>10}  {bar}")
    
    print()
    
    # 统计亮星和暗星
    print(f"{'='*50}")
    print(f"特殊统计")
    print(f"{'='*50}")
    
    a1 = [s for s in all_stars if s.get('app_mag', 0) < 0.5]
    a2 = [s for s in all_stars if 0.5 <= s.get('app_mag', 0) < 1.5]
    a3 = [s for s in all_stars if 1.5 <= s.get('app_mag', 0) < 2.5]
    a4 = [s for s in all_stars if 2.5 <= s.get('app_mag', 0) < 3.5]
    a5 = [s for s in all_stars if 3.5 <= s.get('app_mag', 0) < 4.5]
    a6 = [s for s in all_stars if 4.5 <= s.get('app_mag', 0) < 5.5]
    a7 = [s for s in all_stars if 5.5 <= s.get('app_mag', 0) < 6.5]
    a8 = [s for s in all_stars if s.get('app_mag', 0) >= 6.5]
    
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
        brightest = min(all_stars, key=lambda s: s.get('app_mag', 999))
        dimmest = max(all_stars, key=lambda s: s.get('app_mag', -999))
        
        print(f"最亮星: {brightest.get('id', 'N/A')} (视星等: {brightest.get('app_mag', 'N/A'):.2f})")
        print(f"最暗星: {dimmest.get('id', 'N/A')} (视星等: {dimmest.get('app_mag', 'N/A'):.2f})")
        
        # 平均视星等
        avg_mag = sum(s.get('app_mag', 0) for s in all_stars) / len(all_stars)
        print(f"平均视星等: {avg_mag:.2f}")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "star_map.json"
    analyze_star_distribution(path)
