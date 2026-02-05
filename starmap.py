import json
import math
import random
import numpy as np

# ==========================================
# 1. 配置参数与邻居数据
# ==========================================
OUTPUT_FILE = "stars_data_1000.json"
TOTAL_STARS = 14000
GALAXY_TYPE = "SBbc-Spiral-Arm"

# 颜色映射 (近似 RGB Hex)
COLOR_MAP = {
    "O": "#9bb0ff", # 蓝
    "B": "#aabfff", # 蓝白
    "A": "#cad7ff", # 白
    "F": "#f8f7ff", # 黄白
    "G": "#fff4ea", # 黄 (太阳)
    "K": "#ffd2a1", # 橙
    "M": "#ffcc6f", # 红 (为了视觉可见度，稍微调亮)
    "D": "#e0e0e0", # 白矮星 (暗淡灰白)
}

# 恒星物理半径参考 (单位: 太阳半径 R_sol)
RADIUS_MAP = {
    "O": 15.0, "B": 7.0, "A": 2.5, "F": 1.3, 
    "G": 1.0, "K": 0.8, "M": 0.3, "D": 0.01,
    "Supergiant": 100.0, "Giant": 20.0
}

# 24 个邻居数据
fixed_neighbors = [
    {"id": "Sys_1", "dist": 4.79, "type": "M", "pos": [0.03, -4.78, -0.34]},
    {"id": "Sys_2", "dist": 11.57, "type": "M", "pos": [-6.23, -5.99, -7.7]},
    {"id": "Sys_3", "dist": 10.24, "type": "K", "pos": [-0.83, 10.09, 1.51], "abs_mag": 7.0},
    {"id": "Sys_4", "dist": 10.79, "type": "M", "pos": [-3.12, 7.93, 6.62]},
    {"id": "Sys_5", "dist": 10.75, "type": "M", "pos": [4.34, 9.82, 0.53]},
    {"id": "Sys_6", "dist": 5.64, "type": "G", "pos": [-5.12, 0.79, 2.22], "abs_mag": 4.8}, # Bright
    {"id": "Sys_7", "dist": 8.65, "type": "M", "pos": [-3.6, -0.32, -7.86]},
    {"id": "Sys_8", "dist": 7.09, "type": "F", "pos": [-1.36, 0.2, 6.95], "abs_mag": 3.0}, # Brightest
    {"id": "Sys_9", "dist": 9.91, "type": "M", "pos": [2.58, 9.56, 0.36]},
    {"id": "Sys_10", "dist": 10.73, "type": "M", "pos": [3.25, 7.71, -6.72]},
    {"id": "Sys_11", "dist": 10.61, "type": "M", "pos": [-6.66, 4.21, 7.1]},
    {"id": "Sys_12", "dist": 7.97, "type": "K", "pos": [-4.93, -5.26, -3.39], "abs_mag": 7.0},
    {"id": "Sys_13", "dist": 11.68, "type": "G", "pos": [10.67, 4.28, -2.09], "abs_mag": 4.8},
    {"id": "Sys_14", "dist": 5.53, "type": "M", "pos": [-0.35, -2.12, -5.1]},
    {"id": "Sys_15", "dist": 5.53, "type": "K", "pos": [-0.65, -2.47, 4.91], "abs_mag": 7.0},
    {"id": "Sys_16", "dist": 11.73, "type": "M", "pos": [8.97, -4.35, -6.18]},
    {"id": "Sys_17", "dist": 4.77, "type": "M", "pos": [-2.09, 4.28, 0.35]}, # Closest
    {"id": "Sys_18", "dist": 11.33, "type": "M", "pos": [6.96, -0.86, -8.9]},
    {"id": "Sys_19", "dist": 9.82, "type": "M", "pos": [4.42, 6.91, -5.39]},
    {"id": "Sys_20", "dist": 9.56, "type": "M", "pos": [4.45, 7.92, -2.98]},
    {"id": "Sys_21", "dist": 10.65, "type": "M", "pos": [3.06, 5.59, -8.53]},
    {"id": "Sys_22", "dist": 11.87, "type": "M", "pos": [-3.23, -10.5, 4.49]},
    {"id": "Sys_23", "dist": 10.99, "type": "M", "pos": [0.29, -2.82, -10.62]},
    {"id": "Sys_24", "dist": 11.14, "type": "M", "pos": [-10.4, 0.59, -3.94]},
]

# ==========================================
# 2. 辅助函数
# ==========================================
def cartesian_to_spherical(x, y, z):
    dist = math.sqrt(x**2 + y**2 + z**2)
    # Dec (Declination): -90 to 90
    dec = math.degrees(math.asin(z / dist)) if dist != 0 else 0
    # RA (Right Ascension): 0 to 360
    ra = math.degrees(math.atan2(y, x))
    if ra < 0: ra += 360
    return ra, dec

def calculate_apparent_mag(abs_mag, dist_ly):
    # dist_pc = dist_ly / 3.2616
    # m = M + 5 * log10(d_pc) - 5
    if dist_ly <= 0: return -26.7 # Avoid error for home star
    d_pc = dist_ly / 3.2616
    return abs_mag + 5 * math.log10(d_pc) - 5

def calculate_angular_diameter_mas(radius_sol, dist_ly):
    # 粗略计算: 太阳在1ly处的视直径约为 15 mas (milliarcseconds)
    # theta = (R_star / R_sol) / (Dist / 1ly) * 15
    return (radius_sol / dist_ly) * 15.0

# ==========================================
# 3. 核心生成逻辑
# ==========================================
stars_list = []

# --- 步骤 A: 处理 24 个固定邻居 ---
for neighbor in fixed_neighbors:
    ra, dec = cartesian_to_spherical(*neighbor["pos"])
    
    # 默认绝对星等
    abs_mag = neighbor.get("abs_mag", 12.0) # M型星默认为12
    
    # 物理半径
    radius = RADIUS_MAP.get(neighbor["type"], 0.3)
    
    app_mag = calculate_apparent_mag(abs_mag, neighbor["dist"])
    ang_diam = calculate_angular_diameter_mas(radius, neighbor["dist"])
    
    stars_list.append({
        "id": neighbor["id"],
        "is_neighbor": True,
        "type": neighbor["type"],
        "ra": round(ra, 4),
        "dec": round(dec, 4),
        "dist_ly": neighbor["dist"],
        "abs_mag": abs_mag,
        "app_mag": round(app_mag, 2),
        "color_hex": COLOR_MAP.get(neighbor["type"], "#ffffff"),
        "angular_diameter_mas": round(ang_diam, 4),
        "pos_cartesian": neighbor["pos"] # 保存直角坐标方便引擎使用
    })

# --- 步骤 B: 生成背景星 (Voronoi/Fibonacci Sampling) ---
num_bg_stars = TOTAL_STARS - len(fixed_neighbors)
phi = (1 + math.sqrt(5)) / 2  # 黄金比例

for i in range(num_bg_stars):
    # 1. 斐波那契球采样 (生成均匀的方向向量)
    y = 1 - (i / float(num_bg_stars - 1)) * 2
    radius_at_y = math.sqrt(1 - y * y)
    theta = phi * i * 2 * math.pi
    
    x = math.cos(theta) * radius_at_y
    z = math.sin(theta) * radius_at_y
    
    # 这里 x, z, y 是单位向量坐标 (y 为天轴)
    # 我们映射为: X=x, Y=z, Z=y (Z为Declination轴)
    vec = np.array([x, z, y])
    
    # 2. 距离生成 (加权随机: 远处星星更多)
    dist = random.triangular(40, 1500, 800)
    
    # 3. 旋臂环境模拟 (SBbc Context)
    # 银河平面判定: Z轴接近 0 (即 y 接近 0)
    # sin(lat) = y component
    sin_lat = abs(y)
    
    roll = random.random()
    
    # 默认参数 (背景填充星)
    s_type = "K" # 橙巨星
    radius = RADIUS_MAP["Giant"]
    abs_mag = 0.5 
    color = COLOR_MAP["K"]
    
    # 逻辑: 靠近平面 -> 更多蓝超巨星
    if sin_lat < 0.25: # 银河带内 (~15度)
        if roll < 0.20: # 20% 概率是明亮的蓝星
            s_type = "B"
            radius = RADIUS_MAP["Supergiant"]
            abs_mag = -4.5 + random.uniform(-1, 1) # 极亮
            color = COLOR_MAP["B"]
        elif roll < 0.50:
            s_type = "A"
            radius = 30.0
            abs_mag = -1.0 + random.uniform(-0.5, 0.5)
            color = COLOR_MAP["A"]
    else: # 银晕/高纬度
        if roll < 0.10: # 偶尔有蓝星跑出来
            s_type = "B"
            radius = RADIUS_MAP["B"]
            abs_mag = -1.5
            color = COLOR_MAP["B"]
            
    # 计算视参数
    app_mag = calculate_apparent_mag(abs_mag, dist)
    
    # 剔除过暗的星 (视星等 > 6.5 肉眼不可见)
    if app_mag > 7.0:
        continue
        
    ang_diam = calculate_angular_diameter_mas(radius, dist)
    
    # 计算最终坐标
    final_pos = vec * dist
    ra, dec = cartesian_to_spherical(final_pos[0], final_pos[1], final_pos[2])
    
    stars_list.append({
        "id": f"Bg_{i}",
        "is_neighbor": False,
        "type": s_type,
        "ra": round(ra, 4),
        "dec": round(dec, 4),
        "dist_ly": round(dist, 1),
        "abs_mag": round(abs_mag, 2),
        "app_mag": round(app_mag, 2),
        "color_hex": color,
        "angular_diameter_mas": round(ang_diam, 4),
        "pos_cartesian": [round(val, 2) for val in final_pos]
    })

# ==========================================
# 4. 输出 JSON
# ==========================================
output_data = {
    "meta": {
        "galaxy_type": GALAXY_TYPE,
        "total_stars": len(stars_list),
        "notes": "Generated with Fibonacci sampling + SBbc Spiral Arm density logic."
    },
    "stars": stars_list
}

with open(OUTPUT_FILE, "w", encoding='utf-8') as f:
    json.dump(output_data, f, indent=2)

print(f"成功生成 {len(stars_list)} 颗恒星数据，已保存至 {OUTPUT_FILE}")
print("包含关键字段: id, ra, dec, app_mag, angular_diameter_mas, color_hex")