import json
import matplotlib.pyplot as plt
import os

# =========================
# 配置项
# =========================
INPUT_FILE = "star_map_complete.json"
OUTPUT_HR_IMG = "validation_hr_diagram.png"
OUTPUT_DIST_IMG = "validation_dist_mag.png"
OUTPUT_REPORT = "validation_report.txt"

# =========================
# 1. 加载数据
# =========================
if not os.path.exists(INPUT_FILE):
    print(f"错误: 找不到文件 {INPUT_FILE}，请先运行生成脚本。")
    exit()

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

stars = data["stars"]
print(f"成功加载 {len(stars)} 颗恒星数据。")

# =========================
# 2. 异常检测逻辑 (Astrophysical Sanity Check)
# =========================
flagged_stars = []
valid_stars = []

for s in stars:
    reasons = []

    # --- 规则 A: 不可能的红矮星 (The Impossible M-Dwarf) ---
    # M型主序星(V)极暗，肉眼可见距离通常<10pc。如果超过20pc还能看见，说明物理参数错了。
    if (
        s["spectral_type"] == "M"
        and s["luminosity_class"] == "V"
        and s["distance_pc"] > 20
    ):
        reasons.append(f"M型主序星距离过远 ({s['distance_pc']}pc) - 物理上应不可见")

    # --- 规则 B: 超爱丁顿光度极限 (Super-Eddington Limit) ---
    # 恒星光度极少超过 1,000,000 L_sun (除了极不稳定的高光度蓝变星)
    if s["luminosity_solar"] > 1_500_000:
        reasons.append(
            f"光度异常过高 ({s['luminosity_solar']:.1e} Lsun) - 超过稳定极限"
        )

    # --- 规则 C: 错误的分类 (Classification Mismatch) ---
    # 超巨星(0/I)的绝对星等应该很亮 (Mv < -3)
    # 亮巨星(II)允许更暗，但不应超过 -1.0 (保留天文真实性)
    # 注意：不能用 "I" in class，会误匹配 III/IV/VI
    if s["luminosity_class"] in ["0", "I"] and s["abs_mag"] > -3:
        reasons.append(f"超巨星绝对星等过暗 (Mv={s['abs_mag']})")
    elif s["luminosity_class"] == "II" and s["abs_mag"] > -1.0:
        reasons.append(f"亮巨星绝对星等过暗 (Mv={s['abs_mag']})")

    # --- 规则 D: 错误的温度-光度关系 (主序带偏离) ---
    # 如果标为主序星(V)，但像红巨星一样冷且亮
    if s["luminosity_class"] == "V" and s["temperature_K"] < 4000 and s["abs_mag"] < 5:
        reasons.append("主序星异常: 温度低但过亮 (疑似应为巨星)")

    if reasons:
        s["flag_reasons"] = reasons
        flagged_stars.append(s)
    else:
        valid_stars.append(s)

print(f"检测完成: 正常恒星 {len(valid_stars)} 颗, 异常恒星 {len(flagged_stars)} 颗。")

# =========================
# 3. 绘图与保存
# =========================


# 提取绘图数据 (分为正常组和异常组)
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

# --- 图表 1: 赫罗图 (H-R Diagram) ---
plt.figure(figsize=(12, 10))
ax = plt.gca()

# 画正常星
plt.scatter(
    v_data["temp"],
    v_data["abs_mag"],
    c=v_data["color"],
    s=15,
    alpha=0.6,
    edgecolors="none",
    label="Valid Stars",
)

# 画异常星 (如果有)
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

# 坐标轴与装饰
plt.gca().invert_xaxis()  # 温度逆序
plt.gca().invert_yaxis()  # 星等逆序
plt.xscale("log")
plt.title(
    f"Hertzsprung-Russell Diagram Validation\n(Flagged: {len(flagged_stars)} stars)",
    fontsize=16,
)
plt.xlabel("Temperature (K)", fontsize=12)
plt.ylabel("Absolute Magnitude (Mv)", fontsize=12)
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend()

# 标注参考线 (Main Sequence Roughly)
ax.set_xticks([30000, 10000, 6000, 3000])
ax.set_xticklabels(["30k", "10k", "6k", "3k"])

# 保存
plt.savefig(OUTPUT_HR_IMG, dpi=150, bbox_inches="tight")
print(f"图表已保存: {OUTPUT_HR_IMG}")
plt.close()  # 关闭以释放内存

# --- 图表 2: 距离 vs 视星等 (Malmquist Bias) ---
plt.figure(figsize=(12, 8))

# 画正常星
plt.scatter(
    v_data["dist"], v_data["app_mag"], c="blue", s=10, alpha=0.3, label="Valid Stars"
)

# 画异常星
if f_data["dist"]:
    plt.scatter(
        f_data["dist"],
        f_data["app_mag"],
        c="red",
        marker="x",
        s=40,
        label="Problematic Stars",
    )

# 画肉眼极限线
plt.axhline(
    y=6.5, color="green", linestyle="--", linewidth=2, label="Naked Eye Limit (6.5)"
)

plt.xlabel("Distance (pc)", fontsize=12)
plt.ylabel("Apparent Magnitude (m)", fontsize=12)
plt.title("Distance vs. Visibility Check", fontsize=16)
plt.gca().invert_yaxis()
plt.legend()
plt.grid(True, alpha=0.3)

# 保存
plt.savefig(OUTPUT_DIST_IMG, dpi=150, bbox_inches="tight")
print(f"图表已保存: {OUTPUT_DIST_IMG}")
plt.close()

# =========================
# 4. 生成文本报告
# =========================
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

    # 统计数据
    f.write("\n--- 关键统计 ---\n")
    f.write(f"最远恒星: {max([s['distance_pc'] for s in stars]):.1f} pc\n")
    f.write(f"最大光度: {max([s['luminosity_solar'] for s in stars]):.1e} L_sun\n")

    # M型星统计
    m_stars = [s for s in stars if s["spectral_type"] == "M"]
    m_giants = [s for s in m_stars if s["luminosity_class"] in ["I", "II", "III"]]
    m_dwarfs = [s for s in m_stars if s["luminosity_class"] == "V"]
    f.write(
        f"M型星总数: {len(m_stars)} (巨星: {len(m_giants)}, 矮星: {len(m_dwarfs)})\n"
    )

print(f"文本报告已保存: {OUTPUT_REPORT}")
print("验证完成。")
