# Worldbuilding Starmap

从盘状恒星总体生成背景恒星，再根据观测者位置、星际消光和视星等筛出可见星。默认输出 **9000–9500 颗、视星等 ≤6.5 的全天可见背景星**；不反向抽取目标星等或距离。

“至少 9000”是生成和保存时的硬约束。先将总体密度调整到区间附近，再接受数量符合要求的**整份随机星空**；最多尝试 8 次，失败不输出星表。每次尝试使用独立随机子流，记录所有尝试数量，不补星、删星或修改视星等。输出因此服从**数量条件下的盘状星族模型**。

最新 4.4 数据采用设定的 **30,712 光年银心距（9,416.342 pc）**，包含 **9,356 颗点源**及共龄星团、发射/反射云和尘埃云。V1/V2 四个页面均已加入 **15 个黄道天区，每区 24°**。[生成方法、科学依据与边界](data/galactic_sky_science.md)、[15 个地标设计](design/landmark_objects.md)、[进动章动候选与数值核验](design/terrax_precession_nutation.md)分别说明已实现内容与仅设计内容。

当前默认形态批次为 `output_20260915_nebula_03`：同一物理星空保留五类气体形态，先生成不规则发光轮廓与局部烟雾，再按固定角尺度平滑整个发光层，改善细纹与硬边。最后按球面光量归一化；恒星点像、星团、原星表及物理预算保留。`_nebula_02` 仍可显式打开比较。[形态方法与边界](data/nebula_morphology.md)说明其作为投影发光近似的适用范围。

[星座抽卡工作台](web/constellations/)基于同一份星表，反复尝试十五个星形与不等宽天区，支持种子复现、锁定重抽、收藏比较和意象笔记。当前 **draw-9** 用滑块调整初始黄道采样带，默认两侧各 **40°**，范围 15°–60°；参考赤道总览实时显示两条曲线和候选数量。每区前三亮与全部不暗于 3 等的恒星进入完整星形及骨架，最终按完整星表复查。边界沿 1° 参考赤经／赤纬网格，包络外余量不超过 8°；**先求各区独立最少拐点，再按局部额外折角协调公共边界**。至少十四座占黄道 16°–44°，最多一座可窄至 6°。

候选窗口内提供**编辑边界、编辑星座与连线**：可加入背景星、连接两颗星、删除连线或成员，移除成员会同步清除关联线。新的 manual-3 不再受自动星等、成员数量、亮星必选、8° 余量与黄道宽度限制；只保留源星真实性、正交唯一归属以及对全部成员和连线的包围检查。草稿可先越界，再调整边界；确认应用，取消恢复，支持撤销重做和视野拖动缩放。“本区重新生成”仅在当前实际边界内重新采样和连线。**本地仅保存收藏**：新结果先临时显示，收藏后才会在刷新后恢复；旧未收藏记录会清理。旧 draw-1 至 draw-9 及 manual-1/2 配方仍可恢复，正式四个页面尚未替换。[收藏与本区重新生成](design/zodiac_regional_sampling.md)、[当前手动操作](design/zodiac_manual_editing.md)、[工作流](design/zodiac_workflow.md)及[本轮核验](design/zodiac_regional_validation.json)说明具体范围；自动采样详见 [draw-9 方法](design/zodiac_candidate_editing.md)。

历史 4.3 在 24 个根种子中实测 **9080–9336** 颗，平均 **9197.21** 颗，旧示例为 **9327** 颗；这些统计不冒充新版的重复实验。BSC5 同一 `V ≤6.5` 口径为 **8404** 颗，数量合同有意略多。[早期生成和图片验收](data/generation_and_rendering_validation.md)及 [4.2 校准报告](data/calibration_report.md)保留为历史基线。

当前代码继续按用途分开放：生成脚本在 `src/`，科学输入在 `data/`，生成星表在 `output/`，候选设计在 `design/`。网页进一步明确分为 **`web/v1/` 标准星图、`web/v2/` 肉眼观看、`web/shared/` 公共计算与绘制基础**。[网页目录说明](web/README.md)列出职责和入口。根首页提供版本选择；旧 `web/` 和 `web/eye/` 的 HTML 入口只做跳转。仓库不恢复 `v4/` 目录；GitHub Pages 的旧 `/v4/` 书签由 `404.html` 转到新入口。

## 使用

需要 Python 3.11 或更新版本，以及支持 OpenGL 的环境（图片导出时）。浏览星云图层还需要支持浮点纹理的 WebGL 浏览器。

```sh
python3 -m pip install -r requirements.txt
python3 src/generate_galactic_sky.py --seed 20260915
python3 -m http.server 8000
```

在浏览器打开 `http://localhost:8000/`。每次生成另存为 `output/output_<批次>/`，旧星表和图片不覆盖。随机种子、配置、参考表指纹和实际验证统计一起保存；同样的种子、配置、代码和 NumPy 版本可重现星体数据。批次 ID 是独立的 UTC 时间戳。

打开首页后选择版本：

| 版本 | 网页 1 · 平面星图 | 网页 2 · 沉浸天球 |
|---|---|---|
| V1 · 标准星图 | [打开](web/v1/sky_atlas.html) | [打开](web/v1/star_map.html) |
| V2 · 肉眼观看 | [打开](web/v2/sky_atlas.html) | [打开](web/v2/star_map.html) |

“V1/V2”指观察版本，“网页 1/2”指平面/沉浸视图，二者独立。每个版本内两页共用时间、地点和可编辑轨道初值；两个版本的观察设置分别保存。天体以真实角径为基准应用统一显示增强，面板始终显示物理角径；两页均可放大观察。从 Terrax 观察 Sol、六颗其他行星及 Luna / Echo，在 **140.49 地球年**内播放、暂停或跳转。全天星图展示完整天球；选择“地表天空”后遮挡地平线下的天体，并可模拟昼夜明暗。详细计算依据见[动态天空验证说明](data/sky_motion_validation.md)和 [V2 说明](web/v2/README.md)。

双月倾角现已按用户选择解释为 **第 0 日相对 Terrax 初始赤道的瞬时轨道倾角**，允许后续受扰演化。这一选择只用于独立动力学核验；约四万年进动仍是候选量级，尚未接入四个正式页面。

```sh
# 新版：场星、共龄星团、云气以及默认数量合同
python3 src/generate_galactic_sky.py --seed 42

# 研究用：显式关闭数量条件，检查无条件总体（不保证 9000 颗）
python3 src/generate_galactic_sky.py --seed 42 --unconditioned

# 只检查已有数据，不修改它
python3 src/star_generator.py --validate output/output_<批次>/star_map_<批次>.json

# 导出南北天球星图；省略 --color 时保留白色符号图
python3 src/star_shader.py output/output_<批次>/star_map_<批次>.json --res 4k --color
```

图片已经存在时，使用 `--output 新文件.png` 另存。每次图片导出都执行逐像素核验和 PNG 重新打开检查，并附带 `.render.json` 报告，记录每颗星的半球、像素坐标、大小和验收像素。`--no-lines` 可省略网格线；图片 `--limit-mag` 默认采用星表阈值。`--no-plots` 只跳过生成器的两张诊断图。更多选项见 `--help`。星表程序入口为 `GenerationConfig`、`generate_catalog`、`save_catalog`；导入模块不会生成或保存数据。

## 模型与边界

新版星团等时线、旋臂、三维云消光、发射/反射预算及弥散光图见[4.4 科学说明](data/galactic_sky_science.md)。下面介绍沿用的场星及静态点星图片模型；`star_shader.py` 仍只输出点星图，包含新图层的图片请在网页中导出。

- 场星空间总体是非均匀 Poisson 点过程，密度按银河径向和离盘高度指数衰减。新默认银心距 9,416.342 pc，高度暂定 20 pc；旧星表保留其原有参数。不同恒星群体使用不同尺度高度。坐标以观测者为原点，+x 指向银心，+z 指向银北极。
- 总体范围默认为 1–10000 pc。宿主恒星、近距离伴星、行星和卫星不在该背景星表中。相对群体丰度由 BSC5 与四组训练样本估计的可见体积共同标定；4.3 将原密度乘以 `9250/8404`，得到约 0.120035 颗/pc³。详细亮度分布和演化星参数仍是近似，没有完整的恒星形成史、年龄或金属丰度模型。
- 主序星的质量、温度、光度与热光校正沿同一条 [参考序列](data/README.md) 抽样和插值。演化星使用受质量、半径、表面重力与光度约束的经验范围。
- 旧入口的消光沿垂向指数尘埃盘解析积分，局部消光率 0.7 mag/kpc。4.4 入口改用径向/垂向盘与三维高斯云，局部平滑项 0.62 mag/kpc，沿有限视线积分；云后的背景源才受相应消光。颜色红化尚未建模。
- 每颗恒星先有空间位置和物理参数，再正向计算 `m = Mv + 5 log10(d/10) + Av`；仅输出 `m ≤ 极限星等` 的恒星，没有人为的最亮截断。坐标和物理量保存完整浮点精度，显示时可自行取舍小数。
- 改变极限星等、恒星密度、观测者位置或消光等参数，会改变无条件可见数量。数量合同仍生效，因此不相容的参数可能耗尽尝试并报错；研究时可显式选择 `--unconditioned`。历史星表按其中保存的参数验证，不强套新下限。
- 为避免生成必定看不见的海量暗星，按每类恒星最亮的可能绝对星等计算**无消光可见地平线**，只省去其外必不可见的体积；地平线内仍先生成总体，再逐颗筛选。这个裁剪不改变可见分布，但报告中的总体样本数仅指这些体积内实际抽到的恒星，不是整个星系的星数。资源预算不足会报错，不保存不完整结果。
- 验证覆盖实际字段、单位换算、空间几何、距离模数、Stefan–Boltzmann 关系、主序关系和演化阶段范围。通过表示满足这些已实现约束，不等于完成了全部天体物理验证。此前版本的旧星表可以继续查看，无法因此获得新版验证通过标记。
- 网页相机固定在观察坐标原点，拖动改变朝向，滚轮改变视场。动态网页可选择 Terrax 地表或质心；静态 PNG 仍为南北天球的方位等距投影。赤道朝向沿用原先近似地球 J2000 的定义，黄赤交角为 25°。动态轨道为固定 Kepler 椭圆，未作多体扰动或精密食象模拟。
- 静态 PNG 的星点大小与光晕用于阅读星图，图片不是测光图。PNG 最小直径 2.5 px 防止漏采样；赤道圆内缩，给最亮星的完整符号留出黑色余量。赤道星仅归北图；相邻星可能重叠为同一光点，RGB8 叠加可能饱和。动态网页采用另一套统一角尺度和像素覆盖采样，月面按实际角径绘制；两种输出都未模拟真实人眼曝光或未分辨银河背景。

## 检查

```sh
python3 -m unittest discover -s tests -v
python3 src/verify_stat.py output/output_<批次>/star_map_<批次>.json
python3 src/analyze_star_distribution.py output/output_<批次>/star_map_<批次>.json

# 完整生成多个根种子，检查新数量合同和物理关系
python3 tests/check_population.py --runs 24 --output /tmp/population-check.json

# 需要真实 OpenGL；验证亚像素暗星、亮度单调性及 2K/4K PNG
python3 tests/check_rendering.py --output /tmp/rendering-check.json
```

本地预览启动后，打开 `http://localhost:8000/tests/browser_checks.html` 检查相机位置、缩放、反向聚焦、极点、重置与宽屏选星。

动态计算、投影与资源可运行 `node --test tests/test_sky_motion.mjs tests/test_sky_projection.mjs tests/test_atlas_resources.mjs`。打开 `http://localhost:8000/tests/motion_render_checks.html` 可核验真实 Canvas / GPU 的可见性、盘面遮挡、同角尺度、月相与两种沉浸投影；`http://localhost:8000/tests/atlas_playback_checks.html` 检查一分钟连续播放与暂停后的绘图恢复。

新增科学检查位于 `tests/test_galactic_sky.py` 和 `tests/test_zodiac.mjs`；[深空像素验收页](tests/deep_sky_render_checks.html)检查方向、面亮度、缩放、月面暗侧遮挡和地平裁剪。
