# 选星与连线：十三个黄道星座的形态参照

2026-09-17。本文保留十三星座调研、第二代 `regional-figure-sampling-2` 方法和当时的验证。当前已升级为第三代清晰度采样，并用于 draw-10 种子生成，默认标准，见 [当前方法](zodiac_figure_clarity.md)。[打开工作台](../web/constellations/?revision=clarity-draw-10) · [第二代历史验证](zodiac_morphology_validation.json) · [第二代共同尺度对照图](../reports/zodiac_morphology/comparison/Z01.png)

以下第二代工作仅改变实际边界内的成员选择、图形组织和骨架简化，当时整轮仍为 draw-9。星表中的 ID、方向、颜色、距离和星等不变；下文旧指标与复现命令不作为当前版本的新验证。

## 调研带来的判断

现实参照不是十二个等分扇区，也不是十二张必须照抄的轮廓。黄道穿过十三个现代星座区域，包括蛇夫座；星座区域与组内连线是不同层次。IAU 没有规定唯一的星形连线，同一批星可以有不同画法。

Kemp、Hamacher、Little 和 Cropper 的 [2022 年论文](https://arxiv.org/html/2010.06108v3)整理了 27 种文化的星群。其 Graph Clustering 模型把亮度和距离结合，并按局部背景重新定标：亮星间的较长联系与暗星间的较短联系都可能显著。模型合并 3.5、4.0、4.5 等阈值下的 Delaunay 图，而不是只用一张最短树。

这里有两个必须保留的边界：论文研究的是**哪些星被分成一组**，并未解决组内怎样组织成图；其文化间比较中的最小生成树是分析表示，并不等于各文化的真实画法。论文中增加良好连续性也没有改善其分组模型，因此本次对连续折线的偏好是绘图设计，不能说成论文已经证明的最优连法。

[Stellarium 的 H. A. Rey 画法说明](https://github.com/Stellarium/stellarium/blob/master/skycultures/modern_rey/description.md)提供了另一种启发：相同天空可以组织出更明确的头、躯干、肢体和尾部关系，甚至重新解释亮星在形象中的角色。这支持“先形成可读的线条关系，再由人命名”，不要求生成器预置动物轮廓。

## 十三座逐项参照

下表的形态描述是对常见画法的概括；最后一列是直接计算两套 Stellarium 连线数据所得的 **成员数 / 独立闭环数**，顺序为 Modern → H. A. Rey。它们没有统一裁剪到 4.5 等，也不是所有现实画法的统计分布。中文名链接为各座的文字资料；括号内的组织方式是本项目提炼的设计启发。

| 星座 | 可借鉴的图形关系 | Modern → Rey：星数 / 环数 |
| --- | --- | --- |
| [白羊 Ari](https://www.constellation-guide.com/constellation-list/aries-constellation/) | 三颗明显恒星形成扁三角的点位关系，简画常是一条短折线；少量节点也能成立，不必每区都填成大树 | 4 / 0 → 10 / 3 |
| [金牛 Tau](https://www.constellation-guide.com/constellation-list/taurus-constellation/) | 毕星团的 V 形头部与向外延伸的双角；紧凑核心和长肢可以使用不同角尺度 | 12 / 1 → 20 / 1 |
| [双子 Gem](https://www.constellation-guide.com/constellation-list/gemini-constellation/) | 北河二、北河三构成明显的双锚点，两组大致并列的肢干从它们延伸；不需要精确镜像对称 | 17 / 0 → 18 / 0 |
| [巨蟹 Cnc](https://www.constellation-guide.com/constellation-list/cancer-constellation/) | 整区较暗，简单分叉比复杂外轮廓更有用；说明区内相对亮星不能被统一的亮度阈值排除 | 6 / 0 → 6 / 1 |
| [狮子 Leo](https://www.constellation-guide.com/constellation-list/leo-constellation/) | 镰刀般弯曲的前部与较紧凑的后部组合；曲线和闭合形可以共同组成一座 | 9 / 2 → 16 / 3 |
| [室女 Vir](https://www.constellation-guide.com/constellation-list/virgo-constellation/) | 跨度较大的分叉主干，角宿一作为醒目的端部或局部锚点；允许不均衡的枝长与亮度 | 12 / 1 → 15 / 3 |
| [天秤 Lib](https://www.constellation-guide.com/constellation-list/libra-constellation/) | 四边形点位与悬挂部分；轮廓闭合配少量外接支线，不需大量交织短边 | 5 / 1 → 6 / 1 |
| [天蝎 Sco](https://www.constellation-guide.com/constellation-list/scorpius-constellation/) | 头部的短分叉、心宿二锚点和连续弯折的长尾；主干方向的连贯性比总边长最短更重要 | 13 / 0 → 20 / 0 |
| [人马 Sgr](https://www.constellation-guide.com/constellation-list/sagittarius-constellation/) | 茶壶的壶身、盖、嘴和柄是紧凑局部结构的组合；少量三角、四边或更多顶点的环都有用途 | 21 / 4 → 18 / 5 |
| [摩羯 Cap](https://www.constellation-guide.com/constellation-list/capricornus-constellation/) | 宽而较扁的三角点位、两端角部；图形不必各方向尺度接近，暗星也能补出转折 | 9 / 1 → 13 / 2 |
| [宝瓶 Aqr](https://www.constellation-guide.com/constellation-list/aquarius-constellation/) | 小水罐星群与延伸的水流形成层次；局部醒目的小组未必包含全座最亮星 | 15 / 0 → 21 / 3 |
| [双鱼 Psc](https://www.constellation-guide.com/constellation-list/pisces-constellation/) | 两端星群或小环通过长线相连，中间有结点；双主体不应该被最短距离聚成一个紧密团块 | 20 / 2 → 19 / 2 |
| [蛇夫 Oph](https://www.constellation-guide.com/constellation-list/ophiuchus-constellation/) | 大尺度多边形主体与少量外接部分；人物与所持的蛇是文化图像关系，蛇座仍是另一个正式星座 | 7 / 1 → 20 / 2 |

计算时把每条折线的相邻 HIP 编号转为无向边，去重后统计 `V`、`E`、连通分量 `C`，独立闭环数为 `E−V+C`。两套数据的这十三座均为单一连通图。原始来源、Git blob、SHA-256 及完整统计保存在 [references.json](../reports/zodiac_morphology/references.json)，也纳入[验证记录](zodiac_morphology_validation.json)的 `sourceReferences`，便于随代码保存。数据源为 [Modern](https://github.com/Stellarium/stellarium/blob/master/skycultures/modern/index.json) 和 [Modern (Rey)](https://github.com/Stellarium/stellarium/blob/master/skycultures/modern_rey/index.json)，归属 Stellarium contributors，源数据许可 CC BY-SA 4.0；本项目没有把这些 HIP 连线作为生成模板。

## 为什么选择这套算法

| 方法 | 优点与限制 | 本次采用方式 |
| --- | --- | --- |
| 多阈值 Delaunay / 视觉分组 | 能在不同亮度与空间尺度上寻找自然星群；单独使用不能决定主干、分叉和环 | 借鉴局部尺度与亮度对比，保留当前候选区，不重新聚类整片天空 |
| 空间排斥采样、DPP | 能减少重复选中同一密集小团；强排斥可能漏掉真实的近邻亮星，DPP 也增加核与条件采样成本 | 必选亮星不参与淘汰，仅对可选星施加柔和的近邻排斥和有正下限的随机空间权重 |
| 随机最小生成树 | 简单、连通，但一味压低边长会重复相似的小枝，且不能独立表达闭环 | 保留森林作为连接阶段，但每一步根据当前度数和转向重新评分，再选择有面积的闭环 |
| 图形结构偏好与随机搜索 | 可以分别形成链、分叉、轮廓和多主体；硬套图案会忽视真实星位 | 随机选择软结构偏好和连续参数，用真实恒星实现；比较八个候选，不贴合十二星座模板 |

## 实际实现

1. **硬性保留亮星。** 从完整源星表按当前区域的真实格子归属取星。区内前三亮，以及全部视星等 `m≤3` 的星，必须同时进入完整星形和骨架。这里“绝对亮”指固定视星等阈值，不是恒星物理字段 `abs_mag`。邻座占用的重要星仍报错，不抢夺成员。
2. **提高相对显著性。** 其他成员仍从 `m≤4.5` 的候选里取；比较最近六颗候选的中位星等与自身星等，对局部突出的星加权。每次抽样的全局亮度权重系数在 0.18–0.38 间变化，局部亮度对比系数为 0.4。它们是绘图权重，不修改测光值，也不是拟合得到的心理物理参数。
3. **让选星本身具有空间变化。** 随机选显著星作焦点，产生走廊、环带、双焦点或局部团块的柔和偏好；近邻排斥只用于可选星，全部候选始终有正的抽样权重。这不保证每颗可选星都能通过后续图形约束而进入最终结果。以带种子的指数竞赛做加权不放回抽样。完整成员目标为丰富 7–15、适中 5–11；目标数在一次重生成内固定，不能靠挑选评分把所有结果偏向某个大小。必选星超过目标时全部保留。
4. **随机改变图结构。** 链、分叉、单环、双主体、带环主体五种软倾向控制节点分叉代价、转向偏好、局部组间联系和闭环目标，参数继续随机变化。边长按端点附近第二近邻的角距定标，允许稀疏亮星间的主干比密集部位的支线更长。随机扰动采用 Gumbel 形式，与选星的指数竞赛相互独立；不承诺均匀抽取所有合法图。
5. **只保留合法线条。** 连线沿真实短大圆弧；解析检查全部经过的归属格，排除越界、交叉、距第三成员不足 0.04°、重合／对跖端点以及节点度数超过四的情况。可以形成森林，不越过分片或洞强制连接。闭环允许 3–12 个节点，检查围成的面积，避免退化的细缝；宽到不适合局部面积投影的环直接不采用。节点度四和这些环条件是本工具的审美选择，现实 Rey 画法也有度五节点。
6. **保住形态核心。** 在加环后的完整图上剔除较暗的非必选叶节点。重要亮星、连通它们所需的中间节点及闭环都留下；骨架以七星为软目标，可能因重要星或结构需要而更多，完整图本身不足七星时不补凑。旧方法在加环前截取森林，所以骨架永远没有环。
7. **比较实际差异。** 每次产生八个建议，先选其中连通分量最少的一组，再优先采用与当前图不同的结果，按成员与边的 Jaccard 距离、分叉／端点／闭环与主干占比的变化以及边长选择。评分含随机项，不总取某一确定性最优形。找不到合格的新组合就返回 `changed=false`。这只避免候选集合中可避免的断开，不证明存在的所有合法连接都会被找到。

五种倾向只是内部搜索参数，并不等于成功生成了五种规定轮廓。区域很窄、候选很少或重要亮星已经占满时，自由度本来就有限。相同方法版本、实际草稿、完整源星表、复杂度和种子可复现；与当前图比较意味着“当前草稿”也是输入。

手动放大区域后，必选亮星可能远超过通常的成员目标。一次重生成的八个建议会复用相同成员序列的几何候选；缓存只存活于这次调用。成员不超过 32 颗时检查全部点对；超过 32 颗时，候选边改为 3、4、4.5 等及全部成员四个层次的八近邻边并集，加上角距最小生成树的跨团桥接边，再执行原有几何检查。稀疏候选减少重复计算，但不保证找到全部合法连接；最小生成树桥接也不能越过实际边界、其他成员或交叉限制。

## 对照结果与边界

固定已保存的 draw-9 区域，十五区 × 两种复杂度 × 32 个相同的测试种子，新旧各 **960 套**。统计定义与完整结果见 [验证记录](zodiac_morphology_validation.json)。旧实现来自提交 `498b052cb3de66ce9f13b9439b5bf0067553378b`，旧数据和收藏没有被重生成覆盖。

| 指标 | 原方法 | 新方法 |
| --- | ---: | ---: |
| 每组 32 次的平均骨架结构种类 | 3.57 | 10.50 |
| 可选成员平均 Jaccard 距离，越高差异越大 | 0.612 | 0.685 |
| 连线集合平均 Jaccard 距离 | 0.802 | 0.831 |
| 每组 32 次的平均完整图拓扑种类 | 15.57 | 14.50 |
| 平均独立闭环数：完整图 / 骨架 | 1.22 / 0 | 0.73 / 0.73 |
| 平均成员数 | 10.61 | 9.41 |
| 断开的结果数 | 36 | 0 |
| 平均边长 | 7.49° | 8.50° |
| 最长边 | 38.96° | 35.75° |

“结构种类”按连通分量、独立环数、端点数和分叉节点度数组合计数，去掉恒星身份以及度二节点的数量；它不是精确图同构检测，也没有包含曲率和朝向。**不能声称每项多样性指标都增加**：完整图的纯拓扑种类略少，新结果更常出现疏朗的长链，平均边长稍长。改进主要在可选成员变化、长链与局部结构的组合，以及不再被削成相似树形的骨架。

另外以十二颗全部必选的固定星位运行 24 个种子，测试在成员集合和数量完全不变时仍有至少八种拓扑，并出现长链、多个分叉和至少两个环，避免把“换星”和“换人数”当成连线多样性。几何测试另以密集弧采样和球面交点检查线条留空与交叉。

大成员压力检查中，50、100、200 颗全部必选的合成星区，新版加入缓存与稀疏候选前分别约需 0.83、5.61、38.78 秒，优化后约为 0.17、0.54、2.50 秒。这里比较的是本次新版的性能优化前后，不是上表的第一代算法；单次本机计时也不是隔离环境的基准。另以两个相距较远的星团共 48 颗必选星核验跨团连接、亮星保留、节点度数及全部连线合法性，不设置容易波动的测试耗时门槛。

共同尺度图展示前三个指定区域（Z01、Z06、Z11）的前四个种子，没有从结果里挑最好看的例子：[Z01](../reports/zodiac_morphology/comparison/Z01.png)、[Z06](../reports/zodiac_morphology/comparison/Z06.png)、[Z11](../reports/zodiac_morphology/comparison/Z11.png)。每张图的四行依次为旧完整、新完整、旧骨架、新骨架。金色圆环标识必选亮星。

这些是固定星表上的软件与形态检查，不是文化认同或肉眼观测实验；参数仍需结合用户看到的候选继续评审。图形目前仍在独立工作台内，没有应用到正式 V1/V2 天球。

## 复现

```sh
node --test tests/test_constellation_morphology.mjs tests/test_constellation_regional_sampling.mjs
node src/audit_regional_figures.mjs --baseline-ref 498b052cb3de66ce9f13b9439b5bf0067553378b --samples 32 --out /tmp/regional-before.json
node src/audit_regional_figures.mjs --samples 32 --out /tmp/regional-after.json
MPLCONFIGDIR=/tmp/zodiac-matplotlib .venv/bin/python src/plot_regional_figures.py --before /tmp/regional-before.json --after /tmp/regional-after.json --out /tmp/regional-comparison
```

审计命令拒绝覆盖已有 JSON；比较使用同一份提交内的星表与区域配方。历史基线只从指定提交读取局部采样器，公共几何模块仍是当前工作区版本。本轮已核对相关公共模块与输入未受这次修改；该命令不保证任意未来共享几何变更下都能重现这里的旧数字。
