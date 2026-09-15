# 主序星参考序列

`main_sequence.csv` 提取自 Eric Mamajek 的 **A Modern Mean Dwarf Stellar Color and Effective Temperature Sequence**，版本 **2022.04.16**，访问日期 **2026-09-13**。

来源：https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt

按照原表要求引用：Pecaut & Mamajek (2013), ApJS, 208, 9。在线表补充了论文原表没有的部分物理参数。

仅保留 O3V 至 M9.5V 的 87 行，字段逐一对应原表的 SpT、Teff、Msun、logL、BCv。没有改动这些观测参考值。半径与绝对星等由这些值及 IAU 2015 热光零点重新推导，以避免原表各列舍入误差进入恒等式。

插值沿 log(Teff) 轴进行；质量与光度取对数插值，BC 线性插值。A 型星质量的原始经验估计不完全单调，不能简单按质量排序再连接亚型。

这是一条经验均值序列，不是含年龄、金属丰度和双星演化的等龄线网格。主序亚型的抽样权重采用分段 Kroupa 型 IMF（低于 0.5 M☉ 指数 −1.3，以上 −2.3）以及恒定形成率下的近似存活比例 `min(1, max(0.0003, M^-2.5))`。其作用是避免把罕见高质量亚型当作等概率；不是精确恒星形成史。每个参考区间以中点权重近似积分。

演化星不使用本表冒充演化轨道，仍采用代码中明确记录的经验范围，并同时约束质量、半径、表面重力和光度。

## 地球亮星参照与总体校准

- `bsc5_reference.json`：从 [BSC5 原始目录](https://cdsarc.cds.unistra.fr/ftp/cats/V/50/catalog.gz) 提取的统计摘要，来源为 Hoffleit & Warren (1991)。记录原始压缩文件与解压内容的 SHA-256。总目录 9110 条，其中有效恒星 9096 条；严格按 Vmag 字段取 `≤6.5` 是 8404 条，不能用总条目数代替同星等样本数。
- `population_calibration_pilots.json`：4.1 模型在种子 20260913、42、1369、2718 下的四组完整生成统计，保存训练参数与恒星物理代码指纹。它们只用于估计各群体每单位密度能产生多少可见星。
- `population_profile.json`：由 `calibrate_population.py` 从上述统计估计出的**总体密度**、归一化丰度与尺度高度。生成器加载这些固定先验，随后继续生成 Poisson 盘状总体和逐星亮度筛选。
- `population_acceptance.json`：4.2 校准后用种子 0–11 进行的独立验收结果，含数量、星等累计数、粗光谱数、银纬统计和物理验证。与训练种子无重合，作为历史基线保留。
- `count_contract_acceptance.json`：4.3 的 24 个根种子数量合同验收；默认密度乘以 `9250/8404`，只接受 9000–9500 颗的整份总体实现，记录所有尝试。
- `rendering_acceptance.json`：实际 GPU 的亚像素、符号单调性、2K/4K 合成方向锚点与 PNG 读回结果。临时测试图可以用 `tests/check_rendering.py` 重建；成品另有逐星 `.render.json`。
- [生成与图片正确性说明](generation_and_rendering_validation.md)：当前合同、数学依据、实测证据与限制。
- `final_validation.json`：最终成品的物理验证、图片验收摘要、代码与成品指纹对应、旧17个成品的保留核验。

目录中的复合谱、范围和不确定光度级不强行指定演化阶段。缺失的阶段比例由同光谱型的已分类样本估计，并加入 5 颗样本强度的弱先验以避免把罕见群体置零。光度级 I 与 0 共享拟合；M 矮星的可见样本太少，保留其原有局部密度先验。C/S/W/N 等本模型没有实现的粗光谱类别及未识别条目共 25 条，不伪造为其他类别。

此校准涉及历史目录条目与部分未分辨多星系统，不能等同于现代精密亮星表或每个人实际肉眼可见的星数。空间尺度高度、尘埃模型和各类恒星内部的细分物理分布没有随此次数量校准重拟合。更亮星等区间仍有偏差，见 [完整校准报告](calibration_report.md)。

下载原始 `catalog.gz` 后，可离线重建校准文件：

```sh
python3 src/calibrate_population.py /path/to/catalog.gz data/population_calibration_pilots.json --output data
```
