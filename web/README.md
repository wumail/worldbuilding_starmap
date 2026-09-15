# 网页版本与公共模块

从[版本选择首页](../index.html)进入。**V1/V2 是观察版本；平面/沉浸是每个版本内的两种视图。**

| 目录 | 职责 | 入口 |
|---|---|---|
| `v1/` | 原有标准星图，Canvas 平面绘制与 Three.js 沉浸绘制 | [平面](v1/sky_atlas.html) · [沉浸](v1/star_map.html) |
| `v2/` | 肉眼观看版本，观看尺度、大气天光、可调光扩散；保留专属测试和验证资料 | [平面](v2/sky_atlas.html) · [沉浸](v2/star_map.html) |
| `shared/` | 唯一的恒星系参数及 Kepler 计算、目录加载、投影数学、点星采样、弥散光与黄道天区、共同界面样式 | 由两版引用，没有独立页面 |

两版 `sky_state.mjs`、`sky_controls.mjs`、`sky_view.js` 及各自渲染器保留独立职责；同名不表示可以合并。恒星系参数原先重复存储的 `solar_system.mjs` 已统一到 `shared/`，避免两个版本的物理参数漂移。观察设置键保持不变，因此目录移动不会清空用户配置，也不会把 V1 与 V2 的配置混在一起。

生成代码、参考数据和星表继续位于根目录 `src/`、`data/`、`reference/`、`output/`，不按观察版本复制。

根目录的 `sky_atlas.html`、`star_map.html` 以及 `eye/` 中的少量 HTML 仅为旧链接跳转，没有渲染代码；正式源码只放在上述三类目录。GitHub Pages 的历史 `v4/` 页面通过根 `404.html` 和 `legacy_routes.mjs` 跳转。本地普通静态服务器应从新首页或这些 `web/` 跳转入口进入；它不会自动执行 GitHub Pages 的自定义 404 路由。

过去的验证 JSON 中仍保留当时的路径和指纹，作为历史证据，不能用当前路径去改写原始记录。当前目录迁移验收与新页面截图保存在 `../data/layout_validation_20260915/`。

随后完成的辅助标注透明度、两版高倍星点一致性和连续深空剖面修正记录在 `../data/profile_validation_20260915/`。深空候选的生成数据保持不变；网页绘制与生成方法的验证分别记录。

气体云显示边缘、V1 投影反方向覆盖及辅助线强度的后续修正记录在 `../data/haze_validation_20260915/`；两份记录分别保留修正前后的证据。

星云形态位于 `shared/nebula_morphology.mjs`，由两版四页共同使用。默认目录为 `output_20260915_nebula_03`，采用先噪声后平滑的固定角尺度发光场；记住的原始 4.4、`_nebula_01` 或 `_nebula_02` 目录会升级到这个副本。显式 `?catalog=` 始终优先，因此可以继续打开三个旧批次比较。原始恒星 ID、物理值和星团账本保持不变，形态数据另存；方法见 `../data/nebula_morphology.md`，新的核验见 `../data/nebula_filter_validation_20260915/summary.md`。
