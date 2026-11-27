# E4-S1 冻结协议（GPT E4 DESIGN SIGN-OFF 修订版）

定位（GPT DEC-01）：**不同来源运动知识包在固定预算下的任务适应能力筛选实验**。不宣称能独立识别 H1/H2 的因果贡献。

## 1. 实验臂

| 臂 | 先验 | host / 命令（启动自检实测） | 归一化 | 角色 |
|---|---|---|---|---|
| A0 | `hurdles-fast-walker@1` | walk，vx 0.19、head 0.6 | HFW | 匹配 + 强 T0（5.68 s） |
| A1 | `base-run-forward@2` | **raw**，vx 0.7 | 该 move | 同轴、最强 T0（4.52 s） |
| A2 | `base-hop-forward@2` | **raw**，vx 0.4 | 该 move | 同轴、跳步态（6.75 s） |
| A3 | `base-sidestep-left@1` | **raw**，vy 0.2 | 该 move | 正交方向（T0 0/32） |
| A4 | 同 A3 程序与配置，`DUCK_SCRATCH=1` | raw，vy 0.2 | **与 A3 相同** | A3 的同期随机权重对照（仅 Actor 权重不同） |

接口门禁（M1）已核：全部 61→14、4 层、196,864 MACs；host/command 由各自 `duckbench start sprint --move …` 的程序继承，preflight 日志见 `r4/artifacts/pre_a*.out`。
A4 的 `DUCK_DUMP_INIT_AFTER` 每 seed 保存（`r4/artifacts/a4_init_s<seed>.pt`）；s107 抽检：四层 SD 0.07404 / 0.02553 / 0.03619 / 0.05087，与默认初始化理论值一致，normalizer 与 A3 完全相同。

Stand Still 不再作为臂：其 T0 在 32/32 episode 中跌倒共 69 次（本机复核与 GPT 复算一致），不是稳定站立先验。

## 2. 种子与顺序（GPT 指定）

| seed | 顺序 |
|---|---|
| 107 | A0 → A1 → A2 → A3 → A4 |
| 108 | A2 → A3 → A4 → A0 → A1 |
| 109 | A4 → A3 → A1 → A2 → A0 |

15 run × 5 min = 75 min 纯训练；本轮总机时上限 100 min。

## 3. 评价（每 run 三项）

| 项 | 内容 |
|---|---|
| T0 | 训练前策略：A0–A3 为该先验 move 直接上 Sprint；A4 为该 seed 的随机初始化权重（导出后评价） |
| Best | 按冻结规则导出的 `submission.json` |
| Last | `last.pt` 导出（完整 5 min 后的权重） |

指标：best-half、mean、worst-half、完赛率、torso_reach、x/y 位移（终值与最大值）、偏航变化、跌倒事件与 episode 数、环境步数、loadavg1、in-loop 曲线。

## 4. 统计与解释边界（GPT 给定）

- n=3 只做探索性筛选；不宣称普遍规律；不使用任意 1 s 门槛；不自动扩样（E4-S2 未批准）。
- A3 vs A4 是重点受控对照（共享 host/command/normalizer/配置，仅权重不同）。
- E3 的 HFW/Scratch 六次运行只作历史锚点，不并入本次统计。
- 允许的解释表（来自 GPT）：A3>A4 且完赛 → 非前向知识有迁移价值；两者都未完赛但 A3 运动更好 → 部分迁移；差异不明显 → 当前预算未识别；A4 明显优于 A3 → 负迁移候选；A1>A0 / A2>A1 只能说明该知识包优势，不能归因匹配度。

## 5. 执行与失败纪律

- 零登录、零提交、不改 Plant、不用 fresh seeds（40–71 封存）、不启动 E2/E4-S2。
- 所有失败、中断、退化运行保留并登记；不替换种子。
- 训练驱动 `r4/scripts/e4_driver.sh`；评价 `r4/scripts/e4_eval.sh`；统计 `r4/scripts/e4_stats.py`。