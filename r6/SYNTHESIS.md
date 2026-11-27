# R6 — 研究档案综合（E1 / E3 / E4 / E2 / E2.1）

状态：**SYNTHESIS（零训练、零提交）**。依据 GPT DEC-05：E2.1 CLOSED；进入 R6-SYNTHESIS，整理完整研究档案、正负结果与证据等级。

## 1. 时间线与资源

| 阶段 | 内容 | 机时（训练） | 结果 |
|---|---|---|---|
| R0 | 平台/资产/论文取证 | 0 | 平台纠正（duckbench.app）、执行器 XL330、EMA、BAM pin |
| R1/E1 | sprint 5 seeds × 5 min | 25 min | best-half 3.77–4.36 s；sd 0.234 s |
| R2 | 三项门禁（执行器/EMA/版本） | 0 | 全部 PASS 但范围限定 |
| R3/E3 | warm-start vs scratch，6 对 | 60 min | A 192/192 完赛 vs B 0/192 |
| R4/E4 | 五个知识包 × 3 种子 | 75 min | run-forward 3.31 s 最优；sidestep 部分迁移 vs scratch 0 |
| R5 | E2 跨步态拼接 + 诊断 | 0 | 跨步态拼接显著变差；拆分非中性 |
| R6/E2.1 | 命令条件化 T0/T1/T2 × 3 种子 | 45 min | T1 低速 +13.5 %；T2 无额外收益 |

累计训练机时：约 205 min（kit 预算账本 224.38 min，含 preflight 与探针）。

## 2. 结论登记（含证据等级）

| ID | 结论 | 证据 | 等级 |
|---|---|---|---|
| K/P 系列 | DuckBench 平台与提交契约（duckbench.app、duckscript、duck-mlp/1） | R0-1 | 文档+实测 |
| R0 结论（未编号） | 平台/资产/论文复算（Table 1a/1b 全中；"95% CI" 实为 2×SEM） | R0-1、R0-3 | 独立复算 |
| FINDING-005 | 相同 30 s 惩罚掩盖不同运动行为（B105 最终检查点 280 次跌倒 vs 早期 0） | R3.1 | 实测 |
| FINDING-006 | 非前向运动先验（sidestep）相对随机权重有部分迁移（10.7/32 vs 0/32） | E4 | 探索性（n=3，CI 跨 0） |
| FINDING-007 | 计分最优 checkpoint ≠ 全程最稳（A1 best→last 跌倒 75→38，时间 +2.6 %） | E4 | 实测 |
| FINDING-008 | 节点拆分改变闭环行为（同策略 26/25/28 完赛随阈值） | R5.1 | 实测（确定性已排除噪声） |
| FINDING-009 | 所测策略、切换条件与评价种子下，跨步态组合出现明显任务性能退化，且退化随切换高度阈值升高而增加（HFW→Steps 10/7/2 完赛） | R5.1 | 行为层（非统计显著性） |
| FINDING-010 | 命令分布随机化改善低速命令表现（7.75→6.71 s，三个训练种子同向） | E2.1 | EXPLORATORY POSITIVE / n=3 TRAINING RUNS |
| FINDING-011 | 5 min 预算下阶段命令训练无额外收益 | E2.1 | 探索性负结果 |
| FINDING-012 | 命令值分布适应 ≠ 时序调度能力（T1 每 episode 抽一次） | E2.1 | 候选/未验证 |
| 观察 | E3：warm-start 192/192 vs scratch 0/192（scratch 最终检查点亦 0） | E3 | 强（6 对，CI 远离 0） |

## 3. 资源口径（不能只看 5 min）

| 实验 | 臂 | wall-clock | 环境步数（均值） | 备注 |
|---|---|---|---|---|
| E3 | warm-start | 301 s | 7,606,272 | 6 run |
| E3 | scratch | 301 s | 6,355,627 | 6 run（−16.4 % 步数） |
| E4 | A3 sidestep | 301 s | 9,065,813 | 3 run |
| E4 | A4 sidestep-scratch | 301 s | 7,531,179 | 3 run（−16.9 %） |
| E2.1 | T0 | 301 s | 8,972,971 | 3 run |
| E2.1 | T1 | 301 s | 8,907,435 | 3 run；命令分布 U(0.10,0.19) |
| E2.1 | T2 | 301 s | 9,011,200 | 3 run；命令 0.10→0.19 |

结论：相同 5 min 不代表相同学习机会；跨臂比较须并列步数与命令分布（GPT R6 方法学要求）。

## 4. 可复现性限制（GPT 列出 + 本地补充）

1. **R5 包缺独立文件级 SHA256 清单的争议**：本包在 zip 根部提供 `PACKAGE-SHA256.txt`（覆盖所有登记文件，除自身）；若评审要求"包外清单"，需单独附件交付。
2. 训练引用的 `programs/steps.json`、`programs/steps_vx010.json` 已补入 `r5/artifacts/`；但历史 run 的实际程序以各 run 的 `program.plain.json` 为准。
3. 上游知识成本（HFW、steps-starter 的谱系步数/时间）来自日志谱系，未独立核验。
4. TS 内部状态（`prev_action`、头 EMA）不可观测；Python 参考实现的读码证据不等于 TS 逐位一致。
5. kit 的三处最小补丁（`lab/learn.py` 的 scratch/dump、`train.py` walk-host COMMAND_RANGES、`duck_rl/env.py` 命令抽样/CMD_STAGE）逐 run 记录于 `kit_files.json` 与 `kit_diff.patch`。
6. E1 的方差含本机负载竞争（load1 10–29），不是纯种子方差。
7. 所有结论限于 DuckBench 冻结 Plant、Sprint v6 / Ducky Steps v4、5 min 预算。

## 5. 失败与负面结果（全部保留）

- E3：scratch 6/6 全 0/192（负结果）；a103 基础设施中断后重跑（已披露）。
- E4：A3 部分迁移但 CI 跨 0；A1 计分与稳定性不一致（FINDING-007）。
- E2：跨步态拼接 C4/C5 显著变差；Z3 的"收益"因 0 s 切换被撤回。
- E2.1：T2 无额外收益；T1 的低速收益伴随冲线后跌倒增加（冲线前未见恶化）。
- 评价协议错误一次（staged 只绑单节点 → 全 0/32），已修正并记录。

## 6. 方法与纪律沉淀（可复用）

1. 初始化取证必须在 reset 之后 dump；否则会把库存权重误当随机初始化（E3 教训，GPT 抓到）。
2. 跌倒必须拆成 episode/事件与冲线前/后；30 s 惩罚会把不同行为压成同一数字（FINDING-005）。
3. 跨臂对照要冻结评价协议并同口径（E4 A4 plain program 等价性、E2.1 staged 双节点绑定）。
4. 阈值/边界先冻结再跑，不做结果驱动搜索（R5.1）。
5. 同一程序重跑逐 seed 一致 → 可以把差异归因于程序变化（D1 确定性）。
6. 训练包最小补丁必须可审计（kit_diff + kit_files）。

## 7. 下一步候选（均未启动）

| 候选 | 问题 | 预算估计 |
|---|---|---|
| T1 扩展 | 命令范围更宽（0.05–0.19）与速度-稳定性权衡 | 3 arms × 6 seeds × 5 min = 90 min |
| T2′ | episode 内连续命令调度（真正的时序适应） | 需先设计调度接口，预算另议 |
| 跨挑战迁移 | steps 上训练的 T1 策略迁到 hurdles/cliff | 零训练评价 + 可选微调 |
| 竞赛提交 | 需用户登录与授权 | 未授权 |

## 8. 交付清单

- 报告：`r0/`（R0 系列）、`r1/E1.md`、`r2/G0*.md`、`r3/E3.md`+`R31.md`+`FINDING-005.md`、`r4/E4.md`+`RECIPE-CHECK.md`、`r5/E2-DESIGN.md`+`R51.md`+`E21-D1.md`+`E21.md`、本文件。
- 数据：各阶段 `artifacts/*.json`（TS 原始评价）、训练日志与 `summary.json`、初始化张量。
- 清单：`r1/PACKAGE-SHA256.txt`、`r1/SOURCE-REGISTRY.md`。