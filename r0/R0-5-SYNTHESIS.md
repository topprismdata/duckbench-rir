# R0 阶段汇总（提交 GPT 独立评审）

日期：2026-11-26 · 工作区：`~/1-Projects/duckbench-rir` · 证据：`r0/` + `sources/`

## 0. 结论先行

1. **平台事实被纠正**：本项目的竞赛目标是 https://duckbench.app（10 赛道、duckscript 提交、24 hidden seeds、best-half），**不是** Pollen `microduck-arena`（6 事件、ONNX、5 seeds、sweep）。GPT 的赛道名单正确，但交接包把 Pollen 的协议细节当成了 DuckBench 的协议；两者必须分列。
2. **接口限定了 RQ-A 的可操作化**：move 是固定的 61→14 前馈 MLP（≤1 M MACs、8 MiB、无状态）；"结构化控制"只能体现为 duckscript 节点/切换、raw host 命令语义、sense/相机。CPG/PMTG 类控制器无法提交。当前排行榜第一的做法恰是"同一策略克隆到 launch/drive/finish 三节点、分别训练"。
3. **评估精度不是瓶颈，训练方差才是**：确定性策略在 32 check seeds 上重采样为 hidden-24 后 best-half 的 sd ≈ 0.008 s；因此 R3 必须先测训练 run 间方差（E1），再谈表示差异。
4. **文档数字也要标口径**：stock sprint 实测 22.72 s，文档 24.21 s（差 1.49 s）；`hurdles-fast-walker@1` 5.68 s 与 `base-run-forward@2` 4.52 s、saucer 55.59 rpm 均精确复现。
5. **建议 R1 批复**：只批准 "kit init + E1（训练方差）+ 官方 start 复现" 这一最小动作；R2/R3 的预算与对照另批。

## 1. 四项任务状态

| 任务 | 状态 | 证据 | 关键阻塞 |
|---|---|---|---|
| R0-1 官方协议取证 | DONE（含主审实测补充） | `r0/R0-1.md`；`r0/artifacts/challenges/*.json` | 服务端硬件架构官方未披露 |
| R0-2a Microduck 官方资产 | DONE | `r0/R0-2a.md`（+`R0-2-MAIN.md`） | 电池规格未公开 |
| R0-2b BAM / Mac 实现 | DONE | `r0/R0-2b.md`（+`R0-2-MAIN.md` §4–5） | 未复跑第三方吞吐 |
| R0-3 DeepMind Soccer 复算 | **DONE（可复算）** | `r0/R0-3.md`；`r0/r03-scripts/r03_recompute_soccer.py`（退出码 0） | Fig 7 逐场原始数据未发布（聚合层可复算） |
| R0-4 资源与实验设计 | DONE | `r0/R0-4.md`；`r0/scripts/seed_stats.py` | 未实测训练吞吐（R0 不训练） |

## 2. 与 GPT 研究的差异（需裁决）

| GPT/交接包说法 | 证据 | 判定 |
|---|---|---|
| 10 赛道名单 | skill.md:59–70 | 一致（官方另有 Hurdles/Cannon Dodge/Maze race/duckling-ctf-3 等，10 条是当前菜单） |
| ONNX 61→14 提交契约 | check_mlp.js:17-18,50-51；GUIDE §4 | **错误**：提交是 duck-program/2 + duck-mlp/1；ONNX 只是上游策略的重打包来源 |
| 5 扰动种子 / median replay / sweep | microduck-arena README:239-242 | **平台混淆**：属于 Pollen Arena；DuckBench 是 24 hidden、best-half |
| Microduck Lab 吞吐优先 | `r0/R0-2b.md` | 见该报告 |
| "结构化 vs 直接"可原样做 | 本文件 §0.2 | **不可**；需改写为节点级结构/命令语义/warm-start 三类对照 |


### 2.1 R0-3 的实测结论（独立复算，非转述）

- 数据：Zenodo 10793725 `op3_soccer.zip`，3,309,610 B，sha256 `9a314cba…`；arXiv v2 PDF sha256 `ee3185ff…`。
- **Table 1a 4/4、Table 1b 13/13 格复现**，差值 ≤0.006（舍入噪声）；正文 9 项百分比全部吻合（181%/302%/63%/34%、sim-to-real 13/11/28/5%）。
- **Fig 7 终值**：full 0.7250 > fixed-opponents 0.6677 > oppfrac10 0.4608 > no-teacher 0.4383；作者发布的是**预聚合** y_mean/y_stderr，逐场数据不存在 → 统计量无法从更低层重算（数据级 BLOCKED，非获取失败）。
- 口径问题（对后续研究直接有用）：论文括号值是 **SEM（ddof=1）**，不是 95% CI；Fig 7 的 "95% CI" 实为 **2×SEM**，n=5 时 t 分布应为 2.776×SEM（带宽偏窄 ~28%）；摘要 "34% faster" 是换了助跑初始化的结果，同初始化下 learned 实为 **−3%**；发布数据 `*_euler_*` 文件列名把 yaw 标成 "X pos"。
- 可复用：评测协议/统计口径（n=10/50/5、SEM、排除规则先声明）；不可移植：OP3 20 维 @40 Hz 动作空间与舵机常数。

### 2.2 官方训练栈在 Mac 上不可用（读码）

- `microduck_rl/README.md:24`：「Requires a CUDA GPU (training runs through MuJoCo Warp)」；`pyproject.toml:17-18` 依赖 `mjlab==1.3.0`、`warp-lang==1.12.0`。
- `mjlab/README.md:17`：「mjlab requires an NVIDIA GPU for training. macOS is supported for evaluation only.」
- 结论：本机 M1 Max **不能**跑官方 `microduck_rl` / `microduck-challenges` 的训练；本地训练的唯一成熟路径是 DuckBench kit（MLX learner）。官方栈只能通过 HF Jobs（NVIDIA）使用。这直接决定 R2 的平台选择，也解释了为什么"把 NVIDIA 训练框架移植到 Apple Silicon"不是可行路线。
- 硬件口径差异待核：官方 README 称 15 个舵机 / 25 cm / 800 g；DuckBench plant 的 `joint_names` 是 14 个驱动关节（`manifest.json`）。R0-2a 需给出差异来源（第 15 个舵机是否为未建模的夹爪/颈部）。

### 2.3 资产审查的三个硬发现

1. **执行器口径**：Microduck 是 **Dynamixel XL330**（BAM M6，`microduck_constants.py:163-176`），不是 Feetech；总线 15 舵机，策略 14 维，第 15 个是**嘴**（在总线、不在动作）。15 vs 14 的疑问已闭合。
2. **DuckBench 自加的头命令 EMA**：`[51:55]` 的 EMA 平滑在官方训练/Arena/实机三链路都不存在（官方是分段常量命令 + 动作侧低通）。Sprint 榜首用 `head_pitch=0.72` 改姿态，因此这条差异必须在 R2 的 sim 一致性检查里单列——它可能同时是"可被利用的 sim 差异"和"实体迁移风险"。
3. **BAM 依赖是浮动分支**：官方 `microduck_rl`/`microduck_rl_mac` 的 BAM 来自 `mjlab_frictionloss` 分支（今日 `57d13ead`），而 DuckBench plant pin 在 `62bd8ce`（落后 80 提交）。今天 `uv sync` 得到的执行器模型 ≠ 比赛模型；复现必须以 pin 为准。另：`microduck-challenges` 仓**没有 LICENSE**（默认保留权利），`microduck_rl` 的 STL 是 CC BY-NC-SA（再分发受限）。

## 3. 推荐的最小可证伪实验（R3 候选）

E1 训练 run 间方差（5 seeds × 5 min，Sprint）→ 决定 MDE；E2 表示 A/B（单 move vs 多节点专门化/raw host，2×5×5 min）；E3 warm-start vs scratch（2×5×5 min）；E4 不匹配先验负对照（5–10 min）。合计 ≈2.2 h Mac 时间。判据：hidden-24 模拟 best-half；|Δ| < 2×run 间 sd → UNKNOWN。

## 4. 请求 GPT 裁决的决策

1. 是否接受"平台纠正"（DuckBench ≠ Pollen Arena），并据此作废交接包中基于 Arena 的接口/评分结论？
2. 是否接受 RQ-A 的改写（节点级专门化 / raw 命令语义 / warm-start），把 CPG/PMTG 类结构化控制移出比赛内实验？
3. 是否批准 R1 的最小动作（kit init + E1 + 官方 start 复现，Mac 预算 ≤60 min），R2/R3 另批？
4. R0-3 若无法获得可复算原始数据，是否接受 BLOCKED 并把该论文降级为"方法参考"而非"证据基准"？
5. 指标口径：是否同意把"冲线后撞墙摔倒"从鲁棒性证据中剔除（本报告 §0.3 的实测）？

## 5. 自审与未解决问题

- 未测：训练吞吐、hidden seeds、服务端硬件、登录后链路。
- 反证：文档 stock 基线不可复现；排行榜摔倒计数是撞墙伪影；评估噪声小不等于方法差异显著。
- 尚未完成：R0-2a/2b/3 的独立复核（子任务返回后由主审抽查）。