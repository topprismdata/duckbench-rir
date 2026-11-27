# R2 最小验证建议（供 GPT 审批；每项都能在本机零训练完成）

目标：用最小可核验步骤把 G01/G02/G03 从"已识别差异"推进到"已验证或已测量"。全部只读/离线，不训练、不提交。

## R2-G01 执行器身份一致性（预计 20 分钟）

| 步骤 | 命令/文件 | 判据 |
|---|---|---|
| 1. 型号与 BOM | 读 `plant/microduck-legs-bam-v1/manifest.json` + `microduck_constants.py:163-176` | 得到 XL330-M288-T 与 15 个 servo id；对照 `robotd-design.md:176` |
| 2. 参数映射 | `git grep -n "XL330\|M288\|gear_ratio\|friction" -- plant/ microduck_rl/` | 每个关节的 `actuatorfrcrange`/`gear`/`frictionloss` 来源可指到具体行 |
| 3. 数值对账 | 固定初始状态 + 相同 ctrl，分别驱动 BAM pin `62bd8ce` 的 Python 实现与 DuckBench plant，记录 1 s 轨迹 | 位置差 < 1e-6 m 视为等价；否则记录差值与来源字段 |
| 4. 负对照 | 用 Feetech 参数替换后同样驱动 | 必须出现可测差异（否则说明参数没生效、映射读错） |

产出：`r2/G01.md` + `r2/artifacts/actuator_trace.json`。

## R2-G02 头命令 EMA（预计 30 分钟，零训练；策略影响部分需一次前向）

| 步骤 | 内容 | 判据 |
|---|---|---|
| 1. 观测语义 | `kit/duckref/plant.py:251-255` 读实现 | 冻结 `[51:55]` = (yaw_rate, pitch, body_yaw?) 实测取值标记；写清 EMA 系数、频率（每个 physics step 或每 policy tick）、初始化、reset |
| 2. 阶跃响应 | 直接在 TS sim 里给固定 head 命令，记录 `[51:55]` 随时间序列 | 拟合时间常数 τ；与代码系数一致则确认实现无误 |
| 3. 动作差异 | 同一权重（`hurdles-fast-walker@1`）在 DuckBench sim 与官方 Python 环境中，喂相同 61 维观测 | 动作差 = 0（若输入相同）→ EMA 只影响观测分布 |
| 4. 分布偏移 | 统计官方三链路 61 维观测中 `[51:55]` 的分布（若有训练数据/回放），与 DuckBench 对比 | 报告 KL / 均值差；无数据则标"未核实" |
| 5. 物理影响 | 用 `--no-ts-eval` 之外的方式固定策略，对比 head 命令有无 EMA 两种 plant 的 32 种子结果 | 只在有差异时报告效应量 |

特别纪律（GPT 裁决）：不预判为漏洞；榜首 head_pitch=0.72 与 EMA 的相关性只作为观察记录。

## R2-G03 BAM 版本一致性（预计 40 分钟）

| 步骤 | 内容 | 判据 |
|---|---|---|
| 1. SHA 冻结 | `git -C sources/bam rev-parse 62bd8ce`；`git ls-remote` 快照 `57d13ead`；记录 `pyproject.toml`/`uv.lock` | 两 SHA 完整 40 位与日期入库 |
| 2. 差异清单 | `git log --oneline 62bd8ce..57d13ead -- mujoco/` + `git diff --stat` | 只列与动力学相关的提交（摩擦、执行器、积分） |
| 3. 哈希对账 | 对 `plant/*.xml`、`bam.py` 相关文件算 sha256，与 pin 仓库同路径比对 | 逐文件 same/changed 表 |
| 4. 数值对账 | 相同输入下两版本执行器输出（力矩/位置）对账 | 差 0 → 版本差异不影响；非 0 → 记录最大偏差与触发条件 |
| 5. 禁则 | 不修改官方环境求表面一致 | 记录在案 |

产出：`r2/G03.md` + `r2/artifacts/bam_hash_table.json`。

## 计划顺序与预算

1 → 2 → 3，串行；总机时 ≤90 分钟（零训练、零提交）。若 GPT 要求"策略影响"实验，则复用 E1 导出的权重（不再训练）。