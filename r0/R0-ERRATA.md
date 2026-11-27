# R0 整改回执（追加式勘误，不覆盖原档案）

依据：GPT R0 裁决 DEC-01「知识资产采用追加式勘误」。原件 `duckbench-research-handoff/` 保持只读，本文件只追加。

## 1. 受影响的知识声明

| 原 ID | 原陈述 | 处置 | 证据 |
|---|---|---|---|
| K092 | 「Microduck 官方训练使用 BAM，DuckBench 未核验」 | **核验完成**：DuckBench plant 明确使用 BAM | `plant/microduck-legs-bam-v1/manifest.json`（pin `Rhoban/bam@62bd8ce`）、`plant/*/LICENSE-bam-Apache-2.0.txt`、`kit/duckref/bam.py` |
| （新增）K102 | DuckBench 提交物是 duckscript（duck-program/2）+ duck-mlp/1（61→14），不是 ONNX | 追加 | R0-1 §2.2/§3.2；`dist/check_mlp.js:17-18,50-51` |
| （新增）K103 | 官方 microduck_rl / mjlab 训练要求 CUDA；macOS 仅评估 | 追加 | `microduck_rl/README.md:24`；`mjlab/README.md:17` |
| （新增）K104 | Microduck 执行器为 Dynamixel XL330（BAM M6）；总线 15 舵机、策略 14 维（嘴不在动作） | 追加 | `microduck_constants.py:163-176`；`duck-control/src/obs.rs:8,53-54`；`robotd-design.md:176` |
| （新增）K105 | DuckBench 对头部命令做 EMA（`[51:55]`），官方训练/Arena/实机均无命令侧 EMA | 追加（平台差异，非漏洞判定） | `kit/duckref/plant.py:251-255`；R0-2a §"与 DuckBench plant 的差异" |
| （新增）K106 | 官方 BAM 依赖是浮动分支（R0 快照 `57d13ead`），DuckBench pin 在 `62bd8ce`；**commit 计数（修正见 K108）** | 追加（复现性风险） | `microduck_rl/pyproject.toml [tool.uv.sources]`；`git ls-remote` 快照 |
| （新增）K108 | **提交计数勘误**：`62bd8ce` 是 `57d13ead` 的祖先；`git rev-list --count 62bd8ce..57d13ead` = **27**，反向 = 0。此前 ERRATA 写的"落后 80 提交"错误，属混用不同过滤条件/查询时刻；R0-2b 的 "ahead 27 / behind 0" 正确 | 修正 K106 | 本地 `sources/bam`：`git merge-base --is-ancestor 62bd8ce 57d13ead` 退出码 0；`git rev-list --count` 27/0；`62bd8ce` 2026-07-13 16:42:35 +0200 "Using stiffer frictionloss in BamActuator"，`57d13ead` 2026-07-16 17:21:03 +0200 "Updating favicon" |
| （新增）K107 | 现有排行榜证据显示 Sprint 领先做法是"多节点 + 每节点独立训练 move"（#1 三节点、#3 四节点） | 追加（观察，非因果） | `duckbench result 4evrui8rkyk4 --duckscript` / `q7s8hm7gk49v --duckscript` |

## 2. 需要作废/限定的旧表述（保留原文，仅在此标注）

| 位置 | 旧表述 | 限定 |
|---|---|---|
| 交接包 `research/01-synthesis.md`「DuckBench 定位」 | 「10 赛道…规则/评分/策略格式/隐藏种子/Elo/资源约束尚未官方核验」 | 现已闭环（R0-1 §2–4）；名单一致，另发现 Hurdles/Cannon Dodge/Maze race/duckling-ctf-3 等菜单外名称 |
| 交接包 `agent/START-HERE.md` R0-1 | 「观测动作、状态记忆、推理限制」按 Arena/ONNX 写 | 以 duckbench.app 协议为准（duckscript + duck-mlp/1；24 hidden、best-half） |
| 交接包「ONNX 61/14」 | 作为提交契约 | 仅 Pollen Arena 成立；DuckBench 是 duck-mlp/1（61→14），ONNX 只是上游重打包来源 |
| 交接包 `02-literature.md` P0 资产表 | 把 `microduck-lab` 当作本地训练候选 | 它是第三方 SB3/自定义环境，非 DuckBench 官方 kit；**当前已识别的最直接可提交训练路径**是官方 kit（不排除存在其他未验证路径） |

## 3. 假设台账（不改变状态）

H001–H042 **仍全部为 `UNTESTED_NOT_FROZEN`**。R0 只做证据调查与复算，未产生任何可支持/反驳假设的训练或对照实验。GPT 裁决要求 E2/E3/E4 暂不批准，因此本轮不得把任何假设改为 SUPPORTED/REFUTED。

## 4. 新增"接口例外"

- 相机 move（`duck-cam/1`）是**实验性**接口：61 维 stock 输入 + emb1 特征；因此"61 维是所有 move 的绝对上限"这一表述不成立，应写"标准 move 61 维；实验性相机 move 额外带相机特征"。

## 5. 版本与许可冻结（供 R2 使用）

| 资产 | 冻结值 |
|---|---|
| DuckBench 包 | `0.3.87`，tgz sha256 `5a1437de…`，大小 30,569,928 B |
| Plant | `microduck-legs-bam-v1`，BAM pin `62bd8ce121…`，microduck_rl pin `cb70b792…`，mujoco 3.14.0 |
| 官方 microduck_rl | 本地 HEAD `0daab160…`；R0 报告以 pin `cb70b792…` 与 HEAD 的差异已核对（robot xml/actuator/velocity cfg/constants 不变） |
| 许可 | microduck_rl 代码 Apache-2.0 + STL CC BY-NC-SA 4.0；microduck-challenges **无 LICENSE**；microduck-lab / microduck_rl_mac / bam Apache-2.0；bam-feetech 无 LICENSE |
## 6. GPT R1 评审提出的整改（追加）

| 项 | 处置 |
|---|---|
| 跌倒口径 episode vs event 混用 | `r1/scripts/e1_stats.py` v2 已拆为 `episodes_with_pre/post_finish_fall`、`pre/post_finish_fall_events`、`episodes_mixed_unattributable`；本轮 208 事件、147 episode、91.9 % |
| best-half 精度 | 改用 TS `result` 字段（3 dp），与 GPT 复算一致：4.3547/3.7663/4.3044/4.1803/4.0661，sd 0.2344 |
| 本机资源竞争 | 每 run 记录 `loadavg1`（median/max：13.81/18.67、13.58/16.36、17.36/29.03、15.64/24.80、15.20/19.07）与 steps/s；E1 测量的是"固定墙钟预算 + 本机负载"下的总方差，不是纯种子方差 |
| commit 计数矛盾（27 vs 80） | 见 K108：27 ahead / 0 behind，`62bd8ce` 为祖先 |
| manifest 混用 | 待拆为 `PACKAGE-SHA256`（制品）与 `SOURCE-REGISTRY`（外部来源），见 R1 补证包 |
