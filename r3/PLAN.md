# E3 协议（冻结；GPT R3/E3 LIMITED GO 修订版）

问题：固定 5 min 预算下，`hurdles-fast-walker@1` 知识包相对随机初始化带来多少迁移收益？同时记录初始能力、学习过程、最终性能与累计知识成本。

## 1. 条件

| | A（Warm-start） | B（Scratch） |
|---|---|---|
| 配置 | `configs/sprint/baseline.py` | `configs/sprint/scratch.py`（`WARM_MLP=None`）+ `DUCK_SCRATCH=1` |
| Actor 权重 | HFW `mv_f37c68365a05` | 逐层 `nn.Linear.reset_parameters()` 重抽 |
| 输入归一化 | HFW 的 sub/div | 相同（实测两份 dump 的 norm 逐位相同） |
| Actor 架构 / Critic 初始化 / INIT_STD / LR / 奖励 / 环境 / 预算 / checkpoint 规则 | 相同 | 相同 |

初始化证据（`r3/artifacts/init_check.json`）：A 的初始化权重与 HFW 文件**逐位相同**（每层 max 0.0，含 bias）；B 与 HFW 的 RMS 差 0.006–0.008（max 0.027–0.068）。两者 normalizer 逐位相同。

## 2. 种子与顺序（GPT 指定）

| 对 | 种子 | 顺序 |
|---|---|---|
| P1 | 101 | A → B |
| P2 | 102 | B → A |
| P3 | 103 | A → B |
| P4 | 104 | B → A |
| P5 | 105 | A → B |
| P6 | 106 | B → A |

每 run 5 min，共 12 run = 60 min 纯训练。本轮总机时上限 90 min（含初始化检查与评价）。

## 3. 评价

TS seeds 8–39（fresh 40–71 封存）；miss = 30 s。主指标 = 导出策略 best-half；辅助 = 全种子均值、worst-half、完赛率、冲线前后跌倒（episode 与事件分开）、T0 与完整曲线、环境步数、loadavg1。

## 4. 统计

d_i = A_i − B_i（负值 = Warm-start 更快）。报告 6 对差值、均值、SD、95% t 区间。**0.2 s 仅作暂定实际意义参考**；0.474 s 不再作为显著性门槛；CI 跨 0 ≠ 等效。Scratch 大量未完赛时单独报告完成率与 30 s 惩罚成绩。

## 5. 成本核算

- 新增：12 × 5 min + 加载/评价机时（预算账本为准），环境步数按 run 记录。
- 上游：HFW 谱系记录 37,450,000 环境步 / 2214 s 训练时间（日志谱系信息，未独立核验）。
- 累计：新增 + 上游；预训练被多任务复用时的摊销价值只作定性讨论。

## 6. 实施记录

- 本机 kit 被最小改动：`lab/learn.py build()` 增加 `DUCK_SCRATCH=1`（重抽 linears）与 `DUCK_DUMP_INIT[_AFTER]`（初始化取证）。两个条件共用同一改动；每次 run 的 `kit_files.json` / `kit_diff.patch` 记录该差异。
- kit 的 `build()` **无论如何都先加载 `WARM_START` 的库存权重**（`lab/learn.py:188`），所以只把 `WARM_MLP` 置空并不是 Scratch——GPT 的 P0-E3-02 在该 kit 中确实成立，已由初始化 dump 证实并规避。
- 不因结果调整奖励、LR、探索强度；失败运行保留。
- 零登录、零提交，不改 Plant，不启动 E2/E4。