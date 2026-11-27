# E3 最小对照设计（提案，等待审批；不自动启动）

问题：有限预算（5 min/run）下，warm-start（`hurdles-fast-walker@1`）相对 scratch 贡献多少？

## 设计（配对、交替、同一评价集）

| 项 | 取值 |
|---|---|
| 条件 | A = warm-start（`configs/sprint/baseline.py` 不改）；B = scratch（同配置，`WARM_MLP` 置空 + `INIT_STD` 默认） |
| 训练种子 | 预注册 6 对：`(1,2) (3,4) (5,6) (7,8) (9,10) (11,12)`，同对内两个条件共用种子号 → 按对分析差值 |
| 顺序 | 每对交替：奇数对 A→B，偶数对 B→A（抵消负载漂移） |
| 预算 | 每 run 5 min；6 对 × 2 = 12 run = **60 min 纯训练** |
| 评价 | TS seeds 8–39；报告 best-half、mean、worst-half、完赛率、跌倒（前/后，episode 与事件分开） |
| 记录 | loadavg1 中位/峰值、steps、actual minutes、checkpoint 曲线、每次 run 哈希 |
| 分析 | 配对差值 d_i = A_i − B_i（秒）；报告均值、sd(d)、n=6 的 95% CI（t 分布）；不把 2×SD 当显著性；若 CI 跨 0 则结论为"不可辨识"，不追加预算 |

## 成本核算（GPT 要求）

- 新增训练成本：60 min 机时。
- 上游知识成本：warm-start 父策略 `hurdles-fast-walker@1` 的谱系记录约 3745 万环境步 / 2214 s 训练时间（日志谱系信息，未独立核验）；E3 报告须同时列"新增成本"与"累计知识成本"。

## 判据

- 若 |mean(d)| > 0.474 s（E1 MDE 量级）且 CI 不跨 0 → 可辨识的迁移效应；
- 若 CI 跨 0 → 写"6 对不足"，不申请自动扩样；
- 任何"更便宜"的替代（减少 run 数、改 5 min）都需另行批准。

## 不批准则不执行

本文件仅为提案；未获批准前不启动任何训练。