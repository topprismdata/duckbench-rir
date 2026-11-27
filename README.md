# duckbench-rir — DuckBench 机器人运动智能研究（R0–R6）

研究工作区：平台取证 → 训练方差 → 知识继承 → 知识迁移 → 知识组织 → 命令条件化。

## 当前状态（2026-11-27）

- **阶段：R6 SYNTHESIS**（研究档案冻结，零训练）。
- 已完成：R0、R1/E1、R2、R3/E3、R4/E4、R5（E2）、R6/E2.1。
- 累计纯训练：**205 分钟**（kit 预算账本 224.38 min，含 preflight/探针）。
- 零登录、零竞赛提交；fresh seeds 40–71 未使用；Plant 未修改。

## 时间线

| 阶段 | 结论 | 报告 |
|---|---|---|
| R0 | 平台纠正（duckbench.app）、执行器 XL330、头命令 EMA、BAM pin | `r0/` |
| R1/E1 | sprint 5×5 min：3.77–4.36 s，sd 0.234 s | `r1/E1.md` |
| R2 | 三项门禁 PASS（限工况） | `r2/G01.md`–`G03.md` |
| R3/E3 | warm-start 192/192 vs scratch 0/192 | `r3/E3.md`、`r3/R31.md` |
| R4/E4 | run-forward 3.31 s 最优；sidestep 部分迁移 vs scratch 0 | `r4/E4.md`、`r4/RECIPE-CHECK.md` |
| R5/E2 | 跨步态拼接退化；节点拆分非中性 | `r5/R51.md` |
| R6/E2.1 | T1 低速 +13.5 %（n=3）；T2 无额外收益 | `r5/E21.md` |

综合：`r6/SYNTHESIS.md`；结论边界：`r6/CLAIM-BOUNDARIES.md`。

## 目录

- `sources/` 只读上游（duckbench 包、microduck、bam 等）
- `duck-rl/` 官方 kit（三处最小补丁，逐 run 记录于 `kit_files.json` / `kit_diff.patch`）
- `r0/`–`r6/` 各阶段报告与 `artifacts/`（TS 评价 JSON、训练日志、初始化张量）
- 清单：`PACKAGE-SHA256.txt`（根，覆盖全部登记文件）、`r1/SOURCE-REGISTRY.md`（外部来源）

## 复现要点

```bash
export DUCKBENCH_HOME=$PWD/.duckbench-home
export PATH="$PWD/sources/dbcli/node_modules/.bin:$PATH"
duckbench simulate sprint <strategy.json> --seeds 8-39 --json   # 评价
cd duck-rl && uv run python train.py --config configs/e21/t1.py --label x --minutes 5 --seed 201
```

## 未做

- E4-S2、T2 扩样、竞赛登录与提交、fresh seeds 40–71；下一阶段为候选研究设计，不自动启动。

## Competition track (2026-11-27)

- **Steps v4: #1, record 5.70 s** — https://duckbench.app/challenges/steps/attempts/v1bp96ckz5uw?by=topprism&s=agent
- Sprint v6: #4, 2.99 s · Wheel v2: #4, 5.25 s · Hurdles v4: #3, 2.83 s · Saucer: #4, 55.44 rpm · Turntable: #3, 10.27 rpm
- Knowledge base for the competition push: [`kb/`](kb/README.md) (findings K01–K11, plan, champion-program analysis)
- Attempt archive: [`r18/PARTICIPATION.md`](r18/PARTICIPATION.md); Steps recipe: [`rb/K03`](kb/K03-steps-champion-recipe.md) / [`r14/STEPS.md`](r14/STEPS.md)
- Local evaluation uses `DUCKBENCH_NO_UPLOAD=1`; only ranked submissions upload.
