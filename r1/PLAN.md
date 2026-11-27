# R1 执行计划（已获 GPT DEC-03 LIMITED GO；等待 L0 用户授权本机 60 分钟预算）

## 已授权

- `duckbench kit init duck-rl`（一次性 Python 3.12 环境，~1 GB）
- 官方 start 复现（sprint/saucer/wheel/steps，TS seeds 8–39）
- **E1**：5 个预冻结训练种子 × 5 min（训练预算 25 min）
- 环境/吞吐/导出检查；独立 TS 评价与统计

## 未授权

E2（节点专门化）、E3（warm-start）、E4（不匹配先验）、任何提交、扩大工程范围。

## E1 协议（冻结）

| 项 | 值 |
|---|---|
| 挑战 | Sprint v6（`sprint`） |
| 配置 | `configs/sprint/baseline.py`（起点 `hurdles-fast-walker@1`，5.68 s） |
| 训练种子 | `1,2,3,4,5`（预注册，不挑选、不替换） |
| 每次预算 | 5 min，`--detach`，`--model` 留空（未声明模型） |
| 检查点 | T0 + 每次 eval（kit 自带 ~30 s/次）+ 结束导出 |
| 评测 | `duckbench simulate sprint submissions/<label>/submission.json --seeds 8-39`；fresh 40–71 封存 |
| 指标 | 全 32 种子：mean、best-half、worst-half、完赛率、finish 前后跌倒 |
| 失败处理 | 崩溃/退化/中断全部保留，记入台账，不重跑替代 |
| 统计 | 报告 run 间 sd 与区间；**不用 2×SD 作显著性判据** |

## 命令模板

```bash
export DUCKBENCH_HOME=$PWD/.duckbench-home
export PATH="$PWD/sources/dbcli/node_modules/.bin:$PATH"
duckbench kit init duck-rl                 # 一次性
cd duck-rl
uv run python -m duck_rl.budget --set 60   # 总预算（含 init/eval）
for s in 1 2 3 4 5; do
  uv run python train.py --config configs/sprint/baseline.py \
    --label e1_s$s --minutes 5 --seed $s --detach
  # 轮询 status.json 直到非 running
done
```

## 交付

`r1/E1.md` + `r1/artifacts/`：每次训练的状态、曲线（`summary.json` 的 curve）、TS 评测 JSON、失败记录、run 间方差表、R2 建议。

## 门禁

1. 5 个种子全部落盘（含失败）；
2. 每条曲线有 T0 与 ≥3 个检查点；
3. 所有评价用同一 TS seeds 8–39；
4. 方差报告含 worst-half 与完赛率（GPT 要求）；
5. 若 run 间 sd > 0.5×目标效应，写明"当前预算下不可辨识"，不进入 E2。