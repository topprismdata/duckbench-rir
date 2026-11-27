# DuckBench 研究知识库（KG v0.1）

设计依据：GPT 的《DuckBench Research Ontology & Knowledge Graph v1.0》。

## 结构

- `ontology/ontology.json`：29 个实体类型（ID 前缀 + 必填字段）、关系类型、枚举。
- `data/*.jsonl`：**唯一权威事实源**（Git 版本化）：
  - `domain.jsonl`（Competition/Task/Metric/Environment/Host）
  - `assets.jsonl`（RewardVersion 等资产）
  - `experiments.jsonl`（Experiment/TrainingRun/EvaluationRun/MetricResult）
  - `evaluations.jsonl`（同上补充）
  - `knowledge.jsonl`（Hypothesis/Claim/Counterexample/Exclusion/Decision/Literature/Session/Budget/Scope/SeedSet/Protocol/Artifact/Attempt/Leaderboard）
  - `relations.jsonl`（SUPPORTS/REFUTES/TESTED_BY/…）
- `scripts/kb/validate.py`：结构 + 引用 + 闭环门禁（claim 必须有证据、closed 假设必须有 TESTED_BY、排除必须有 reason）。
- `scripts/kb/build.py`：生成 `generated/graph.sqlite` 与 `generated/GRAPH.md`（只读生成物）。
- `scripts/kb/query.py`：`support <id>` / `open` / `exclusions` / `chain <id>`。
- `scripts/kb/test_gates.py`：变异测试，证明门禁真的会拦（4/4 通过）。

## 纪律

1. 事实只写 JSONL；`generated/` 不手改；旧 K 文档移入 `legacy/` 或保留原文并在 `source_refs` 引用。
2. 每轮实验：EXP → RUN/EV/RES → CLM（或 CEX）→ 更新 relations；`validate.py` 必须 exit 0。
3. 结论被推翻不删，新增 CEX/REFUTES 与 `supersedes` 边。
4. 提交 champion 前，`query.py support <CLM>` 必须能列出到 ATT/LB/EV 的证据链。

## 当前规模

88 实体 / 32 关系；claims 13；hypotheses 6；exclusions 3；champions：Steps 5.70、Hurdles 2.03（Sprint 2.40 #4）。
