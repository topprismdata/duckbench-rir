# 知识库的认识论（v0.2，GPT 定稿框架）

主张：这个项目表面是机器人竞赛，实质是一个**新手如何成为专家**的可观测过程；比赛是测量仪器，知识库是成长档案。企业知识库/新人培养可以用同一套结构描述。

## 1. 三种知识来源（认识论三层）

| 层 | 名称 | 例子 | 特征 | 现有实体 |
|---|---|---|---|---|
| K1 | **陈述性知识**（Declarative） | 论文、GUIDE、plant 规范、物理约束 | 可引用、可验证但非亲历 | `LiteratureSource`、`Task`、`MetricDefinition`、`HostContract` |
| K2 | **见证性知识**（Testimonial/External） | 冠军公开程序、榜单、他人经验、Arena bot | 来源可信度差异大，可能不可复现 | `ProgramAsset`（他人）、`LeaderboardSnapshot`、`CompetitionAttempt`（他人） |
| K3 | **经验性知识**（Experiential/Procedural） | 自己的训练、失败、诊断、技巧 | 有一手证据与指纹，可复现性最高 | `Experiment`、`TrainingRun`、`EvaluationRun`、`Claim`（本库） |

## 2. 成长阶段（能力阶梯）

| 阶段 | 能力 | 证据 |
|---|---|---|
| Novice | 复述规则、读文献 | 能引用 K1 且不误用 |
| Apprentice | 复现基线 | 本地评价复现文档数字 |
| Practitioner | 可控实验、排除法 | 单变量实验 + 证伪入库 |
| Expert | 机制解释、迁移 | 假设被支持/证伪并可跨任务 |
| Master | 创造新方法/新赛事方法 | 方法被复用、冠军可归因 |

## 3. 学习事件（可记录的动作）

Read → Imitate → Probe → Experiment → Reflect（写 K） → Generalize（跨赛事） → Teach（清单/方法）。

每条事件是一条边，带时间、成本、结果、证据。

## 4. 度量（成长可计算）

- **覆盖率**：任务 × 阶段 的知识是否齐备。
- **证据强度分布**：ANECDOTAL/EXPLORATORY/REPLICATED/STRONG/FALSIFIED。
- **假设吞吐**：提出 → 证伪/支持的周期（小时/实验数）。
- **迁移率**：某任务学到的做法在另一任务被复用并有效。
- **复用率**：决策（DEC）引用了多少已有证据；重复错误（EXC 违反）次数。
- **路径长度**：从零到某赛事冠军的最短学习路径（节点数）。

## 5. 防止把哲学当口号

- 所有"成长"论断必须落到实体/边与证据；无证据的阶段标记 `UNSUPPORTED`。
- 外部知识（K2）与自证知识（K3）冲突时，保留双方（SUPPORTS/REFUTES），不静默覆盖。
- 被推翻的结论保留为 CEX/REFUTES，历史不可改写。

## 6. v0.2 落地点（已实现）

- **四维分类**取代"三层等级"：`origin`（FOUNDATIONAL/EXTERNAL/SELF_GENERATED）、`form`（DECLARATIVE/PROCEDURAL/CONDITIONAL/METACOGNITIVE）、`acquisition`（STUDY/TESTIMONY/IMITATION/EXPERIENCE/INFERENCE）、`epistemic_status`（PROPOSED/SUPPORTED/CONTESTED/REFUTED/SUPERSEDED/RETRACTED）。来源与可信度分离。
- **11 类学习实体**：Learner、Competency、LearningActivity（READ/EXPLAIN/IMITATE/REPRODUCE/PROBE/EXPERIMENT/REFLECT/GENERALIZE/TEACH/ASSESS/REASSESS）、AssessmentSpec、CompetencyAssessment、LearningObjective、LearningResource、TransferAssessment 等。
- **能力只能由 CAS 派生**：没有 `DEMONSTRATED_BY` 证据的能力声明不成立（门禁强制）。
- **迁移分型**：权重（负）/方法（正）/知识/技能，避免"迁移失败"混为一谈。
- **成长指标**：M01–M08（覆盖率、达标率、证据分布、假设周期/吞吐、实验有效率、迁移成功率、知识驱动决策率、传承）。
- 自动门禁：`scripts/kb/validate.py` + 6 项变异测试；成长面板 `scripts/kb/growth.py`。

## 7. 实施的边界（防口号）

- 理论（Ryle 知道-如何、Dreyfus 阶段、双环学习、Crossan 组织学习、PROV-O）只作设计依据，不作为"我们已经具备该能力"的证据；
- 团队是学习主体（LRN:duckbench-team），冠军不归给单个 Agent；
- 未通过 CAS 的层级标记为未获得。