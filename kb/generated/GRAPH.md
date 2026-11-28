# Knowledge graph (generated — do not edit)

Entities: 188 · relations: 137 · types: 33

## Claims and their evidence

- **CLM:A2:01** — Saucer radius effect is weak *(EXPLORATORY)*
  - ← OBS:saucer:radius-drdt (Observation: Saucer radius vs rpm growth)
- **CLM:BATCH:01** — Larger PPO batch improves sprint score *(EXPLORATORY)*
  - ← OBS:batch16k (Observation: 256 envs × 64 steps beats 128 envs)
  - ← LIT:rudin2022 (LiteratureSource: Rudin et al. 2022 Learning to Walk in Minutes)
  - ← CLM:PLATEAU:01 (Claim: Flat-start arms read null by construction)
  - ← OBS:ctl128-2.41 (Observation: 128-env equal-budget control: 2.41, flat)
- **CLM:K03:01** — Steps recipe *(STRONG)*
  - ← EV:s2b:server (EvaluationRun: steps champion hidden)
  - ← ATT:steps (CompetitionAttempt: Steps champion attempt)
  - ← LB:steps (LeaderboardSnapshot: Steps LB)
- **CLM:K13:01** — Cross-task weights fail *(REPLICATED)*
  - ← ART:t2p:cross (EvidenceArtifact: cross-task eval)
- **CLM:K17:01** — Hurdles reward fix improves *(REPLICATED)*
- **CLM:K18:01** — Hurdles champion 2.03 *(STRONG)*
  - ← ATT:hurdles (CompetitionAttempt: Hurdles champion attempt)
  - ← LB:hurdles (LeaderboardSnapshot: Hurdles LB)
- **CLM:K18:02** — Local-hidden consistency *(REPLICATED)*
  - ← EV:hfast50:local (EvaluationRun: hfast50 on 8-71)
- **CLM:K19:01** — Sprint clip fix *(REPLICATED)*
  - ← ART:sprintf20:json (EvidenceArtifact: sprintf20 8-71 eval)
- **CLM:K19:02** — Sprint 2.40 *(STRONG)*
  - ← EV:sprintf80:local (EvaluationRun: sprintf80 on 8-71)
  - ← ATT:sprint (CompetitionAttempt: Sprint 2.40 attempt)
  - ← ART:sprintf50:json (EvidenceArtifact: sprintf50 8-71 eval)
- **CLM:K21:01** — Handoff finish no gain *(FALSIFIED)*
- **CLM:K24:01** — Accel penalty no gain *(FALSIFIED)*
  - ← EV:sprintf80:local (EvaluationRun: sprintf80 on 8-71)
  - ← RES:sprintf80 (MetricResult: sprint 2.41)
- **CLM:PLATEAU:01** — Flat-start arms read null by construction *(EXPLORATORY)*
  - ← OBS:nofall-2.38 (Observation: no-fall-termination arm also 2.38)
  - ← OBS:ctl128-2.41 (Observation: 128-env equal-budget control: 2.41, flat)
- **CLM:PLATEAU:02** — Fine-tune plateau at 2.38 for sprintf80 family *(EXPLORATORY)*
  - ← OBS:b64k-2.38 (Observation: 512 envs same 2.38: batch saturates at 16k)
  - ← OBS:update-scaling (Observation: Scratch-phase update-count hypothesis (scoped))
- **CLM:REP:01** — Local evaluation is deterministic per seed *(REPLICATED)*
  - ← ART:hfast50:rep1 (EvidenceArtifact: hfast50 repeat eval #1)
  - ← ART:hfast50:rep2 (EvidenceArtifact: hfast50 repeat eval #2)
- **CLM:S3:01** — Capacity arm needs weight grafting *(EXPLORATORY)*
  - ← OBS:S3:scratch-wide (Observation: Wide net from scratch cannot walk)
- **CLM:S3:02** — Net2Wider is function-preserving here *(REPLICATED)*
  - ← OBS:net2wider (Observation: Net2Wider preserves the policy function)
  - ← OBS:wide:seed-flips (Observation: Widened policy fall flips)
  - ← OBS:wide:noisy-start (Observation: Symmetry-broken wide start keeps 2.41)
- **CLM:S3:03** — Capacity arm requires identical reward and start function *(ANECDOTAL)*
  - ← OBS:S3:null-test (Observation: Identical-replica net2wider is a null test)
- **CLM:S3:04** — Capacity arm: wider net gives no gain at 10 min *(EXPLORATORY)*
- **CLM:SCRATCH:01** — Scratch asymptote ~2.50 at 11.6k updates *(EXPLORATORY)*
  - ← OBS:scratch-2.50-final (Observation: scratch 165min final: 2.50, flattening; campaign closed)
- **CLM:STACK:01** — Mac arena stack is convertible but ~15x slower *(EXPLORATORY)*
  - ← OBS:arena-mac (Observation: Arena Mac port runs here and converts to duck-mlp)
- **CLM:STACK:02** — Arena PPO hyperparams break warm-started policies *(EXPLORATORY)*
  - ← OBS:arena-graft (Observation: Arena graft faithful pre-training, broken by first update)
- **CLM:STACK:03** — Arena stack matches kit score with fewer transitions *(EXPLORATORY)*
  - ← OBS:arena-vs-kit-5m (Observation: Arena 10 updates matches kit 5.6M transitions)
- **CLM:STACK:04** — Arena 5M transitions does not beat kit 5.65M *(EXPLORATORY)*
- **CLM:STACK:05** — Arena objective does not transfer to DuckBench TS *(EXPLORATORY)*
- **CLM:T1:01** — Turntable bank is representative by construction *(ANECDOTAL)*
  - ← OBS:T1:bank (Observation: Turntable bank recorded from stock pre-roll)
- **CLM:W1:01** — Wheel gap is sustained rim speed *(EXPLORATORY)*
  - ← OBS:wheel:rim-saturation (Observation: Wheel rim speed saturates at 0.66 m/s)
- **CLM:W1:02** — Wheel bottleneck is deceleration events *(EXPLORATORY)*
  - ← OBS:wheel:accel-balance (Observation: Wheel rim accel/decel balance)

## Hypotheses

- HYP:A2 [PARTIAL] 接触半径/力矩臂可提升 saucer rpm
- HYP:A2b [FALSIFIED] 冠军的 'exact clones' 来自在三节点结构内训练的 base：单策略绑全部三节点（TRAINED_NODE 列表）训练 10 min，若 ≤2.38 则结构成本可被适应抵消，之后专门化才有 fair base；若 >2.38 则冠军 base 来自结构外配方
- HYP:H1 [SUPPORTED] 奖励的参考节奏约束是 Hurdles 瓶颈
- HYP:PPO:ADAPTIVE [TESTABLE] 固定 lr 1e-4 + clip_frac 0.325 的持续搅动构成伪平台;DKL=0.01 自适应可释放精修空间(H-P1')
- HYP:PPO:EXPLORATION [TESTABLE] 当前 std 0.0895 探索过保守;温和恢复至 0.12(保持维度比例)改善局部搜索
- HYP:RUN2:CEILING [TESTABLE] base-run-forward@2 的优化上限低于 hfw@1 平台(2.376)
- HYP:S1 [FALSIFIED] 前 2 s 加速不足是 Sprint 差距主因
- HYP:S3 [FALSIFIED] Sprint 2.40→2.17 的差距来自策略容量或训练栈，而不是奖励塑形（H-S1 已证伪）
- HYP:S4 [FALSIFIED] finish 节点在前序状态分布上训练可提升组合
- HYP:T1 [REFUTED] roll 入口状态代表性限制 turntable
- HYP:W1 [PARTIAL] 轮缘切向加速度奖励可提升 wheel

## Exclusions (do not repeat without reopening condition)

- EXC:multinode: 节点拆分/异权重拼接 — reopen if 有状态分布训练支持
- EXC:saucer:reward-weights: w_spin/w_peak/end_s 调整 — reopen if 改机制（角动量/接触）
- EXC:sprint:lr3e-4: REW:sprint:fast + LR_FIXED=3e-4 — reopen if 换 KL/critic 配置后

## Types

- AssessmentSpec: 3
- BudgetAllocation: 2
- Claim: 27
- Competency: 3
- CompetencyAssessment: 3
- Competition: 1
- CompetitionAttempt: 3
- Counterexample: 1
- Decision: 3
- Environment: 2
- EvaluationProtocol: 2
- EvaluationRun: 11
- EvidenceArtifact: 9
- Exclusion: 3
- Experiment: 12
- HostContract: 2
- Hypothesis: 11
- LeaderboardSnapshot: 2
- Learner: 1
- LearningActivity: 7
- LearningObjective: 1
- LearningResource: 4
- LiteratureSource: 5
- MetricDefinition: 6
- MetricResult: 11
- Observation: 27
- ResearchSession: 1
- RewardVersion: 7
- Scope: 6
- SeedSet: 3
- Task: 6
- TrainingRun: 1
- TransferAssessment: 2
