# Growth dashboard (generated)

entities=154 relations=121 tasks=6 claims=18 hypotheses=7

## Evidence distribution

- REPLICATED: 6
- EXPLORATORY: 5
- STRONG: 3
- FALSIFIED: 2
- ANECDOTAL: 2

## Hypothesis status

- FALSIFIED: 3
- PARTIAL: 2
- SUPPORTED: 1
- REFUTED: 1

## Coverage per task

- TASK:duckbench:hurdles:v4: claims 4, hypotheses 0
- TASK:duckbench:saucer:v1: claims 1, hypotheses 1
- TASK:duckbench:sprint:v6: claims 9, hypotheses 1
- TASK:duckbench:steps:v4: claims 1, hypotheses 0
- TASK:duckbench:turntable:v1: claims 1, hypotheses 1
- TASK:duckbench:wheel:v2: claims 2, hypotheses 1

## Exclusions (reopen condition)

- EXC:sprint:lr3e-4: 换 KL/critic 配置后
- EXC:saucer:reward-weights: 改机制（角动量/接触）
- EXC:multinode: 有状态分布训练支持

## Decision evidence use

- DEPENDS_ON: 4 edges

## Orphan claims (no SUPPORTS/REFUTES)

- CLM:K17:01
- CLM:K21:01
- CLM:S3:04

## Growth metrics (v0.2, DB-KG-04 honest denominators)

- M01a_task_registration: 6/6 tasks have ≥1 claim or hypothesis (registration, not knowledge coverage)
- M01b_target_knowledge_coverage: NO_TARGET_LIST — 未定义每任务目标知识点，无法计算真正覆盖率
- M02_competency_attainment: 3/3 competencies passed by a CAS whose spec matches
- M03_evidence_distribution: {'REPLICATED': 6, 'STRONG': 3, 'FALSIFIED': 2, 'ANECDOTAL': 2, 'EXPLORATORY': 5}
- M04_hypothesis_cycle: 40 min (n=5)
- M05_hypothesis_throughput: 5 closed in session SES:2026-11-28 (window 20:00Z→06:00Z)
- M06_experiment: registered 10/11; protocol-valid 9/11; repeated 1/11
- M07_transfer: METHOD: 1/1 positive; POLICY: 0/1 positive
- M08_decision_evidence: 1/2 decisions cite in-scope, non-retracted, pre-decision evidence
