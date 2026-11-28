# Growth dashboard (generated)

entities=133 relations=99 tasks=6 claims=12 hypotheses=6

## Evidence distribution

- REPLICATED: 4
- STRONG: 3
- FALSIFIED: 2
- EXPLORATORY: 2
- ANECDOTAL: 1

## Hypothesis status

- FALSIFIED: 2
- PARTIAL: 2
- SUPPORTED: 1
- REFUTED: 1

## Coverage per task

- TASK:duckbench:hurdles:v4: claims 3, hypotheses 0
- TASK:duckbench:saucer:v1: claims 1, hypotheses 1
- TASK:duckbench:sprint:v6: claims 5, hypotheses 0
- TASK:duckbench:steps:v4: claims 1, hypotheses 0
- TASK:duckbench:turntable:v1: claims 1, hypotheses 1
- TASK:duckbench:wheel:v2: claims 1, hypotheses 1

## Exclusions (reopen condition)

- EXC:sprint:lr3e-4: 换 KL/critic 配置后
- EXC:saucer:reward-weights: 改机制（角动量/接触）
- EXC:multinode: 有状态分布训练支持

## Decision evidence use

- DEPENDS_ON: 2 edges

## Orphan claims (no SUPPORTS/REFUTES)

- CLM:K17:01
- CLM:K21:01

## Growth metrics (v0.2)

- M01_knowledge_coverage: 6/6 tasks with ≥1 claim
- M02_competency_attainment: 2/3 competencies passed by a CAS
- M03_evidence_distribution: {'REPLICATED': 4, 'STRONG': 3, 'FALSIFIED': 2, 'ANECDOTAL': 1, 'EXPLORATORY': 2}
- M04_hypothesis_cycle: 40 min (median closed hypotheses)
- M05_hypothesis_throughput: 4 closed / session
- M06_experiment_efficiency: 9/10 experiments with an execution record
- M07_transfer_success: 1/2 transfer assessments positive
- M08_knowledge_driven_decisions: 1/2 decisions cite evidence
