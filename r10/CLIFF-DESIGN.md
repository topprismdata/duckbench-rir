# Cliff 跨挑战候选设计（CANDIDATE，零训练）

> **R10 状态（GPT 裁决）**：设计保留为候选，执行未授权（新增训练 0 min、新增仿真 0 次）。本文件按 R10 意见修订，不再提交评审。

## 1. 已核实的挑战定义

`get-challenge cliff`：两鸭对向攀爬四级台阶（riser 4/6/8/10 cm，共 28 cm），**对局**（先登顶者胜）；`climb` 节点；库内参照 `cliff-climber@1` 为 **raw host**（2.873 s、64/64、38 falls）。

## 2. 接口兼容性（仅候选检查，未授权执行）

我们的 9 个产物是 **walk host**（61→14），Cliff 的 kit start 程序把 move 绑在 `climb` 节点且参照是 raw host。可能的不兼容：

1. `climb` 节点的 skill 是否接受 walk-host move；
2. 命令槽语义（raw 13 槽 vs walk vx/vy/wz）是否可映射；
3. 对局评价（duel）与我们的 solo 评价协议不同。

**门禁顺序**（零训练，逐条取证）：

- G1（候选）：`duckbench start cliff --move steps-starter@1` 是否接受该 walk move。
- G2（候选）：program 的节点 skill/host 与命令语义。
- **G3 及以后未授权**（GPT DEC-04）：任何实际对局评价都需新的立项。
- **不得自行裁决 raw↔walk 命令槽映射**；映射若需要，必须由平台文档或 GPT 裁决给出。
- 未来立项需先冻结：Host 兼容性、对手、初始位置、duel 评价指标。

## 3. 若 G1 通过的评价矩阵

| 项 | 值 |
|---|---|
| 策略 | 9 个 T2′ Last 产物 + `steps-starter@1` |
| 命令 | vx 0.19 与 0.10（若 climb 节点只接受 raw，则按 card 的 knob 映射，登记换算） |
| 评价 | seeds 8–39；对局时间/完赛；报告 best-hard 与完成率 |
| 指标 | 不跨挑战平均；以 starter 为参照做相对差 |
| 预算 | 20 次 simulate（≈3 min），零训练 |

## 4. 判定与限制

- 若 G1 拒绝 walk move → Cliff 只能与 raw-host 策略比较，**本题材 BLOCKED**（需另训 raw 策略，未授权）。
- Cliff 是四级变高度台阶，比 Steps 更接近"攀爬"；但 4/6/8/10 cm 与 Steps 的 2.5 cm 不同，属**中迁移**而非近迁移。
- 不因结果改选任务；不追加训练。

## 5. 请求

当前不请求执行任何检查；本文件仅作为候选设计保留。