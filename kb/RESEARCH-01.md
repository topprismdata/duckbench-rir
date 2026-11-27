# RESEARCH-01 — 三个未夺冠赛事的机制候选（文献定位）

用途：为 Wheel / Saucer / Turntable（以及 Sprint 的 0.24 s 差距）提出可检验机制，替代"同配方加时间"。

## Sprint（2.40 vs 2.17）

观察：2.41 策略在 t=2 s 已跑 3.48 m、瞬时速度 2.22 m/s，但全程平均 1.87 m/s；冠军 2.07 m/s。差距主要在起步加速与前 2 s。

文献要点：
- 高速奔跑的 RL 常用**速度跟踪 + 步态频率/相位塑形**，而非仅距离奖励（*Learning Accurate and Robust Velocity Tracking for Quadrupedal Robots*, JFR 2025；*Learning Quadrupedal High-Speed Running on Uneven Terrain*, Biomimetics 2024）。
- 单一奖励易收敛到单一速度/步态；相位或频率项可推动更高步频（*Deep RL of quadrupedal gaits without motion references*, Adv. Robotics 2025；*Real-Time Gait Adaptation via MPC+RL*, arXiv 2510.20706）。

可检验假设：
- H-S1：加入**速度目标跟踪项**（目标 2.2 m/s，误差平方惩罚）可提升前 2 s 加速。
- H-S2：加入**步频/相位奖励**（每步推进量或摆动频率）可提高峰值速度。

## Wheel（5.25 vs 4.60）

观察：轮内跑步；冠军 4.60 s = 完成 1 圈。我们的策略 rim 速度 ~0.8 m/s 平台（奖励 clip 已解除仍无改善）。

文献要点：
- 跑步机/旋转表面行走的接触时序与切向力是核心（split-belt / treadmill locomotion 研究）；策略需要**切向加速相位**而非恒定推力。
- 高速步态需要与表面速度匹配的步频（同上 Sprint 文献）。

可检验假设：
- H-W1：奖励加入**轮缘切向加速度**项（而非仅 rim 速度）可提高转数速率。
- H-W2：**步频课程**（随轮速提高步频目标）可避免打滑平台。

## Saucer（55.4 vs 59.24 rpm）

观察：盘面倾斜 4 cm、鸭子站在盘上；得分是 0–10 s 内 top-0.5 s 平均 |rpm|。我们的策略在 55.5 平台；加大 spin/peak 权重均退化。

文献要点：
- 旋转平台上的平衡控制需要**角动量管理**：身体姿态变化产生反作用力矩（*Balance Control of a Biped Robot on a Rotating Platform*, IEEE/CAA JAS 2019；*Stability Control on a Dynamic Platform*, Sensors 2020）。
- 旋转物体操作强调**接触点位置与摩擦锥**（*Lessons from Learning to Spin "Pens"*, CoRL 2024）。

可检验假设：
- H-A1：奖励应直接对齐**盘面角速度**（当前用峰值增量），并惩罚"盘面减速"。
- H-A2：奖励**接触点半径**（鸭子离盘心距离）以增大力矩臂。

## Turntable（11.31 vs 12.78）

观察：程序 out→face→backpedal→…→roll（stand 节点）；冠军 12.78。我们 60 min 平台 11.4。

文献要点：
- 平台旋转的**阶段衔接**决定初始角速度（Xi 2019/2020 的 phase-based 控制）；末端节点训练忽略前置状态分布。
- 角动量注入需要身体与盘面的相对运动（spin-pens 2024）。

可检验假设：
- H-T1：训练应从**前置阶段结束时的真实状态分布**开始（handoff bank at roll 入口），而非固定 settle 状态。
- H-T2：奖励加入**roll 入口角速度**项（让前置动作把盘先转起来）。

## 下一步

每个赛事只批准一个最小实验（≤20 min），先做零训练可测的（如 handoff bank 状态分布统计、接触点半径测量），再决定是否训练。