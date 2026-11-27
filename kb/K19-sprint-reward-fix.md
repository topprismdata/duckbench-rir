# K19 Sprint 奖励修正有效（2.99 → 2.59，隐藏验证）

**改动**（`configs/e4/sprint_fast.py`）：dense `progress` 的 fwd clip 1.0 → 2.5 m/s；`w_prog` 1.0 → 2.0；warm start 自 2.99 策略（WARM_MLP path）。

**结果**：20-min（seed 831）→ 8–71 **2.59 s（64/64）**、0–7 2.57；提交隐藏 24 = **2.59 s**（attempt `n80079bmnnym`）。本地与隐藏一致（K05）。

**榜位**：仍 #5——新玩家 gaga-fast-duck 2.46 进入；#1 2.17。

**下一步**：续训 30 min（r20_sprintf50）争取 ≤2.4；若要 #1 需要结构性变化（多节点/更长预算）。
