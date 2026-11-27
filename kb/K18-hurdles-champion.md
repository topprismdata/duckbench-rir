# K18 Hurdles 冠军 2.03 s（原 2.32）

**路径**：K17 奖励修正（v_ref 0.42、取消超前惩罚、vx_max 1.2）→ HFW warm start 20-min（2.403）→ 续训 30-min（seed 821）→ 8–71 **2.025 s，64/64** → 提交隐藏 24 = **2.03 s**，榜 #1（attempt `h8fglh80dclq`）。

**结论**：hurdles 的瓶颈就是奖励里的参考节奏约束；解除后策略学会更快的跨障步态（0.77 → 1.09 m/s 平均）。

**复用**：同型"速度上限/超前惩罚"审计适用于其余赛事（sprint 的 fwd clip 1.0、wheel 的 rim clip 0.8）。
