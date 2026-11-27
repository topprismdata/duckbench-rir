# K22 Sprint 提高 LR 导致崩溃

**证据**：sprint_fast + LR_FIXED 3e-4（warm start 2.41 策略，10-min，seed 892）：checkpoint 2.87 → 2.96 → 3.25 → 16.34 → 28.66（训练发散）。

**结论**：当前 PPO 配置下 actor 步长不能加大；2.41 的策略已接近该优化器设置的可达区间。

**失效条件**：换 KL 目标 / critic 配置 / 更小 batch 后重测。
