# K07 Saucer 训练平台 ~55.5 rpm

**证据**：库内 `saucer-spinner@1` 55.4–55.6；5-min 训练 53.6（best）/51.8（last）；30-min gentle（INIT_STD 0.05、LR 5e-5、WARM_MLP）逐 checkpoint 55.0–55.6，未超过起点（`ts_eval.jsonl`）。

**结论**：当前 kit 奖励/环境组合下，saucer 策略在 55.5 rpm 附近平台；冠军 59.24 rpm 需要不同资产或训练目标，而非更多同配方预算。

**失效条件**：改进奖励（持续转速/接触滑移）或换起点后重测。
