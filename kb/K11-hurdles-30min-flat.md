# K11 Hurdles 30-min gentle 仍无增益（提前终止）

**证据**：HFW 起点 2.84（8–71）；gentle 30-min（INIT_STD 0.05，WARM_MLP）中途 checkpoint 2.844–2.954，未优于起点；按淘汰纪律（连续 5-min 窗口无改善）在 ~3M steps 终止。

**结论**：HFW 在当前奖励/配方下已近平台；2.32 s 冠军需要不同训练目标（跨障专项奖励）或更长预算（GPT 预算 90 min，已用 ~40）。
