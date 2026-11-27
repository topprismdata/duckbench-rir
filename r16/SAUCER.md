# Saucer 尝试（失败，记录）

基线 `saucer-spinner@1`：55.42–55.49 rpm（8–71），冠军 59.24。

5-min 训练（S731）：导出 best 53.64、last 51.75 rpm——**训练反而退化**（in-loop 首个 checkpoint 52.97 < 起点 55.42）。S732 因预算耗尽未跑（已补预算）。

判断：该 kit 的 saucer baseline 奖励/命令与 `saucer-spinner@1` 的强起点不匹配，短训不能提升；继续需专门调 reward 或更长预算，暂缓，优先 Wheel/Hurdles。
