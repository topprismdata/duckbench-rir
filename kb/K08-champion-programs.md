# K08 冠军的 duckscript 结构并不复杂

**证据**（`duckbench result <id> --duckscript`，公开 attempt）：
- Wheel #1（4.60）：`run → recover → cruise` 三节点，vx 0.19，均为默认 walk 命令；
- Hurdles #1（2.32）：单 walk 节点 head 0.6；
- Saucer #1（59.24）：单 walk 节点 vx 0.19、vy +0.03、wz 0；
- Turntable #1（12.78）：out/face/backpedal/... 序列，与 kit 程序节点一致。

**结论**：冠军的程序基本就是 kit 模板；差异在**训练出的 move**（"trained moves" 标签），不在脚本结构。

**推论**：
- 我们不需要写复杂脚本，需要更强的单 move（或每节点专用 move）。
- 命令语义与 move 绑定：Saucer 冠军的 vy +0.03 用在我们的 spinner 上是 26.8 rpm（spinner 训练于 vy −0.03）——不能直接抄命令。

**失效条件**：某些赛事（Sprint #1）确实用了三节点各自独立权重。
