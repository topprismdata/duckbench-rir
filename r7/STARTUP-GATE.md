# T2′ 启动门禁（GPT DEC-04）

目标：证明 `CMD_SCHEDULE` 真的改变 Actor observation，而不是只改配置字段；并建立命令暴露统计。

## 1. 命令进入观测（实测）

`duck_rl/env.py` 的 walk host 命令经 `WalkController.command()` 写入 `[48:51]` twist，再由 `build_obs` 拼进 61 维观测。探针在 k=100 打印实际 twist：

```
[cmd] obs twist at k=100: [0.4,    0.0, -0.0279] (walk cmd vx=0.19)
[cmd] obs twist at k=100: [0.2154, 0.0,  0.5589] (walk cmd vx=0.1)
```

0.19→0.4 与 0.10→0.2154 正是 steps-starter 卡片的 walk 速度映射（vx 0.19 ⇒ policy vx 0.4）。→ **命令确实进入 Actor 输入**。

## 2. 三种调度模式的运行证据（`DUCK_CMD_DEBUG=1`）

| 模式 | 探针输出 |
|---|---|
| episode（C1） | `exposure={0.1: 154, 0.19: 0}` / `{0.1: 0, 0.19: 79}` … 逐 episode 变化 |
| time（C2） | `switches=1 resamples=1 dwell=[1.5] exposure={0.1: 8, 0.19: 75}` |
| state（C3） | `switches=1 exposure={0.1: 96, 0.19: 0}`；`switches=4 exposure={0.1: 83, 0.19: 39}` |

记录字段：重采样次数、真实切换次数、驻留时长（s）、按命令的 tick 暴露量。每个训练 run 的 `train_*.out` 含逐 episode 行，可离线汇总。

## 3. 已知限制

- 计数是训练侧自测；它不是 TS 侧的同名统计（TS 只在评价时运行该程序）。
- C1 的逐 episode 抽样在早期 episode 呈偏斜（随机），最终分布以汇总为准。
- `exposure` 是策略 tick 计数（50 Hz）；评价端的暴露由评价程序与 seeds 决定。

## 4. 评价条件（V1–V7，冻结）

| ID | 命令表 |
|---|---|
| V1 | 固定 0.19 |
| V2 | 固定 0.10 |
| V3 | 固定 0.14（未训练取值插值） |
| V4 | height>0.15 触发 0.10→0.19 |
| V5 | t=3 s 0.10→0.19 |
| V6 | t=3 s 0.19→0.10 |
| V7 | t=3 s / 6 s：0.10→0.19→0.10 |

主要指标：V7 Last 的 C2−C1 配对 best-half 差（按训练种子配对）。