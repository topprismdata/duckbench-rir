# R0-2（主审版）：官方资产与 Mac 实现审查

日期：2026-11-26 · 只读克隆在 `sources/`（未修改、未 push）· 与 R0-2a/2b（子代理报告）互为补充

## 1. 仓库、HEAD 与许可【实测】

| 仓库 | HEAD（2026-11-26） | 许可（GitHub API） | 用途 |
|---|---|---|---|
| `pollen-robotics/microduck` | `4024f6ee43…`（2026-11-25） | Apache-2.0 | 机器人本体固件/控制（RK3566，50 Hz） |
| `pollen-robotics/microduck_rl` | `0daab160d3…`（2026-11-10，develop） | Apache-2.0 | 官方训练（mjlab + MuJoCo Warp） |
| `pollen-robotics/microduck-challenges` | `5cee942d8b…`（2026-10-30） | Apache-2.0 | Arena 六个训练环境 |
| `Rhoban/bam` | `279c4828c4…`（2026-11-23） | Apache-2.0 | 舵机扩展摩擦模型 |
| `jonathanhawkins/microduck-lab` | `8dad921806…`（2026-10-23） | Apache-2.0 | Mac 本地训练 lab（SB3/自定义） |
| `keeeeenw/microduck_rl_mac` | `60f0b26a55…`（2026-09-26） | Apache-2.0 | CPU MuJoCo + MPS PPO |

**DuckBench 的 pin（`sources/duckbench-pkg/package/plant/microduck-legs-bam-v1/manifest.json`）**：`microduck_rl @ cb70b792312d…`（2026-09-14）、`microduck-policies @ 1b56c396825c…`、`Rhoban/bam @ 62bd8ce121…`（2026-07-13）。三个 commit 均已在 GitHub 上核实存在【实测：API `/commits/<sha>`】。**注意：pin ≠ 今日 HEAD**，DuckBench 冻结的是 9 月的训练栈；引用时必须写 pin。

## 2. 接口与物理（读码）

- DuckBench plant：14 个驱动关节（`manifest.json:joint_names`），`timestep=0.005`，`decimation=4` → 策略 50 Hz；`plant.xml` 用的积分器 `implicitfast`。
- 61 维观测（`kit/duckref/plant.py:230-265`）：`[0:3]` 躯干角速度、`[3:6]` 重力方向、`[6:20]` 关节位置−默认、`[20:34]` 关节速度（滞后 1 tick）、`[34:48]` 上一动作、`[48:51]` twist、`[51:55]` 头部命令（EMA）、`[55:61]` body pose；合计 61，源码注释逐项对应官方 `microduck_rl` 的观测实现。
- 动作：14 维关节目标；`duck-mlp/1` 样例 61→512→256→128→14（ELU）；单 move ≤1 M MACs、≤16 层。
- 固件侧（`microduck/README.md:26`）："a 50 Hz control loop driving **fifteen** servos"；官方策略同样只输出 14 维——**第 15 个舵机是嘴（jaw），在总线上但不在策略动作里**（`duck-control/src/obs.rs:8,53-54`、`robotd-design.md:176`）。R0-2a 已闭合该差异。
- 执行器是 **Dynamixel XL330**，BAM M6 模型（`microduck_constants.py:163-176`），**不是 Feetech**；`bam-feetech` 仓与本机器人无关（我此前的推测已作废）。
- **DuckBench 与官方的一处真实差异**：DuckBench 在 `[51:55]` 对头部命令做 EMA 平滑（`kit/duckref/plant.py` 注释），而官方训练/Arena/实机三链路都**没有命令侧 EMA**（官方为分段常量命令 + 动作侧低通）。这改变头部命令的输入分布；Sprint 榜首正是靠 `head_pitch` 0.72 改变姿态，因此该差异必须在 R2 的 sim-vs-官方一致性检查里单列。

## 3. 决定性约束：Mac 无法跑官方训练栈【读码】

- `microduck_rl/README.md:24`：「Requires a **CUDA GPU** (training runs through MuJoCo Warp)」；`pyproject.toml:17-18`：`mjlab==1.3.0`、`warp-lang==1.12.0`。
- `mjlab/README.md:17`：「mjlab requires an **NVIDIA GPU** for training. macOS is supported for **evaluation only**.」
- 因此：官方 `microduck_rl` 与 `microduck-challenges` 在本机只能做评估/回放，训练必须走 HF Jobs（NVIDIA）或 NVIDIA 机器。**"把 NVIDIA 训练框架移植到 Apple Silicon"这条路线应正式排除。**

## 4. Mac 侧两个第三方实现的真实能力【读码】

### 4.1 `jonathanhawkins/microduck-lab`（GPT 引用的 "Microduck Lab"）

README:220-232 原文（自报，Apple M-series）：

- `~16.5k env-steps/s`（32-env 配方，约 1 min / 1M steps）；throughput 配置峰值 `~27k steps/s`。
- 64 envs 内存 `41 GB → 1.5 GB`（共享编译模型 + fork/COW）。
- 两个吞吐"优化"（overlapped updates、big-batch）使 steps/s +25–40%，但在 **seed-matched A/B** 中 **单位样本学习效果减半**，因此只作 opt-in。
- 依赖：numba 融合的 BAM actuator kernel（与 numpy 参考逐位一致）、可选 MPS PPO。

判定：GPT 对这四个数值的引用**逐字准确**；但它们是作者自报、特定 32-env 配方与 M 系列机器，不能直接外推到 64 GB M1 Max 或 DuckBench 的 TS sim。该项目**不是** DuckBench 官方 kit，接口是 SB3/自定义 env，不是 `duck-mlp/1` + duckscript。

### 4.2 `keeeeenw/microduck_rl_mac`

README 原文：CPU MuJoCo + Torch MPS PPO；M1 Max 32 GB（24-core GPU）上 4,096 envs × 6,000 updates 训练出能走的策略，实测持续速度 ~0.18 m/s（指令 0.30）；观测吞吐 `1,912 transitions/s`，实验性 Unified Metal 为 `2,507–2,801/s`（作者注明是不同 env 数/阶段的历史运行，不是受控对比）。作者自述 "precise speed, heading and low-speed command tracking remain incomplete"。

判定：可行的 Mac 路线，但吞吐比 DuckBench kit 文档的 ~13k steps/s 低一个量级，且没有官方接口/评测；只能当"CPU 物理 + MPS PPO 可行性"证据，不能当比赛基线。

## 5. BAM 与 DuckBench 的关系【读码】

- DuckBench plant 目录直接带 `LICENSE-bam-Apache-2.0.txt`、`plant_id=microduck-legs-bam-v1`、pin `Rhoban/bam@62bd8ce`，并在 `kit/duckref/bam.py` 实现同一模型。**K092「Microduck 官方训练使用 BAM，DuckBench 未核验」→ 可升级为已验证：DuckBench 也使用 BAM。**
- BAM 提供扩展摩擦（load-dependent、Stribeck、directional）与 MuJoCo CPU 接入（`bam.mujoco.MujocoController`）；Microduck 的模型是 `motor_name="xl330"`（BAM M6）。`zeroth-robotics/bam-feetech` 是另一类舵机（Feetech）的参数仓，与本机器人无关。

## 6. 复用结论（R1 平台裁决的建议输入）

| 资产 | 处置 | 理由 |
|---|---|---|
| DuckBench 官方 kit（MLX） | **唯一本地训练路径** | 官方栈 CUDA-only；kit 与比赛共用 TS 评估 |
| 官方 `microduck_rl` / `microduck-challenges` | 只读参考 + 可选 HF Jobs | 观测/奖励/执行器实现是权威来源，但本机不能训练 |
| `microduck-lab` | 方法参考（吞吐/内存/AB 纪律） | 非官方接口，数值不可外推 |
| `microduck_rl_mac` | 可行性证据 | 吞吐低一个量级，无官方评测 |
| BAM / bam-feetech | 已在 DuckBench plant 内 | 不需另装 |

## 7. BLOCKED / 未核实

1. ~~15 舵机 vs 14 驱动关节~~ → 已闭合（嘴在总线、不在策略）。
2. 官方训练栈在 HF Jobs 上的实际成本与排队（未登录、未申请）。
3. `microduck-lab` 的 16.5k/27k 数值未在本机复跑（需要安装其依赖并训练，R0 禁止）。
4. DuckBench pin 与今日 HEAD 之间的行为差异（未做 diff）。

## 8. 反证

- 反证 A：GPT 把 `microduck-lab` 的吞吐当作"本地训练可行"的主证据，但它不是 DuckBench 官方 kit，接口/评测都不同；真正可提交的本地训练只有 kit。
- 反证 B：官方 README 的 "fifteen servos" 与 DuckBench 的 14 关节不一致，说明"官方模型 = 比赛模型"这句话需要限定到 pin 与具体关节集。
- 反证 C：pin 是 2026-09 的 commit，而两个仓库 HEAD 已更新到 11 月；用 HEAD 阅读会得到与比赛不一致的实现细节。