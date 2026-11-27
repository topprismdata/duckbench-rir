# SOURCE-REGISTRY — 外部来源（与制品清单分开）

| 来源 | 版本/标识 | 许可 | 用途 | 核验方式 |
|---|---|---|---|---|
| duckbench CLI 包 | `duckbench-0.3.87-5a1437de.tgz`，sha256 `5a1437def69604d44820e10ebd4e52e5686705fee6422134af639086077a5cd3`，30,569,928 B | 见包内 LICENSE | 官方 CLI、plant、moves、kit | 下载后本地 sha256 |
| DuckBench plant | `microduck-legs-bam-v1`（manifest 内 `source: github.com/Rhoban/bam@62bd8ce`） | Apache-2.0（BAM）+ CC BY-NC-SA（模型）+ Apache-2.0（策略） | 训练/仿真物理 | manifest + 文件哈希 |
| Rhoban/bam | `62bd8ce12154340be97e06f7f41a0ca8f116d967`（plant pin）；`57d13ead53206a6bf0db3d66f86506ae8c2ce01a`（官方浮动分支快照） | Apache-2.0 | 执行器模型对账 | `git archive` 到 /tmp 后比对哈希 |
| microduck | `4024f6ee` | 见仓库 | 执行器身份/常量 | R0-2a |
| microduck_rl | `0daab160`（本地 HEAD）；pin `cb70b792312d559a4da09064d92009079671815f` | Apache-2.0（代码）+ CC BY-NC-SA（STL） | 官方训练栈（CUDA） | R0-2b |
| microduck-challenges | `5cee942d` | **无 LICENSE** | 挑战定义 | R0-2b |
| microduck-lab | `8dad9218` | Apache-2.0 | 第三方训练候选 | R0-2b |
| microduck_rl_mac | `60f0b26a` | Apache-2.0 | 评估工具 | R0-2b |
| bam-feetech | `61b4d0b0` | **无 LICENSE** | 负对照素材（未使用） | R0-2b |
| DeepMind soccer | `soccer-deepmind/op3_soccer.zip` sha256 `9a314cba…`；`soccer-paper-arxiv-2304.13653v2.pdf` sha256 `ee3185ff…` | 论文/数据原许可 | R0-3 复算 | 本地哈希 |
| 本机工具链 | Python 3.12.12（kit venv）、uv.lock sha256 `851e88b263c249b733c61c00cf5d03122eff935b88303d2985d9a6f…`（见 locks.txt 全值）、mujoco 3.14.0、torch 2.9.1、mlx 0.32.2 | — | 训练 | locks.txt |

说明：Artifact 哈希在 `r1/PACKAGE-SHA256.txt`；本文件只登记**外部**来源，二者不再混用。
