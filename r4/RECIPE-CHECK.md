# A3/A4 recipe 哈希差异核验（GPT P0：唯一剩余阻塞）

结论：**差异来自配置文件字节中的一行注释**。有效训练参数逐字段相同；normalizer 逐位相同。不需要重训。

## 1. recipe_hash 的输入（源码）

`train.py:462-467`：

```python
def recipe_hash(knobs, extra, cfg_path):
    h = sha256(json.dumps(dict(knobs=knobs, **extra), sort_keys=True, default=str))
    h.update(cfg_path.read_bytes())   # 配置文件原始字节也进哈希
```

`extra` = `dict(workers, per_worker, learner, ckpt_every, init, resume, overrides)`（`train.py:2291-2293`）。

所以哈希不同只有两种可能：参数不同，或配置文件字节不同。

## 2. 配置文件差异：一行注释

`diff configs/e4/a3.py configs/e4/a4.py` 只有 1 行不同：

```
- WARM_MLP = None   # E4 a3: the program node carries its own move (...)
+ WARM_MLP = None   # E4 a4 (Sidestep Scratch: same program/config as a3; DUCK_SCRATCH=1 at run time): ...
```

剥离行尾注释后两份文件**文本完全相同**（`normalized_equal: true`，sha256 相同）。

机制演示（同一 knobs/extra 下）：

| 输入 | recipe |
|---|---|
| a3.py 原样 | `sha256:641fcbe940a681e2` |
| a4.py 原样 | `sha256:cc68907d36fb5cd6` |
| a3.py 去注释 | `sha256:da12141855743da5` |
| a4.py 去注释 | `sha256:da12141855743da5` |

（这里的绝对哈希与实际 run 不同，因为实际 run 的 `extra` 还含 CLI 解析后的 `ckpt_every` 等；机制相同：唯一可变输入是文件字节。）

实际 run 的哈希：A3 三 seed 均 `899740816acbacf8`，A4 三 seed 均 `408f997accecabfc`——组内一致，组间差一个注释字节。

## 3. 有效参数逐字段比对

从 `home/runs/e4_a{3,4}_s{107,108,109}/summary.json` 抽取设置类字段：

| 字段 | A3 | A4 |
|---|---|---|
| workers / per_worker | 8 / 16 | 8 / 16 |
| learner / overlap / iw_cap | mlx / true / 3.0 | 同 |
| knobs（全部学习器参数） | 完全相同 | 同 |
| env / curriculum / rollout_steps | 相同 | 同 |

差异仅出现在**运行期派生量**，不是设置：

| 字段 | 含义 | 为何不同 |
|---|---|---|
| `released_at_it`（A3 5–20，A4 恒 4） | actor 解除固定的迭代点（按 KL 自适应） | 取决于初始化与梯度，随机权重下收敛路径不同 |
| `value_bias_shift` | critic 偏置漂移诊断 | 取决于随机初始化 |
| `setup_s` | 环境构建耗时 | 机器计时噪声 |

## 4. A3 的 normalizer：与 A4 逐位相同

A3 的 normalizer 来自 sidestep move 的 `policy.mlp.json` 的 `norm` 段：

- sub sha256 `a2a7d0aa…`，div sha256 `88356def…`
- A4 三个 seed 的 `initB/a4_init_s*.pt` 中的 `obs_normalizer._mean/_std` 与上述**逐位相等**（107/108/109 全部 true）

因此 A3/A4 的差异在有效层面只剩 **Actor 线性权重**（A4 随机重置），与实验声明一致。

## 5. 限制

- A3 没有同期初始化张量（它不重置），其权重直接等于 sidestep move 文件；已用文件字节哈希与 A4 dumps 对照。
- `released_at_it` 的差异是训练动力学的后果而非可设置参数；不构成单变量破坏，但应在因果声明中保留"自适应释放时点不同"这一事实。

## 6. 产物

`r4/artifacts/recipe_check.json`（配置 diff、归一化哈希、运行时字段 diff、normalizer 比对）、`r4/artifacts/recipe_mechanism.json`（机制演示）、`r4/artifacts/configs/e4/a{1..4}.py` 副本。