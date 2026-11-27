# K01 `--init` 覆盖 `INIT_STD`

**观察**：wheel/saucer 用 `--init <ckpt>` 续训时，改配置里的 `INIT_STD` 似乎无效。

**证据（读码）**：`lab/learn.py:init_actor_from` 对含 `actor` 的 checkpoint 执行 `alg.actor.load_state_dict(ck["actor"])`，其中包含 `distribution.log_std_param`；随后 `build()` 不再重置 std。

**结论**：用 `--init` 续训时，探索强度 = checkpoint 的 std，`INIT_STD` 仅对 `WARM_MLP`（全新优化器）或纯 `ckpt.pt`（plain weights，无 actor 分支）生效。

**操作性**：想真正降低探索，必须走 `WARM_MLP` 路径（不 `--init`），或在代码里显式覆写 `log_std`。

**失效条件**：kit 版本变化导致 checkpoint 结构不同。
