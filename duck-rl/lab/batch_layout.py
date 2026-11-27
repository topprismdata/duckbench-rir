"""The MLX learner's batch columns for any obs width (numpy only: imported by lab/overlap.py, which packs the batch, and by
lab/mlx_ppo.py, which reads it; PR #1 B5, 2026-11-14).

A batch row is [policy obs | critic-only obs | action | old mu | old sigma | old log-prob | advantage | old value | return],
float32. The widths come from the nets themselves (pack_params' arrays): the actor's first layer reads the policy obs,
the critic's reads [policy obs, critic-only obs], log-std has one entry per action. The stock move is 61 + 6 + 3 x 14 + 4 =
113 columns (the layout mlx_ppo had fixed until kit 1.27.0); an ENCODER move's policy obs is 61 + the encoder's numbers.
"""
from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np


class Layout(NamedTuple):
    n_obs: int  # the policy obs (the actor's input)
    n_priv: int  # the critic-only obs (the critic reads [policy obs, these])
    n_act: int

    @property
    def O(self) -> slice:  # noqa: N802  (mlx_ppo's column names)
        return slice(0, self.n_obs)

    @property
    def CRIT(self) -> slice:  # noqa: N802
        return slice(0, self.n_obs + self.n_priv)

    @property
    def A(self) -> slice:  # noqa: N802
        s = self.n_obs + self.n_priv
        return slice(s, s + self.n_act)

    @property
    def MU(self) -> slice:  # noqa: N802
        s = self.A.stop
        return slice(s, s + self.n_act)

    @property
    def SG(self) -> slice:  # noqa: N802
        s = self.MU.stop
        return slice(s, s + self.n_act)

    @property
    def LP(self) -> int:  # noqa: N802
        return self.SG.stop

    @property
    def ADV(self) -> int:  # noqa: N802
        return self.LP + 1

    @property
    def VAL(self) -> int:  # noqa: N802
        return self.LP + 2

    @property
    def RET(self) -> int:  # noqa: N802
        return self.LP + 3

    @property
    def width(self) -> int:
        return self.LP + 4


def layout_of_params(p: dict) -> Layout:
    """The layout of the nets in `p` (pack_params' arrays: torch [out, in] weights). Stops on nets that disagree."""
    n_obs = int(np.shape(p["a.mlp.0.weight"])[1])
    n_crit = int(np.shape(p["c.mlp.0.weight"])[1])
    n_act = int(np.size(p["a.log_std"]))
    if n_crit < n_obs:
        raise ValueError(f"the critic reads {n_crit} inputs, fewer than the actor's {n_obs}: not [policy obs, critic obs]")
    if np.size(p["a.sub"]) != n_obs or np.size(p["a.div"]) != n_obs:
        raise ValueError(f"the actor's normaliser has {np.size(p['a.sub'])} / {np.size(p['a.div'])} entries, its first "
                         f"layer {n_obs} inputs")
    if np.size(p["c.mean"]) != n_crit:
        raise ValueError(f"the critic's normaliser has {np.size(p['c.mean'])} entries, its first layer {n_crit} inputs")
    last = max(int(k.split(".")[2]) for k in p if k.startswith("a.mlp.") and k.endswith(".weight"))
    if int(np.shape(p[f"a.mlp.{last}.weight"])[0]) != n_act:
        raise ValueError(f"the actor outputs {np.shape(p[f'a.mlp.{last}.weight'])[0]} actions, its log-std has {n_act}")
    return Layout(n_obs, n_crit - n_obs, n_act)


def pack_columns(cols: Sequence[np.ndarray], n: int, layout: Layout | None = None) -> np.ndarray:
    """[policy obs, critic obs, actions, mu, sigma, log-prob, advantages, values, returns], each [..., w] with n rows in
    all -> float32 [n, sum of w]; with `layout`, the widths must be its own (else ValueError)."""
    widths = [int(np.shape(t)[-1]) for t in cols]
    if layout is not None:
        want = [layout.n_obs, layout.n_priv, layout.n_act, layout.n_act, layout.n_act, 1, 1, 1, 1]
        if widths != want:
            raise ValueError(f"batch column widths {widths}, the nets' layout {want}")
    out = np.empty((n, sum(widths)), np.float32)
    o = 0
    for t, w in zip(cols, widths):
        out[:, o:o + w] = np.asarray(t).reshape(n, w)
        o += w
    return out
