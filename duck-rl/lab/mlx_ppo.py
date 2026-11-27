"""T1 (Mac-speed research, opt-in): rsl_rl 5.5.1 PPO.update re-implemented in MLX on the Apple GPU, as a server process.

Run by lab/overlap.py MlxLearner with an MLX venv's python (numpy + mlx only; no torch): `python mlx_ppo.py <fd>`,
fd = one end of a socketpair speaking multiprocessing.connection. Messages:
  ("init", params, seed)                      -> "ok"        params: numpy arrays named as in pack_params (overlap.py),
                                                             plus "oa.*" / "oc.*" = a saved Adam state (--resume)
  ("set", params, seed)                       -> "ok"        (tests) replace weights, keep the optimiser state
  ("load", params, None)                      -> "ok"        replace weights + critic normaliser; the optimiser state
                                                             and the RNG stay (VALUE_BIAS_INIT)
  ("update", batch[N,W] float32, cfg)         -> (losses, lr, params, t_update_s)
  ("opt_state", None, None)                   -> {"oa.<key>": array, "oc.<key>": array}: both Adam states (moments,
                                                             step counts, learning rates), for checkpoints
  ("grad", batch[N,W] float32, cfg)           -> (losses, grads)  (selftest: one minibatch at the current weights, no step)
  ("close", None, None)
Math = lab.learn.train_cfg's PPO.update: 5 epochs x 4 minibatches over one randperm; actor obs-512-256-128-14 ELU
behind the frozen stock (x - sub) / div; Gaussian with learned log-std; critic 256-256-128-1 ELU on [policy, critic]
obs behind the running normaliser (x - mean) / (std + 0.01), updated after the epochs; per minibatch KL(old || new) ->
adaptive LR (if cfg["adaptive"]) before the step, clipped surrogate 0.2, clipped value loss (coef 1), no entropy
bonus, grad-norm clip 1.0 for actor (+ log-std) and critic separately; Adam (torch defaults, bias correction) with
separate step counts for actor and critic (as torch's per-parameter state: the actor's only starts when it is
released after the critic warm-up). ELU backward from the output (docs/.../mac-speed/learner/ppo_mlx.py elu_cvjp).
cfg["decoupled"] (the overlapped loop's one-update lag): see Learner._decouple; cfg["iw_cap"] caps its weight.
Per-run knobs in the update cfg (kit 1.6.2; absent = the constants below, i.e. the pre-1.6.2 update exactly): "dkl"
(the adaptive LR's KL target, DKL), "value_clip" (CLIP; None = an unclipped value loss), "critic_lr" (the critic's own
fixed LR; None = the actor's LR, written into both Adams as before), "critic_epochs" (EPOCHS; epochs past the actor's
are critic-only), "std_floor" (the log-std clamped to log(floor) after each update).
losses also carry per-update diagnostics (kit 1.6.2): kl_mb (= kl: the mean minibatch KL over the actor epochs),
kl_final / dmu_rms (KL(proximal -> final weights) and the rms action-mean shift over the batch: one extra actor
forward), lr_a_* / lr_c_* (mean/min/max of the LR each minibatch step used), clip_frac (|ratio - 1| > CLIP),
vclip_frac (|v - vold| > the value clip; with the clip off, > CLIP: where it would bind), gn_actor / gn_critic (the
mean grad norm before clipping; 0 for a frozen net).
The log-ratio is clamped at LOG_RATIO_MAX before exp() (a float32 overflow guard, see there; rsl_rl has none);
losses["ratio_clamped"] counts the clamped rows over the update's minibatch steps.
Debug: MLX_PPO_DUMP=<dir> saves an update that ends non-finite (batch, permutation, cfg, the weights and both Adam
states it started from) to <dir>/fail-*.npz, enough to replay it outside training.
Batch columns (lab/batch_layout.py, widths from the init params; W = their sum): policy obs, critic-only obs, action,
old mu, old sigma, old log-prob, advantage, old value, return. The stock move: obs 0:61, priv 61:67, action 67:81, old mu
81:95, old sigma 95:109, old log-prob 109, advantage 110, old value 111, return 112 (W = 113); an ENCODER move's obs is
wider (61 + the encoder's numbers) and every later column moves along (PR #1 B5, kit 1.27.0).
"""
from __future__ import annotations

import math
import os
import sys
import time
from multiprocessing.connection import Connection

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

try:  # run as a script (overlap.py MlxLearner: `python mlx_ppo.py <fd>`, lab/ on sys.path) or imported as lab.mlx_ppo
    from batch_layout import Layout, layout_of_params
except ImportError:
    from .batch_layout import Layout, layout_of_params

EPOCHS, MINIBATCHES, CLIP, VCOEF, MAXGN, DKL, NORM_EPS = 5, 4, 0.2, 1.0, 1.0, 0.01, 1e-2
STOCK = Layout(61, 6, 14)  # the stock move's batch (the layout before PR #1 B5; selftest's grad default)
LOG2PI = math.log(2 * math.pi)
# exp() of a log-ratio above ~88.7 is inf in float32, and 0 * inf = NaN in the surrogate (forward and backward). Saucer
# 2026-10-01: rows from out-of-distribution states (normalised obs up to 75 sigma, actor means +-33) sit 20-30 sigma
# from the proximal mean; with sigma 0.08 their log-ratio passed 88 within one update, their (decoupled, w ~ 0)
# advantage was -0.0, and the NaN reached every actor weight. Above e^20 the ratio is clamped (no gradient through
# those rows); everything below is untouched.
LOG_RATIO_MAX = 20.0


@mx.custom_function
def elu(x):
    return mx.where(x > 0, x, mx.expm1(mx.minimum(x, 0)))


@elu.vjp
def _elu_vjp(x, cot, y):
    return (cot * mx.where(x > 0, 1.0, y + 1.0),)


class MLP(nn.Module):
    def __init__(self, p: dict, tag: str):
        super().__init__()
        self.layers = []
        for i in (0, 2, 4, 6):
            W = p[f"{tag}.mlp.{i}.weight"]  # torch layout [out, in] = MLX Linear's
            lin = nn.Linear(W.shape[1], W.shape[0])
            lin.weight, lin.bias = mx.array(W), mx.array(p[f"{tag}.mlp.{i}.bias"])
            self.layers.append(lin)

    def __call__(self, x):
        for i, lin in enumerate(self.layers):
            x = lin(x)
            if i < 3:
                x = elu(x)
        return x


class Nets(nn.Module):
    def __init__(self, p: dict):
        super().__init__()
        self.actor, self.critic = MLP(p, "a"), MLP(p, "c")
        self.log_std = mx.array(p["a.log_std"])


def make_loss_fn(vclip: float | None = CLIP, L: Layout = STOCK):
    """The PPO loss with the value clip `vclip` (None: unclipped) on batches of layout `L`. Returns (loss, kl, surr,
    vloss, ent, n_clamped, clip_frac, vclip_frac); only the first is differentiated."""
    O, A, MU, SG, LP, ADV, VAL, RET = L.O, L.A, L.MU, L.SG, L.LP, L.ADV, L.VAL, L.RET

    def loss_fn(model, mb, a_sub, a_div, c_mean, c_std):
        mu = model.actor((mb[:, O] - a_sub) / a_div)
        log_std = model.log_std
        std = mx.exp(log_std)
        act = mb[:, A]
        logp = (-((act - mu) ** 2) / (2 * std ** 2) - log_std - 0.5 * LOG2PI).sum(-1)
        ent = (0.5 + 0.5 * LOG2PI + log_std).sum()
        v = model.critic((mb[:, L.CRIT] - c_mean) / (c_std + NORM_EPS)).squeeze(-1)
        mus, stds, osg, omu = mx.stop_gradient(mu), mx.stop_gradient(std), mb[:, SG], mb[:, MU]
        kl = (mx.log(stds / osg) + (osg ** 2 + (omu - mus) ** 2) / (2 * stds ** 2) - 0.5).sum(-1).mean()
        log_r = logp - mb[:, LP]
        ratio = mx.exp(mx.minimum(log_r, LOG_RATIO_MAX))
        adv = mb[:, ADV]
        surr = mx.maximum(-adv * ratio, -adv * mx.clip(ratio, 1 - CLIP, 1 + CLIP)).mean()
        vold, ret = mb[:, VAL], mb[:, RET]
        if vclip is None:
            vloss = ((v - ret) ** 2).mean()
        else:
            vc = vold + mx.clip(v - vold, -vclip, vclip)
            vloss = mx.maximum((v - ret) ** 2, (vc - ret) ** 2).mean()
        clip_frac = (mx.abs(mx.stop_gradient(ratio) - 1.0) > CLIP).astype(mx.float32).mean()
        vclip_frac = (mx.abs(mx.stop_gradient(v) - vold) > (CLIP if vclip is None else vclip)).astype(mx.float32).mean()
        return surr + VCOEF * vloss, kl, surr, vloss, ent, (log_r > LOG_RATIO_MAX).sum(), clip_frac, vclip_frac
    return loss_fn




def _assign(tree, key: str, value) -> None:
    """A tree_flatten dotted key -> set that leaf of a nested dict/list in place."""
    *path, last = key.split(".")
    for k in path:
        tree = tree[int(k)] if isinstance(tree, list) else tree[k]
    if isinstance(tree, list):
        tree[int(last)] = value
    else:
        tree[last] = value


class Learner:
    def __init__(self, p: dict, seed: int):
        mx.random.seed(seed)
        self.L = layout_of_params(p)  # the batch's columns: from the nets' widths (any obs width)
        self.model = Nets(p)
        self.opt_a = optim.Adam(learning_rate=1e-3, bias_correction=True)  # actor + log-std
        self.opt_c = optim.Adam(learning_rate=1e-3, bias_correction=True)  # critic
        tp = self.model.trainable_parameters()
        self.opt_a.init({"actor": tp["actor"], "log_std": tp["log_std"]})
        self.opt_c.init(tp["critic"])
        for tag, opt in (("oa", self.opt_a), ("oc", self.opt_c)):  # a saved Adam state (--resume), in place
            saved = {k[len(tag) + 1:]: v for k, v in p.items() if k.startswith(tag + ".")}
            if saved:
                from mlx.utils import tree_flatten
                have = dict(tree_flatten(opt.state))
                if set(saved) != set(have) or any(np.size(saved[k]) != have[k].size for k in have):
                    raise ValueError(f"the saved {tag} Adam state does not fit this network")
                for k, v in saved.items():
                    _assign(opt.state, k, mx.array(np.asarray(v).reshape(have[k].shape)).astype(have[k].dtype))
        self.a_sub, self.a_div = mx.array(p["a.sub"]), mx.array(p["a.div"])
        self.c_mean, self.c_var, self.c_std = (mx.array(p[f"c.{k}"]) for k in ("mean", "var", "std"))
        self.c_count = int(p["c.count"])
        self.state = [self.model.state, self.opt_a.state, self.opt_c.state]
        self.steps = {}

    def _step(self, train_actor: bool, adaptive: bool, train_critic: bool = True, own_critic_lr: bool = False,
              dkl: float = DKL, vclip: float | None = CLIP):
        """One compiled minibatch step per configuration (actor trainable, adaptive LR, critic trainable, the critic on
        its own LR, the KL target, the value clip). Returns [kl, surr, vloss, ent, n_clamped, clip_frac, vclip_frac,
        gn_actor, gn_critic, lr_actor, lr_critic]: the grad norms before clipping (0 for a frozen net), the LRs this
        step used."""
        key = (train_actor, adaptive, train_critic, own_critic_lr, dkl, vclip)
        if key not in self.steps:
            model, opt_a, opt_c = self.model, self.opt_a, self.opt_c
            lvg = nn.value_and_grad(model, make_loss_fn(vclip, self.L))

            def step(mb, a_sub, a_div, c_mean, c_std):
                (_, kl, surr, vl, ent, nclamp, cf, vcf), g = lvg(model, mb, a_sub, a_div, c_mean, c_std)
                if adaptive:  # rsl_rl: from the KL before this step, one LR for every param group (own_critic_lr:
                    lr = opt_a.learning_rate  # the critic keeps its own)
                    lr = mx.where(kl > 2 * dkl, mx.maximum(lr / 1.5, 1e-5),
                                  mx.where((kl < dkl / 2) & (kl > 0), mx.minimum(lr * 1.5, 1e-2), lr))
                    opt_a.learning_rate = lr
                    if not own_critic_lr:
                        opt_c.learning_rate = lr
                gnc = gna = mx.array(0.0, mx.float32)
                if train_critic:
                    gc, gnc = optim.clip_grad_norm(g["critic"], MAXGN)
                    opt_c.update(model.critic, gc)
                if train_actor:  # (warm-up: the actor's grads are computed and dropped, like requires_grad=False)
                    ga, gna = optim.clip_grad_norm({"actor": g["actor"], "log_std": g["log_std"]}, MAXGN)
                    opt_a.update(model, ga)
                return mx.stack([kl, surr, vl, ent, nclamp.astype(mx.float32), cf, vcf, gna.astype(mx.float32),
                                 gnc.astype(mx.float32), opt_a.learning_rate.astype(mx.float32),
                                 opt_c.learning_rate.astype(mx.float32)])

            self.steps[key] = mx.compile(step, inputs=self.state, outputs=self.state)
        return self.steps[key]

    def _check(self, batch: np.ndarray) -> None:
        if batch.ndim != 2 or batch.shape[1] != self.L.width:
            raise ValueError(f"batch of shape {batch.shape}: the nets' layout {self.L} has {self.L.width} columns")

    def update(self, batch: np.ndarray, cfg: dict):
        self._check(batch)
        L = self.L
        dkl = float(cfg.get("dkl", DKL))
        vclip = cfg.get("value_clip", CLIP)
        vclip = None if vclip is None else float(vclip)
        critic_lr = cfg.get("critic_lr")
        c_epochs = int(cfg.get("critic_epochs") or EPOCHS)
        if cfg.get("lr") is not None:
            self.opt_a.learning_rate = mx.array(cfg["lr"], mx.float32)
            self.opt_c.learning_rate = mx.array(cfg["lr"], mx.float32)
        if critic_lr is not None:
            self.opt_c.learning_rate = mx.array(critic_lr, mx.float32)
        train_actor, adaptive = bool(cfg["train_actor"]), bool(cfg["adaptive"])
        data = mx.array(batch)
        if cfg.get("decoupled"):
            data = self._decouple(data, cfg.get("iw_cap"))
        n = data.shape[0]
        mbs = n // MINIBATCHES
        idx = mx.array(cfg["perm"]) if cfg.get("perm") is not None else mx.random.permutation(n)  # perm: tests
        outs = []
        for e in range(max(EPOCHS, c_epochs)):
            # the first EPOCHS epochs as before (the actor once released; the critic while e < c_epochs); later ones
            # critic-only, with no adaptive-LR change (no actor step)
            step = self._step(train_actor and e < EPOCHS, adaptive and e < EPOCHS, e < c_epochs, critic_lr is not None,
                              dkl, vclip)
            for i in range(MINIBATCHES):
                o = step(data[idx[i * mbs:(i + 1) * mbs]], self.a_sub, self.a_div, self.c_mean, self.c_std)
                outs.append(o)
                mx.async_eval(self.state, o)
        # critic normaliser (rsl_rl EmpiricalNormalization.update over the whole batch, after the epochs)
        x = data[:, L.CRIT]
        mean_x, var_x = x.mean(0, keepdims=True), x.var(0, keepdims=True)
        self.c_count += n
        rate = mx.array(n, mx.float32) / mx.array(self.c_count, mx.float32)
        delta = mean_x - self.c_mean
        self.c_mean = self.c_mean + rate * delta
        self.c_var = self.c_var + rate * (var_x - self.c_var + delta * (mean_x - self.c_mean))
        self.c_std = mx.sqrt(self.c_var)
        floor = cfg.get("std_floor")
        if floor:  # STD_FLOOR: the learned log-std never below log(floor) (a projection; Adam's moments untouched)
            self.model.log_std = mx.maximum(self.model.log_std, math.log(float(floor)))
        # how far the policy moved in this update: KL(proximal -> final weights) and the rms action-mean shift over the
        # batch (proximal = the batch's mu/sigma columns: the update's start weights when decoupled, else the behaviour's)
        mu_o, sg_o = data[:, L.MU], data[:, L.SG]
        mu_f = self.model.actor((data[:, L.O] - self.a_sub) / self.a_div)
        sg_f = mx.exp(self.model.log_std)
        kl_f = (mx.log(sg_f / sg_o) + (sg_o ** 2 + (mu_o - mu_f) ** 2) / (2 * sg_f ** 2) - 0.5).sum(-1).mean()
        dmu = mx.sqrt(((mu_f - mu_o) ** 2).mean())
        per = mx.stack(outs)
        mx.eval(self.state, per, self.c_mean, self.c_var, self.c_std, kl_f, dmu)
        rows = np.array(per)  # [minibatch steps, 11], epoch-major
        na, nc = EPOCHS * MINIBATCHES, c_epochs * MINIBATCHES

        def f32sum(col) -> float:  # as the pre-1.6.2 accumulator summed them: float32, in step order
            acc = np.float32(0.0)
            for v in col:
                acc = np.float32(acc + v)
            return float(acc)
        kl, surr, ent = (f32sum(rows[:na, j]) / na for j in (0, 1, 3))
        vl = f32sum(rows[:nc, 2]) / nc
        losses = dict(value=vl, surrogate=surr, entropy=ent, kl=kl, ratio_clamped=f32sum(rows[:na, 4]))  # (rows x steps)
        la, lc = rows[:na, 9].astype(np.float64), rows[:nc, 10].astype(np.float64)
        losses.update(kl_mb=kl, kl_final=float(kl_f.item()), dmu_rms=float(dmu.item()),
                      clip_frac=float(rows[:na, 5].mean()), vclip_frac=float(rows[:nc, 6].mean()),
                      gn_actor=float(rows[:na, 7].mean()), gn_critic=float(rows[:nc, 8].mean()),
                      lr_a_mean=float(la.mean()), lr_a_min=float(la.min()), lr_a_max=float(la.max()),
                      lr_c_mean=float(lc.mean()), lr_c_min=float(lc.min()), lr_c_max=float(lc.max()),
                      lr_c=float(self.opt_c.learning_rate.item()), actor_trained=train_actor,
                      lr_end=float(self.opt_a.learning_rate.item()))  # (this update's own state: logged with it)
        if cfg.get("decoupled"):  # (of the uncapped weights; iw_capped = the share of samples at the cap)
            w, cap = self.last_w, cfg.get("iw_cap")
            losses.update(w_mean=float(w.mean().item()), w_max=float(w.max().item()),
                          w_logstd=float(mx.std(mx.log(w)).item()),
                          iw_capped=float((w > cap).mean().item()) if cap else 0.0)
        return losses, float(self.opt_a.learning_rate.item())

    def opt_state(self) -> dict:
        """Both Adam states (moments, step counts, learning rates) as numpy arrays, keyed "oa.<key>" / "oc.<key>"."""
        from mlx.utils import tree_flatten
        out = {}
        for tag, opt in (("oa", self.opt_a), ("oc", self.opt_c)):
            out.update({f"{tag}.{k}": np.array(v) for k, v in tree_flatten(opt.state)})
        return out

    def _decouple(self, data, iw_cap=None):
        """Decoupled PPO (Hilton, Cobbe & Schulman 2021, "Batch size-invariance for policy optimization") for a batch
        collected by an older (behaviour) policy: the clip and the adaptive-LR KL are taken around the policy at the
        start of this update (the proximal policy), and the behaviour -> proximal off-policyness is corrected by the
        importance weight w = pi_prox / pi_behav on the advantage: L = w * max(-A r, -A clip(r)), r = pi / pi_prox.
        Implemented as a column rewrite (old mu / sigma / log-prob -> proximal; advantage -> w * advantage). The old
        values (the value clip's anchor) stay the behaviour rollout's: anchoring the clip to the start-of-update critic
        instead learned worse on sprint-opus at 1 M steps (3 seeds: reward/step -9%, falls/1k +63%, 40-episode passes
        18.3 vs 23.7; M, kit-v2/ACCEPTANCE.md "Review fixes").
        iw_cap: w is truncated at this value (as V-trace's rho-bar); uncapped, single samples reached w = 13-46 (M)."""
        L = self.L
        mu = self.model.actor((data[:, L.O] - self.a_sub) / self.a_div)
        log_std = self.model.log_std
        std = mx.broadcast_to(mx.exp(log_std), mu.shape)
        logp = (-((data[:, L.A] - mu) ** 2) / (2 * std ** 2) - log_std - 0.5 * LOG2PI).sum(-1)
        w = mx.exp(mx.minimum(logp - data[:, L.LP], LOG_RATIO_MAX))  # (uncapped, an overflow would be inf * advantage)
        self.last_w = w
        if iw_cap:
            w = mx.minimum(w, iw_cap)
        return mx.concatenate([data[:, :L.MU.start], mu, std, logp[:, None], (data[:, L.ADV] * w)[:, None],
                               data[:, L.VAL:]], axis=1)

    def grad(self, batch: np.ndarray, cfg: dict):
        """(selftest) The loss and the gradients after the per-group clipping for one minibatch (the whole batch) at
        the current weights, without a step: what one minibatch step of update() feeds to Adam."""
        self._check(batch)
        mb = mx.array(batch)
        if cfg.get("decoupled"):
            mb = self._decouple(mb, cfg.get("iw_cap"))
        (_, _kl, surr, vl, _ent, _n, _cf, _vcf), g = nn.value_and_grad(self.model, make_loss_fn(CLIP, self.L))(
            self.model, mb, self.a_sub, self.a_div, self.c_mean, self.c_std)
        gc, _ = optim.clip_grad_norm(g["critic"], MAXGN)
        ga, _ = optim.clip_grad_norm({"actor": g["actor"], "log_std": g["log_std"]}, MAXGN)
        out = {"a.log_std": np.array(ga["log_std"])}
        for tag, layers in (("a", ga["actor"]["layers"]), ("c", gc["layers"])):
            for j, lin in zip((0, 2, 4, 6), layers):
                out[f"{tag}.mlp.{j}.weight"], out[f"{tag}.mlp.{j}.bias"] = np.array(lin["weight"]), np.array(lin["bias"])
        return dict(value=vl.item(), surrogate=surr.item()), out

    def params(self) -> dict:
        out = {}
        for tag, net in (("a", self.model.actor), ("c", self.model.critic)):
            for j, lin in zip((0, 2, 4, 6), net.layers):
                out[f"{tag}.mlp.{j}.weight"], out[f"{tag}.mlp.{j}.bias"] = np.array(lin.weight), np.array(lin.bias)
        out["a.log_std"] = np.array(self.model.log_std)
        out["c.mean"], out["c.var"], out["c.std"] = np.array(self.c_mean), np.array(self.c_var), np.array(self.c_std)
        out["c.count"] = np.int64(self.c_count)
        return out


def device():
    """DUCK_RL_MLX_DEVICE (research only, docs/research/novel-moves/rl-code/mac-speed/pcores/GPU.md): "cpu" runs the
    update on MLX's CPU backend (one stream thread, matmuls in Accelerate; mx.compile builds its fused kernels with g++
    once), leaving the GPU to the person's 3D viewer; unset, "gpu" or anything else = the GPU, as always."""
    name = os.environ.get("DUCK_RL_MLX_DEVICE", "gpu")
    if name not in ("gpu", "cpu"):
        print(f"[mlx_ppo] DUCK_RL_MLX_DEVICE={name!r} is not gpu|cpu: using the GPU", file=sys.stderr, flush=True)
    return mx.cpu if name == "cpu" else mx.gpu


def serve(fd: int) -> None:
    conn = Connection(fd)
    mx.set_default_device(device())
    learner = None
    while True:
        cmd, data, cfg = conn.recv()
        if cmd == "init":
            try:
                learner = Learner(data, cfg)
            except ValueError as e:  # (a saved Adam state that does not fit: the client retries without it)
                conn.send(f"error: {e}")
                continue
            conn.send("ok")
        elif cmd == "set":  # weights + critic normaliser from the CPU copy (tests; the optimiser state is kept)
            fresh = Learner(data, cfg)
            learner.model.update(fresh.model.parameters())
            learner.c_mean, learner.c_var, learner.c_std, learner.c_count = fresh.c_mean, fresh.c_var, fresh.c_std, fresh.c_count
            conn.send("ok")
        elif cmd == "load":  # new weights + critic normaliser; the optimiser state and the RNG stay (no Learner())
            learner.model.update(Nets(data).parameters())
            learner.c_mean, learner.c_var, learner.c_std = (mx.array(data[f"c.{k}"]) for k in ("mean", "var", "std"))
            learner.c_count = int(data["c.count"])
            conn.send("ok")
        elif cmd == "opt_state":
            conn.send(learner.opt_state())
        elif cmd == "grad":
            conn.send(learner.grad(data, cfg))
        elif cmd == "update":
            dump = os.environ.get("MLX_PPO_DUMP")  # (debug) a dir: an update that ends non-finite saves its inputs there
            if dump:
                from mlx.utils import tree_flatten
                cfg = dict(cfg, perm=cfg.get("perm") if cfg.get("perm") is not None else np.random.permutation(len(data)))
                pre = {f"p.{k}": v for k, v in learner.params().items()}
                pre["p.a.sub"], pre["p.a.div"] = np.array(learner.a_sub), np.array(learner.a_div)
                for tag, opt in (("oa", learner.opt_a), ("oc", learner.opt_c)):
                    pre.update({f"{tag}.{k}": np.array(v) for k, v in tree_flatten(opt.state)})
            t = time.perf_counter()
            losses, lr = learner.update(data, cfg)
            p = learner.params()
            if dump and not all(np.isfinite(v).all() for v in p.values()):
                import json
                os.makedirs(dump, exist_ok=True)
                np.savez(os.path.join(dump, f"fail-{os.getpid()}-{time.time_ns()}.npz"), batch=data, perm=cfg["perm"],
                         cfg=json.dumps({k: v for k, v in cfg.items() if k != "perm"}), **pre)
            conn.send((losses, lr, p, time.perf_counter() - t))
        else:
            conn.close()
            return


if __name__ == "__main__":
    serve(int(sys.argv[1]))
    sys.stdout.flush()
    os._exit(0)  # (skip MLX's static destructors at interpreter exit; see kit.py main)
