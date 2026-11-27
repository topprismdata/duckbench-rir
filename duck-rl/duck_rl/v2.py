"""duck-program/2 in Python (kit 1.15.0): the expression evaluator, the memory slots and the program state of
packages/schema (expr.ts evaluateExpr / Memory, program.ts ProgramRunner / pickTransition / feedValues / headTargets)
and the route planner of packages/sim (runtime/planner.ts ScenePlanner.plan), operation for operation, so a Python match
replays a TS one (tools/duel_parity.py: the logic replay must agree tick for tick).

  compile_expr(e)   a compiled Expr (ts/prepare.ts --program-full: {k: ...}) -> fn(ctx) -> float | bool, with JS
                    semantics: division by zero gives +-Infinity / NaN, Math.min/max/round/floor/sign, === between a
                    bool and a number is false, Math.hypot as V8 computes it (jsmath.js_hypot2);
  Memory            one slot per memory call, updated in slot order every program tick (avg, delta, held, since, last,
                    count, ever, sum), reading the live slots (outer calls see the inner ones' new values);
  ProgramState      ProgramRunner: memory, vars, BACK's target, the pending `set` of the start node; tick(node, s)
                    -> (go, via, rule) | None with BACK resolved; enter(node, s);
  Planner           ScenePlanner's Dijkstra (the same binary heap, so ties and float sums come out the same) and its
                    3-cell downhill waypoint, over the occupancy grid ts/prepare.ts exports (info["routes"]: computed
                    by the TS planner itself, so the walls' growth never differs); RouteWorld = path / path_len bound to
                    a duck's pose at the program tick.
"""
from __future__ import annotations

import base64
import math

from .jsmath import js_hypot2, js_round

TICKS_PER_S = 10
FEED_ABS_MAX = 4.0
HEAD_ABS_MAX = 1.5
HEAD_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
INF = math.inf
NAN = math.nan


# ---- JS number semantics ------------------------------------------------------------------------------------------
def _isb(x) -> bool:
    return isinstance(x, bool)


def js_div(a: float, b: float) -> float:
    try:
        return a / b
    except ZeroDivisionError:
        if a != a or a == 0:
            return NAN
        neg = (a < 0) != (math.copysign(1.0, b) < 0)
        return -INF if neg else INF


def js_min(a: float, b: float) -> float:
    if a != a or b != b:
        return NAN
    if a == b == 0:
        return -0.0 if (math.copysign(1, a) < 0 or math.copysign(1, b) < 0) else 0.0
    return a if a < b else b


def js_max(a: float, b: float) -> float:
    if a != a or b != b:
        return NAN
    if a == b == 0:
        return 0.0 if (math.copysign(1, a) > 0 or math.copysign(1, b) > 0) else -0.0
    return a if a > b else b


def js_sign(a: float) -> float:
    if a != a or a == 0:
        return a
    return 1.0 if a > 0 else -1.0


def js_floor(a: float) -> float:
    return a if not math.isfinite(a) else float(math.floor(a))


def js_round_f(a: float) -> float:
    return a if not math.isfinite(a) else float(js_round(a))


def js_mod_pct(a: float, b: float) -> float:
    """JS a % b (the dividend's sign; NaN for b == 0 or a infinite)."""
    if b == 0 or not math.isfinite(a) or b != b:
        return NAN
    if not math.isfinite(b):
        return a
    return math.fmod(a, b)


def wrap_angle(a: float) -> float:
    """schema wrapAngle: (-pi, pi]; Infinity / NaN -> NaN."""
    if not math.isfinite(a):
        return NAN
    w = a - 2 * math.pi * math.floor((a + math.pi) / (2 * math.pi))
    return math.pi if w == -math.pi else w


def _exp(a: float) -> float:
    try:
        return math.exp(a)
    except OverflowError:
        return INF


def _sqrt(a: float) -> float:
    m = js_max(a, 0.0)
    return NAN if m != m else math.sqrt(m)


def _round_even(a: float) -> float:
    """expr.ts round: Math.round, then half to even (torch.round)."""
    r = js_round_f(a)
    if not math.isfinite(a):
        return r
    return r - 1 if abs(js_mod_pct(a, 1.0)) == 0.5 and js_mod_pct(r, 2.0) != 0 else r


def _jsmod(a: float, b: float) -> float:
    """expr.ts mod: a - b * floor(a / b)."""
    return a - b * js_floor(js_div(a, b))


def _eq(a, b) -> bool:
    return _isb(a) == _isb(b) and a == b


_CMP = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
        "==": _eq, "!=": lambda a, b: not _eq(a, b)}
_ARITH = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b, "/": js_div}
_PURE = {
    "abs": lambda v: abs(v[0]), "sign": lambda v: js_sign(v[0]), "min": lambda v: js_min(v[0], v[1]),
    "max": lambda v: js_max(v[0], v[1]), "clamp": lambda v: js_min(js_max(v[0], v[1]), v[2]),
    "wrap": lambda v: wrap_angle(v[0]), "sin": lambda v: math.sin(v[0]) if math.isfinite(v[0]) else NAN,
    "cos": lambda v: math.cos(v[0]) if math.isfinite(v[0]) else NAN, "atan2": lambda v: math.atan2(v[0], v[1]),
    "hypot": lambda v: js_hypot2(v[0], v[1]), "round": lambda v: _round_even(v[0]), "floor": lambda v: js_floor(v[0]),
    "sqrt": lambda v: _sqrt(v[0]), "mod": lambda v: _jsmod(v[0], v[1]), "exp": lambda v: _exp(v[0]),
}
_MEM_BOOL = {"ever"}


class Ctx:
    """EvalContext: signals (dict), vars (dict), lets (name -> compiled fn), slots (Memory.out), world (RouteWorld)."""
    __slots__ = ("signals", "vars", "lets", "slots", "world")

    def __init__(self, signals, vars_, lets, slots, world):
        self.signals, self.vars, self.lets, self.slots, self.world = signals, vars_, lets, slots, world


def compile_expr(e: dict):
    """A compiled Expr -> fn(Ctx), as expr.ts evaluateExpr."""
    k = e["k"]
    if k == "num":
        v = float(e["v"])
        return lambda c: v
    if k == "const":
        v = bool(e["v"])
        return lambda c: v
    if k == "sig":
        name = e["name"]

        def sig(c):
            v = c.signals.get(name)
            if v is None:
                raise KeyError(f"evaluateExpr: signal '{name}' has no value")
            return v
        return sig
    if k == "let":
        name = e["name"]
        return lambda c: c.lets[name](c)
    if k == "var":
        name = e["name"]
        return lambda c: c.vars.get(name, 0.0)
    if k == "neg":
        a = compile_expr(e["a"])
        return lambda c: -a(c)
    if k == "arith":
        f, a, b = _ARITH[e["op"]], compile_expr(e["a"]), compile_expr(e["b"])
        return lambda c: f(a(c), b(c))
    if k == "cmp":
        f, a, b = _CMP[e["op"]], compile_expr(e["a"]), compile_expr(e["b"])
        return lambda c: f(a(c), b(c))
    if k == "and":
        args = [compile_expr(x) for x in e["args"]]
        return lambda c: all(f(c) for f in args)
    if k == "or":
        args = [compile_expr(x) for x in e["args"]]
        return lambda c: any(f(c) for f in args)
    if k == "not":
        a = compile_expr(e["a"])
        return lambda c: not a(c)
    if k == "pure":
        fn, args = e["fn"], [compile_expr(x) for x in e["args"]]
        if fn == "choose":
            cnd, x, y = args
            return lambda c: x(c) if cnd(c) else y(c)
        if fn in ("path", "path_len"):
            px, py = args
            idx = 0 if fn == "path" else 1

            def route(c):
                if c.world is None:
                    return 0.0
                return c.world.plan(px(c), py(c))[idx]
            return route
        f = _PURE[fn]
        return lambda c: f([g(c) for g in args])
    if k == "mem":
        slot = int(e["slot"])
        if e["fn"] in _MEM_BOOL:
            return lambda c: c.slots[slot] != 0
        return lambda c: c.slots[slot]
    raise ValueError(f"expression kind {k!r}")


def expr_signals(e: dict, out: set) -> set:
    """expr.ts exprSignals."""
    if e["k"] == "sig":
        out.add(e["name"])
    for key in ("a", "b"):
        if key in e:
            expr_signals(e[key], out)
    for x in e.get("args", ()):
        expr_signals(x, out)
    return out


class Memory:
    """expr.ts Memory."""

    def __init__(self, calls: list[dict]):
        self.calls = calls
        self.args = [[compile_expr(a) for a in m["args"]] for m in calls]
        n = len(calls)
        self.out = [0.0] * n
        self.a = [0.0] * n
        self.b = [0.0] * n
        self.rings = [([0.0] * int(m["window"]) if m["fn"] == "avg" else [0.0] * (int(m["window"]) + 1) if m["fn"] == "delta"
                       else None) for m in calls]
        self.reset()

    def reset(self) -> None:
        n = len(self.calls)
        self.out[:] = [0.0] * n
        self.a[:] = [0.0] * n
        self.b[:] = [0.0] * n
        for r in self.rings:
            if r is not None:
                r[:] = [0.0] * len(r)
        for i, m in enumerate(self.calls):
            if m["fn"] == "since":
                self.out[i] = INF

    def update(self, c: Ctx) -> None:
        for i, m in enumerate(self.calls):
            fn = m["fn"]
            x = self.args[i][0](c)
            if fn in ("avg", "delta"):
                ring = self.rings[i]
                W = len(ring)
                seen = int(self.a[i])
                ring[seen % W] = float(x)
                self.a[i] = seen + 1
                n = min(seen + 1, W)
                if fn == "avg":
                    s = 0.0
                    for j in range(n):
                        s += ring[(seen - j + W) % W]
                    self.out[i] = s / n
                else:
                    self.out[i] = float(x) - ring[(seen - (n - 1) + W) % W]
            elif fn == "held":
                self.a[i] = self.a[i] + 1 if x else 0.0
                self.out[i] = self.a[i] / TICKS_PER_S
            elif fn == "since":
                if x:
                    self.b[i], self.a[i] = 1.0, 0.0
                elif self.b[i]:
                    self.a[i] = self.a[i] + 1
                self.out[i] = self.a[i] / TICKS_PER_S if self.b[i] else INF
            elif fn == "last":
                if self.args[i][1](c):
                    self.out[i] = float(x)
            elif fn == "count":
                if x and not self.b[i]:
                    self.out[i] = self.out[i] + 1
                self.b[i] = 1.0 if x else 0.0
            elif fn == "ever":
                if x:
                    self.out[i] = 1.0
            elif fn == "sum":
                if math.isfinite(x):
                    self.out[i] = self.out[i] + float(x) / TICKS_PER_S


class ProgramState:
    """program.ts ProgramRunner for a compiled duck-program/2 (program.py CompiledProgram with v2 = True)."""

    def __init__(self, program):
        self.program = program
        self.memory = Memory(program.memory_calls)
        self.vars: dict = {}
        self.world = None
        self.reset()

    def reset(self) -> None:
        self.memory.reset()
        self.vars.clear()
        self.vars.update(self.program.var_init)
        self.pending = None
        self.back = None

    def context(self, s: dict) -> Ctx:
        return Ctx(s, self.vars, self.program.lets, self.memory.out, self.world)

    def enter(self, node, s: dict | None) -> None:
        self.pending = None
        if node is None or not node.set:
            return
        if s is None:
            self.pending = node
        else:
            self._assign(node, self.context(s))

    def _assign(self, node, c: Ctx) -> None:
        for name, f in node.set:
            self.vars[name] = f(c)

    def tick(self, node, s: dict):
        """(go, via, rule) or None; BACK resolved to a node id (the start node before any any-rule left a node)."""
        c = self.context(s)
        if self.pending is not None:
            self._assign(self.pending, c)
            self.pending = None
        if self.memory.calls:
            self.memory.update(c)
        if node is None:
            return None
        t = self._pick(node, c)
        if t is None:
            return None
        go, via, rule = t
        if via == "any" and node.id not in self.program.handlers:
            self.back = node.id
        if go == "BACK":
            go = self.back if self.back is not None else self.program.start
        return go, via, rule

    def _pick(self, node, c: Ctx):
        for rule, (f, go, exc) in enumerate(self.program.any):
            if go == node.id or node.id in exc:
                continue
            if f(c):
                return go, "any", rule
        for rule, (f, go, _) in enumerate(node.until):
            if f(c):
                return go, "until", rule
        if node.max_time is not None:
            ts = c.signals.get("t_state")
            if ts is None:
                raise KeyError("nextNode: signal 't_state' has no value")
            if ts >= node.max_time[0]:
                return node.max_time[1], "max_time", None
        return None

    def feed_values(self, node, s: dict) -> list:
        """program.ts feedValues: [(slot index, value)], non-finite -> 0, clamped to +-FEED_ABS_MAX."""
        c = self.context(s)
        out = []
        for index, f in node.feed:
            v = f(c)
            out.append((index, js_max(-FEED_ABS_MAX, js_min(FEED_ABS_MAX, v)) if math.isfinite(v) else 0.0))
        return out

    def head_targets(self, node, s: dict) -> list:
        """program.ts headTargets: 4 values (NaN = the move's own), clamped to +-HEAD_ABS_MAX."""
        c = self.context(s)
        out = [NAN] * 4
        for index, f in node.head:
            v = f(c)
            if math.isfinite(v):
                out[index] = js_max(-HEAD_ABS_MAX, js_min(HEAD_ABS_MAX, v))
        return out


# ---- routes (packages/sim/src/runtime/planner.ts) ----------------------------------------------------------------
class Planner:
    """ScenePlanner over the TS-exported occupancy grid: info["routes"] = {n, x0, cell, occ (base64, one byte a cell)}."""

    def __init__(self, routes: dict):
        self.n = int(routes["n"])
        self.x0 = self.y0 = float(routes["x0"])
        self.cell = float(routes["cell"])
        self.occ = base64.b64decode(routes["occ"])
        if len(self.occ) != self.n * self.n:
            raise ValueError("routes.occ does not match n")
        self.fields: dict[int, list] = {}

    def cell_of(self, x: float, y: float) -> int:
        n, c = self.n, self.cell
        i = min(n - 1, max(0, math.floor(js_div(x - self.x0, c)) if math.isfinite(x) else 0))
        j = min(n - 1, max(0, math.floor(js_div(y - self.y0, c)) if math.isfinite(y) else 0))
        return j * n + i

    def field(self, target: int) -> list:
        hit = self.fields.get(target)
        if hit is not None:
            return hit
        n, occ, C = self.n, self.occ, self.cell
        d = [INF] * (n * n)
        d[target] = 0.0
        hd, hc = [0.0], [target]

        def push(dist, cell):
            k = len(hd)
            hd.append(dist)
            hc.append(cell)
            while k > 0:
                p = (k - 1) >> 1
                if hd[p] <= hd[k]:
                    break
                hd[p], hd[k] = hd[k], hd[p]
                hc[p], hc[k] = hc[k], hc[p]
                k = p

        def pop():
            top = (hd[0], hc[0])
            ld, lc = hd.pop(), hc.pop()
            if hd:
                hd[0], hc[0] = ld, lc
                k, L = 0, len(hd)
                while True:
                    a = 2 * k + 1
                    b = a + 1
                    m = k
                    if a < L and hd[a] < hd[m]:
                        m = a
                    if b < L and hd[b] < hd[m]:
                        m = b
                    if m == k:
                        break
                    hd[m], hd[k] = hd[k], hd[m]
                    hc[m], hc[k] = hc[k], hc[m]
                    k = m
            return top

        diag = C * math.sqrt(2)  # Math.SQRT2 = the double nearest sqrt(2): math.sqrt(2) is that double
        while hd:
            dist, cell = pop()
            if dist > d[cell]:
                continue
            i = cell % n
            j = (cell - i) // n
            for dj in (-1, 0, 1):
                for di in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ii, jj = i + di, j + dj
                    if ii < 0 or jj < 0 or ii >= n or jj >= n:
                        continue
                    c = jj * n + ii
                    if occ[c] == 1:
                        continue
                    if di != 0 and dj != 0 and (occ[j * n + ii] == 1 or occ[jj * n + i] == 1):
                        continue
                    nd = dist + (diag if (di != 0 and dj != 0) else C)
                    if nd < d[c]:
                        d[c] = nd
                        push(nd, c)
        if len(self.fields) >= 32:
            self.fields.pop(next(iter(self.fields)))
        self.fields[target] = d
        return d

    def plan(self, x: float, y: float, heading: float, tx: float, ty: float) -> tuple[float, float]:
        n = self.n

        def straight():
            return wrap_angle(math.atan2(ty - y, tx - x) - heading), js_hypot2(tx - x, ty - y)
        tc = self.cell_of(tx, ty)
        if self.occ[tc] == 1:
            return straight()
        d = self.field(tc)
        cur = self.cell_of(x, y)
        here = d[cur]
        if not math.isfinite(here):
            best = INF
            i = cur % n
            j = (cur - i) // n
            for dj in range(-2, 3):
                for di in range(-2, 3):
                    ii, jj = i + di, j + dj
                    if ii < 0 or jj < 0 or ii >= n or jj >= n:
                        continue
                    c = jj * n + ii
                    if d[c] < best:
                        best, cur = d[c], c
            if not math.isfinite(best):
                return straight()
            here = best
        for _ in range(3):
            i = cur % n
            j = (cur - i) // n
            best, nxt = d[cur], cur
            for dj in (-1, 0, 1):
                for di in (-1, 0, 1):
                    ii, jj = i + di, j + dj
                    if (di == 0 and dj == 0) or ii < 0 or jj < 0 or ii >= n or jj >= n:
                        continue
                    c = jj * n + ii
                    if d[c] < best:
                        best, nxt = d[c], c
            if nxt == cur:
                break
            cur = nxt
        wi = cur % n
        wj = (cur - wi) // n
        wx = tx if cur == tc else self.x0 + (wi + 0.5) * self.cell
        wy = ty if cur == tc else self.y0 + (wj + 0.5) * self.cell
        return wrap_angle(math.atan2(wy - y, wx - x) - heading), here


_PLANNERS: dict[int, Planner] = {}


class RouteWorld:
    """planner.ts DuckRoutes: path / path_len bound to the duck's pose at the program tick (x, y, heading)."""

    def __init__(self, routes: dict):
        key = id(routes)
        if key not in _PLANNERS:
            _PLANNERS[key] = Planner(routes)
        self.planner = _PLANNERS[key]
        self.x = self.y = self.heading = 0.0

    def plan(self, tx: float, ty: float) -> tuple[float, float]:
        return self.planner.plan(self.x, self.y, self.heading, tx, ty)
