"""JS Math semantics where the sim's numbers depend on them and Python's differ (stdlib only):

  js_round  Math.round: the nearest integer, halves up (Python's round() rounds halves to even: round(2.5) == 2);
  js_hypot  V8's Math.hypot: the arguments scaled by the largest, squares summed with Kahan compensation, sqrt, scaled
            back (Python's math.hypot is a different algorithm, so the two can differ in the last bit);
  js_fixed  +x.toFixed(digits): exact decimal rounding, ties away from zero (Python's format rounds ties to even).
The match runtime (duck_rl/program.py, duel.py) and the judge (metrics.py MatchPoints) use them where TS calls these.
"""
from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal


def js_round(x: float) -> int:
    """Math.round(x) for a finite x: floor(x), plus 1 when the fraction is >= 0.5 (exact: x - floor(x) is)."""
    r = math.floor(x)
    return r + 1 if x - r >= 0.5 else r


def js_hypot(*xs: float) -> float:
    """V8's Math.hypot (src/builtins/math.tq MathHypot), operation for operation."""
    absval, nan, mx = [], False, 0.0
    for v in xs:
        v = float(v)
        if v != v:
            nan = True
            absval.append(0.0)
        else:
            a = abs(v)
            absval.append(a)
            if a > mx:
                mx = a
    if mx == math.inf:
        return math.inf
    if nan:
        return math.nan
    if mx == 0:
        return 0.0
    s = comp = 0.0
    for a in absval:
        n = a / mx
        summand = n * n - comp
        prelim = s + summand
        comp = (prelim - s) - summand
        s = prelim
    return math.sqrt(s) * mx


def js_hypot2(a: float, b: float) -> float:
    """js_hypot(a, b) for two finite-or-not floats, without the argument list (the hot path)."""
    if a != a or b != b:
        return math.inf if (a == math.inf or a == -math.inf or b == math.inf or b == -math.inf) else math.nan
    a, b = abs(a), abs(b)
    mx = a if a > b else b
    if mx == math.inf:
        return math.inf
    if mx == 0:
        return 0.0
    n = a / mx
    s = n * n  # the first Kahan step from sum 0, compensation 0 (exact: s = n*n, comp = 0)
    n = b / mx
    summand = n * n
    return math.sqrt(s + summand) * mx


def js_fixed(x: float, digits: int = 6) -> float:
    """+x.toFixed(digits) (|x| < 1e21): the exact value of the double rounded to `digits` decimals, a tie away from zero
    (toFixed picks the larger n of |x|), then parsed back; a negative x that rounds to 0 gives -0.0, as +"-0.000000"
    does, and -0.0 itself gives 0.0 ((-0).toFixed is "0.000000")."""
    if x == 0:
        return 0.0
    q = Decimal(float(x)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
    return float(q)
