"""The attempt's number, as the server ranks it (kit 1.14.0; Emil 2026-10-30, packages/schema rank.ts summariseResults).

A solo Challenge ranks by the mean of the best half of its seeds: sort the per-seed natural results in the Challenge's
direction (a race seed that never finishes counts as the Challenge's duration, 30 s, like any other result) and average
the best ceil(n/2) (12 of 24 hidden seeds, 4 of 8 practice, 8 of 16, 16 of 32). A crowd Challenge (a field: the mean
points per round, tie time) keeps the plain mean; duels (Sumo) keep their own mean points. ts/eval.ts writes this number
as the summary's `result` (bestHalfMean there); this is its Python mirror for code that holds per-seed results.
"""
from __future__ import annotations

import math
from collections.abc import Iterable


def rank_result(results: Iterable[float | None], lower_is_better: bool, crowd: bool = False) -> float | None:
    """The attempt's ranking number: the best-half mean (crowd: the plain mean). None for no seeds or a None / non-finite one."""
    xs = list(results)
    if not xs or any(x is None or not math.isfinite(x) for x in xs):
        return None
    if crowd:
        return sum(xs) / len(xs)
    k = math.ceil(len(xs) / 2)
    best = sorted(xs, reverse=not lower_is_better)[:k]
    return sum(best) / k


def plain_mean(results: Iterable[float | None]) -> float | None:
    """The plain mean over all seeds (the secondary number beside rank_result); None as rank_result."""
    xs = list(results)
    if not xs or any(x is None or not math.isfinite(x) for x in xs):
        return None
    return sum(xs) / len(xs)
