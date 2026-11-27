"""A busy machine, noticed while it trains (newbie test 2026-11-02, N25): on a loaded Mac the checkpoint evals grew
17 s -> 214 s and training fell from ~5000 to ~200 env steps/s with no word in the log. BusyWatch compares the run with
its own start, so the Challenge, the config and the machine's speed don't matter:

- training: the median steps/s of the last minute against the median of the first minute (iterations 1+, the first
  one carries the pool's start-up);
- the in-loop evals: wall seconds per CPU second of each eval against the first eval's (an eval's length depends on the
  episodes, its wall/CPU ratio only on how much CPU it gets).

When either is SLOW_RATIO times worse it returns one line for the log, at most once per REPEAT_S; summary() is what
summary.json carries. Information only (docs/rl-rules.md): it never changes the training. Stdlib only.
"""
from __future__ import annotations

import statistics

SLOW_RATIO = 3.0  # this many times slower than the run's own start: the line
WINDOW_S = 60.0  # the first minute (the baseline) and the trailing minute (now)
MIN_SPAN_S = 40.0  # the trailing window must cover this much training time (a sustained slowdown, not one slow iteration)
REPEAT_S = 300.0  # at most one line per this many seconds of training


def busy_line(ratio: float) -> str:
    """The one line the agent relays to its human."""
    return f"Mac is busy: training runs ~{ratio:.0f}× slower than at its start (other programs use the CPU)"


class BusyWatch:
    def __init__(self, ratio: float = SLOW_RATIO, window_s: float = WINDOW_S, repeat_s: float = REPEAT_S):
        self.ratio, self.window_s, self.repeat_s = ratio, window_s, repeat_s
        self.rows: list[tuple[float, float]] = []  # (training s, steps/s) per iteration
        self.base_sps: float | None = None
        self.base_eval: float | None = None  # the first eval's wall / CPU seconds
        self.worst = 1.0  # the largest slowdown seen (either measure)
        self.worst_sps = None
        self.worst_eval = None
        self.lines: list[dict] = []  # every line printed: {t, ratio, source}
        self.last_at: float | None = None

    def _maybe(self, t: float, ratio: float, source: str) -> str | None:
        self.worst = max(self.worst, ratio)
        if ratio < self.ratio or (self.last_at is not None and t - self.last_at < self.repeat_s):
            return None
        self.last_at = t
        self.lines.append(dict(t=round(t, 1), ratio=round(ratio, 1), source=source))
        return busy_line(ratio)

    def sps(self, elapsed_s: float, sps: float, it: int | None = None) -> str | None:
        """One training iteration (its training seconds and steps/s); a line when the last minute is SLOW_RATIO slower."""
        if it == 0 or not sps or sps <= 0:
            return None
        self.rows.append((float(elapsed_s), float(sps)))
        if self.base_sps is None:
            if elapsed_s < self.window_s:
                return None
            first = [s for t, s in self.rows if t <= self.window_s] or [self.rows[0][1]]
            self.base_sps = statistics.median(first)
            return None
        recent = [(t, s) for t, s in self.rows if t >= elapsed_s - self.window_s]
        if recent[-1][0] - recent[0][0] < MIN_SPAN_S:
            return None
        ratio = self.base_sps / statistics.median(s for _, s in recent)
        self.worst_sps = max(self.worst_sps or 0.0, ratio)
        return self._maybe(elapsed_s, ratio, "training")

    def eval(self, elapsed_s: float, eval_s: float | None, cpu_s: float | None) -> str | None:
        """One finished in-loop eval (wall and CPU seconds); a line when it got SLOW_RATIO less CPU than the first one."""
        if not eval_s or not cpu_s or cpu_s <= 0:
            return None
        per = float(eval_s) / float(cpu_s)
        if self.base_eval is None:
            self.base_eval = per
            return None
        ratio = per / self.base_eval
        self.worst_eval = max(self.worst_eval or 0.0, ratio)
        return self._maybe(elapsed_s, ratio, "eval")

    def summary(self) -> dict | None:
        """summary.json `busy`: null when the run never slowed SLOW_RATIO times; else the worst slowdown and its line."""
        if self.worst < self.ratio:
            return None
        return dict(line=busy_line(self.worst), worst_ratio=round(self.worst, 1),
                    training_ratio=round(self.worst_sps, 1) if self.worst_sps else None,
                    eval_ratio=round(self.worst_eval, 1) if self.worst_eval else None,
                    start_sps=round(self.base_sps) if self.base_sps else None, printed=self.lines)
