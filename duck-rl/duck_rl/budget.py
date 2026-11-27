"""The owner's Mac-time budget in a player's kit (docs/rl-rules.md A.4: the budget is the owner's Mac wall-clock, and
the owner sets it). Nothing else is limited.

  budget.json          {"limit_minutes": N}, the owner's limit (null: none, still metered). Edit it, or
                       `python -m duck_rl.budget --set N` (`--set none` removes the limit)
  budget-ledger.jsonl  one line per charged command once it has ended: {"kind", "label", "start", "seconds", "run", "end"}
  home/budget-runs/    one <pid>-<start ms>.json per charged command that has not closed yet: {"pid", "kind", "label",
                       "start", "last_alive"}; a thread refreshes last_alive every BEAT_S seconds
  python -m duck_rl.budget   the limit, what is spent and what is left

Charged, from the command's start (imports included) to its end: train.py, make_bank.py, probe.py, tools/eval_ts.py
(not `train.py --help`, nor the short command `train.py --detach` that starts the training: the training it starts is).
Each opens its record with begin() before its heavy imports (torch, mujoco: ~25 s on a cold start), so one killed
while importing is charged too; its charged() block adopts that record. Spent =the ledger + each command still running (to now) + each one that died without closing its record (killed,
crashed, its terminal closed: to its last heartbeat; the next charged command moves it into the ledger). So commands
running side by side share what is left, and train.py re-reads it while it trains: it refuses to start once the limit
is spent and stops at it (training ends --reserve-s before, then the export). A refused command is not charged.

Active in a player's kit (duckbench-kit.json) and in any kit dir that holds a budget.json; in the monorepo without one
it does nothing (experiments meter with spikes/duck-rl/runner/budget.py instead). Stdlib only.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import threading
import time
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
BUDGET = KIT / "budget.json"
LEDGER = KIT / "budget-ledger.jsonl"
RUNS = KIT / "home" / "budget-runs"  # the kit's own home (not DUCK_RL_HOME), so every command of this kit sees the others
BEAT_S = 10.0  # heartbeat period
STALE_S = 600.0  # no heartbeat for this long: the record is dead even if its pid is alive again (pid reuse)


def active() -> bool:
    # duck-arena-kit.json: a kit folder made before the DuckBench rename (bridge.LEGACY_KIT_JSON)
    return BUDGET.exists() or (KIT / "duckbench-kit.json").exists() or (KIT / "duck-arena-kit.json").exists()


def limit_s() -> float | None:
    """The owner's limit in seconds, None for no limit."""
    if not BUDGET.exists():
        return None
    v = json.loads(BUDGET.read_text()).get("limit_minutes")
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
        raise SystemExit(f"{BUDGET}: limit_minutes {v!r}: a number of minutes >= 0, or null for no limit")
    return 60.0 * v


def _ledger() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]


def _records() -> list[tuple[Path, dict]]:
    out = []
    for f in sorted(RUNS.glob("*.json")) if RUNS.is_dir() else []:
        with contextlib.suppress(OSError, ValueError):  # closed meanwhile
            out.append((f, json.loads(f.read_text())))
    return out


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _live(r: dict, now: float) -> bool:
    return now - float(r["last_alive"]) < STALE_S and _pid_alive(int(r["pid"]))


def usage(now: float | None = None) -> tuple[float, int]:
    """(seconds spent, commands running): the ledger, plus each record, a running one to now and a dead one to its last
    heartbeat."""
    now = time.time() if now is None else now
    spent, live = sum(float(r["seconds"]) for r in _ledger()), 0
    for _, r in _records():
        alive = _live(r, now)
        live += alive
        spent += max(0.0, (now if alive else float(r["last_alive"])) - float(r["start"]))
    return spent, live


def spent_s() -> float:
    return usage()[0]


def left_s() -> float | None:
    """Seconds left under the limit (may be negative), None for no limit."""
    lim = limit_s()
    return None if lim is None else lim - spent_s()


def _close(r: dict, seconds: float, end: str, run: str) -> None:
    rec = dict(kind=r["kind"], label=r["label"], start=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(float(r["start"]))),
               seconds=round(seconds, 1), run=run, end=end)
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")


def reap() -> None:
    """Move the records of commands that died without closing them into the ledger, charged to their last heartbeat."""
    now = time.time()
    for f, r in _records():
        if _live(r, now):
            continue
        try:
            f.unlink()  # the claim: of two commands reaping at once, one charges it
        except FileNotFoundError:
            continue
        _close(r, max(0.0, float(r["last_alive"]) - float(r["start"])), "last heartbeat", f.stem)


class Run:
    """A charged command's record in RUNS, kept alive by a heartbeat thread, closed into the ledger by close()."""

    def __init__(self, kind: str, label: str, start: float):
        self.start = start
        self.file = RUNS / f"{os.getpid()}-{int(start * 1000)}.json"
        self.rec = dict(pid=os.getpid(), kind=kind, label=label, start=start, last_alive=time.time())
        self.lost = False  # another command closed the record (no heartbeat for STALE_S): never write it again
        self._lock, self._stop = threading.Lock(), threading.Event()
        RUNS.mkdir(parents=True, exist_ok=True)
        self._write()
        self._thread = threading.Thread(target=self._beat_loop, name="budget-heartbeat", daemon=True)
        self._thread.start()

    def _write(self) -> None:
        tmp = self.file.with_name(f".{self.file.name}.tmp")
        tmp.write_text(json.dumps(self.rec))
        os.replace(tmp, self.file)

    def beat(self, label: str | None = None) -> None:
        with self._lock:
            if label is not None:
                self.rec["label"] = label
            if self.lost or not self.file.exists():
                self.lost = True
                return
            self.rec["last_alive"] = time.time()
            self._write()

    def _beat_loop(self) -> None:
        while not self._stop.wait(BEAT_S):
            with contextlib.suppress(OSError):
                self.beat()

    def deadline(self) -> float | None:
        """The epoch this command should end by, the limit re-read: now + what is left, shared with the other commands
        running (each gets an equal part); None for no limit."""
        lim = limit_s()
        if lim is None:
            return None
        now = time.time()
        spent, live = usage(now)
        return now + (lim - spent) / max(1, live)

    def close(self) -> float:
        """Charge this command's wall-clock (minus what a reaper already charged) to the ledger: the seconds charged."""
        self._stop.set()
        self._thread.join()
        with contextlib.suppress(FileNotFoundError):
            self.file.unlink()
        done = sum(float(r["seconds"]) for r in _ledger() if r.get("run") == self.file.stem)
        seconds = max(0.0, time.time() - self.start - done)
        _close(self.rec, seconds, "exit", self.file.stem)
        return seconds


def banner() -> str:
    """One line: the kit version, the mode (bridge.py) and the budget, so a stray duckbench-kit.json or budget.json
    shows."""
    from . import KIT_VERSION, bridge
    head = f"duck-rl kit {KIT_VERSION} · {bridge.MODE} · budget"
    if not active():
        return f"{head} off"
    lim = limit_s()
    spent, live = usage()
    b = f"no limit ({spent / 60:.1f} min spent)" if lim is None else f"{(lim - spent) / 60:.1f}/{lim / 60:.1f} min"
    return f"{head} {b}" + (f" · {live} other charged command(s) running" if live else "")


def _open(kind: str, label: str, start: float | None) -> Run | None:
    """The banner, then the command's Run from `start` (or now); None when the budget is not active. SystemExit if the
    limit is spent (nothing charged)."""
    t0 = time.time() if start is None else start
    if active():
        reap()
    print(banner(), flush=True)
    if not active():
        return None
    left = left_s()
    if left is not None and left <= time.time() - t0:
        raise SystemExit(f"[budget] the limit is spent: {spent_s() / 60:.1f} of {limit_s() / 60:.1f} min ({BUDGET.name}); "
                         "only the owner raises it")
    return Run(kind, label, t0)


_BEGUN: dict[str, Run | None] = {}  # kind -> the Run begin() opened, until charged(kind) adopts it


def begin(kind: str, start: float) -> None:
    """Open the command's record now, at its start and before its heavy imports, so a command killed while importing
    is charged too (to its last heartbeat); the command's charged(kind, label) adopts it and fills in the label. The
    scripts call it right after their stdlib imports. SystemExit if the limit is spent (nothing charged)."""
    _BEGUN[kind] = _open(kind, "", start)


@contextlib.contextmanager
def charged(kind: str, label: str = "", start: float | None = None):
    """Charge the block's wall-clock, from `start` (epoch s; the scripts pass their start, before their imports) or now,
    or adopt the record begin(kind) opened. SystemExit before the block if the limit is spent (nothing charged). Yields
    the Run (its deadline() is the epoch to end by, the limit re-read), or None when the budget is not active."""
    run = _BEGUN.pop(kind) if kind in _BEGUN else _open(kind, label, start)
    if run is None:
        yield None
        return
    run.beat(label)  # an adopted record gets its label
    try:
        yield run
    finally:
        print(f"[budget] {kind} charged {run.close():.0f} s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="The owner's Mac-time budget (budget.json + budget-ledger.jsonl)")
    ap.add_argument("--set", metavar="MINUTES", help="set the limit (the owner's call); 'none' removes it")
    a = ap.parse_args()
    if a.set is not None:
        BUDGET.write_text(json.dumps({"limit_minutes": None if a.set == "none" else float(a.set)}) + "\n")
    lim = limit_s()
    spent, live = usage()
    print(json.dumps({"limit_minutes": None if lim is None else round(lim / 60, 2), "spent_minutes": round(spent / 60, 2),
                      "left_minutes": None if lim is None else round((lim - spent) / 60, 2), "running": live}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
