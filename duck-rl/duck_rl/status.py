"""status.json beside train.log (kit 1.26.0): the one record of whether a training runs, ended well, crashed or was stopped.

    {"format": "duck-rl-status/1", "state": "running" | "done" | "crashed" | "stopped", "label": ..., "session": ...,
     "pid": 1234, "started_at": ISO, "heartbeat_at": ISO, "ended_at": ISO?, "exit_code": 0?, "error": "one line"?}

`state` is "running" from the start until train.py's last line; then "done" (exit 0), "stopped" (SIGTERM, Ctrl-C) or
"crashed" (any other exit, `error` says why). While running, a daemon thread rewrites `heartbeat_at` every
HEARTBEAT_S. A process that dies without a word (SIGKILL, out of memory, a closed lid that never came back) leaves
"running" behind with a dead `pid` and an old `heartbeat_at`: readers (the CLI's training card) call that crashed
(process gone). Every write is atomic (a temp file, then a rename). Stdlib only; never raises.
"""
from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from pathlib import Path

FILE = "status.json"
FORMAT = "duck-rl-status/1"
STATES = ("running", "done", "crashed", "stopped")
HEARTBEAT_S = 15.0
ERROR_MAX = 300  # one line, cut


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def one_line(error: str | None) -> str | None:
    """The error's first non-empty line, at most ERROR_MAX characters (None stays None)."""
    if error is None:
        return None
    line = next((x.strip() for x in str(error).splitlines() if x.strip()), "") or "unknown error"
    return line if len(line) <= ERROR_MAX else line[: ERROR_MAX - 1] + "…"


def read(run_dir: Path | str) -> dict | None:
    """The run dir's status.json, or None (none, or unreadable)."""
    try:
        s = json.loads((Path(run_dir) / FILE).read_text())
    except (OSError, ValueError):
        return None
    return s if isinstance(s, dict) else None


def pid_alive(pid: object) -> bool:
    """A process with this pid exists (os.kill(pid, 0); another user's process counts as alive)."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def live_pid(run_dir: Path | str) -> int | None:
    """The pid of a training that still runs in `run_dir` (its train.pid, from --detach, or status.json's pid while its
    state is "running"), else None. A status.json that says the training ended wins over a stale train.pid; this
    process's own pid never counts."""
    d = Path(run_dir)
    s = read(d)
    if s is not None and s.get("state") in STATES and s.get("state") != "running":
        return None
    pids = []
    with contextlib.suppress(OSError, ValueError):
        pids.append(int((d / "train.pid").read_text().strip()))
    if s is not None and isinstance(s.get("pid"), int):
        pids.append(s["pid"])
    return next((p for p in pids if p != os.getpid() and pid_alive(p)), None)


def write(run_dir: Path | str, rec: dict) -> None:
    """status.json = rec, atomically (best effort)."""
    d = Path(run_dir)
    tmp = d / f".{FILE}.{os.getpid()}.tmp"
    try:
        tmp.write_text(json.dumps(rec, indent=1) + "\n")
        os.replace(tmp, d / FILE)
    except OSError:
        with contextlib.suppress(OSError):
            tmp.unlink()


class Status:
    """This process's status.json in `run_dir`: start() writes "running" and starts the heartbeat, end() the end state
    (once; later calls do nothing)."""

    def __init__(self, run_dir: Path | str, label: str, pid: int | None = None, session: str | None = None,
                 started_at: str | None = None):
        self.dir = Path(run_dir)
        self.rec = dict(format=FORMAT, state="running", label=label, session=session, pid=pid or os.getpid(),
                        started_at=started_at or now_iso(), heartbeat_at=now_iso())
        self.ended = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, heartbeat: bool = True) -> "Status":
        self._write()
        if heartbeat and self._thread is None:
            self._thread = threading.Thread(target=self._beat, name="duck-rl-status", daemon=True)
            self._thread.start()
        return self

    def set(self, **fields) -> None:
        with self._lock:
            self.rec.update(fields)
        self._write()

    def end(self, state: str, exit_code: int | None, error: str | None = None) -> None:
        if self.ended:
            return
        self.ended = True
        self._stop.set()
        with self._lock:
            self.rec.update(state=state, ended_at=now_iso(), heartbeat_at=now_iso(), exit_code=exit_code)
            if state == "crashed":
                self.rec["error"] = one_line(error) or f"exit {exit_code}"
            else:
                self.rec.pop("error", None)
        self._write()

    def _beat(self) -> None:
        while not self._stop.wait(HEARTBEAT_S):
            with self._lock:
                if self.ended:
                    return
                self.rec["heartbeat_at"] = now_iso()
            self._write()

    def _write(self) -> None:
        with self._lock:
            write(self.dir, dict(self.rec))


def end_state(exit_code: int, error: str | None, signalled: bool) -> str:
    """train.py's end: "stopped" after SIGTERM / Ctrl-C (whatever the export did), "done" on exit 0, else "crashed"."""
    if signalled:
        return "stopped"
    return "done" if exit_code == 0 and error is None else "crashed"
