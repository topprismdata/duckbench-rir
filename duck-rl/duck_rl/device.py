"""train.device: a short human name of the training machine, e.g. "M3 MacBook Air" (PLAN L2c.7, MR-18: the ranked
caption "trained by <model> · <N> min · <device> (self-reported)"). Stdlib only; it ships in the player's kit.

The server keeps 1-32 chars of [A-Za-z0-9 ._@/+:-] after trim and silently drops anything else (the SITE contract,
2026-10-05), so the auto-detected name is sanitised to fit; a name the player gives (--device, DUCKBENCH_DEVICE, the
config's DEVICE) is kept as given, with a warning when the server would drop it. An empty string means "leave it out".

    python duck_rl/device.py     # prints what this machine is detected as
"""
from __future__ import annotations

import functools
import os
import platform
import re
import subprocess
import sys

MAX_LEN = 32
ALLOWED = re.compile(r"[A-Za-z0-9 ._@/+:-]{1,32}")  # the server's check, after trim
_BAD = re.compile(r"[^A-Za-z0-9 ._@/+:-]+")


def sanitise(s: str) -> str | None:
    """Disallowed chars -> space, spaces collapsed, trimmed, cut to MAX_LEN at a word boundary; None if nothing is left."""
    s = re.sub(r"\s+", " ", _BAD.sub(" ", s)).strip()
    if len(s) > MAX_LEN:
        cut = s[:MAX_LEN + 1].rsplit(" ", 1)[0] if " " in s[:MAX_LEN + 1] else s[:MAX_LEN]
        s = cut[:MAX_LEN].strip()
    return s or None


def _run(cmd: list[str], timeout: float = 5.0) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:  # noqa: BLE001  (missing tool, timeout: no name from it)
        return ""


BATCH_ENVS = 128  # envs in all (workers x per_worker); the learner's batch is 64 steps x these = 8192
FALLBACK_LAYOUT = (8, 16)


def performance_cores() -> int | None:
    """The Mac's performance-core count (sysctl hw.perflevel0.physicalcpu, Apple silicon), else None."""
    if platform.system() != "Darwin":
        return None
    try:
        n = int(_run(["sysctl", "-n", "hw.perflevel0.physicalcpu"]).strip())
    except ValueError:  # an Intel Mac has no perflevels
        return None
    return n if n > 0 else None


def default_layout() -> tuple[int, int, str]:
    """(workers, per_worker, why) when train.py gets no --workers (kit 1.6.0, LB-56): one env worker per performance
    core, ~BATCH_ENVS envs in all, so the batch stays ~8192 steps. The workers step in lock-step, so a worker on an
    efficiency core holds back the rest: 8 workers on a 4P+4E Air trained only ~6% faster on an idle Mac and no faster
    with the person's apps open, while their apps' work ran ~1.4x slower (experiments/2026-10-06-pcore-training).
    Elsewhere 8x16."""
    p = performance_cores()
    if not p:
        return (*FALLBACK_LAYOUT, "no performance-core count on this machine")
    return p, max(1, round(BATCH_ENVS / p)), f"one per performance core ({p})"


def _short_cpu(brand: str) -> str:
    """'Intel(R) Core(TM) i7-8559U CPU @ 2.70GHz' -> 'Intel Core i7-8559U'; 'AMD Ryzen 9 7950X 16-Core Processor' -> 'AMD Ryzen 9 7950X'."""
    s = re.sub(r"\((R|TM|tm|r)\)", "", brand)
    s = re.sub(r"@.*$", "", s)
    s = re.sub(r"\b(CPU|Processor|\d+-Core)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _mac() -> str | None:
    brand = _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
    m = re.search(r"Model Name:\s*(.+)", _run(["system_profiler", "SPHardwareDataType"], 10.0))
    model = m.group(1).strip() if m else None
    if brand.startswith("Apple "):  # "Apple M3" -> "M3", "Apple M3 Pro" -> "M3 Pro"
        chip = brand[len("Apple "):].strip()
        return " ".join(x for x in (chip, model) if x) or None
    cpu = _short_cpu(brand) if brand else None
    return " ".join(x for x in (model or "Mac", cpu) if x) or None


def _linux() -> str | None:
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return _short_cpu(line.split(":", 1)[1])
    except OSError:
        pass
    return None


@functools.lru_cache(maxsize=1)
def detect() -> str | None:
    """The auto-detected name, sanitised to the server's rule (cached: system_profiler takes ~0.5 s); None if nothing is found."""
    try:
        sysname = platform.system()
        name = _mac() if sysname == "Darwin" else _linux() if sysname == "Linux" else None
        if not name:
            name = platform.processor() or platform.machine() or None
        return sanitise(name) if name else None
    except Exception:  # noqa: BLE001  (never crash a training run over its caption)
        return None


def resolve(cli: str | None, config: str | None = None) -> str | None:
    """train.device: --device (cli), else env DUCKBENCH_DEVICE, else the config's DEVICE, else detect(). The first one
    that is set wins; an empty string there means "leave it out" (None). A given name is kept as given (trimmed); a
    warning says when the server would drop it."""
    for source, v in (("--device", cli), ("DUCKBENCH_DEVICE", os.environ.get("DUCKBENCH_DEVICE")), ("DEVICE", config)):
        if v is None:
            continue
        v = str(v).strip()
        if v and not ALLOWED.fullmatch(v):
            print(f"[duck-rl] warning: device {v!r} ({source}) is not 1-{MAX_LEN} chars of [A-Za-z0-9 ._@/+:-]: the server "
                  f"will drop train.device (e.g. {sanitise(v)!r} would pass)", file=sys.stderr, flush=True)
        return v or None
    return detect()


if __name__ == "__main__":
    print(detect())
