"""What a training ran, written beside its logs at start (kit 1.22.0, LB-277; stdlib only). The CLI uploads them with the
training's logs (PLAYER_TRAINING_FILES) when the player is logged in:

  config.py        the config file as it was when the training started (the reward, REWARD_PARAMS, ENV, PROGRAM,
                   WARM_MLP, the learner knobs); --set overrides are in kit_files.json
  kit_files.json   duck-rl-kit-files/1: the session, kit version, config name, --set overrides, the eye the training
                   used (kit 1.23.0: "eye" = its id with VISION = True; always null since the eye's removal, LB-330),
                   the frozen camera encoder (kit
                   1.27.0: "encoder" = {name, id} with ENCODER, else null), and each kit source file's
                   sha256 against the release's own copy (the CLI package's kit template, when this kit folder came from
                   `duckbench kit init` of the same kit version): "same", "changed" or "added"; "removed" lists the
                   release's files this kit lacks. No release copy (the repo's kit, another version): sha256 only.
  kit_diff.patch   the unified diff of the changed and added files against the release (only when there are some), at
                   most DIFF_MAX bytes.

write() never fails the training: a problem is one printed line.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
import shutil
from pathlib import Path

FORMAT = "duck-rl-kit-files/1"
#: The kit's own code (not its configs, programs or local state): what a player may change to change how training works.
SOURCE_GLOBS = ("*.py", "duck_rl/**/*.py", "lab/**/*.py", "duckref/**/*.py", "tools/**/*.py")
#: kit_diff.patch at most (a longer diff is cut, with a last line saying so).
DIFF_MAX = 256 * 1024
_SKIP = {"__pycache__", ".venv", "venv", "home", "submissions", "banks", "tmp", "node_modules", ".git"}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def source_files(kit: Path) -> list[str]:
    """The kit's source files (SOURCE_GLOBS), as posix paths relative to `kit`, sorted."""
    out = set()
    for g in SOURCE_GLOBS:
        for p in kit.glob(g):
            rel = p.relative_to(kit)
            if p.is_file() and not (_SKIP & set(rel.parts)):
                out.add(rel.as_posix())
    return sorted(out)


def kit_version_of(kit: Path) -> str | None:
    try:
        m = re.search(r'^KIT_VERSION = "([^"]+)"', (kit / "duck_rl" / "__init__.py").read_text(), re.M)
    except OSError:
        return None
    return m.group(1) if m else None


def release_dir(config: dict | None, kit_version: str) -> tuple[Path | None, str]:
    """The release's own kit (the CLI package's kit/ template beside its dist/main.js) and why/why not."""
    if config is None:
        return None, "repo kit: no release copy"
    try:
        tpl = Path(config["cli"][-1]).expanduser().resolve().parent.parent / "kit"
    except (KeyError, IndexError, TypeError):
        return None, "no CLI in duckbench-kit.json"
    if not tpl.is_dir():
        return None, "the CLI package has no kit template"
    v = kit_version_of(tpl)
    if v != kit_version:
        return None, f"the CLI's kit is {v}, this folder's {kit_version}"
    return tpl, "found"


def _text(b: bytes) -> list[str]:
    return b.decode("utf-8", errors="replace").splitlines(keepends=True)


def compare(kit: Path, tpl: Path | None) -> tuple[dict, list[str], str]:
    """{rel: {sha256, release?, release_sha256?}}, the release files this kit lacks, and the diff text (uncut)."""
    files: dict[str, dict] = {}
    diffs: list[str] = []
    for rel in source_files(kit):
        mine = (kit / rel).read_bytes()
        ent: dict = {"sha256": _sha(mine)}
        if tpl is not None:
            ref = tpl / rel
            if not ref.is_file():
                ent["release"] = "added"
                diffs.append("".join(difflib.unified_diff([], _text(mine), "/dev/null", f"kit/{rel}")))
            else:
                base = ref.read_bytes()
                if base == mine:
                    ent["release"] = "same"
                else:
                    ent.update(release="changed", release_sha256=_sha(base))
                    diffs.append("".join(difflib.unified_diff(_text(base), _text(mine), f"release/{rel}", f"kit/{rel}")))
        files[rel] = ent
    removed = [rel for rel in source_files(tpl) if rel not in files] if tpl is not None else []
    return files, removed, "".join(d if d.endswith("\n") else d + "\n" for d in diffs if d)


def write(out: Path, cfg_path: Path, kit: Path, *, session: str, label: str, kit_version: str, mode: str,
          config: dict | None, overrides: dict | None = None, eye: str | None = None, encoder: dict | None = None) -> None:
    """config.py, kit_files.json and (when the kit differs from its release) kit_diff.patch in the run dir `out`.
    `eye`: kept for the record's shape (kit 1.23.0's VISION eye id); None since the ray eye's removal (LB-330). `encoder`: {name, id} of the frozen camera encoder an ENCODER training
    reads (kit 1.27.0), else None."""
    try:
        shutil.copyfile(cfg_path, out / "config.py")
    except OSError as e:
        print(f"[duck-rl] could not copy the config into the run dir ({e}): its upload has no config.py", flush=True)
    try:
        tpl, why = release_dir(config, kit_version)
        files, removed, diff = compare(kit, tpl)
        try:
            cfg_name = cfg_path.resolve().relative_to(kit.resolve()).as_posix()
        except ValueError:
            cfg_name = cfg_path.name  # a config outside the kit: its name only (no local paths leave the machine)
        changed = sorted(r for r, e in files.items() if e.get("release") in ("changed", "added"))
        doc = dict(format=FORMAT, session=session, label=label, kit_version=kit_version, mode=mode, config=cfg_name,
                   overrides=overrides or {}, eye=eye, encoder=encoder, release=why, changed=changed, removed=removed, files=files)
        (out / "kit_files.json").write_text(json.dumps(doc, indent=1, default=str))
        if diff:
            data = diff.encode("utf-8")
            if len(data) > DIFF_MAX:
                data = data[:DIFF_MAX].decode("utf-8", errors="ignore").encode("utf-8") + f"\n# kit_diff.patch cut at {DIFF_MAX} bytes\n".encode()
            (out / "kit_diff.patch").write_bytes(data)
    except Exception as e:  # noqa: BLE001  (never fail a training over its provenance)
        print(f"[duck-rl] could not record the kit's files ({type(e).__name__}: {e})", flush=True)
