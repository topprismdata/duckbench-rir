"""The sim's seeded randomness in Python, bit for bit (packages/sim/src/prng.ts): xoshiro128** seeded from
sha256(label), and the draws a match makes from it, so a Python episode can replay a TS seed exactly.

  label        "<challenge_id>:<version>:<seed index>" (a public seed), "<challenge_id>:<version>:key:<seed key>" (a hidden ranked
               seed), or any other base a caller picks (training: "sumo:3:train:<env seed>:<n>", duel_env.py); a match
               appends ":duck:<slot>" for a duck slot's own stream (its randomisation, its sensor noise) and ":match"
               for the draws both ducks share (the spawn-line rotation), match_prng;
  state        the first 16 bytes of sha256(label), four little-endian uint32 words;
  uniform      lo + (hi - lo) * (next_u32() / 2^32), in float64 like the JS arithmetic.
draw_randomisation is episode.ts drawRandomisation (the per-slot draws, in their order). Sensor noise is drawn by
program.py from the duck's own stream with Xoshiro128.uniform. Stdlib only.

    python duck_rl/prng.py sumo:3:0:duck:0 sumo:3:0:match   # the first 20 draws of each label (checked against TS)
"""
from __future__ import annotations

import hashlib
import sys

_M32 = 0xFFFFFFFF
_TWO32 = 4294967296.0


def _rotl(x: int, k: int) -> int:
    return ((x << k) | (x >> (32 - k))) & _M32


class Xoshiro128:
    """xoshiro128** (Blackman & Vigna), as prng.ts Xoshiro128: four uint32 words, not all zero."""

    __slots__ = ("s0", "s1", "s2", "s3")

    def __init__(self, state) -> None:
        w = [int(x) & _M32 for x in state]
        if len(w) != 4 or not any(w):
            raise ValueError("xoshiro128** needs 4 words, not all zero")
        self.s0, self.s1, self.s2, self.s3 = w

    def next_u32(self) -> int:
        s0, s1, s2, s3 = self.s0, self.s1, self.s2, self.s3
        result = (_rotl((s1 * 5) & _M32, 7) * 9) & _M32
        t = (s1 << 9) & _M32
        s2 ^= s0
        s3 ^= s1
        s1 ^= s2
        s0 ^= s3
        s2 ^= t
        self.s0, self.s1, self.s2, self.s3 = s0, s1, s2, _rotl(s3, 11)
        return result

    def uniform(self, lo: float, hi: float) -> float:
        """Uniform in [lo, hi): lo + (hi - lo) * u, u = next_u32() / 2^32 (prng.ts uniform, float64)."""
        return lo + (hi - lo) * (self.next_u32() / _TWO32)


def label_prng(label: str) -> Xoshiro128:
    """The PRNG of `label`: state = the first 16 bytes of sha256(UTF-8 label), little-endian words."""
    h = hashlib.sha256(label.encode("utf-8")).digest()
    return Xoshiro128([int.from_bytes(h[4 * i:4 * i + 4], "little") for i in range(4)])


def rng_key(challenge: dict) -> str:
    """The name a Challenge's PRNG labels are made from (schema challengeRngKey): the info's challenge `rng_key` (set only on a
    renamed Challenge: its old id, so its runs stay the same; LB-370), else its id."""
    return str(challenge.get("rng_key") or challenge["id"])


def episode_label(challenge_id: str, version: int, seed_index: int | None = None, seed_key: str | None = None) -> str:
    """prng.ts episodeLabel: "<challenge_id>:<version>:<seed index>", or "<challenge_id>:<version>:key:<seed key>" (the index then
    does not enter the PRNG)."""
    if seed_key == "":
        raise ValueError("seed_key is empty")
    if seed_key is not None:
        return f"{challenge_id}:{version}:key:{seed_key}"
    if seed_index is None:
        raise ValueError("an episode label needs a seed index or a seed key")
    return f"{challenge_id}:{version}:{int(seed_index)}"


def match_label(base: str, stream) -> str:
    """prng.ts matchPrng's label: `base` + ":duck:<slot>" (stream 0 or 1) or ":match" (stream "match")."""
    if stream == "match":
        return base + ":match"
    if stream in (0, 1):
        return f"{base}:duck:{stream}"
    raise ValueError(f"match stream {stream!r}: 0, 1 or 'match'")


def match_prng(base: str, stream) -> Xoshiro128:
    """A match episode's PRNG for one stream (prng.ts matchPrng) from its label base (episode_label, or a training base)."""
    return label_prng(match_label(base, stream))


def draw_randomisation(prng: Xoshiro128, dr: dict) -> dict:
    """episode.ts drawRandomisation with constants.ts DR (info duel.runtime.dr), in its order: dx, dy, dyaw, sole
    friction, trunk mass scale."""
    xy, yaw = float(dr["spawnXY"]), float(dr["spawnYaw"])
    f0, f1 = (float(x) for x in dr["soleFriction"])
    m0, m1 = (float(x) for x in dr["trunkMass"])
    return dict(dx=prng.uniform(-xy, xy), dy=prng.uniform(-xy, xy), dyaw=prng.uniform(-yaw, yaw),
                friction=prng.uniform(f0, f1), mass=prng.uniform(m0, m1))


if __name__ == "__main__":
    # The first 20 draws of each label as uint32 and as uniform(0, 1), one JSON line per label (the TS check prints the same).
    import json
    for lab in sys.argv[1:] or ["sumo:3:0:duck:0", "sumo:3:0:duck:1", "sumo:3:0:match"]:
        a, b = label_prng(lab), label_prng(lab)
        print(json.dumps({"label": lab, "u32": [a.next_u32() for _ in range(20)],
                          "uniform": [b.uniform(-0.03, 0.03) for _ in range(20)]}))
