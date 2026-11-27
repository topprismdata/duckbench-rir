#!/usr/bin/env python
"""Tests of the senses kit (duck_rl/sense): the reference evaluator's copy on the repo's parity fixture, the collect
reader (Row.at / prev, None labels, dict labels), each template trained on a small synthetic collect dir (the written file
through ref.py == the torch forward pass within 1e-5; valid; within the MACs budget), the site's own evaluator (sim
sense-eval.ts via ts/check_sense.ts) on the same held-out rows, the Python mirror of senseProblem against the TS one on
broken specs, and refit. The TS parts need the repo and node (skipped in a player's kit). ~40 s:

  python -m unittest tools/sense_test.py        (from the kit dir, the kit's python)
"""
from __future__ import annotations

import copy
import json
import math
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(KIT))
from duck_rl import bridge  # noqa: E402
from duck_rl.sense import check as C  # noqa: E402
from duck_rl.sense import ref as R  # noqa: E402
from duck_rl.sense import train as T  # noqa: E402
from duck_rl.sense.data import Data  # noqa: E402
from duck_rl.sense.synth import make  # noqa: E402

REPO = Path(bridge.REPO) if bridge.REPO is not None else None
CHECK_TS = REPO / "spikes/duck-rl/ts/check_sense.ts" if REPO else None
HAS_TS = bool(CHECK_TS and CHECK_TS.exists() and shutil.which("node"))

NEAR = '''# the rival within 0.3 m in 0.5 s (true poses; README "Senses")
import math

def label(row):
    later = row.at(0.5)
    if later is None:
        return None
    w = later.world
    return math.hypot(w["other0_x"] - w["true_x"], w["other0_y"] - w["true_y"]) < 0.3
'''
MULTI = '''OUTPUTS = {"dist": {"unit": "m"}, "near": {"activation": "sigmoid"}}
def label(row):
    later = row.at(0.5)
    return {"dist": min(row.world["rival_dist"], 3.0), "near": None if later is None else float(later.world["rival_dist"] < 0.3)}
'''


def ts(jp: Path, inputs: Path | None = None) -> dict:
    cmd = ["node", str(CHECK_TS), str(jp)] + (["--inputs", str(inputs)] if inputs else [])
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, timeout=120)
    return json.loads(r.stdout.strip().splitlines()[-1])


class SenseKit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="sense-test-"))
        cls.a = make(cls.tmp / "a", list(range(0, 8)), ticks=100, challenge="synth")
        cls.b = make(cls.tmp / "b", list(range(30, 34)), ticks=100, challenge="synth2")
        cls.trained = {}
        for t in T.TEMPLATES:
            ev = T.train([str(cls.a)], NEAR, "near.py", t, str(cls.tmp / f"near_{t.replace('-', '_')}"), log=lambda *a: None)
            cls.trained[t] = ev

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_ref_copy_on_repo_fixture(self):
        fx = REPO / "packages/sim/test/fixtures/sense" if REPO else None
        if not (fx and fx.exists()):
            self.skipTest("no repo fixture")
        self.assertLessEqual(R.check(fx), 1e-6)
        pc = REPO / "packages/sim/test/fixtures/sense-prev-conv"
        if pc.exists():  # inputs.prev_conv: the previous tick's conv features
            self.assertLessEqual(R.check(pc), 1e-6)
            self.assertTrue(C.check(pc / "sense.sense.json", ts=False)["ok"])

    def test_rows(self):
        d = Data([str(self.a), str(self.b)])
        r = d.row(0)
        self.assertIsNone(r.prev)
        later = r.at(0.5)
        self.assertEqual((later.seed, later.duck, later.tick), (r.seed, r.duck, r.tick + 5))
        self.assertEqual(later.prev.tick, r.tick + 4)
        last = d.row(int(d.seqs[0][-1]))
        self.assertIsNone(last.at(0.1))
        self.assertIn("result", last.world)
        self.assertEqual(d.group_names(), ["synth", "synth2"])
        names, Y = T.run_label(T.load_label(NEAR), d, None, "all")
        self.assertEqual(names, ["value"])
        self.assertEqual(int(np.isnan(Y).sum()), 5 * len(d.seqs))  # the last 0.5 s of every sequence has no label
        _, Y0 = T.run_label(T.load_label(NEAR), d, None)  # default: only duck 0 (the strategy's own) is labelled
        self.assertTrue(np.isnan(Y0[d.duck != 0]).all())
        np.testing.assert_array_equal(Y0[d.duck == 0], Y[d.duck == 0])

    def test_templates_parity_valid_budget(self):
        for t, ev in self.trained.items():
            with self.subTest(template=t):
                self.assertTrue(ev["parity"]["ok"], ev["parity"])
                self.assertLessEqual(ev["parity"]["max_abs_diff"], 1e-5)
                self.assertGreater(ev["parity"]["rows"], 50)
                res = C.check(ev["_files"]["json"], ts=False)
                self.assertTrue(res["ok"], res)
                self.assertLessEqual(res["macs"], C.MACS_PER_TICK)
                self.assertIsNotNone(ev["test"]["value"]["auc"])
                j = json.loads(Path(ev["_files"]["json"]).read_text())
                self.assertEqual(j["definition"]["label"], NEAR)
                self.assertIn("duckbench sense collect synth synthetic --seeds 0-7", j["definition"]["collect"])
                self.assertEqual("conv" in j, t != "mlp-cam")

    def test_emb1_block_is_the_kit_encoder(self):
        from duck_rl import encoder as E
        enc = E.load("emb1")
        j = json.loads(Path(self.trained["emb1-head"]["_files"]["json"]).read_text())
        w = np.fromfile(Path(self.trained["emb1-head"]["_files"]["json"]).with_suffix(".bin"), "<f4")
        ref = R.Sense(j, w)
        pics = Data([str(self.a)]).pictures(np.arange(3))
        for p in pics:
            np.testing.assert_array_equal(ref.conv_features(p), enc.latent(p))

    def test_ts_evaluator_matches(self):
        if not HAS_TS:
            self.skipTest("no repo / node")
        d = Data([str(self.a)])
        for t, ev in self.trained.items():
            with self.subTest(template=t):
                jp = Path(ev["_files"]["json"])
                j = json.loads(jp.read_text())
                s = d.seqs[-1][:40]
                body, cam = j["inputs"]["body"], j["inputs"]["cam"]
                cur = d.current(body, cam)
                ticks = []
                for k, i in enumerate(s):
                    bd, at = {}, 0
                    for n in body:
                        w = C.body_width(n)
                        bd[n] = [float(x) for x in cur[i, at:at + w]] if n in C.BODY_GROUPS else float(cur[i, at])
                        at += w
                    ticks.append({"body": bd, "cam": {n: float(cur[i, at + q]) for q, n in enumerate(cam)}})
                pics = d.pictures(s)
                inp = self.tmp / f"inputs_{t}.json"
                pics.astype("<f4").tofile(self.tmp / f"pics_{t}.bin")
                inp.write_text(json.dumps({"pictures": f"pics_{t}.bin", "ticks": ticks}))
                got = ts(jp, inp)
                self.assertIsNone(got["problem"])
                self.assertEqual(got["macs"], C.macs(j))
                self.assertEqual(got["id"], C.check(jp, ts=False)["id"])
                ref = R.Sense(j, np.fromfile(jp.with_suffix(".bin"), "<f4"))
                want = [ref.step(pics[k], tk["body"], tk["cam"]) for k, tk in enumerate(ticks)]
                self.assertLess(float(np.abs(np.asarray(got["outputs"], float) - np.asarray(want)).max()), 1e-5)

    def test_python_mirror_agrees_with_ts(self):
        if not HAS_TS:
            self.skipTest("no repo / node")
        jp = Path(self.trained["conv"]["_files"]["json"])
        base = json.loads(jp.read_text())
        bb = jp.with_suffix(".bin").read_bytes()
        muts = {
            "ok": lambda j: None,
            "name": lambda j: j.update(name="Bad-Name"),
            "body": lambda j: j["inputs"]["body"].append("height"),
            "cam": lambda j: j["inputs"]["cam"].append("cam_nope"),
            "picture": lambda j: j.update(picture="ray-rgbd-palette-4x16x32/1"),
            "nopicture": lambda j: j.pop("picture"),
            "mlp_in": lambda j: j["mlp"][0].update({"in": j["mlp"][0]["in"] + 1}),
            "outputs": lambda j: j["outputs"][0].update(activation="relu"),
            "gate": lambda j: j.update(gate=3),
            "ops": lambda j: j.update(ops=[{"op": "class_frac", "class": "self_accent", "rows": [0, 1], "cols": [0, 1]}]),
            "conv_k": lambda j: j["conv"]["convs"][0].update(k=9),
            "offset": lambda j: j["mlp"][-1].update(bias_offset=10 ** 7),
            "budget": lambda j: j["conv"]["convs"].__setitem__(0, {**j["conv"]["convs"][0], "out": 64}),
            "extra": lambda j: j.update(foo=1),
            "layers": lambda j: j.update(mlp=j["mlp"] + [{**j["mlp"][-1], "in": j["mlp"][-1]["out"]}] * 16),
        }
        for k, f in muts.items():
            with self.subTest(mutation=k):
                j = copy.deepcopy(base)
                f(j)
                p = self.tmp / f"mut_{k}.sense.json"
                p.write_text(json.dumps(j))
                p.with_suffix(".bin").write_bytes(bb)
                py = C.problem(j, len(bb))
                tsr = ts(p)["problem"]
                self.assertEqual(py is None, tsr is None, (k, py, tsr))
                if k == "ok":
                    self.assertIsNone(py)

    def test_nonfinite_outputs_and_std(self):
        """Review finding 2: a non-finite output reads its none, a non-finite gate reads closed; std >= 1e-6 (float32)."""
        def spec(gate):
            j = {"format": "duck-sense/1", "name": "nan", "inputs": {"body": ["t", "speed"], "cam": [], "prev": False, "prev_out": False},
                 "norm": {"mean_offset": 0, "std_offset": 2}, "mlp": [{"in": 2, "out": 2, "weight_offset": 4, "bias_offset": 8, "activation": "none"}],
                 "outputs": [{"name": "g", "unit": "", "min": 0, "max": 1, "none": 1, "activation": "sigmoid"},
                             {"name": "y", "unit": "", "min": -10, "max": 10, "none": 7, "activation": "linear"}]}
            if gate:
                j["gate"] = 0
            return j
        w = np.array([0, 0, 1e-6, 1e-6, 1e6, -1e6, 1e6, -1e6, 0, 0], "<f4")
        ticks = [{"body": {"t": 1e300, "speed": 1e300}, "cam": {}}, {"body": {"t": 1e300, "speed": 0.0}, "cam": {}}, {"body": {"t": 2e-6, "speed": 1e-6}, "cam": {}}]
        want = {False: [[1, 7], [1, 7], [1, 10]], True: [[1, 7], [1, 7], [1, 10]]}
        w2 = w.copy()
        w2[6:8] = 0  # y = 0: finite while the gate is NaN
        want2 = {False: [[1, 0], [1, 0], [1, 0]], True: [[1, 7], [1, 0], [1, 0]]}
        for weights, wants in ((w, want), (w2, want2)):
            for gate in (False, True):
                with self.subTest(gate=gate, y_zero=weights is w2):
                    j = spec(gate)
                    ref = R.Sense(j, weights)
                    got = [list(map(float, ref.step(None, tk["body"], tk["cam"]))) for tk in ticks]
                    self.assertEqual(got, wants[gate])
                    if HAS_TS:
                        p = self.tmp / f"nan_{gate}_{weights is w2}.sense.json"
                        p.write_text(json.dumps(j))
                        weights.tofile(p.with_suffix(".bin"))
                        inp = self.tmp / f"nan_{gate}_{weights is w2}.inputs.json"
                        inp.write_text(json.dumps({"ticks": ticks}))
                        self.assertEqual(ts(p, inp)["outputs"], wants[gate])
        j = spec(False)
        bad = w.copy()
        bad[2] = 5e-7
        self.assertIsNotNone(C.value_problem(j, bad))
        self.assertIsNone(C.value_problem(j, w))  # float32(1e-6) passes
        big = np.concatenate([w, np.array([2e6], "<f4")])  # review finding 6: |w| <= 1e6 anywhere in the .bin
        self.assertIn("at most 1000000", C.value_problem(j, big))
        if HAS_TS:
            p = self.tmp / "big.sense.json"
            p.write_text(json.dumps(j))
            big.tofile(p.with_suffix(".bin"))
            self.assertIn("at most 1000000", ts(p)["problem"])
        if HAS_TS:
            p = self.tmp / "std.sense.json"
            p.write_text(json.dumps(j))
            bad.tofile(p.with_suffix(".bin"))
            self.assertIn("every std must be", ts(p)["problem"])

    def test_split_unit_is_challenge_seed(self):
        """Review finding 4: two collect dirs of one Challenge with the same seeds: each (Challenge, seed) in one split."""
        a2 = make(self.tmp / "a2", list(range(0, 8)), ticks=20, challenge="synth")
        m = json.loads((a2 / "meta.json").read_text())
        m["challenge"] = {"id": "synth", "version": 1}  # as `duckbench sense collect` writes it
        (a2 / "meta.json").write_text(json.dumps(m))
        d = Data([str(self.a), str(a2), str(self.b)])
        self.assertEqual(d.dirs[1].challenge, "synth")
        for k in range(5):
            tr, va, te = T.split(d, k)
            for m in (tr, va, te):
                both = {(d.challenge_of(i), int(d.seed[i])) for i in np.flatnonzero(m)}
                others = {(d.challenge_of(i), int(d.seed[i])) for i in np.flatnonzero(~m)}
                self.assertFalse(both & others, k)

    def test_canonical_id_matches_ts(self):
        """Review finding 5: the id is over the behavioural fields as canonical JSON, numbers as JavaScript writes them."""
        nums = [0, -0.0, 1, 1.0, -3, 0.1, 1.5, 1e-6, 1e-7, 1.234e-7, 123456.789, 1e20, 1e21, 1.5e21, 2 ** 53, 2 ** 60, 5e-324, 1.7976931348623157e308, -2.5e-8, 0.000001234, 12345678901234567890.0]
        got = [C._js_number(x) for x in nums]
        if shutil.which("node"):
            r = subprocess.run(["node", "-e", f"console.log(JSON.stringify({json.dumps(nums)}.map((x) => JSON.stringify(x))))"], capture_output=True, text=True, timeout=60)
            self.assertEqual(got, json.loads(r.stdout))
        self.assertEqual(C._js_number(1e21), "1e+21")
        self.assertEqual(C._js_number(1e-7), "1e-7")
        self.assertEqual(C._js_number(1.0), "1")
        jp = Path(self.trained["mlp-cam"]["_files"]["json"])
        j = json.loads(jp.read_text())
        bb = jp.with_suffix(".bin").read_bytes()
        sid = C.sense_id(j, bb)
        self.assertEqual(sid, self.trained["mlp-cam"]["_files"]["id"])
        self.assertEqual(C.sense_id({**j, "eval": {"x": 1}, "definition": {"label": "", "collect": ""}}, bb), sid)
        self.assertNotEqual(C.sense_id({**j, "name": "other"}, bb), sid)
        if HAS_TS:
            p = self.tmp / "restyled.sense.json"
            p.write_text(json.dumps({**j, "eval": None}, indent=3, sort_keys=True))
            p.with_suffix(".bin").write_bytes(bb)
            self.assertEqual(ts(p)["id"], sid)
            self.assertEqual(C.check(p, ts=False)["id"], sid)

    def test_multi_output_worlds_and_refit(self):
        out = self.tmp / "rival"
        ev = T.train([str(self.a), str(self.b)], MULTI, "multi.py", "mlp-cam", str(out), log=lambda *a: None)
        self.assertTrue(ev["parity"]["ok"], ev["parity"])
        self.assertEqual(set(ev["test"]), {"dist", "near"})
        self.assertIn("mae", ev["test"]["dist"])
        self.assertIn("auc", ev["test"]["near"])
        self.assertEqual(set(ev["per_world"]), {"synth", "synth2"})
        j = json.loads(out.with_suffix(".sense.json").read_text())
        self.assertEqual([o["activation"] for o in j["outputs"]], ["linear", "sigmoid"])
        ev2 = T.refit(str(out), [str(self.b), str(make(self.tmp / "c", [40, 41, 42], ticks=60))], str(self.tmp / "rival2"), log=lambda *a: None)
        j2 = json.loads((self.tmp / "rival2.sense.json").read_text())
        self.assertEqual(j2["inputs"], j["inputs"])
        self.assertEqual([o["name"] for o in j2["outputs"]], ["dist", "near"])
        self.assertEqual(ev2["train"]["template"], "mlp-cam")
        self.assertEqual((ev["train"]["ducks"], ev2["train"]["ducks"]), ("own", "own"))
        self.assertEqual(j2["definition"]["label"], MULTI)
        self.assertTrue(ev2["parity"]["ok"])
        # audit F3: names, not local paths, in eval.train
        self.assertEqual(j["eval"]["train"]["data"], [self.a.name, self.b.name])
        self.assertEqual(j["eval"]["train"]["label_file"], "multi.py")
        self.assertNotIn(str(self.tmp), json.dumps(j["eval"]))

    def test_collect_meta_camera_world_and_command(self):
        """Audit F3 / F9: the collect command as the CLI recorded it (else challenge@version, no dict); camera-world and
        sim disagreements between dirs and with the current camera world are warned about and recorded in eval.train."""
        def remeta(path, **kw):
            m = json.loads((path / "meta.json").read_text())
            m.update(kw)
            (path / "meta.json").write_text(json.dumps(m))
            return path
        h = lambda c: "sha256:" + c * 64  # noqa: E731
        x = remeta(make(self.tmp / "x", list(range(0, 8)), ticks=20, challenge="synth"),
                   challenge={"id": "synth", "version": 3}, camera_world=h("a"), sim_version="duck-arena-sim/0.5.0")
        self.assertEqual(Data([str(x)]).dirs[0].collect_command(), "duckbench sense collect synth@3 synthetic --seeds 0-7 --out x")
        y = remeta(make(self.tmp / "y", list(range(8, 16)), ticks=20, challenge="synth"),
                   challenge={"id": "synth", "version": 3}, camera_world=h("b"), sim_version="duck-arena-sim/0.4.0",
                   command="duckbench sense collect synth s.json --seeds 8-15 --out y")
        self.assertEqual(Data([str(y)]).dirs[0].collect_command(), "duckbench sense collect synth s.json --seeds 8-15 --out y")
        d = Data([str(x), str(y)])
        w = T.data_warnings(d, current=lambda ch, pic: h("a"))
        self.assertEqual(len(w), 3, w)
        self.assertIn("disagree on the camera world", w[0])
        self.assertIn("disagree on the sim", w[1])
        self.assertTrue(w[2].startswith("y: collected on camera world"), w[2])
        self.assertEqual(T.data_warnings(Data([str(x)]), current=lambda ch, pic: h("a")), [])
        self.assertEqual(T.data_warnings(Data([str(self.a)]), current=lambda ch, pic: 1 / 0), [])  # no camera_world: no lookup
        if HAS_TS:
            self.assertEqual(bridge.camera_world("synth", C.PICTURE), None)  # an unknown Challenge
            self.assertRegex(bridge.camera_world("sumo", C.PICTURE) or "", r"^sha256:[0-9a-f]{64}$")
        logs = []
        ev = T.train([str(x), str(y)], MULTI, "/somewhere/multi.py", "mlp-cam", str(self.tmp / "cw"), log=logs.append)
        self.assertEqual(ev["train"]["camera_world"], [h("a"), h("b")])
        self.assertEqual(ev["train"]["sim"], ["duck-arena-sim/0.5.0", "duck-arena-sim/0.4.0"])
        self.assertEqual(ev["train"]["label_file"], "multi.py")
        self.assertTrue(any("disagree on the camera world" in m for m in logs if isinstance(m, str)), logs[:5])
        self.assertEqual(json.loads((self.tmp / "cw.sense.json").read_text())["definition"]["collect"],
                         "duckbench sense collect synth@3 synthetic --seeds 0-7 --out x; duckbench sense collect synth s.json --seeds 8-15 --out y")


if __name__ == "__main__":
    unittest.main()
