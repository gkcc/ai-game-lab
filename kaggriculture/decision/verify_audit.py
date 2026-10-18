"""Targeted regressions for ordered auditing, official effects and episode isolation."""
import argparse
from copy import deepcopy
import io
import json
from pathlib import Path
import runpy
import unittest

from decision.api import observe, HistorySession, FrozenAdapter, act
from decision.semantics import canonical, digest
from decision.full_audit import reexecute
from decision.execution import ExecutionTrace
from decision.verify_mechanisms import env, obs, action, rules


class AuditRegressions(unittest.TestCase):
    def inventories(self, order, stored=99):
        e = env()
        p = e.state[0].observation.private
        p["shed"]["WHEAT"] = stored
        p["inventories"][0] = dict((item, 1) for item in order)
        return e

    def test_ordered_semantics_full_and_nonfull_drop_and_night(self):
        for stored in (0, 99):
            for night in (False, True):
                with self.subTest(stored=stored, night=night):
                    a, b = (self.inventories(order, stored) for order in (("MILK", "WOOL"), ("WOOL", "MILK")))
                    self.assertEqual(digest(a.state, 1), digest(b.state, 1))
                    self.assertNotEqual(digest(a.state), digest(b.state))
                    self.assertNotEqual(canonical(a.state), canonical(b.state))
                    for e in (a, b):
                        if night:
                            rules._drop_inventories_to_shed(e.state[0].observation.private, 100)
                        else:
                            rules._apply_unit_action(e.state[0].observation.farms[0], e.state[0].observation.private,
                                                     0, ["DROP"], 10, 0, 24, 100)
                    pa, pb = (e.state[0].observation.private for e in (a, b))
                    self.assertEqual(pa["shed"] == pb["shed"], stored == 0)

    def test_ordinary_json_fields_and_timing_remain_immaterial(self):
        e = env()
        a = deepcopy(e.state)
        b = deepcopy(a)
        b[0]["observation"] = dict(reversed(list(b[0]["observation"].items())))
        b[0]["observation"]["remainingOverageTime"] = -123
        self.assertEqual(digest(a), digest(b))
        self.assertEqual(canonical(a), canonical(b))
        with self.assertRaises(ValueError):
            digest(a, 99)

    def test_inventory_mapping_type_is_not_erased(self):
        a = self.inventories(("MILK", "WOOL")).state
        b = deepcopy(a)
        b[0]["observation"]["private"]["inventories"][0] = [["MILK", 1], ["WOOL", 1]]
        self.assertNotEqual(canonical(a), canonical(b))
        self.assertNotEqual(digest(a), digest(b))

    def test_reexecute_rejects_only_inventory_order_tampering(self):
        e = env()
        for a in (action(market=[["BUY_PRODUCT", "WHEAT", 1], ["BUY_PRODUCT", "FERTILIZER", 1]]),
                  action(["PICKUP", "WHEAT"]), action(["PICKUP", "FERTILIZER"])):
            e.step([a, action()])
        d = deepcopy(e.toJSON())
        before = digest(d["steps"], 1)
        inv = d["steps"][-1][0]["observation"]["private"]["inventories"][0]
        self.assertEqual(len(inv), 2)
        d["steps"][-1][0]["observation"]["private"]["inventories"][0] = dict(reversed(list(inv.items())))
        self.assertEqual(before, digest(d["steps"], 1))
        with self.assertRaisesRegex(AssertionError, "ordered state transition mismatch"):
            reexecute(d)

    def trace(self, commands, seeds=2, tile=None):
        e = env()
        f, p = e.state[0].observation.farms[0], e.state[0].observation.private
        f["hands"] = [[4, 4] for _ in commands[1:]]
        p["inventories"] = [{} for _ in commands]
        p["seeds"]["WHEAT"] = seeds
        f["tiles"][4][4] = tile
        trace = ExecutionTrace(rules)
        with trace.installed(e):
            e.step([action(commands[0], hands=commands[1:]), action()])
        return e, trace

    def test_same_tile_duplicate_plant_is_one_effect(self):
        e, t = self.trace([["PLANT", "WHEAT"], ["PLANT", "WHEAT"]])
        rows = [r for r in t.records if r["player"] == 0]
        self.assertEqual([r["status"] for r in rows], ["executed", "no_effect"])
        self.assertEqual([r["before"]["seeds"]["WHEAT"] for r in rows], [2, 1])
        self.assertEqual(sum(t.plants.values()), 1)
        self.assertEqual(e.state[0].observation.private["seeds"]["WHEAT"], 1)

    def test_dig_plant_and_plant_dig_plant_preserve_sequence(self):
        for commands, tile, count in (([["DIG"], ["PLANT", "WHEAT"]], {"kind": "WEED"}, 1),
                                      ([["PLANT", "WHEAT"], ["DIG"], ["PLANT", "WHEAT"]], None, 2)):
            e, t = self.trace(commands, tile=tile)
            self.assertEqual(sum(t.plants.values()), count)
            self.assertEqual(e.state[0].observation.farms[0]["tiles"][4][4]["crop"], "WHEAT")

    def test_atomic_seed_rewrite_and_extra_hand(self):
        _, t = self.trace([["PLANT", "WHEAT"], ["PLANT", "WHEAT"]], seeds=1)
        self.assertEqual(sum(t.plants.values()), 0)
        self.assertEqual(t.counts["unit:blocked"], 2)
        e = env()
        e.state[0].observation.private["seeds"]["WHEAT"] = 2
        t = ExecutionTrace(rules)
        with t.installed(e):
            e.step([action(["PLANT", "WHEAT"], hands=[["PLANT", "WHEAT"]]), action()])
        self.assertEqual(sum(t.plants.values()), 1)
        ghost = next(r for r in t.records if r.get("unit") == 1)
        self.assertEqual((ghost["position"], ghost["status"]), (None, "no_effect"))

    def test_partial_pickup_and_drop_loss(self):
        e = env()
        e.state[0].observation.private["shed"]["WHEAT"] = 1
        t = ExecutionTrace(rules)
        with t.installed(e):
            e.step([action(["PICKUP", "WHEAT", 3]), action()])
        row = next(r for r in t.records if r["player"] == 0)
        self.assertEqual((row["status"], row["transferred_units"]), ("partial", 1))
        e = self.inventories(("MILK", "WOOL"))
        t = ExecutionTrace(rules)
        with t.installed(e):
            e.step([action("DROP"), action()])
        row = next(r for r in t.records if r["player"] == 0)
        self.assertEqual((row["status"], row["transferred_units"], row["discarded_units"]), ("partial", 1, 1))

    def test_market_partial_no_effect_atomic_and_order_limit(self):
        e = env()
        e.state[0].observation.farms[0]["money"] = rules.CROPS["WHEAT"]["seed"] * 2
        t = ExecutionTrace(rules)
        with t.installed(e):
            raw = {"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "WHEAT", 3], ["BUY_LAND"], ["BAD"]] + [["HIRE"]] * 9}
            e.step([raw, action()])
        rows = [r for r in t.records if r["phase"] == "market" and r["player"] == 0]
        first = next(r for r in rows if r["order"] == 0)
        self.assertEqual((first["status"], first["executed_units"], first["requested_units"]), ("partial", 2, 3))
        self.assertEqual(next(r for r in rows if r["order"] == 1)["status"], "no_effect")
        self.assertEqual(sum(r["status"] == "ignored" for r in rows), 2)

    def test_official_animal_place_ignores_quantity_payload(self):
        a, b = env(), env()
        for e in (a, b):
            e.state[0].observation.farms[0]["tiles"][4][4] = {"kind": "COOP"}
            e.state[0].observation.private["inventories"][0] = {"GOOSE": 1}
        raw = {"farmer": ["PLACE", "GOOSE", "ignored_quantity"], "hands": [], "market": []}
        a.step([raw, action()])
        t = ExecutionTrace(rules)
        with t.installed(b):
            b.step([raw, action()])
        self.assertEqual(canonical(a.state), canonical(b.state))
        self.assertEqual(next(r for r in t.records if r["player"] == 0)["status"], "executed")

    def test_market_lockstep_quotes_and_repeatable_effect_digest(self):
        traces = []
        for _ in range(2):
            e = env()
            for s in e.state:
                s.observation.private["shed"]["WHEAT"] = 2
            t = ExecutionTrace(rules)
            with t.installed(e):
                e.step([action(market=[["SELL", "WHEAT", 2]]), action(market=[["SELL", "WHEAT", 2]])])
            rows = [r for r in t.records if r["phase"] == "market"]
            self.assertEqual([r["executed_units"] for r in rows], [2, 2])
            self.assertEqual(rows[0]["first_quote"], rows[1]["first_quote"])
            self.assertEqual(rows[0]["last_quote"], rows[1]["last_quote"])
            for r in rows:
                self.assertEqual(r["cash_delta"], r["after"]["money"] - r["before"]["money"])
            traces.append(t.effect_hash.hexdigest())
        self.assertEqual(*traces)

    def test_tracing_is_transparent_and_restores_on_exception(self):
        a, b = env(), env()
        original = rules._apply_unit_action
        actions = [action(market=[["BUY_SEED", "WHEAT", 3], ["HIRE"], ["BUY_LAND"]]), action(market=[["BUY_SEED", "WHEAT", 2]])]
        a.step(actions)
        with ExecutionTrace(rules).installed(b):
            b.step(actions)
        self.assertEqual(canonical(a.state), canonical(b.state))
        self.assertIs(rules._apply_unit_action, original)
        with self.assertRaisesRegex(RuntimeError, "probe"):
            with ExecutionTrace(rules).installed(b):
                raise RuntimeError("probe")
        self.assertIs(rules._apply_unit_action, original)
        with ExecutionTrace(rules).installed(b):
            with self.assertRaises(RuntimeError):
                with ExecutionTrace(rules).installed(a):
                    pass

    def test_history_requires_episode_seat_and_monotonic_frames(self):
        a, b = env(), env()
        token = object()
        old = observe(obs(a), a.configuration, episode=token)
        b.step([action(), action()])
        for episode in (None, object()):
            with self.assertRaises(ValueError):
                observe(obs(b), b.configuration, old, episode=episode)
        with self.assertRaises(ValueError):
            observe(obs(b, 1), b.configuration, old, episode=token)
        current = observe(obs(b), b.configuration, old, episode=token)
        for previous in (current,):
            with self.assertRaises(ValueError):
                observe(obs(b), b.configuration, previous, episode=token)
            with self.assertRaises(ValueError):
                observe(obs(a), a.configuration, previous, episode=token)

    def test_session_reset_excludes_identity_from_features(self):
        a, b = env(), env()
        s = HistorySession(0)
        v = s.observe(obs(a), a.configuration)
        a.step([action(), action()])
        self.assertTrue(s.observe(obs(a), a.configuration).history)
        with self.assertRaises(ValueError):
            s.observe(obs(b), b.configuration)
        with self.assertRaises(ValueError):
            s.observe(obs(b, 1), b.configuration)
        s.reset(0)
        new = s.observe(obs(b), b.configuration)
        self.assertEqual(new.history, {})
        self.assertEqual(v.encode(), new.encode())
        self.assertIsNot(v._episode, new._episode)

    def test_opponent_private_counterfactual_encoding_and_action(self):
        e = env()
        baseline = Path(__file__).resolve().parents[1] / "models/land_probe/baseline.py"
        policy = FrozenAdapter(runpy.run_path(str(baseline))["agent"])
        features, out = observe(obs(e), e.configuration).encode(), act(policy, obs(e), e.configuration)
        other = e.state[1].observation.private
        other["shed"]["COW"], other["seeds"]["MELON"] = 87, 91
        other["inventories"][0] = {"GOOSE": 9}
        self.assertEqual(features, observe(obs(e), e.configuration).encode())
        self.assertEqual(out, act(policy, obs(e), e.configuration))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Do not overwrite retained regressions"
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AuditRegressions))
    report = {"passed": result.wasSuccessful(), "tests_run": result.testsRun,
              "test_names": unittest.defaultTestLoader.getTestCaseNames(AuditRegressions),
              "failures": [{"test": t.id(), "details": detail} for t, detail in result.failures + result.errors]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(stream.getvalue())
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
