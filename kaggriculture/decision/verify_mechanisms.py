"""Executable, pinned official-interpreter mechanism contract checks.

Boundary fixtures intentionally modify input state; only the official interpreter
produces next state. They are not claims that fixtures arose in ordinary play.
"""
import argparse
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import unittest
from contextlib import redirect_stdout, redirect_stderr

_saved_stderr = os.dup(2)
with open(os.devnull, "w") as _null:
    try:
        os.dup2(_null.fileno(), 2)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            from kaggle_environments import make, __version__
            from kaggle_environments.envs.kaggriculture import kaggriculture as rules
    finally:
        os.dup2(_saved_stderr, 2)
        os.close(_saved_stderr)

from decision.api import Atom, Joint, UNIT_OPS, MARKET_OPS, CROPS, ANIMALS, observe, assess_unit, assess_joint

ROOT = Path(__file__).resolve().parents[1]
ENGINE_SHA = "bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e"
PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def env():
    return make("kaggriculture", configuration={"seed": 6137181}, debug=True)


def obs(e, player=0):
    return e._Environment__get_shared_state(player).observation


def step(e, action=PASS, rival=PASS, at=None):
    # Explicit interpreter boundary, avoiding a second implementation of rules.
    number = e.state[0].observation.step if at is None else at
    for p, a in enumerate((action, rival)):
        e.state[p].action = deepcopy(a)
        e.state[p].observation.day = number // e.configuration.turnsPerDay
        e.state[p].observation.hour = number % e.configuration.turnsPerDay
    e.state[0].observation.step = number
    rules.interpreter(e.state, e)
    e.state[0].observation.step = number + 1  # framework counter bookkeeping


def action(unit="PASS", market=(), hands=()):
    return Joint(Atom(unit) if isinstance(unit, str) else Atom.parse(unit),
                 tuple(Atom.parse(h) for h in hands), tuple(Atom.parse(o, market=True) for o in market)).wire()


class Mechanisms(unittest.TestCase):
    def setUp(self):
        self.e = env()
        self.f = self.e.state[0].observation.farms[0]
        self.p = self.e.state[0].observation.private

    def view(self):
        return observe(obs(self.e), self.e.configuration)

    def test_01_codec_complete_domains(self):
        for op in UNIT_OPS:
            a = [op, "WHEAT"] if op in ("PICKUP", "PLACE", "PLANT") else [op]
            self.assertEqual(Atom.parse(a).wire(), a)
        for op in MARKET_OPS:
            a = [op] if op in ("HIRE", "BUY_LAND") else [op, "GOOSE" if op == "BUY_ANIMAL" else "WHEAT", 99999]
            self.assertEqual(Atom.parse(a, market=True).wire(), a)
        raw = action(["PLACE", "FERTILIZER", 31], [["SELL", "MILK", 12345], ["HIRE"], ["BUY_LAND"]])
        self.assertEqual(Joint.parse(raw).wire(), raw)
        self.assertEqual(len(UNIT_OPS), 18)
        self.assertEqual(len(MARKET_OPS), 6)

    def test_02_purchase_cannot_fund_same_turn_plant(self):
        step(self.e, action(["PLANT", "WHEAT"], [["BUY_SEED", "WHEAT", 1]]))
        self.assertIsNone(self.f["tiles"][4][4])
        self.assertEqual(self.p["seeds"]["WHEAT"], 1)
        step(self.e, action(["PLANT", "WHEAT"]))
        self.assertEqual(self.f["tiles"][4][4]["crop"], "WHEAT")
        self.assertEqual(self.p["seeds"]["WHEAT"], 0)
        step(self.e, action("WATER"))
        self.assertTrue(self.f["tiles"][4][4]["watered_today"])
        step(self.e, action("HARVEST"), at=48)
        self.assertEqual(self.p["inventories"][0]["WHEAT"], 1)
        self.assertIsNone(self.f["tiles"][4][4])

    def test_03_all_crops_encode_and_plant(self):
        for crop in CROPS:
            with self.subTest(crop=crop):
                self.f["tiles"][4][4] = None
                self.p["seeds"][crop] = 1
                a = Atom("PLANT", crop)
                self.assertEqual(assess_unit(self.view(), 0, a)[0], "available")
                step(self.e, action(a.wire()))
                self.assertEqual(self.view().encode()["observation"]["farms"][0]["tiles"][4][4]["crop"], crop)

    def test_04_movement_locked_and_boundary(self):
        step(self.e, action("EAST"))
        self.assertEqual(self.f["farmer"], [5, 4])
        self.assertEqual(self.f["tiles"][4][5], "LOCKED")
        step(self.e, action("BUILD_COOP"))
        self.assertEqual(self.f["tiles"][4][5], "LOCKED")
        self.p["shed"]["GOOSE"] = 1
        step(self.e, action(["PICKUP", "GOOSE", 1]))
        self.assertEqual(self.p["inventories"][0]["GOOSE"], 1)
        step(self.e, action("DROP"))
        self.assertEqual(self.p["shed"]["GOOSE"], 1)
        self.f["farmer"] = [0, 0]
        self.assertEqual(assess_unit(self.view(), 0, Atom("WEST"))[0], "unavailable")
        step(self.e, action("WEST"))
        self.assertEqual(self.f["farmer"], [0, 0])

    def test_05_joint_seed_shortage_includes_ghost_hands(self):
        self.p["seeds"]["WHEAT"] = 1
        a = action(["PLANT", "WHEAT"], hands=[["PLANT", "WHEAT"]])
        report = assess_joint(self.view(), Joint.parse(a))
        self.assertEqual(report["atomic_seed_blocked_crops"], ["WHEAT"])
        self.assertEqual(report["ignored_extra_hands"], 1)
        step(self.e, a)
        self.assertIsNone(self.f["tiles"][4][4])
        self.assertEqual(self.p["seeds"]["WHEAT"], 1)

    def test_06_joint_dig_then_plant_not_statically_masked(self):
        self.f["hands"] = [[4, 4]]
        self.p["inventories"].append({})
        self.f["tiles"][4][4] = {"kind": "WEED"}
        self.p["seeds"]["WHEAT"] = 1
        a = action("DIG", hands=[["PLANT", "WHEAT"]])
        self.assertEqual(assess_unit(self.view(), 1, Atom("PLANT", "WHEAT"))[0], "unavailable")
        self.assertIn("conditional", assess_joint(self.view(), Joint.parse(a))["unit_joint_status"])
        step(self.e, a)
        self.assertEqual(self.f["tiles"][4][4]["crop"], "WHEAT")

    def test_07_fertilize_then_water_order(self):
        self.f["hands"] = [[4, 4]]
        self.p["inventories"] = [{"FERTILIZER": 1}, {}]
        self.f["tiles"][4][4] = rules._new_plant("MELON", -6, 24)
        step(self.e, action("FERTILIZE", hands=[["WATER"]]))
        tile = self.f["tiles"][4][4]
        self.assertEqual(tile["yield_units"], 3)
        self.assertEqual(tile["fertilized_until_day"], 2)
        self.assertNotIn("FERTILIZER", self.p["inventories"][0])
        step(self.e, action("WATER"))
        self.assertEqual(tile["yield_units"], 3)

    def test_08_ongoing_production_and_lifespan(self):
        tile = rules._new_plant("TOMATO", -7, 24)
        tile.update(watered_today=True, fertilized_until_day=0)
        self.f["tiles"][4][4] = tile
        step(self.e, at=23)
        self.assertEqual(tile["yield_units"], 2)
        for day in (1, 2, 3):
            tile["watered_today"] = True
            step(self.e, at=day * 24 + 23)
        self.assertEqual(tile["yield_units"], 4)
        self.assertEqual(tile["max_lifespan_step"], 120)
        step(self.e, at=120)
        self.assertEqual(tile["yield_units"], 3)

    def test_09_unwatered_plant_turns_to_weed(self):
        self.f["tiles"][4][4] = rules._new_plant("WHEAT", 0, 24)
        step(self.e, at=23)
        self.assertEqual(self.f["tiles"][4][4], {"kind": "WEED"})

    def test_10_animal_complete_chain(self):
        step(self.e, action("BUILD_COOP", [["BUY_ANIMAL", "GOOSE", 1]]))
        self.assertEqual(self.f["money"], 2700)
        self.assertEqual(self.f["tiles"][4][4], {"kind": "COOP"})
        step(self.e, action(["PICKUP", "GOOSE"]))
        step(self.e, action(["PLACE", "GOOSE"]))
        tile = self.f["tiles"][4][4]
        self.assertEqual(tile["animal"], "GOOSE")
        step(self.e, action("DIG"))
        self.assertIn("animal", self.f["tiles"][4][4])
        step(self.e, action("FEED"))
        self.assertFalse(tile["fed_today"])
        bank = self.f["money"]
        step(self.e, action(market=[["BUY_PRODUCT", "WHEAT", 2]]))
        self.assertLess(self.f["money"], bank)
        step(self.e, action(["PICKUP", "WHEAT", 2]))
        step(self.e, action("FEED"))
        step(self.e, action("CARE"))
        tile["placed_day"] = -3
        step(self.e, at=23)
        self.assertEqual(tile["yield_units"], 1)
        self.assertEqual(tile["pending_care_bonus"], 1)
        self.assertTrue(tile["fertilizer_available"])
        step(self.e, action("COLLECT_FERTILIZER"))
        self.assertEqual(self.p["inventories"][0]["FERTILIZER"], 1)
        step(self.e, action(["PICKUP", "WHEAT", 1], [["BUY_PRODUCT", "WHEAT", 1]]))
        step(self.e, action(["PICKUP", "WHEAT", 1]))
        step(self.e, action("FEED"))
        step(self.e, action("CARE"))
        step(self.e, at=47)
        self.assertEqual(tile["yield_units"], 3)
        step(self.e, action("HARVEST"))
        self.assertEqual(self.p["inventories"][0]["EGG"], 3)
        step(self.e, action("DROP"))
        before = self.f["money"]
        step(self.e, action(market=[["SELL", "EGG", 3], ["SELL", "FERTILIZER", 1]]))
        self.assertGreater(self.f["money"], before)
        self.assertEqual(self.p["shed"]["EGG"], 0)

    def test_11_all_animal_species_and_escape(self):
        for species in ANIMALS:
            with self.subTest(species=species):
                tile = rules._new_animal(species, 1 - rules.ANIMALS[species]["first_yield_day"])
                tile["fed_today"] = True
                self.f["tiles"][4][4] = tile
                step(self.e, at=23)
                self.assertEqual(tile["yield_units"], 1)
                step(self.e, at=47)
                step(self.e, at=71)
                self.assertEqual(self.f["tiles"][4][4], {"kind": rules.ANIMALS[species]["structure"]})

    def test_12_pasture_placement_and_selective_shed_place(self):
        step(self.e, action("BUILD_PASTURE"))
        self.p["inventories"][0] = {"COW": 1}
        step(self.e, action(["PLACE", "COW"]))
        self.assertEqual(self.f["tiles"][4][4]["animal"], "COW")
        self.p["inventories"][0] = {"MILK": 3}
        step(self.e, action(["PLACE", "MILK", 2]))
        self.assertEqual(self.p["shed"]["MILK"], 2)
        self.assertEqual(self.p["inventories"][0]["MILK"], 1)

    def test_13_land_order_prices_and_limit(self):
        self.f["money"] = 10000
        for quadrant, remaining, cells in [("NE", 9000, 50), ("SW", 7000, 75), ("SE", 3000, 100)]:
            step(self.e, action(market=[["BUY_LAND"]]))
            self.assertEqual(self.f["unlocked_quadrants"][-1], quadrant)
            self.assertEqual(self.f["money"], remaining)
            self.assertEqual(sum(t != "LOCKED" for row in self.f["tiles"] for t in row), cells)
        step(self.e, action(market=[["BUY_LAND"]]))
        self.assertEqual(self.f["money"], 3000)

    def test_14_ordered_cash_budget_and_hiring(self):
        self.f["money"] = 1000
        step(self.e, action(market=[["HIRE"], ["BUY_LAND"]]))
        self.assertEqual(len(self.f["hands"]), 1)
        self.assertEqual(self.f["unlocked_quadrants"], ["NW"])
        self.assertEqual(self.f["money"], 999)
        step(self.e, action(market=[["HIRE"], ["HIRE"], ["HIRE"]]))
        self.assertEqual(self.f["money"], 993)
        self.assertEqual(self.f["hires_today"], 4)
        self.assertEqual(self.f["hands"][0], [5, 4])

    def test_15_hired_unit_acts_next_turn_only(self):
        step(self.e, action(market=[["HIRE"]], hands=[["EAST"]]))
        self.assertEqual(self.f["hands"][0], [5, 4])
        step(self.e, action(hands=[["EAST"]]))
        self.assertEqual(self.f["hands"][0], [6, 4])

    def test_16_lockstep_market_quotes(self):
        market = self.e.state[0].observation.market
        market["inventory"]["MELON"] = 10100
        for p in range(2):
            self.e.state[p].observation.private["shed"]["MELON"] = 2
        a = action(market=[["SELL", "MELON", 2]])
        step(self.e, a, a, at=1)
        self.assertEqual([f["money"] for f in self.e.state[0].observation.farms], [3296, 3296])
        self.assertEqual(market["inventory"]["MELON"], 10104)

    def test_17_price_floor_sales_do_not_add_stock(self):
        market = self.e.state[0].observation.market
        market["inventory"]["MELON"] = 10200
        self.p["shed"]["MELON"] = 3
        step(self.e, action(market=[["SELL", "MELON", 3]]), at=1)
        self.assertEqual(self.f["money"], 3003)
        self.assertEqual(market["inventory"]["MELON"], 10200)

    def test_18_buy_sell_round_trip_and_restricted_products(self):
        step(self.e, action(market=[["BUY_PRODUCT", "WHEAT", 3], ["SELL", "WHEAT", 3], ["BUY_PRODUCT", "MILK", 2]]), at=1)
        self.assertEqual(self.f["money"], 3000)
        self.assertEqual(self.p["shed"]["MILK"], 0)

    def test_19_storage_caps_separate_seeds_and_order_limit(self):
        self.p["shed"]["WHEAT"] = 100
        step(self.e, action(market=[["BUY_ANIMAL", "GOOSE", 1], ["BUY_PRODUCT", "FERTILIZER", 1], ["BUY_SEED", "WHEAT", 1]]))
        self.assertEqual(self.f["money"], 2990)
        self.p["inventories"][0] = {"MILK": 4}
        step(self.e, action("DROP"))
        self.assertEqual(self.p["inventories"][0], {})
        self.assertEqual(self.p["shed"].get("MILK", 0), 0)
        step(self.e, action(market=[["BUY_SEED", "WHEAT", 1]] * 11))
        self.assertEqual(self.p["seeds"]["WHEAT"], 11)

    def test_20_town_duplicate_shop_demand(self):
        o = self.e.state[0].observation
        o.town["unlocked_shops"] = ["YARN_STORE", "YARN_STORE"]
        step(self.e, at=4)
        self.assertEqual(o.market["inventory"]["WOOL"], 9996)
        self.assertEqual(o.market["inventory"]["FERTILIZER"], 10000)

    def test_21_night_drop_reset_overflow_and_shops(self):
        self.p["shed"]["WHEAT"] = 99
        self.p["inventories"][0] = {"MILK": 2}
        self.f["hands"] = [[5, 4]]
        self.f["hires_today"] = 1
        self.p["inventories"].append({"WOOL": 2})
        step(self.e, at=71)
        self.assertEqual(self.p["shed"]["MILK"], 1)
        self.assertEqual(self.p["shed"]["WOOL"], 0)
        self.assertEqual(self.f["hands"], [])
        self.assertEqual(self.p["inventories"], [{}])
        self.assertEqual(self.f["hires_today"], 0)
        self.assertEqual(len(self.e.state[0].observation.town["unlocked_shops"]), 1)

    def test_22_weeds_and_deterministic_rng(self):
        a, b = env(), env()
        for e in (a, b):
            e.configuration.weedSpawnChance = 1
            step(e, at=71)
        self.assertEqual(obs(a)["farms"], obs(b)["farms"])
        self.assertEqual(obs(a)["town"], obs(b)["town"])
        self.assertTrue(all(t == {"kind": "WEED"} for row in obs(a)["farms"][0]["tiles"][:5] for t in row[:5]))

    def test_23_terminal_cash_only_no_final_drop(self):
        self.p["inventories"][0] = {"MILK": 4}
        step(self.e, at=718)
        self.assertTrue(self.e.done)
        self.assertEqual(self.e.state[0].reward, 3000)
        self.assertEqual(self.p["inventories"][0]["MILK"], 4)
        self.assertEqual(self.p["shed"]["MILK"], 0)

    def test_24_runtime_visibility_and_sanitized_history(self):
        self.p["shed"]["GOOSE"] = 7
        self.e.state[1].observation.private["shed"]["COW"] = 9
        seen = []
        def probe(o, cfg):
            v = observe(o, cfg)
            self.assertNotIn("seed", v.configuration)
            self.assertNotIn("private", v.observation["farms"][1-o.player])
            self.assertEqual(v.private["shed"]["GOOSE" if o.player == 0 else "COW"], (7 if o.player == 0 else 6) if o.step else 0)
            self.assertEqual(v.private["shed"]["COW" if o.player == 0 else "GOOSE"], 0)
            seen.append(o.player)
            return action(market=[["BUY_ANIMAL", "GOOSE" if o.player == 0 else "COW", 7 if o.player == 0 else 6]]) if o.step == 0 else PASS
        self.e.run([probe, probe])
        self.assertEqual(len(seen), 1438)
        a = env()
        before = observe(obs(a), a.configuration)
        step(a)
        after = observe(obs(a), a.configuration, before)
        self.assertIn("net_public_inventory_delta", after.encode()["history"])
        fake = obs(a)
        fake["farms"][1]["private"] = {"OPPONENT_ONLY_SECRET": "hidden"}
        self.assertNotIn("OPPONENT_ONLY_SECRET", json.dumps(observe(fake, a.configuration).encode()))

    def test_25_unknown_commands_vs_schema_errors(self):
        self.e.step([{"farmer": ["TELEPORT"]}, PASS])
        self.assertEqual(self.e.state[0].status, "ACTIVE")
        self.assertEqual(self.f["farmer"], [4, 4])
        with self.assertRaises(ValueError):
            Joint.parse({"farmer": ["TELEPORT"]})
        other = env()
        other.step(["not-an-object", PASS])
        # utils.default_schema substitutes the default object before validation.
        self.assertEqual(other.state[0].status, "ACTIVE")
        self.assertEqual(other.state[0].action, PASS)
        broken = env()
        broken.step([RuntimeError("contract probe"), PASS])
        self.assertEqual(broken.state[0].status, "ERROR")
        self.assertIsNone(broken.state[0].reward)

    def test_26_isolated_assessments_match_official_effects(self):
        for op in UNIT_OPS:
            with self.subTest(op=op):
                e = env()
                f, p = e.state[0].observation.farms[0], e.state[0].observation.private
                item = "WHEAT" if op in ("PLANT", "PICKUP") else "GOOSE" if op == "PLACE" else None
                a = Atom(op, item)
                p["seeds"]["WHEAT"] = 1
                p["shed"]["WHEAT"] = 1
                p["inventories"][0] = {"WHEAT": 1, "FERTILIZER": 1, "GOOSE": 1}
                if op in ("WATER", "HARVEST", "FERTILIZE"):
                    f["tiles"][4][4] = rules._new_plant("WHEAT", -2, 24)
                elif op == "DIG":
                    f["tiles"][4][4] = {"kind": "WEED"}
                elif op == "PLACE":
                    f["tiles"][4][4] = {"kind": "COOP"}
                elif op in ("FEED", "CARE", "COLLECT_FERTILIZER"):
                    f["tiles"][4][4] = rules._new_animal("GOOSE", 0)
                    f["tiles"][4][4]["fertilizer_available"] = True
                self.assertEqual(assess_unit(observe(obs(e), e.configuration), 0, a)[0], "available")
                before = deepcopy((f, p))
                rules._apply_unit_action(f, p, 0, a.wire(), 10, 0, 24, 100)
                if op != "PASS":
                    self.assertNotEqual(before, (f, p))
                if op == "PASS":
                    continue
                f["farmer"] = {"SOUTH": [0, 9], "EAST": [9, 0]}.get(op, [0, 0])
                x, y = f["farmer"]
                f["tiles"][y][x] = "LOCKED"
                self.assertEqual(assess_unit(observe(obs(e), e.configuration), 0, a)[0], "unavailable")
                before = deepcopy((f, p))
                rules._apply_unit_action(f, p, 0, a.wire(), 10, 0, 24, 100)
                self.assertEqual(before, (f, p))

    def test_27_observation_keeps_inventory_insertion_order(self):
        self.p["shed"]["WHEAT"] = 99
        self.p["inventories"][0] = {"MILK": 1, "WHEAT": 1}
        v = self.view()
        self.assertEqual(list(v.private["inventories"][0]), ["MILK", "WHEAT"])
        self.assertEqual(v.encode()["inventory_item_order"], [["MILK", "WHEAT"]])
        step(self.e, at=23)
        self.assertEqual(self.p["shed"]["MILK"], 1)
        self.assertEqual(self.p["shed"]["WHEAT"], 99)

    def test_28_official_quantity_coercion_is_preserved(self):
        for quantity in (2.0, 2.8, "2", True):
            e = env()
            raw = action(market=[["BUY_SEED", "WHEAT", quantity]])
            self.assertEqual(Joint.parse(raw).wire(), raw)
            step(e, raw)
            self.assertEqual(e.state[0].observation.private["seeds"]["WHEAT"], int(quantity))
        for quantity in (None, "bad", 0, -1):
            with self.assertRaises(ValueError):
                Atom.parse(["BUY_SEED", "WHEAT", quantity], market=True)

    def test_29_negative_market_inventory_is_not_a_stock_gate(self):
        self.f["money"] = 1000000
        self.e.state[0].observation.market["inventory"]["WHEAT"] = -1
        step(self.e, action(market=[["BUY_PRODUCT", "WHEAT", 1]]), at=1)
        self.assertEqual(self.p["shed"]["WHEAT"], 1)
        self.assertEqual(self.e.state[0].observation.market["inventory"]["WHEAT"], -2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    official = subprocess.check_output(["git", "-C", str(ROOT / "upstream"), "show", "HEAD:kaggle_environments/envs/kaggriculture/kaggriculture.py"])
    assert hashlib.sha256(official).hexdigest() == ENGINE_SHA and Path(rules.__file__).read_bytes() == official
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Mechanisms)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report = {"passed": result.wasSuccessful(), "tests_run": result.testsRun, "engine_package": __version__,
              "engine_sha256": ENGINE_SHA, "unit_operations": list(UNIT_OPS), "market_operations": list(MARKET_OPS),
              "test_names": unittest.defaultTestLoader.getTestCaseNames(Mechanisms),
              "failures": [{"test": t.id(), "details": detail} for t, detail in result.failures + result.errors],
              "boundary": "Constructed official-interpreter transitions plus a full runtime-visibility game; not a substitute server or value proof"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(stream.getvalue())
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
