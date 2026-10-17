"""Read official replays: observable farming/economics and precondition auditing.

This checks the crop-only bots' actions; it is not a second game interpreter.
"""
from collections import Counter

SEED_COST = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
FIRST_AGE = {"WHEAT": 2, "CARROT": 2, "TOMATO": 8, "STRAWBERRY": 10, "MELON": 10}


def summarize(replay, player):
    operations, planted, harvested, sold, bought = (Counter() for _ in range(5))
    invalid, duplicate, seed_shortage = [], [], []
    hire_cost = seed_cost = hires = sales_income = 0
    deaths, depleted, max_shed = Counter(), Counter(), 0
    daily = []
    planting_cohorts, selling_cohorts = {}, {}
    tile_changes, tile_mismatches = Counter(), []
    reset_overwrites_skipped = 0
    for previous, following in zip(replay["steps"], replay["steps"][1:]):
        shared = previous[0]["observation"]
        day, hour, step = shared["day"], shared["hour"], shared["step"]
        farm = shared["farms"][player]
        after = following[0]["observation"]["farms"][player]
        private = previous[player]["observation"]["private"]
        new_private = following[player]["observation"]["private"]
        action = following[player]["action"]
        positions = [farm["farmer"], *farm["hands"]]
        requests, touched = Counter(), set()
        deposit = Counter()
        for i, order in enumerate([action.get("farmer", ["PASS"]), *action.get("hands", [])]):
            op = order[0]
            operations[op] += 1
            if i >= len(positions):
                invalid.append([step, i, "missing unit"])
                continue
            x, y = positions[i]
            tile = farm["tiles"][y][x]
            is_plant = isinstance(tile, dict) and tile.get("kind") == "PLANT"
            valid = True
            if op in ("PLANT", "WATER", "HARVEST", "DIG"):
                if (x, y) in touched:
                    duplicate.append([step, i, x, y])
                touched.add((x, y))
            if op == "PLANT":
                c = order[1]
                requests[c] += 1
                valid = tile is None and private["seeds"].get(c, 0) > 0
                if valid:
                    planted[c] += 1
                    cohort = planting_cohorts.setdefault(str(day), {})
                    cohort[c] = cohort.get(c, 0) + 1
            elif op == "WATER":
                valid = is_plant and not tile["watered_today"]
            elif op == "HARVEST":
                valid = is_plant and tile["yield_units"] > 0 and day - tile["planted_day"] >= FIRST_AGE[tile["crop"]]
                if valid:
                    harvested[tile["crop"]] += tile["yield_units"]
            elif op == "DIG":
                valid = tile is not None and tile != "LOCKED" and "animal" not in tile
            elif op == "DROP":
                h = len(farm["tiles"]) // 2
                valid = x in (h - 1, h) and y in (h - 1, h)
                if valid:
                    deposit.update(private["inventories"][i])
            elif op in ("NORTH", "SOUTH", "EAST", "WEST"):
                dx, dy = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}[op]
                valid = 0 <= x + dx < len(farm["tiles"]) and 0 <= y + dy < len(farm["tiles"])
            if not valid:
                invalid.append([step, i, order])
            elif op in ("PLANT", "WATER", "HARVEST", "DIG"):
                if hour == replay["configuration"]["turnsPerDay"] - 1:
                    reset_overwrites_skipped += 1
                else:
                    now = after["tiles"][y][x]
                    if op == "PLANT":
                        changed = isinstance(now, dict) and now.get("crop") == order[1]
                    elif op == "WATER":
                        changed = isinstance(now, dict) and now.get("watered_today") is True
                    elif op == "DIG":
                        changed = now is None
                    else:
                        changed = now is None or isinstance(now, dict) and (now.get("yield_units") == 0 or now.get("kind") == "WEED")
                    tile_changes[op] += 1
                    if not changed:
                        tile_mismatches.append([step, i, order])
        for c, n in requests.items():
            if n > private["seeds"].get(c, 0):
                seed_shortage.append([step, c, n])
        this_seed_cost = this_hire_cost = 0
        hire_index = farm["hires_today"]
        for order in action.get("market", []):
            if order[0] == "BUY_SEED":
                c, n = order[1:]
                bought[c] += n
                this_seed_cost += SEED_COST[c] * n
            elif order[0] == "HIRE":
                a, b = 1, 1
                for _ in range(hire_index):
                    a, b = b, a + b
                this_hire_cost += a * replay["configuration"]["farmHandCostMult"]
                hire_index += 1
                hires += 1
            elif order[0] == "SELL":
                c, n = order[1:]
                amount = min(n, private["shed"].get(c, 0) + deposit[c])
                sold[c] += amount
                cohort = selling_cohorts.setdefault(str(day), {})
                cohort[c] = cohort.get(c, 0) + amount
        # Cash delta is observed from the server, not calculated from a single quote.
        sales_income += after["money"] - farm["money"] + this_seed_cost + this_hire_cost
        seed_cost += this_seed_cost
        hire_cost += this_hire_cost
        for y, row in enumerate(farm["tiles"]):
            for x, tile in enumerate(row):
                now = after["tiles"][y][x]
                if isinstance(tile, dict) and tile.get("kind") == "PLANT" and isinstance(now, dict) and now.get("kind") == "WEED":
                    if tile["crop"] in ("TOMATO", "STRAWBERRY") and tile["max_lifespan_step"] >= 0:
                        depleted[tile["crop"]] += 1
                    else:
                        deaths[tile["crop"]] += 1
        max_shed = max(max_shed, sum(new_private["shed"].values()))
        if hour == 0:
            daily.append({"day": day, "money": farm["money"], "standing_crops": sum(
                isinstance(t, dict) and t.get("kind") == "PLANT" for row in farm["tiles"] for t in row)})
    last = replay["steps"][-1]
    final_private = last[player]["observation"]["private"]
    final_money = last[0]["observation"]["farms"][player]["money"]
    unsold = sum(final_private["shed"].values()) + sum(sum(inv.values()) for inv in final_private["inventories"])
    return {"unit_action_counts_including_hands": dict(operations), "planted": dict(planted),
            "harvested_units": dict(harvested), "sold_units": dict(sold), "purchased_seeds": dict(bought),
            "hires": hires, "hire_cost": hire_cost, "seed_cost": seed_cost,
            "sales_income_observed_cash_delta_plus_costs": sales_income,
            "net_profit": final_money - replay["configuration"]["startingMoney"],
            "crop_to_weed_transitions": dict(deaths), "exhausted_ongoing_to_weed": dict(depleted),
            "max_shed_items": max_shed, "final_unsold_items": unsold,
            "final_seeds": {c: n for c, n in final_private["seeds"].items() if n},
            "unused_seed_cost": sum(SEED_COST[c] * n for c, n in final_private["seeds"].items()),
            "seed_balance_matches": all(bought[c] == planted[c] + final_private["seeds"].get(c, 0) for c in SEED_COST),
            "planting_cohorts_by_day": planting_cohorts, "selling_cohorts_by_day": selling_cohorts,
            "max_same_crop_planted_per_day": max((n for cohort in planting_cohorts.values() for n in cohort.values()), default=0),
            "post_action_tile_checks": dict(tile_changes), "post_action_tile_mismatches": tile_mismatches,
            "daily_reset_overwrites_skipped": reset_overwrites_skipped,
            "invalid_farming_preconditions": invalid, "duplicate_tile_work": duplicate,
            "atomic_seed_shortages": seed_shortage,
            "farming_preconditions_passed": not (invalid or duplicate or seed_shortage or tile_mismatches),
            "daily": daily}
