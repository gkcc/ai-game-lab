"""Heuristic v1: deterministic coordinated farming, using public observations only.

No engine imports, persistent global state, land buying, animals or training.
The price model is a planning estimate; only the official server executes rules.
"""
import math

CROP_WEIGHTS = {"WHEAT": 1., "CARROT": 1., "TOMATO": 1., "STRAWBERRY": 1., "MELON": 1.}


# seed cost, first harvest age, preferred harvest age, normal total yield
CROPS = {"WHEAT": (10, 2, 4, 4), "CARROT": (20, 2, 3, 3),
         "TOMATO": (50, 8, 11, 4), "STRAWBERRY": (100, 10, 16, 4),
         "MELON": (80, 10, 10, 6)}
PRICE_DEFAULTS = {
    "WHEAT": (25, 400, "sqrt", .80, "log", .20),
    "CARROT": (35, 450, "hinge", 1., "sqrt", .70),
    "TOMATO": (60, 200, "hinge", .40, "sqrt", .60),
    "STRAWBERRY": (120, 100, "sqrt", .70, "linear", 1.60),
    "MELON": (250, 300, "log", .20, "sq", 3.60)}
SHOPS = {"BAKERY": ("EGG", "WHEAT"), "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
         "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"), "YARN_STORE": ("WOOL",),
         "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"), "PET_CAFE": ("CARROT",),
         "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
         "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY")}


def distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def move(a, b):
    if a[0] != b[0]:
        return ["EAST" if a[0] < b[0] else "WEST"]
    if a[1] != b[1]:
        return ["SOUTH" if a[1] < b[1] else "NORTH"]
    return ["PASS"]


def shape(name, value, threshold):
    value = max(0., value)
    if name == "sqrt":
        return math.sqrt(value)
    if name == "log":
        return math.log1p(value)
    if name == "log10":
        return math.log10(1 + value)
    if name == "sq":
        return value * value
    if name == "hinge" and threshold > 0:
        ratio = value / threshold
        return ratio + 8 * max(0., ratio - 1) ** 2
    return value


def quote(crop, inventory, params):
    base, threshold, below, below_target, above, above_target = PRICE_DEFAULTS[crop]
    p = dict(base=base, I0=10000, T=threshold, below_func=below,
             below_target=below_target, above_func=above, above_target=above_target)
    p.update(params.get(crop, {}))
    delta = inventory - p["I0"]
    direction = "above" if delta >= 0 else "below"
    func, target = p[direction + "_func"], p[direction + "_target"]
    amplitude = target * p["base"] / max(1e-12, shape(func, p["T"], p["T"]))
    return max(1, round(p["base"] + (-1 if delta >= 0 else 1) * amplitude * shape(func, abs(delta), p["T"])))


def predicted_yield(crop, age):
    if age < CROPS[crop][1]:
        return 0
    if crop == "TOMATO":
        return min(4, age - 7)
    if crop == "STRAWBERRY":
        return min(4, (age - 10) // 2 + 1)
    if crop == "MELON":
        return min(6, 1 + max(0, age - 5))
    return min(CROPS[crop][3], 1 + max(0, age - 1))


def agent(observation, configuration):
    farm = observation["farms"][observation["player"]]
    private, market = observation["private"], observation["market"]
    day, hour = observation["day"], observation["hour"]
    tpd = configuration["turnsPerDay"]
    remaining = configuration["episodeSteps"] - 1 - observation["step"]
    today = min(tpd - hour, remaining)
    final_day = remaining < tpd - hour
    # Prefer harvest by the penultimate day: final day has no automatic drop.
    last_plant_age = (configuration["episodeSteps"] - 2) // tpd - day - 1
    size = len(farm["tiles"])
    half = size // 2
    accesses = [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]
    units = [farm["farmer"], *farm["hands"]]
    seeds = dict(private["seeds"])
    params = market.get("params", configuration.get("marketParams", {}))
    demand = {c: 1. for c in CROPS}
    for shop in observation["town"].get("unlocked_shops", []):
        products = SHOPS.get(shop, ())
        for c in products:
            if c in demand:
                demand[c] += tpd / configuration["townShopSellInterval"] * (2 if len(products) == 1 else 1)
    supply = {c: private["shed"].get(c, 0) + sum(inv.get(c, 0) for inv in private["inventories"])
              + seeds.get(c, 0) * CROPS[c][3] for c in CROPS}
    plants, empty, weeds = [], [], []
    for y, row in enumerate(farm["tiles"]):
        for x, tile in enumerate(row):
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                plants.append(((x, y), tile))
                c = tile["crop"]
                age = day - tile["planted_day"]
                outstanding = CROPS[c][3]
                if c in ("TOMATO", "STRAWBERRY"):
                    outstanding = max(0, outstanding - predicted_yield(c, age)) + tile["yield_units"]
                supply[c] += outstanding
            elif tile is None:
                empty.append((x, y))
            elif isinstance(tile, dict) and tile.get("kind") == "WEED":
                weeds.append((x, y))

    def crop_score(c):
        cost, first, preferred, _ = CROPS[c]
        age = min(preferred, last_plant_age)
        if age < first:
            return -1.
        amount = predicted_yield(c, age)
        projected = market["inventory"][c] + supply[c] - demand[c] * age
        revenue = sum(quote(c, projected + i, params) for i in range(amount))
        return CROP_WEIGHTS[c] * (revenue - cost - (age + 2) * 1.5) / (age + 1)

    tasks = []
    for pos, tile in plants:
        c, age = tile["crop"], day - tile["planted_day"]
        if c in ("TOMATO", "STRAWBERRY") and age >= CROPS[c][2] and not tile["yield_units"]:
            if last_plant_age >= 2:
                tasks.append((pos, ["DIG"], 150.))
            continue
        ready = tile["yield_units"] > 0 and age >= CROPS[c][1]
        mature = age >= CROPS[c][2] or (c == "MELON" and tile["yield_units"] >= 6)
        if ready and (mature or last_plant_age <= 0):
            needs_bonus = (c in ("WHEAT", "CARROT", "MELON") and not tile["watered_today"]
                           and tile["yield_units"] < predicted_yield(c, age) and not final_day)
            tasks.append((pos, ["WATER"] if needs_bonus else ["HARVEST"], 240.))
        elif not tile["watered_today"] and not final_day:
            tasks.append((pos, ["WATER"], 130. + 50 * tile["consecutive_unwatered"] + 80 / max(1, today)))
    best_crop = max(CROPS, key=lambda c: (crop_score(c), c))
    profitable = crop_score(best_crop) > 0
    if profitable and today >= 4 and not final_day:
        tasks.extend((pos, ["PLANT"], 65.) for pos in empty)
        tasks.extend((pos, ["DIG"], 48.) for pos in weeds)

    actions = [["PASS"] for _ in units]
    available = set(range(len(units)))
    projected_shed = dict(private["shed"])
    carried_total = sum(sum(v.values()) for v in private["inventories"])
    for i, pos in enumerate(units):
        inv = private["inventories"][i]
        if not sum(inv.values()):
            continue
        access = min(accesses, key=lambda a: (distance(pos, a), a))
        dist = distance(pos, access)
        if final_day or sum(projected_shed.values()) + carried_total >= configuration["shedCapacity"] - 8:
            actions[i] = ["DROP"] if dist == 0 else move(pos, access)
            available.remove(i)
            if dist == 0:
                for c, n in inv.items():
                    projected_shed[c] = projected_shed.get(c, 0) + n

    needed_seeds = {}
    # Joint nearest-task assignment; claim an entire tile to avoid duplicate work.
    while available and tasks:
        choices = []
        for i in sorted(available):
            for j, (pos, action, priority) in enumerate(tasks):
                dist = distance(units[i], pos)
                if dist + 1 > today:
                    continue
                if action[0] == "PLANT" and dist + 3 > today:
                    continue
                if final_day and action[0] == "HARVEST":
                    delivery = min(distance(pos, a) for a in accesses)
                    if dist + 1 + delivery + 1 > remaining:
                        continue
                choices.append((priority / (dist + 1.5), -dist, -i, -j, i, j))
        if not choices:
            break
        *_, i, j = max(choices)
        pos, action, _ = tasks.pop(j)
        available.remove(i)
        if action[0] == "PLANT":
            c = max(CROPS, key=lambda c: (crop_score(c), c))
            if crop_score(c) <= 0:
                continue
            if seeds.get(c, 0) <= 0:
                needed_seeds[c] = needed_seeds.get(c, 0) + 1
                supply[c] += CROPS[c][3]
                actions[i] = move(units[i], pos)
                continue
            action = ["PLANT", c]
            if tuple(units[i]) == pos:
                seeds[c] -= 1
            supply[c] += CROPS[c][3]
        actions[i] = action if tuple(units[i]) == pos else move(units[i], pos)

    orders = []
    budget = farm["money"]
    for c, n in sorted(projected_shed.items()):
        if n <= 0:
            continue
        sell = n
        if c in CROPS and not final_day and last_plant_age > 0:
            if market["prices"][c] < PRICE_DEFAULTS[c][0] * .55 and sum(projected_shed.values()) < 70:
                sell = max(0, n - int(demand[c] * 2))
        if sell:
            orders.append(["SELL", c, sell])
    for c, n in sorted(needed_seeds.items()):
        n = min(3, n, max(0, (budget - 60) // CROPS[c][0]))
        if n:
            orders.append(["BUY_SEED", c, n])
            budget -= n * CROPS[c][0]

    if today >= 10 and not final_day:
        work = len(tasks) * 3 + len(plants) + (len(empty) * 2 if profitable else 0)
        target = min(7, max(0, math.ceil(work / max(1, today - 4)) - 1))
        hires = farm["hires_today"]
        a, b = 1, 1
        for _ in range(hires):
            a, b = b, a + b
        hand_count = len(units) - 1
        while hand_count < target and len(orders) < configuration["maxMarketOrdersPerTurn"]:
            cost = a * configuration["farmHandCostMult"]
            if budget < cost + 60 or cost > max(2, today * 1.2):
                break
            orders.append(["HIRE"])
            budget -= cost
            hand_count += 1
            a, b = b, a + b
    return {"farmer": actions[0], "hands": actions[1:],
            "market": orders[:configuration["maxMarketOrdersPerTurn"]]}
