"""Heuristic v2: dated market forecasts, staggered sowing and just-in-time seeds.

No engine imports, persistent global state, land buying, animals or training.
The price model is a planning estimate; only the official server executes rules.
"""
import math


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
    horizon = max(0, last_plant_age)
    pending = {c: [0.] * (horizon + 1) for c in CROPS}
    planted_today = {c: 0 for c in CROPS}
    for c in CROPS:
        pending[c][0] = private["shed"].get(c, 0) + sum(inv.get(c, 0) for inv in private["inventories"])
    plants, empty, weeds = [], [], []
    for y, row in enumerate(farm["tiles"]):
        for x, tile in enumerate(row):
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                plants.append(((x, y), tile))
                c = tile["crop"]
                age = day - tile["planted_day"]
                planted_today[c] += age == 0
            elif tile is None:
                empty.append((x, y))
            elif isinstance(tile, dict) and tile.get("kind") == "WEED":
                weeds.append((x, y))

    # Date standing crops' expected sales, including visible opposing crops.
    # Unrevealed shops are a discounted uniform prior, never future game state.
    for owner, other in enumerate(observation["farms"]):
        for row in other["tiles"]:
            for tile in row:
                if not isinstance(tile, dict) or tile.get("kind") != "PLANT":
                    continue
                c = tile["crop"]
                age = day - tile["planted_day"]
                sale = min(horizon, max(0, CROPS[c][2] - age))
                future_age = age + sale
                if future_age < CROPS[c][1]:
                    continue
                amount = predicted_yield(c, future_age)
                if c in ("TOMATO", "STRAWBERRY"):
                    amount = max(0, amount - predicted_yield(c, age)) + tile["yield_units"]
                pending[c][sale] += amount
                # Short-crop fields will probably be replanted. Ignoring these
                # later cohorts makes every idle worker forecast the same shortage.
                if c in ("WHEAT", "CARROT") and owner == observation["player"]:
                    repeat_sale = sale + CROPS[c][2] + 1
                    while repeat_sale <= horizon:
                        pending[c][repeat_sale] += .75 * CROPS[c][3]
                        repeat_sale += CROPS[c][2] + 1
    for c in CROPS:
        age = min(CROPS[c][2], horizon)
        if age >= CROPS[c][1]:
            pending[c][age] += seeds.get(c, 0) * predicted_yield(c, age)
    unlock_interval = configuration["townShopUnlockInterval"]
    existing_shops = len(observation["town"].get("unlocked_shops", []))
    future_inventory = {}
    for c in CROPS:
        expected_shop_demand = sum(products.count(c) * (2 if len(products) == 1 else 1)
                                   for products in SHOPS.values()) / len(SHOPS)
        expected_shop_demand *= tpd / configuration["townShopSellInterval"]
        inv, path = market["inventory"][c], []
        for offset in range(horizon + 1):
            if offset:
                unlocks = min(8 - existing_shops, (day + offset) // unlock_interval - day // unlock_interval)
                inv -= demand[c] + .5 * max(0, unlocks) * expected_shop_demand
            inv += pending[c][offset]
            path.append(inv)
        future_inventory[c] = path

    def crop_score(c):
        cost, first, preferred, _ = CROPS[c]
        if horizon < first:
            return -1.
        # Compare repeated short crops with long crops over the same remaining season.
        # Candidate sales depress its own later quotes; no instantaneous bulk-sale quote.
        when, already_sold, value, cycle, first_return = 0, 0, 0., 0, 0.
        while horizon - when >= first:
            age = min(preferred, horizon - when)
            when += age
            amount = predicted_yield(c, age)
            # The chosen paid seed is already in pending supply, while this
            # candidate's revenue loop adds its units again. Remove that overlap.
            held_commitment = predicted_yield(c, min(preferred, horizon)) if seeds.get(c, 0) else 0
            projected = future_inventory[c][when] + already_sold - held_commitment
            revenue = sum(quote(c, projected + i, params) for i in range(amount))
            paid_cost = 0 if cycle == 0 and seeds.get(c, 0) else cost
            margin = revenue - paid_cost - (age + 2) * 1.5
            value += margin
            if cycle == 0:
                first_return = margin / (age + 1)
            already_sold += amount
            when += 1  # conservative replant/travel allowance
            cycle += 1
        # Far-future repeat planting is uncertain: immediate crop economics dominate.
        return .85 * first_return + .15 * value / (horizon + 1)

    # Per-day cohorts, rather than a forced full-field monoculture on day zero.
    batch_limits = {"MELON": 16, "STRAWBERRY": 8, "TOMATO": 10, "CARROT": 10, "WHEAT": 10}

    def choose_crop():
        scores = {c: crop_score(c) for c in CROPS}
        eligible = [c for c in CROPS if planted_today[c] < batch_limits[c] and scores[c] > 0]
        owned = [c for c in eligible if seeds.get(c, 0)]
        if owned:
            return max(owned, key=lambda c: (scores[c], c))
        if not eligible:
            return None
        best = max(eligible, key=lambda c: (scores[c], c))
        return best if scores[best] >= .80 * max(scores.values()) else None

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
    profitable = choose_crop() is not None
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
    purchase_crop = None
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
            c = choose_crop()
            if c is None:
                continue
            if seeds.get(c, 0) <= 0:
                # Buy only for a unit already standing on its empty target.
                # One crop per order cycle; consume paid seeds before switching crops.
                if tuple(units[i]) == pos and (purchase_crop is None or purchase_crop == c):
                    purchase_crop = c
                    needed_seeds[c] = needed_seeds.get(c, 0) + 1
                actions[i] = move(units[i], pos)
                continue
            action = ["PLANT", c]
            if tuple(units[i]) == pos:
                seeds[c] -= 1
                planted_today[c] += 1
                # This seed's future sale was already dated in pending above.
        actions[i] = action if tuple(units[i]) == pos else move(units[i], pos)

    orders = []
    budget = farm["money"]
    # A small cash-funded buffer can wait for forecast demand, but harvesting
    # capacity and seed money take priority. Do not park a full shed of produce.
    room = max(0, configuration["shedCapacity"] - 60 - carried_total)
    ripe_units = sum(t["yield_units"] for _, t in plants if day - t["planted_day"] >= CROPS[t["crop"]][1])
    if final_day or horizon <= 1 or budget < 2500 or ripe_units > 40:
        room = 0
    hold = {}
    opportunities = []
    for c, n in projected_shed.items():
        if n > 0 and c in CROPS:
            current = quote(c, market["inventory"][c] + n // 2, params)
            # Pending includes today's stock: this estimates the eventual sale
            # after its price impact, rather than quoting a scarcity before sale.
            future = quote(c, future_inventory[c][-1], params)
            opportunities.append((future - current, c, n, current))
    for gain, c, n, current in sorted(opportunities, reverse=True):
        keep = min(n, room) if gain > max(2, current * .05) else 0
        hold[c] = keep
        room -= keep
    for c, n in sorted(projected_shed.items()):
        if n <= 0:
            continue
        sell = n - hold.get(c, 0)
        if sell:
            orders.append(["SELL", c, sell])
    for c, n in sorted(needed_seeds.items()):
        n = min(3, n, batch_limits[c] - planted_today[c], max(0, (budget - 60) // CROPS[c][0]))
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
