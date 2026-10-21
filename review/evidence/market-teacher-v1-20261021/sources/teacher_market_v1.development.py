"""Heuristic v1: deterministic coordinated farming, using public observations only.

Frozen round2 crop controller plus bounded ongoing-crop fertilizer logistics.
No engine imports, private opponent state, land buying, animals or training.
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


def saleable_bonus_nights(tile, pos, day, configuration):
    """Economic planning window, not a legality mask or a resource guarantee.

    Official production is at day end, then all hands leave and the farmer
    respawns centrally. Budget one farmer's travel, harvest, return and DROP.
    The existing controller can SELL the projected DROP in that same turn.
    Ignore nights outside the episode and yield with no time to realize cash.
    Competing tasks, shed space, water and prices remain heuristic uncertainty.
    """
    first, interval = (8, 1) if tile['crop'] == 'TOMATO' else (10, 2)
    age = day - tile['planted_day']
    tpd = configuration['turnsPerDay']
    last_action = configuration['episodeSteps'] - 2
    half = configuration['boardSize'] // 2
    spawn = (half - 1, half - 1)
    accesses = [(half-1, half-1), (half, half-1), (half-1, half), (half, half)]
    # Travel out + HARVEST + travel back + DROP; SELL follows unit actions.
    cash_actions = distance(spawn, pos) + 1 + min(distance(pos, a) for a in accesses) + 1
    return sum(
        age + d in range(first, first + 4*interval, interval)
        and (day + d)*tpd - 1 <= last_action
        and (day + d)*tpd + cash_actions - 1 <= last_action
        for d in (1, 2, 3)
    )


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

    # Up to three production nights can benefit from one fertilizer.
    # This is an economic preference, never a legality mask for other policies.
    fertilizer_targets = {}
    for pos, tile in plants:
        c = tile["crop"]
        if c not in ("TOMATO", "STRAWBERRY") or tile["fertilized_until_day"] >= day:
            continue
        bonus_nights = saleable_bonus_nights(tile, pos, day, configuration)
        projected_price = quote(c, market["inventory"][c] + supply[c], params)
        if (bonus_nights and remaining > tpd and tile["yield_units"] < 4
                and bonus_nights * projected_price > market["prices"]["FERTILIZER"] + 50):
            fertilizer_targets[pos] = bonus_nights

    tasks = []
    for pos, tile in plants:
        c, age = tile["crop"], day - tile["planted_day"]
        if c in ("TOMATO", "STRAWBERRY") and age >= CROPS[c][2] and not tile["yield_units"]:
            if last_plant_age >= 2:
                tasks.append((pos, ["DIG"], 150.))
            continue
        ready = tile["yield_units"] > 0 and age >= CROPS[c][1]
        mature = age >= CROPS[c][2] or (c == "MELON" and tile["yield_units"] >= 6)
        if c in ("TOMATO", "STRAWBERRY") and tile["fertilized_until_day"] >= day and tile["yield_units"] >= 2:
            mature = True  # Free storage for the next fertilized production night.
        if pos in fertilizer_targets and not tile["watered_today"] and today >= 3:
            tasks.append((pos, ["FERTILIZE"], 245.))
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

    # Reserve shed fertilizer sequentially; buy orders execute later in the official engine.
    stored_fertilizer = private["shed"].get("FERTILIZER", 0)
    if fertilizer_targets and today >= 5 and not final_day:
        for i in sorted(available.copy()):
            if (stored_fertilizer > 0 and tuple(units[i]) in accesses
                    and not private["inventories"][i].get("FERTILIZER", 0)):
                actions[i] = ["PICKUP", "FERTILIZER", 1]
                available.remove(i)
                stored_fertilizer -= 1
                projected_shed["FERTILIZER"] = stored_fertilizer

    needed_seeds = {}
    # Joint nearest-task assignment; claim an entire tile to avoid duplicate work.
    while available and tasks:
        choices = []
        for i in sorted(available):
            for j, (pos, action, priority) in enumerate(tasks):
                if action[0] == "FERTILIZE" and not private["inventories"][i].get("FERTILIZER", 0):
                    continue
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
        tasks = [task for task in tasks if task[0] != pos]
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
        if c == "FERTILIZER" and remaining > tpd:
            continue
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
    stock = private["shed"].get("FERTILIZER", 0) + sum(v.get("FERTILIZER", 0) for v in private["inventories"])
    if fertilizer_targets and hour <= 2 and not final_day:
        buy = max(0, min(2, len(fertilizer_targets)) - stock)
        # A conservative cash reserve; do not assume own sales can fund this order.
        estimate = (market["prices"]["FERTILIZER"] + 5) * buy
        if buy and budget >= estimate + 300 and len(orders) < configuration["maxMarketOrdersPerTurn"]:
            orders.append(["BUY_PRODUCT", "FERTILIZER", buy])
    return {"farmer": actions[0], "hands": actions[1:],
            "market": orders[:configuration["maxMarketOrdersPerTurn"]]}

# Automatically selected constants; no online adaptation.
CROP_WEIGHTS = {'WHEAT': 1.0, 'CARROT': 1.3109056605077691, 'TOMATO': 1.481409970491683, 'STRAWBERRY': 1.2793244685756722, 'MELON': 0.6913358062522903}

# Appended verbatim to the frozen terminal-v2 teacher to build the standalone
# candidate. The prefix is byte-preserved; no farm controller changes.
_frozen_teacher = agent


def shed_after_transfers(observation, configuration, action):
    """Own DROP/PICKUP bookkeeping only; official engine still executes actions.

    DROP discards overflow before market execution, with inventory insertion
    order preserved. This projection never invents space by counting later SELL.
    The frozen teacher uses only DROP and PICKUP for shed transfers.
    """
    private = observation['private']
    farm = observation['farms'][observation['player']]
    shed = dict(private['shed'])
    cap = configuration['shedCapacity']
    half = configuration['boardSize']//2
    accesses = {(half-1,half-1),(half,half-1),(half-1,half),(half,half)}
    carry = 0
    for pos, inv, a in zip([farm['farmer'],*farm['hands']], private['inventories'],
                           [action['farmer'],*action['hands']]):
        carry += sum(inv.values())
        if tuple(pos) not in accesses:
            continue
        if a[0] == 'DROP':
            for c,n in inv.items():
                shed[c] = shed.get(c,0) + min(n,max(0,cap-sum(shed.values())))
            carry -= sum(inv.values())
        elif a[0] == 'PICKUP':
            c,n = a[1:3]
            take = min(n,shed.get(c,0))
            shed[c] = shed.get(c,0)-take
            carry += take
    return shed,carry


def sale_plan(observation, configuration, action):
    """Bounded intraday forecast, NOT official future price or opponent stock.

    Existing SELL slots retain at least one unit, preserving every non-SELL
    slot, including its simultaneous-opponent alignment. No order is displaced.
    Always clear crops at hour tpd-2, after it, and throughout the final day.
    Thus holding cannot roll forward forever; no global or cross-game history.
    """
    shed,carry = shed_after_transfers(observation,configuration,action)
    step,hour,day = observation['step'],observation['hour'],observation['day']
    tpd = configuration['turnsPerDay']
    last = configuration['episodeSteps']-2
    farm = observation['farms'][observation['player']]
    orders = [list(o) for o in action['market']]
    until = max(0,tpd-2-hour)
    # Cash guard uses already banked money, never predicted sale proceeds.
    clear = until == 0 or step//tpd == last//tpd or farm['money'] < 1000
    incoming = sum(o[2] for o in orders if o[0] in ('BUY_PRODUCT','BUY_ANIMAL'))
    # Retain room for carried goods, baseline purchases and another harvest wave.
    capacity = max(0,configuration['shedCapacity']-carry-incoming-16)
    fixed = sum(n for c,n in shed.items() if c not in CROPS)
    keep_budget = max(0,min(capacity-fixed,configuration['shedCapacity']//2))
    market = observation['market']
    params = market.get('params',configuration.get('marketParams',{}))
    si,ci = max(1,configuration['townShopSellInterval']),max(1,configuration['townCenterSellInterval'])
    shop_ticks = sum(t%si == 0 for t in range(step,step+until))
    center_ticks = sum(t%ci == 0 for t in range(step,step+until))
    plans = {}
    for c in sorted(CROPS):
        n = shed.get(c,0)
        if n <= 0:
            continue
        draw = center_ticks
        for shop in observation['town'].get('unlocked_shops',[]):
            products = SHOPS.get(shop,())
            if c in products:
                draw += shop_ticks*(2 if len(products)==1 else 1)
        rival_supply = 0
        for i,other in enumerate(observation['farms']):
            if i == observation['player']:
                continue
            for row in other['tiles']:
                for tile in row:
                    if (isinstance(tile,dict) and tile.get('kind')=='PLANT'
                            and tile.get('crop')==c and day-tile['planted_day']>=CROPS[c][1]):
                        rival_supply += tile['yield_units']
        # Visible ripe yield is a conservative supply scenario, not a private
        # inventory estimate. Unknown carried goods and future actions remain risk.
        net_draw = max(0,draw-rival_supply)
        keep = 0 if clear else min(max(0,n-1),net_draw,keep_budget)
        if keep:
            inventory = market['inventory'][c]
            now = sum(quote(c,inventory+n-keep+i,params) for i in range(keep))
            later = sum(quote(c,inventory+n-keep-net_draw+i,params) for i in range(keep))
            if later-now < max(keep,now*.01):
                keep = 0
        plans[c] = n-keep
        keep_budget -= keep
    seen = set()
    for order in orders:
        if order[0]=='SELL' and order[1] in plans:
            order[2] = plans[order[1]]
            seen.add(order[1])
    # Baseline may have held every unit at a very low price. New sales only use
    # trailing free slots; purchases/hires keep exact indices and quantities.
    for c,n in sorted(plans.items()):
        if c not in seen and n and len(orders)<configuration['maxMarketOrdersPerTurn']:
            orders.append(['SELL',c,n])
    return orders


def agent(observation, configuration):
    action = _frozen_teacher(observation,configuration)
    return {**action,'market':sale_plan(observation,configuration,action)}
