"""Minimal deterministic wheat farmer for the official Kaggriculture engine."""


def agent(observation, configuration):
    farm = observation["farms"][observation["player"]]
    private = observation["private"]
    x, y = farm["farmer"]
    tile = farm["tiles"][y][x]
    seeds = private["seeds"].get("WHEAT", 0)
    market = []
    wheat = private["shed"].get("WHEAT", 0)
    if wheat:
        market.append(["SELL", "WHEAT", wheat])
    if not seeds and farm["money"] >= 10:
        market.append(["BUY_SEED", "WHEAT", 1])

    farmer = ["PASS"]
    if isinstance(tile, dict) and tile.get("kind") == "WEED":
        farmer = ["DIG"]
    elif tile is None and seeds and observation["hour"] < configuration["turnsPerDay"] - 1:
        farmer = ["PLANT", "WHEAT"]
    elif isinstance(tile, dict) and tile.get("kind") == "PLANT":
        if observation["day"] - tile["planted_day"] >= 4 and tile["yield_units"] > 0:
            farmer = ["HARVEST"]
        elif not tile["watered_today"]:
            farmer = ["WATER"]
    return {"farmer": farmer, "hands": [], "market": market}
