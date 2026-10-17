"""Versioned decision interface for official Kaggriculture 1.33.0.

No engine imports, future-state simulation, profitability masks or forced teacher
actions. Assessments are advisory. Joint outcomes remain the server's decision.
"""
from copy import deepcopy
from dataclasses import dataclass
from typing import Protocol

SCHEMA_VERSION = 1
CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
PRODUCTS = (*CROPS, "EGG", "MILK", "WOOL", "FERTILIZER")
ITEMS = (*PRODUCTS, *ANIMALS)
MOVES = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}
UNIT_OPS = (*MOVES, "PASS", "DROP", "PICKUP", "PLACE", "PLANT", "WATER", "HARVEST", "FERTILIZE",
            "DIG", "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER", "CARE")
MARKET_OPS = ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL", "HIRE", "BUY_LAND")
PARAMETER_DOMAINS = {"unit_ops": UNIT_OPS, "market_ops": MARKET_OPS, "crops": CROPS, "animals": ANIMALS,
                     "items": ITEMS, "quantity": "arbitrary positive integer, not restricted to templates",
                     "joint": "one atom per currently observed unit; ordered market atoms, STOP head for a model"}
FARM_KEYS = ("money", "tiles", "farmer", "hands", "unlocked_quadrants", "hires_today")
TILE_KEYS = ("kind", "crop", "planted_day", "watered_today", "consecutive_unwatered", "yield_units",
             "max_lifespan_step", "fertilized_until_day", "animal", "placed_day", "consecutive_unfed",
             "fed_today", "cared_today", "fertilizer_available", "pending_care_bonus")
CONFIG_KEYS = ("episodeSteps", "boardSize", "startingMoney", "maxMarketOrdersPerTurn", "turnsPerDay",
               "shedCapacity", "weedSpawnChance", "townShopUnlockInterval", "townShopSellInterval",
               "townCenterSellInterval", "farmHandCostMult", "marketParams")
PRICE_KEYS = ("base", "I0", "T", "below_func", "below_target", "above_func", "above_target")


def _items(raw, allowed):
    return {k: deepcopy(v) for k, v in raw.items() if k in allowed}


@dataclass
class View:
    observation: dict
    configuration: dict
    history: dict

    @property
    def own(self):
        return self.observation["farms"][self.observation["player"]]

    @property
    def private(self):
        return self.observation["private"]

    @property
    def positions(self):
        return [self.own["farmer"], *self.own["hands"]]

    def encode(self):
        # Lossless structured/sparse features, not a crop-only neural vector.
        return {"schema_version": SCHEMA_VERSION, "observation": deepcopy(self.observation),
                "configuration": deepcopy(self.configuration), "history": deepcopy(self.history),
                "inventory_item_order": [list(i) for i in self.private["inventories"]],
                "shed_item_order": list(self.private["shed"]),
                "opponent_belief": {"private_inventory": "unknown", "seed_inventory": "unknown"}}


def observe(observation, configuration, previous=None):
    """Accept ONE runtime player's observation, never an environment/replay."""
    if "steps" in observation or "info" in observation or "observation" in observation:
        raise ValueError("Expected a single legal runtime observation, not replay or engine state")
    p = int(observation["player"])
    farms = []
    for farm in observation["farms"]:
        public = _items(farm, FARM_KEYS)
        public["tiles"] = [[_items(t, TILE_KEYS) if isinstance(t, dict) else t for t in row] for row in farm["tiles"]]
        farms.append(public)
    priv = observation["private"]
    obs = {"player": p, "step": observation["step"], "day": observation["day"], "hour": observation["hour"],
           "farms": farms, "private": {"shed": _items(priv["shed"], ITEMS), "seeds": _items(priv["seeds"], CROPS),
                                     "inventories": [_items(i, ITEMS) for i in priv["inventories"]]},
           "market": _items(observation["market"], ("inventory", "prices", "params")),
           "town": _items(observation["town"], ("unlocked_shops",))}
    # Filter nested dictionaries as well; future unknown fields fail visibility
    # closed without pretending that their semantics have been modeled.
    for key in ("inventory", "prices", "params"):
        if key in obs["market"]:
            obs["market"][key] = _items(obs["market"][key], PRODUCTS)
    if "params" in obs["market"]:
        obs["market"]["params"] = {k: _items(v, PRICE_KEYS) for k, v in obs["market"]["params"].items()}
    history = {}
    if previous is not None:
        if not isinstance(previous, View) or previous.observation["player"] != p or previous.observation["step"] >= obs["step"]:
            raise ValueError("History must be an earlier sanitized view of the same player")
        history = {"previous_step": previous.observation["step"], "previous_market": deepcopy(previous.observation["market"]),
                   "net_public_inventory_delta": {k: obs["market"]["inventory"][k] - previous.observation["market"]["inventory"][k] for k in PRODUCTS},
                   "interpretation": "Net shared change includes both players and town; not opponent private stock"}
    cfg = _items(configuration, CONFIG_KEYS)
    cfg["marketParams"] = {k: _items(v, PRICE_KEYS) for k, v in cfg.get("marketParams", {}).items() if k in PRODUCTS}
    return View(obs, cfg, history)


@dataclass(frozen=True)
class Atom:
    op: str
    item: str | None = None
    quantity: int | None = None

    def wire(self):
        return [self.op] + ([] if self.item is None else [self.item]) + ([] if self.quantity is None else [self.quantity])

    @classmethod
    def parse(cls, raw, market=False):
        """Canonical documented domain. Rejected junk may be a server no-op,
        not necessarily a Kaggle JSON-schema violation. Never silently repair."""
        if not isinstance(raw, list) or not raw or raw[0] not in (MARKET_OPS if market else UNIT_OPS):
            raise ValueError("Unknown/noncanonical operation")
        op = raw[0]
        n = 1 if op in ("HIRE", "BUY_LAND") or not market and op not in ("PICKUP", "PLACE", "PLANT") else 3 if market else 2
        if len(raw) != n and not (not market and op in ("PICKUP", "PLACE") and len(raw) == 3):
            raise ValueError("Incorrect canonical argument count")
        item = raw[1] if len(raw) > 1 else None
        quantity = raw[2] if len(raw) > 2 else None
        if item is not None and not isinstance(item, str):
            raise ValueError("Item must be a string")
        if quantity is not None and (type(quantity) is not int or quantity <= 0):
            raise ValueError("Quantity must be a positive integer")
        return cls(op, item, quantity)


@dataclass(frozen=True)
class Joint:
    farmer: Atom
    hands: tuple = ()
    market: tuple = ()

    @classmethod
    def parse(cls, raw):
        if not isinstance(raw, dict) or set(raw) - {"farmer", "hands", "market"}:
            raise ValueError("Expected canonical joint action")
        return cls(Atom.parse(raw.get("farmer", ["PASS"])), tuple(Atom.parse(a) for a in raw.get("hands", [])),
                   tuple(Atom.parse(a, market=True) for a in raw.get("market", [])))

    def wire(self):
        # Keep order and over-limit entries. The engine decides truncation and
        # PLANT prevalidation (which counts even nonexistent extra hands).
        Atom.parse(self.farmer.wire())
        for a in self.hands:
            Atom.parse(a.wire())
        for a in self.market:
            Atom.parse(a.wire(), market=True)
        return {"farmer": self.farmer.wire(), "hands": [a.wire() for a in self.hands], "market": [a.wire() for a in self.market]}


def access(position, size):
    h = size // 2
    return tuple(position) in ((h - 1, h - 1), (h, h - 1), (h - 1, h), (h, h))


def assess_unit(view, index, atom):
    """Exact isolated atom preconditions in the current observable state only.
    These are NOT masks for later units after previous atoms mutate the state.
    No effects are simulated. Unprofitable/damaging actions stay available.
    """
    Atom.parse(atom.wire())
    if index >= len(view.positions) or index < 0:
        return "unavailable", "nonexistent unit; official no-op"
    x, y = view.positions[index]
    tile, inv = view.own["tiles"][y][x], view.private["inventories"][index]
    d = tile if isinstance(tile, dict) else {}
    op, item = atom.op, atom.item
    size = view.configuration["boardSize"]
    ok = False
    if op in MOVES:
        dx, dy = MOVES[op]
        ok = 0 <= x + dx < size and 0 <= y + dy < size
    elif op == "PASS":
        ok = True
    elif op == "DROP":
        ok = access((x, y), size)  # Overflow loss is legal, never masked.
    elif op == "PICKUP":
        ok = access((x, y), size) and view.private["shed"].get(item, 0) > 0
    elif op == "PLACE":
        structure = {"GOOSE": "COOP", "COW": "PASTURE", "SHEEP": "PASTURE"}.get(item)
        if structure and d.get("kind") == structure and "animal" not in d:
            ok = inv.get(item, 0) > 0
        else:
            ok = access((x, y), size) and inv.get(item, 0) > 0 and sum(view.private["shed"].values()) < view.configuration["shedCapacity"]
    elif tile == "LOCKED":
        ok = False
    elif op == "PLANT":
        ok = tile is None and item in CROPS and view.private["seeds"].get(item, 0) > 0
    elif op == "WATER":
        ok = d.get("kind") == "PLANT" and not d["watered_today"]
    elif op == "HARVEST":
        first = {"WHEAT": 2, "CARROT": 2, "TOMATO": 8, "STRAWBERRY": 10, "MELON": 10}
        ok = d.get("yield_units", 0) > 0 and ("animal" in d or d.get("kind") == "PLANT" and view.observation["day"] - d["planted_day"] >= first[d["crop"]])
    elif op == "FERTILIZE":
        ok = d.get("kind") == "PLANT" and inv.get("FERTILIZER", 0) > 0
    elif op == "DIG":
        ok = tile is not None and "animal" not in d
    elif op in ("BUILD_COOP", "BUILD_PASTURE"):
        ok = tile is None
    elif op == "FEED":
        ok = "animal" in d and not d["fed_today"] and inv.get("WHEAT", 0) > 0
    elif op == "CARE":
        ok = "animal" in d and not d["cared_today"]
    elif op == "COLLECT_FERTILIZER":
        ok = "animal" in d and d["fertilizer_available"]
    return ("available", "official isolated preconditions satisfied; value not assessed") if ok else ("unavailable", "isolated official no-op; preceding joint actions can change this")


def assess_joint(view, joint):
    """Advisory constraints; never rewrites or silently narrows the policy."""
    joint.wire()
    units = [joint.farmer, *joint.hands]
    counts = {c: sum(a.op == "PLANT" and a.item == c for a in units) for c in CROPS}
    blocked = [c for c, n in counts.items() if n > view.private["seeds"].get(c, 0)]
    return {"atomic_seed_blocked_crops": blocked,
            "unit_isolated_assessments": [assess_unit(view, i, a) for i, a in enumerate(units)],
            "unit_joint_status": "conditional: shared tiles/inventory execute farmer then hands; isolated judgments are not joint masks",
            "market_status": "conditional: ordered queues, cash/storage changes and opponent lockstep prices; no approximate hard mask",
            "market_assessments": [assess_order(view, a) for a in joint.market],
            "ignored_market_orders": max(0, len(joint.market) - view.configuration["maxMarketOrdersPerTurn"]),
            "ignored_extra_hands": max(0, len(joint.hands) - len(view.own["hands"])),
            "economic_value": "not assessed; no strategy preference is a legality constraint"}


def assess_order(view, atom):
    """Only context-independent impossibilities are rejected. Lack of money or
    stock now is not rejection: earlier units/orders can supply them."""
    Atom.parse(atom.wire(), market=True)
    valid_item = {"BUY_SEED": CROPS, "BUY_PRODUCT": ("WHEAT", "FERTILIZER"), "BUY_ANIMAL": ANIMALS, "SELL": PRODUCTS}
    if atom.op in valid_item and atom.item not in valid_item[atom.op]:
        return "unavailable", "unsupported item for official order"
    if atom.op == "BUY_LAND" and len(view.own["unlocked_quadrants"]) == 4:
        return "unavailable", "all quadrants already owned"
    return "conditional", "prior units/orders and opposing quotes determine cash, stock and room"


class Policy(Protocol):
    def decide(self, view: View) -> Joint: ...


def act(policy: Policy, observation, configuration):
    view = observe(observation, configuration)
    joint = policy.decide(view)
    if not isinstance(joint, Joint):
        raise TypeError("Policy must return Joint")
    return joint.wire()


class FrozenAdapter:
    """One optional counterfactual, not a compulsory neural controller."""
    def __init__(self, frozen_agent, buy_first_land=False):
        self.frozen_agent, self.buy_first_land = frozen_agent, buy_first_land

    def decide(self, view):
        raw = self.frozen_agent(deepcopy(view.observation), deepcopy(view.configuration))
        joint = Joint.parse(raw)
        if self.buy_first_land and view.observation["step"] == 0:
            assert view.own["money"] >= 1000 and len(view.own["unlocked_quadrants"]) == 1
            assert len(joint.market) < view.configuration["maxMarketOrdersPerTurn"]
            return Joint(joint.farmer, joint.hands, (Atom("BUY_LAND"), *joint.market))
        return joint
