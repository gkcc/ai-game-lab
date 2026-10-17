"""A one-goose execution fixture, not a competitive animal strategy."""
import argparse
import gzip
import json
from pathlib import Path

from decision.api import Atom, Joint, act
from decision.full_audit import make, reexecute


class GooseFixture:
    """Fresh, public/own-private only; independent of the frozen crop teacher."""
    def decide(self, view):
        o, p = view.observation, view.private
        tile = view.own["tiles"][4][4]
        inv = p["inventories"][0]
        orders = [Atom("SELL", item, n) for item, n in p["shed"].items() if item in ("EGG", "FERTILIZER") and n > 0]
        if o["step"] == 0:
            return Joint(Atom("BUILD_COOP"), market=(Atom("BUY_ANIMAL", "GOOSE", 1),))
        if "animal" not in (tile or {}):
            unit = Atom("PLACE", "GOOSE") if inv.get("GOOSE", 0) else Atom("PICKUP", "GOOSE")
        elif not tile["fed_today"]:
            if inv.get("WHEAT", 0):
                unit = Atom("FEED")
            else:
                unit = Atom("PICKUP", "WHEAT", 1)
                if p["shed"].get("WHEAT", 0) == 0:
                    orders.append(Atom("BUY_PRODUCT", "WHEAT", 1))
        elif not tile["cared_today"]:
            unit = Atom("CARE")
        elif tile["fertilizer_available"]:
            unit = Atom("COLLECT_FERTILIZER")
        elif tile["yield_units"]:
            unit = Atom("HARVEST")
        elif inv.get("EGG", 0) or inv.get("FERTILIZER", 0):
            unit = Atom("DROP")
        else:
            unit = Atom("PASS")
        return Joint(unit, market=tuple(orders))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    policy = GooseFixture()
    env = make("kaggriculture", configuration={"seed": 7137181}, debug=True)
    def agent(obs, cfg):
        return act(policy, obs, cfg)
    env.run([agent, lambda obs: {"farmer": ["PASS"], "hands": [], "market": []}])
    replay = env.toJSON()
    audit = reexecute(replay)
    ops, products = {}, {}
    for frame in replay["steps"][1:]:
        a = frame[0]["action"]
        op = a["farmer"][0]
        ops[op] = ops.get(op, 0) + 1
        for order in a["market"]:
            if order[0] == "SELL":
                products[order[1]] = products.get(order[1], 0) + order[2]
    assert all(ops.get(op, 0) > 0 for op in ("BUILD_COOP", "PICKUP", "PLACE", "FEED", "CARE", "HARVEST", "COLLECT_FERTILIZER", "DROP"))
    assert products.get("EGG", 0) > 0 and products.get("FERTILIZER", 0) > 0
    (args.output / "replay.json.gz").write_bytes(gzip.compress(json.dumps(replay).encode()))
    (args.output / "result.json").write_text(json.dumps({"audit": audit, "unit_action_counts": ops,
                                                        "sold_products": products, "scope": "Execution fixture versus PASS, not competitive value evidence"}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit": audit, "unit_action_counts": ops, "sold_products": products}))


if __name__ == "__main__":
    main()
