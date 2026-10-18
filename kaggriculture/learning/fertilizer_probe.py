"""Bounded real-engine experiment for fertilizer and timely ongoing-crop harvest."""
import argparse
import gzip
import json
from pathlib import Path

from decision.full_audit import make, reexecute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for crop in ("TOMATO", "STRAWBERRY"):
        for fertilize in (False, True):
            def agent(o, c):
                p = o["private"]
                tile = o["farms"][o["player"]]["tiles"][4][4]
                inv = p["inventories"][0]
                orders = [["SELL", crop, p["shed"][crop]]] if p["shed"].get(crop, 0) else []
                op = ["PASS"]
                if o["step"] == 0:
                    orders += [["BUY_SEED", crop, 1]]
                    if fertilize:
                        orders += [["BUY_PRODUCT", "FERTILIZER", 2]]
                elif o["day"] < 2 and tile is None and p["seeds"].get(crop):
                    op = ["PLANT", crop]
                elif fertilize and p["shed"].get("FERTILIZER", 0) and o["day"] < 17:
                    op = ["PICKUP", "FERTILIZER", p["shed"]["FERTILIZER"]]
                elif isinstance(tile, dict) and tile.get("crop") == crop:
                    days = (7, 10) if crop == "TOMATO" else (9, 13)
                    if fertilize and o["day"] in days and tile["fertilized_until_day"] < o["day"] and inv.get("FERTILIZER", 0):
                        op = ["FERTILIZE"]
                    elif not tile["watered_today"]:
                        op = ["WATER"]
                    elif tile["yield_units"]:
                        op = ["HARVEST"]
                return {"farmer": op, "hands": [], "market": orders}
            e = make("kaggriculture", configuration={"seed": 8445705}, debug=True)
            e.run([agent, lambda o,c: {"farmer": ["PASS"], "hands": [], "market": []}])
            replay = e.toJSON()
            audit = reexecute(replay, trace=True)
            sold = sum(int(a[2]) for s in replay["steps"][1:] for a in s[0]["action"]["market"] if a[:2] == ["SELL", crop])
            folder = args.output / f"{crop}-{int(fertilize)}"
            folder.mkdir()
            (folder / "replay.json.gz").write_bytes(gzip.compress(json.dumps(replay).encode()))
            rows.append({"crop": crop, "fertilizer_enabled": fertilize, "sold_units": sold,
                         "net_profit": audit["final_rewards"][0]-3000, "audit": audit})
    (args.output / "results.json").write_text(json.dumps({"rows": rows, "boundary": "One plot versus PASS; real cash including purchase costs, not competitive evidence"}, indent=2)+"\n", encoding="utf-8")
    print(json.dumps([{k:v for k,v in r.items() if k!='audit'} for r in rows]))


if __name__ == "__main__":
    main()
