"""Extract real successful PLANT decisions from frozen v1 wins; no test-panel data."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
old = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else ROOT / "artifacts/crossplay/latest"
out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=True)
summary = json.loads((old / "summary.json").read_text(encoding="utf-8"))
crops = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
shops = ["BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE", "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET"]
X, y, seed_ids, games = [], [], [], []
for g in summary["matches"]:
    if "v1" not in g["agents"]:
        continue
    p = g["agents"].index("v1")
    if g["banks"][p] <= g["banks"][1-p]:
        continue
    replay = json.loads((old / "matches" / g["name"] / "replay.json").read_text(encoding="utf-8"))
    for before, after in zip(replay["steps"], replay["steps"][1:]):
        shared = before[0]["observation"]
        private = before[p]["observation"]["private"]
        farm = shared["farms"][p]
        positions = [farm["farmer"], *farm["hands"]]
        actions = after[p]["action"]
        orders = [actions["farmer"], *actions["hands"]]
        for i, a in enumerate(orders):
            if a[0] != "PLANT":
                continue
            px, py = positions[i]
            tile = after[0]["observation"]["farms"][p]["tiles"][py][px]
            if not isinstance(tile, dict) or tile.get("crop") != a[1]:
                continue
            features = [shared["day"]/30, shared["hour"]/24, np.log1p(farm["money"])/12, px/9, py/9]
            features += [np.log1p(shared["market"]["prices"][c])/7 for c in crops]
            features += [(shared["market"]["inventory"][c]-10000)/300 for c in crops]
            features += [shared["town"]["unlocked_shops"].count(s)/8 for s in shops]
            for owner in (p, 1-p):
                tiles = shared["farms"][owner]["tiles"]
                features += [sum(isinstance(t, dict) and t.get("crop")==c for row in tiles for t in row)/25 for c in crops]
            features += [private["seeds"].get(c,0)/25 for c in crops]
            features += [private["shed"].get(c,0)/100 for c in crops]
            X.append(features)
            y.append(crops.index(a[1]))
            seed_ids.append(g["seed"])
    games.append(g["name"])
X = np.asarray(X, np.float32)
y = np.asarray(y, np.int32)
val = np.isin(seed_ids, [42, 314159])
mu = X[~val].mean(0)
scale = np.maximum(X[~val].std(0), .05)
np.savez_compressed(out / "dataset.npz", x=(X-mu)/scale, y=y, validation=val, feature_mean=mu, feature_scale=scale)
metadata = {"samples": len(y), "train_samples": int((~val).sum()), "validation_samples": int(val.sum()),
            "features": X.shape[1], "crop_labels": crops, "class_counts": np.bincount(y,minlength=5).tolist(),
            "teacher": "frozen original v1; only wins and successful observed PLANT transitions",
            "teacher_games": games, "train_seeds": [0,1,7,2026], "validation_seeds": [42,314159],
            "split": "whole seed groups; validation never normalized into training statistics",
            "scope": "Five-way crop classifier feasibility probe, not a complete game policy; never used by Optuna or independent game test"}
(out / "dataset.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(metadata, ensure_ascii=False))
