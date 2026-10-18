"""Official telemetry in the game environment; JSON stdout protocol."""
import argparse
from contextlib import redirect_stdout
import gzip
import json
from pathlib import Path
from statistics import mean
import sys

def collect(folder, side):
    from decision.full_audit import reexecute
    replay = json.loads(gzip.decompress((folder / "replay.json.gz").read_bytes()))
    execution = reexecute(replay, trace=True)
    p = side
    expanded_plants = actions = 0
    for before, after in zip(replay["steps"], replay["steps"][1:]):
        farm = before[0]["observation"]["farms"][p]
        positions = [farm["farmer"], *farm["hands"]]
        a = after[p]["action"]
        for pos, atom in zip(positions, [a["farmer"], *a["hands"]]):
            actions += atom[0] != "PASS"
    expanded_plants = execution["execution"]["successful_plants_new_land"][p]
    first = replay["steps"][1][0]["observation"]["farms"][p]
    last = replay["steps"][-1][0]["observation"]["farms"][p]
    prices = [s[0]["observation"]["market"]["prices"]["MELON"] for s in replay["steps"]]
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    return {"first_post_action_quadrants": first["unlocked_quadrants"], "final_quadrants": last["unlocked_quadrants"],
            "successful_plant_events_in_new_land": expanded_plants, "non_pass_unit_actions": actions,
            "execution_label_version": execution["execution"]["label_version"], "execution_effect_sha256": execution["execution"]["ordered_effect_sha256"],
            "melon_quote_min": min(prices), "melon_quote_mean": mean(prices),
            "simulation_seconds": result["run_seconds"], "both_bot_seconds": sum(result["bot_seconds"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("side", type=int, choices=(0, 1))
    args = parser.parse_args()
    # Package startup diagnostics must not corrupt the JSON response.
    with redirect_stdout(sys.stderr):
        result = collect(args.folder, args.side)
    print(json.dumps(result))
