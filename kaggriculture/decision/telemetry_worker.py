"""Official telemetry in the game environment; JSON stdout protocol."""
import argparse
from contextlib import redirect_stdout
import gzip
import json
from pathlib import Path
from statistics import mean
import sys

ACTION_COUNT_DEFINITION = "Pre-step existing unit slots; nonempty list with string opcode other than PASS; includes unknown, rejected and no-effect requests; excludes market, ghost slots and malformed atoms; not successful executions"


def count_unit_requests(farm, raw):
    """Request metric only: envelope handling mirrors the official interpreter.

    Parameter validity/economics are deliberately not filters. The official
    engine alone decides whether a well-shaped request has any effect.
    """
    action = raw if isinstance(raw, dict) else {}
    hands = action.get("hands", [])
    hands = hands if isinstance(hands, list) else []
    atoms = [action.get("farmer", ["PASS"]), *hands[:len(farm["hands"])]]
    return sum(isinstance(a, list) and bool(a) and isinstance(a[0], str) and a[0] != "PASS"
               for a in atoms)


def collect(folder, side):
    from decision.full_audit import reexecute
    replay = json.loads(gzip.decompress((folder / "replay.json.gz").read_bytes()))
    execution = reexecute(replay, trace=True)
    p = side
    actions = sum(count_unit_requests(before[0]["observation"]["farms"][p], after[p]["action"])
                  for before, after in zip(replay["steps"], replay["steps"][1:]))
    expanded_plants = execution["execution"]["successful_plants_new_land"][p]
    first = replay["steps"][1][0]["observation"]["farms"][p]
    last = replay["steps"][-1][0]["observation"]["farms"][p]
    prices = [s[0]["observation"]["market"]["prices"]["MELON"] for s in replay["steps"]]
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    return {"first_post_action_quadrants": first["unlocked_quadrants"], "final_quadrants": last["unlocked_quadrants"],
            "successful_plant_events_in_new_land": expanded_plants, "non_pass_unit_actions": actions,
            "non_pass_unit_requests": actions, "action_count_version": 1,
            "action_count_definition": ACTION_COUNT_DEFINITION,
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
