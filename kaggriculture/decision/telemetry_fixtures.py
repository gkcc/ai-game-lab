"""Generate real complete official replays for the isolated telemetry regression."""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import time

from decision.verify_mechanisms import env, action


def generate(out):
    out.mkdir(parents=True, exist_ok=False)
    cases = []
    for side in (0, 1):
        for hired in (False, True):
            e = env()
            schedule = [action(market=[["HIRE"]] if hired else [])]
            schedule += [{"farmer": ["PASS"], "hands": value, "market": []}
                         for value in (None, 1, "ignored", {"ignored": []})]
            schedule += [{"hands": [], "market": []}]
            schedule += [{"farmer": value, "hands": [], "market": []}
                         for value in ([], None, 1, "ignored", {})]
            schedule += [{"farmer": ["UNKNOWN"], "hands": [], "market": []}, action(["PLANT", "WHEAT"])]
            schedule += [{"farmer": ["PASS"], "hands": values, "market": []}
                         for values in ([[], None, 1, "ignored", {}, ["UNKNOWN"]],
                                        [["UNKNOWN"]], [["PASS"], ["UNKNOWN"]],
                                        [[1]], [["PLANT", "WHEAT"]])]
            schedule += [{"farmer": ["PLACE"], "hands": [], "market": []}]
            original = deepcopy(schedule)
            tick = time.perf_counter()
            while not e.done:
                index = len(e.steps)-1
                if 1 <= index < len(schedule):
                    assert len(e.state[0].observation.farms[side]["hands"]) == int(hired)
                pair = [action(), action()]
                pair[side] = deepcopy(schedule[index]) if index < len(schedule) else action()
                e.step(pair)
            elapsed = time.perf_counter()-tick
            assert schedule == original
            replay = e.toJSON()
            assert len(replay["steps"]) == 720 and all(s.status == "DONE" for s in e.state)
            folder = out / f"side{side}-hired{int(hired)}"
            folder.mkdir()
            raw = gzip.compress(json.dumps(replay).encode(), compresslevel=1, mtime=0)
            (folder / "replay.json.gz").write_bytes(raw)
            # Supplied actions, no policy executed: fixture timing is not a Bot benchmark.
            (folder / "result.json").write_text(json.dumps({"run_seconds": elapsed, "bot_seconds": [0, 0], "fixture": True}), encoding="utf-8")
            cases.append({"folder": folder.name, "side": side, "hired": hired, "states": 720,
                          "replay_gzip_sha256": hashlib.sha256(raw).hexdigest(),
                          "expected_requests": 3 + 2*int(hired),
                          "expected_other_side_requests": 0})
    (out / "manifest.json").write_text(json.dumps(cases, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    generate(parser.parse_args().output)
