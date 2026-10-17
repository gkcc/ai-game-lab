"""Privileged OFFLINE audit only: re-execute actions in the official engine.

Never import this module in deployment feature extraction; replay info/private
state belong to evaluation only. No replacement transition rules live here.
"""
import hashlib
import json
from decision.verify_mechanisms import make, ENGINE_SHA, rules
from pathlib import Path


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items() if k != "remainingOverageTime"}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def reexecute(replay):
    assert hashlib.sha256(Path(rules.__file__).read_bytes()).hexdigest() == ENGINE_SHA
    cfg = {**replay["configuration"], "seed": replay["info"]["seed"]}
    official = make("kaggriculture", configuration=cfg, debug=True)
    assert clean(official.state) == clean(replay["steps"][0])
    for expected in replay["steps"][1:]:
        assert not any(s["status"] in ("ERROR", "INVALID", "TIMEOUT") for s in expected)
        official.step([s["action"] for s in expected])
        assert clean(official.state) == clean(expected), "Official state transition mismatch"
    assert official.done and len(official.steps) == 720
    semantic = hashlib.sha256(json.dumps(clean(official.steps), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"passed": True, "official_transitions_reexecuted": len(official.steps) - 1,
            "both_public_and_private_states_match_each_step": True, "semantic_sha256": semantic,
            "final_rewards": [s.reward for s in official.state], "engine_sha256": ENGINE_SHA}
