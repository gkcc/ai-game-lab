"""Privileged OFFLINE audit only: re-execute actions in the official engine.

Never import this module in deployment feature extraction; replay info/private
state belong to evaluation only. No replacement transition rules live here.
"""
import hashlib
from decision.verify_mechanisms import make, ENGINE_SHA, rules
from pathlib import Path
from decision.semantics import clean, canonical, digest, AUDIT_VERSION
from decision.execution import ExecutionTrace
from contextlib import nullcontext


def reexecute(replay, *, trace=False, label_sink=None):
    assert hashlib.sha256(Path(rules.__file__).read_bytes()).hexdigest() == ENGINE_SHA
    cfg = {**replay["configuration"], "seed": replay["info"]["seed"]}
    official = make("kaggriculture", configuration=cfg, debug=True)
    assert canonical(official.state) == canonical(replay["steps"][0])
    execution = ExecutionTrace(rules, label_sink) if trace or label_sink else None
    with execution.installed(official) if execution else nullcontext():
        for expected in replay["steps"][1:]:
            assert not any(s["status"] in ("ERROR", "INVALID", "TIMEOUT") for s in expected)
            official.step([s["action"] for s in expected])
            assert canonical(official.state) == canonical(expected), "Official ordered state transition mismatch"
    assert official.done and len(official.steps) == 720
    result = {"passed": True, "audit_version": AUDIT_VERSION,
            "official_transitions_reexecuted": len(official.steps) - 1,
            "both_public_and_private_states_match_each_step": True, "inventory_order_checked_each_step": True,
            "semantic_sha256": digest(official.steps, 1), "semantic_sha256_version": 1,
            "semantic_sha256_v2": digest(official.steps),
            "final_rewards": [s.reward for s in official.state], "engine_sha256": ENGINE_SHA}
    assert result["semantic_sha256_v2"] == digest(replay["steps"])
    if execution:
        n = len(replay["steps"][0][0]["observation"]["farms"][0]["tiles"])
        result["execution"] = {"label_version": 1, "counts": dict(sorted(execution.counts.items())),
                               "ordered_effect_sha256": execution.effect_hash.hexdigest(),
                               "successful_plants": [sum(v for k, v in execution.plants.items() if k[0] == p) for p in (0, 1)],
                               "successful_plants_new_land": [sum(v for k, v in execution.plants.items() if k[0] == p and k[1] >= n // 2) for p in (0, 1)],
                               "boundary": "Privileged offline effects; market order aggregates and ordered fill digest; no profitability or joint legality guarantee"}
    return result
