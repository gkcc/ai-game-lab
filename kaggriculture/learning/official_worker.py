"""JSON-lines worker in the EXISTING official-engine venv; no external simulator."""
import os
import sys
# Native import warnings must not enter the JSON protocol.
PROTOCOL = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
os.dup2(2, 1)
import cProfile
import gzip
import hashlib
import io
import json
from pathlib import Path
import pstats
import time
import subprocess

PROJECT = Path(os.environ.get("KAGGRICULTURE_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT))
started = time.perf_counter()
from kaggle_environments import make, __version__
from kaggle_environments.envs.kaggriculture import kaggriculture as rules
from replay_metrics import summarize
IMPORT_SECONDS = time.perf_counter() - started
official = subprocess.check_output(["git", "-C", str(PROJECT / "upstream"), "show",
                                   "HEAD:kaggle_environments/envs/kaggriculture/kaggriculture.py"])
assert official == Path(rules.__file__).read_bytes()
ENGINE_SHA = hashlib.sha256(official).hexdigest()
CACHE = {}


def load_agent(path):
    source = Path(path).read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest not in CACHE:
        namespace = {"__name__": "frozen_bot", "__file__": path}
        exec(compile(source, path, "exec"), namespace)
        CACHE[digest] = namespace["agent"]
    return CACHE[digest], digest


def play(request):
    bot, bot_sha = load_agent(request["bot"])
    rival, rival_sha = load_agent(request["rival"])
    elapsed = [0., 0.]
    calls = [0, 0]
    def timed(fn, side):
        def act(obs, config):
            tick = time.perf_counter()
            try:
                return fn(obs, config)
            finally:
                elapsed[side] += time.perf_counter() - tick
                calls[side] += 1
        return act
    agents = [bot, rival] if request["side"] == 0 else [rival, bot]
    agents = [timed(f, p) for p, f in enumerate(agents)]
    t = time.perf_counter()
    env = make("kaggriculture", configuration={"seed": request["seed"]}, debug=True)
    setup = time.perf_counter() - t
    profile = cProfile.Profile() if request.get("profile") else None
    if profile:
        profile.enable()
    tick = time.perf_counter()
    env.run(agents)
    run_seconds = time.perf_counter() - tick
    if profile:
        profile.disable()
    states = env.toJSON()
    errors = [(i, p, s.status) for i, frame in enumerate(env.steps) for p, s in enumerate(frame)
              if s.status in ("ERROR", "INVALID", "TIMEOUT")]
    stderr = [entry.get("stderr") for frame in env.logs for entry in frame if entry.get("stderr")]
    last = env.steps[-1]
    banks = [f["money"] for f in last[0].observation.farms]
    metrics = [summarize(states, p) for p in (0, 1)]
    passed = (env.done and len(env.steps) == 720 and [s.status for s in last] == ["DONE", "DONE"]
              and [s.reward for s in last] == banks and not errors and not stderr
              and all(m["farming_preconditions_passed"] and m["seed_balance_matches"] for m in metrics))
    result = {"passed": passed, "seed": request["seed"], "side": request["side"], "banks": banks,
              "bank": banks[request["side"]], "opponent_bank": banks[1-request["side"]],
              "net_profit": banks[request["side"]] - 3000,
              "margin": banks[request["side"]] - banks[1-request["side"]],
              "states": len(env.steps), "action_turns": len(env.steps)-1,
              "bot_sha256": bot_sha, "rival_sha256": rival_sha, "engine_sha256": ENGINE_SHA,
              "package": __version__, "python": sys.version, "command": request,
              "run_seconds": run_seconds, "bot_seconds": elapsed, "bot_calls": calls,
              "make_seconds": setup, "import_seconds": IMPORT_SECONDS,
              "all_harvested_sold": all(m["harvested_units"] == m["sold_units"] for m in metrics)}
    if request.get("semantic"):
        from decision.semantics import digest
        result["semantic_replay_sha256"] = digest(states["steps"], 1)
        result["semantic_replay_sha256_version"] = 1
        result["semantic_replay_sha256_v2"] = digest(states["steps"])
        result["audit_version"] = 2
    folder = Path(request["folder"])
    folder.mkdir(parents=True, exist_ok=False)
    if profile:
        for order in ("cumulative", "tottime"):
            stream = io.StringIO()
            pstats.Stats(profile, stream=stream).sort_stats(order).print_stats(25)
            (folder / f"profile-{order}.txt").write_text(stream.getvalue(), encoding="utf-8")
    tick = time.perf_counter()
    with gzip.open(folder / "replay.json.gz", "wb", compresslevel=1) as target:
        target.write(json.dumps(states, separators=(",", ":")).encode())
    result["replay_save_seconds"] = time.perf_counter() - tick
    if request.get("html"):
        from localize_replay import localize_html
        (folder / "replay.html").write_text(localize_html(env.render(mode="html", width=1200, height=800, playing=False)), encoding="utf-8")
    for name, value in [("metrics-both.json", metrics), ("agent-logs.json", env.logs), ("result.json", result)]:
        (folder / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    result["total_seconds"] = time.perf_counter() - t
    (folder / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if not passed:
        raise RuntimeError(f"Official full game audit failed: {folder}")
    return result


for line in sys.stdin:
    try:
        request = json.loads(line)
        if request["cmd"] == "close":
            print(json.dumps({"closed": True}), file=PROTOCOL)
            break
        result = play(request)
        print(json.dumps(result), file=PROTOCOL)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(json.dumps({"error": str(exc)}), file=PROTOCOL)
