"""Run the official luxai-s3 CLI with isolated Bot subprocesses."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def bot_path(value):
    p = Path(value)
    if not p.is_absolute():
        p = ROOT / p
    if not p.is_file():
        raise FileNotFoundError(p)
    return str(p.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bot", default="bot/main.py")
    parser.add_argument("--opponent", default="bot/main.py")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--format", choices=("json", "html"), default="json")
    parser.add_argument("--name")
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    name = args.name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    if Path(name).name != name or name in (".", ".."):
        parser.error("--name must be a directory name")
    out = ROOT / "artifacts" / "matches" / name
    out.mkdir(parents=True, exist_ok=False)
    bot, opponent = bot_path(args.bot), bot_path(args.opponent)
    scripts = Path(sys.executable).parent
    executable = scripts / ("luxai-s3.exe" if os.name == "nt" else "luxai-s3")
    child_env = dict(os.environ)
    child_env["PATH"] = str(scripts) + os.pathsep + child_env.get("PATH", "")
    child_env["VIRTUAL_ENV"] = str(scripts.parent)
    child_env["PYTHONNOUSERSITE"] = "1"
    child_env["PYTHONDONTWRITEBYTECODE"] = "1"
    child_env["PYTHONUTF8"] = "1"
    if args.require_gpu:
        child_env["JAX_PLATFORMS"] = "cuda"
        child_env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    probe_code = ("import json,jax; print(json.dumps({'backend':jax.default_backend(),"
                  "'jax':jax.__version__,'devices':[{'platform':d.platform,'kind':d.device_kind} for d in jax.devices()]}))")
    probe = subprocess.run([sys.executable, "-B", "-c", probe_code], env=child_env,
                           capture_output=True, text=True)
    (out / "backend.log").write_text(probe.stdout + probe.stderr, encoding="utf-8")
    if probe.returncode:
        raise RuntimeError(f"Backend initialization failed; see {out / 'backend.log'}")
    backend = json.loads(probe.stdout.strip().splitlines()[-1])
    if args.require_gpu and (backend["backend"] != "gpu"):
        raise RuntimeError(f"CUDA GPU required; got {backend}")
    replay_path = out / ("replay." + args.format)
    entry = [sys.executable, "-u", "-B", str(ROOT / "gpu_entry.py")] if args.require_gpu else [str(executable)]
    command = [*entry, bot, opponent, "--seed", str(args.seed), "--output", str(replay_path), "--verbose", "3"]
    started = time.perf_counter()
    with (out / "run.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=ROOT, env=child_env, stdout=log, stderr=subprocess.STDOUT)
    text = (out / "run.log").read_text(encoding="utf-8", errors="replace")
    if args.format == "json":
        replay = json.loads(replay_path.read_text(encoding="utf-8")) if replay_path.exists() else None
    else:
        html = replay_path.read_text(encoding="utf-8") if replay_path.exists() else ""
        match = re.search(r"window\.episode = (.*);\s*</script>", html, re.DOTALL)
        replay = json.loads(match.group(1)) if match else None
        if replay:
            (out / "replay.json").write_text(json.dumps(replay), encoding="utf-8")
    result = {"game": "luxai-s3", "python": platform.python_version(), "python_executable": sys.executable,
              "bot_python_from_path": shutil.which("python", path=child_env["PATH"]),
              "package_version": importlib.metadata.version("luxai-s3"),
              "upstream_commit": subprocess.check_output(["git", "-C", str(ROOT / "upstream"), "rev-parse", "HEAD"], text=True).strip(),
              "seed": args.seed, "agents": [bot, opponent], "command": command,
              "exit_code": process.returncode, "elapsed_seconds": round(time.perf_counter() - started, 3),
              "reward_log": next((line for line in text.splitlines() if line.startswith("Rewards:")), None),
              "backend": backend, "gpu_required": args.require_gpu, "passed": False}
    source_dir = "setup-gpu" if args.require_gpu else "setup"
    source_check = json.loads((ROOT / "artifacts" / source_dir / "source-verification.json").read_text(encoding="utf-8"))
    distribution = importlib.metadata.distribution("luxai-s3")
    source_check["all_match"] = source_check["all_match"] and all(
        hashlib.sha256(Path(distribution.locate_file(item["file"].removeprefix("src/"))).read_bytes()).hexdigest() == item["sha256"]
        for item in source_check["files"])
    result["engine_commit"] = source_check["commit"] if source_check["all_match"] else None
    result["engine_matches_official_commit"] = source_check["all_match"]
    if replay:
        observations, actions, params = replay["observations"], replay["actions"], replay["params"]
        final = observations[-1]
        expected = (params["max_steps_in_match"] + 1) * params["match_count_per_episode"]
        bad_actions = 0
        nonzero = [0, 0]
        for action in actions:
            for side in range(2):
                rows = action.get(f"player_{side}")
                if not isinstance(rows, list) or len(rows) != params["max_units"] or any(len(row) != 3 for row in rows):
                    bad_actions += 1
                else:
                    nonzero[side] += sum(row[0] != 0 for row in rows)
        movements = [0, 0]
        for before, after in zip(observations, observations[1:]):
            if before["match_steps"] >= params["max_steps_in_match"]:
                continue
            for side in range(2):
                for unit, (a, b) in enumerate(zip(before["units"]["position"][side], after["units"]["position"][side])):
                    if before["units_mask"][side][unit] and after["units_mask"][side][unit] and a != b:
                        movements[side] += 1
        result.update({"recorded_states": len(observations), "action_turns": len(actions), "expected_action_turns": expected,
                       "final_step": final["steps"], "team_wins": final["team_wins"],
                       "bad_action_shapes": bad_actions, "nonzero_unit_actions": nonzero,
                       "actual_unit_movements": movements,
                       "max_observed_team_points": [max(s["team_points"][i] for s in observations) for i in range(2)],
                       "replay_sha256": hashlib.sha256(json.dumps(replay, sort_keys=True, separators=(",", ":")).encode()).hexdigest()})
        result["passed"] = (source_check["all_match"] and process.returncode == 0 and result["reward_log"] is not None and len(actions) == expected
                            and final["steps"] == expected and len(observations) == expected + 1
                            and sum(final["team_wins"]) == params["match_count_per_episode"] and bad_actions == 0
                            and all(movements) and "Traceback" not in text and "invalid action" not in text.lower())
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"artifacts": str(out), **result}, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
