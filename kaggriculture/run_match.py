"""Run official Kaggriculture; keep each replay and its evidence together."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent


@contextlib.contextmanager
def native_output(log):
    saved = [os.dup(1), os.dup(2)]
    try:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        yield
    finally:
        log.flush()
        for fd, original in zip((1, 2), saved):
            os.dup2(original, fd)
            os.close(original)


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def clean_timing(value):
    if isinstance(value, dict):
        return {k: clean_timing(v) for k, v in value.items() if k != "remainingOverageTime"}
    if isinstance(value, list):
        return [clean_timing(v) for v in value]
    return value


def resolve_bot(value):
    if value in ("starter", "random", "pass"):
        return value
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return str(candidate.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bot", default="main.py")
    parser.add_argument("--opponent", default="starter")
    parser.add_argument("--side", type=int, choices=(0, 1), default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", help="Unique artifact directory name")
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts" / "matches",
                        help="Parent directory for match artifacts (relative to project)")
    args = parser.parse_args()
    name = args.name or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8])
    if Path(name).name != name or name in (".", ".."):
        parser.error("--name must be a directory name")
    out = (ROOT / args.output_root).resolve() / name
    out.mkdir(parents=True, exist_ok=False)
    agents = [resolve_bot(args.bot), resolve_bot(args.opponent)]
    if args.side == 1:
        agents.reverse()
    started = time.perf_counter()
    with (out / "run.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log), native_output(log):
        import kaggle_environments
        from kaggle_environments import make
        from kaggle_environments.envs.kaggriculture import kaggriculture as rules

        env = make("kaggriculture", configuration={"seed": args.seed}, debug=True)
        env.run(agents)
        replay = env.toJSON()
        dump(out / "replay.json", replay)
        dump(out / "agent-logs.json", env.logs)
        (out / "final.txt").write_text(env.render(mode="ansi"), encoding="utf-8")
        visualizer = rules.html_renderer(env, "html")
        render_status = "missing official bundled visualizer"
        if visualizer.strip():
            from localize_replay import localize_html
            (out / "replay.html").write_text(localize_html(env.render(mode="html", width=1200, height=800, playing=False)), encoding="utf-8")
            render_status = "official HTML with Chinese UI and action annotations; browser verification is separate"
        semantic = clean_timing(replay["steps"])
        digest = hashlib.sha256(json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        from decision.semantics import digest as semantic_digest
        digest_v2 = semantic_digest(replay["steps"])
        final = env.steps[-1]
        rewards = [s.reward for s in final]
        statuses = [s.status for s in final]
        final_money = [f["money"] for f in final[0].observation.farms]
        counts = [{}, {}]
        errors = []
        for step, logs in enumerate(env.logs):
            for player, entry in enumerate(logs):
                if entry.get("stderr"):
                    errors.append({"step": step, "player": player, "stderr": entry["stderr"]})
        for step in replay["steps"][1:]:
            for player, state in enumerate(step):
                action = state.get("action") or {}
                for order in [action.get("farmer", ["PASS"]), *action.get("market", [])]:
                    key = order[0]
                    counts[player][key] = counts[player].get(key, 0) + 1
        source = Path(rules.__file__)
        upstream = ROOT / "upstream"
        commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
        official_source = subprocess.check_output([
            "git", "-C", str(upstream), "show",
            "HEAD:kaggle_environments/envs/kaggriculture/kaggriculture.py"])
        same_source = source.read_bytes() == official_source
        result = {"game": "kaggriculture", "package_version": kaggle_environments.__version__,
                  "python": platform.python_version(), "python_executable": sys.executable,
                  "upstream_commit": commit, "engine_file": str(source),
                  "engine_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "engine_matches_official_commit": same_source, "seed": env.info.get("seed"),
                  "agents": agents, "agent_file_sha256": {
                      p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in agents if Path(p).is_file()},
                  "bot_side": args.side, "configuration": replay["configuration"],
                  "done": env.done, "recorded_states": len(env.steps), "action_turns": len(env.steps) - 1,
                  "final_step": final[0].observation.step, "statuses": statuses, "rewards": rewards,
                  "final_money": final_money, "action_counts": counts, "agent_errors": errors,
                  "semantic_replay_sha256": digest, "semantic_replay_sha256_version": 1,
                  "semantic_replay_sha256_v2": digest_v2, "audit_version": 2, "render_status": render_status,
                  "elapsed_seconds": round(time.perf_counter() - started, 3),
                  "command": subprocess.list2cmdline([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])}
        result["passed"] = (env.done and statuses == ["DONE", "DONE"] and len(env.steps) == env.configuration.episodeSteps
                            and rewards == final_money and not errors and same_source)
        from replay_metrics import summarize
        # Crop-only audit is diagnostic: other custom agents may legally use animals.
        metrics = summarize(replay, args.side)
        dump(out / "metrics.json", metrics)
        result["bot_farming_preconditions_passed"] = metrics["farming_preconditions_passed"]
        dump(out / "result.json", result)
    print(json.dumps({"artifacts": str(out), **result}, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
