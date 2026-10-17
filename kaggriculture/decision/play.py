"""Trusted local policy launcher with full official replay re-execution.

Unlike Kaggle's raw-file loader, supplies __file__ for multi-file local policies.
No sandbox: do not use for untrusted third-party code.
"""
import argparse
import gzip
import json
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from decision.full_audit import make, reexecute


def load(value):
    if value in ("pass", "starter", "random"):
        return value
    path = Path(value).resolve()
    scope = {"__name__": "local_decision_policy", "__file__": str(path)}
    exec(compile(path.read_bytes(), str(path), "exec"), scope)
    return scope["agent"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bot", required=True)
    parser.add_argument("--opponent", default="pass")
    parser.add_argument("--seed", type=int, default=6774936)
    parser.add_argument("--side", type=int, choices=[0, 1], default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "run.log").open("w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
        agents = [load(args.bot), load(args.opponent)]
        if args.side:
            agents.reverse()
        env = make("kaggriculture", configuration={"seed": args.seed}, debug=True)
        env.run(agents)
        replay = env.toJSON()
        audit = reexecute(replay)
        (args.output / "replay.json.gz").write_bytes(gzip.compress(json.dumps(replay).encode()))
        (args.output / "agent-logs.json").write_text(json.dumps(env.logs), encoding="utf-8")
        (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
