"""Prepare a fresh reproduction directory; never claim published seeds are unseen."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registration", type=Path, default=root.parent / "review/evidence/teacher-terminal-v2-20261018/registration.json")
    parser.add_argument("--round2", type=Path, default=root / "models/tuned_round2.py")
    args = parser.parse_args()
    out = args.output.resolve()
    cfg = json.loads(args.registration.read_text(encoding="utf-8"))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    files = {"baseline.py":root / "bots/teacher_fertilizer_v1.py", "teacher.py":root / "bots/teacher_fertilizer_v1.py", "round2.py":args.round2,
             "candidate.py":root / "bots/teacher_fertilizer_terminal_v2.py",
             "v1.py":root / "bots/heuristic_v1.py", "market.py":root / "bots/heuristic_v1_market.py",
             "worker.py":root / "learning/round2_worker.py",
             "evaluate_holdout.py":root / "learning/evaluate_holdout.py",
             "optimize_round2.py":root / "learning/optimize_round2.py"}
    assert all(p.is_file() for p in files.values())
    out.mkdir(parents=True, exist_ok=False)
    (out / "sources").mkdir()
    for name, path in files.items():
        shutil.copyfile(path, out / "sources" / name)
    cfg["source_sha256"] = {n:sha(out / "sources" / n) for n in files}
    cfg["reproduction_of_registration_sha256"] = sha(args.registration)
    cfg["scope"] = "Published seeds reused for reproduction only; not an independent holdout"
    (out / "registration.json").write_text(json.dumps(cfg,indent=2)+"\n",encoding="utf-8")
    (out / "registration.sha256").write_text(sha(out / "registration.json"))
    freeze = {"candidate_sha256":sha(files["candidate.py"]),
              "experiment_runner_sha256":sha(root / "learning/teacher_fertilizer_experiment.py"),
              "registration_sha256":sha(out / "registration.json"),
              "boundary":"Reproduction, not new selection or unseen evaluation"}
    (out / "candidate-freeze.json").write_text(json.dumps(freeze,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
