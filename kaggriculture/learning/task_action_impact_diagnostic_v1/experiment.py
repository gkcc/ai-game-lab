"""Metadata-only registration for the single saved-data impact diagnosis.

Preparing and freezing never opens a selected replay or trace.  The only old
inputs read here are the explicit metadata allowlist and the root decision.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/task-action-impact-diagnostic-v1-20261027"
PREVIOUS = ROOT / "artifacts/task-anchor-reentry-v1-20261027"
PUBLIC = ROOT.parent / "publication/ai-game-lab"
SOURCE = ROOT / "learning/task_action_impact_diagnostic_v1"
TASK_STARTED_UTC = "2026-10-26T23:31:24+00:00"
TASK_DEADLINE_UTC = "2026-10-27T01:01:24+00:00"
METADATA_LIMIT = 16 * 1024 * 1024
OUTPUT_LIMIT = 8 * 1024 * 1024
CONTROL_RESERVE = 64 * 1024
LOG_RESERVE = 1024 * 1024
PURE_HELPERS = (
    "decision/__init__.py",
    "decision/api.py",
    "decision/semantics.py",
    "learning/task_bc_error_diagnostics_v1/__init__.py",
    "learning/task_bc_error_diagnostics_v1/shadow.py",
    "learning/task_anchor_once_v1/__init__.py",
    "learning/task_anchor_once_v1/processes.py",
)
_META_BASE = int(os.environ.get("TASK_IMPACT_METADATA_BASE_BYTES", "0"))
_META_READ = 0


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def artifact_helper_path():
    value = os.environ.get("TASK_IMPACT_ARTIFACT_HELPER")
    if not value:
        raise RuntimeError("Set TASK_IMPACT_ARTIFACT_HELPER to the user-required scratch ownership utility")
    path = Path(value)
    if not path.is_absolute() or not path.is_file():
        raise ValueError("The scratch ownership utility must be an existing absolute file path")
    return path


def meter():
    return {"inherited_prior_phase_bytes": _META_BASE,
            "this_process_bytes": _META_READ,
            "total_bytes": _META_BASE + _META_READ, "limit_bytes": METADATA_LIMIT}


def _metadata_path(path):
    path = Path(path).resolve()
    if path.suffix.lower() in {".gz", ".npz", ".npy", ".pt", ".pth"}:
        raise ValueError("Raw replay/trace/weight reads are forbidden in metadata helpers")
    if path.is_relative_to(PREVIOUS):
        relative = path.relative_to(PREVIOUS).as_posix()
        names = {"registration.json", "root-result-acceptance.json", "run-summary.json",
                 "holdout-statistics.json", "development-gate.json",
                 "root-next-learning-decision.json", "independent-next-step.json"}
        if relative not in names and not re.fullmatch(
                r"job-results/holdout-[0-9]+-(round2|market)-[01]-(baseline|candidate|teacher_task)\.(row|audit|complete)\.json", relative):
            raise ValueError("Old metadata path is outside the registered allowlist")
    return path


def _bytes(path):
    global _META_READ
    path = _metadata_path(path)
    size = path.stat().st_size
    if _META_BASE + _META_READ + size > METADATA_LIMIT:
        raise RuntimeError("Metadata/source read budget would be exceeded")
    with path.open("rb") as stream:
        data = stream.read(size)
        ending_size = os.fstat(stream.fileno()).st_size
    _META_READ += len(data)
    if len(data) != size or ending_size != size or _META_BASE + _META_READ > METADATA_LIMIT:
        raise RuntimeError("Metadata/source input changed size or exceeded budget")
    return data


def read(path):
    return json.loads(_bytes(path).decode("utf-8-sig"))


def sha(path):
    return hashlib.sha256(_bytes(path)).hexdigest()


def _record(path):
    data = _bytes(path)
    return json.loads(data.decode("utf-8-sig")), hashlib.sha256(data).hexdigest()


def output_bytes(out=OUT):
    return sum(p.stat().st_size for p in Path(out).rglob("*") if p.is_file())


def _write(path, value, reserve=CONTROL_RESERVE + LOG_RESERVE):
    path = Path(path)
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    accounting_root = OUT if path.resolve().is_relative_to(OUT) else path.parent
    if output_bytes(accounting_root) + len(data) > OUTPUT_LIMIT - reserve:
        raise RuntimeError("Owned output budget would be exceeded")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
    return path


def write_new(path, value):
    return _write(path, value)


def write_receipt(path, value):
    return _write(path, value, reserve=0)


def _valid_hash(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def _file_entry(relative, expected_hash):
    if not _valid_hash(expected_hash):
        raise ValueError("Missing certified input hash: " + relative)
    path = PREVIOUS / relative
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(PREVIOUS.resolve()):
        raise ValueError("Missing or redirected selected input: " + relative)
    return {"path": relative, "size": path.stat().st_size, "sha256": expected_hash}


def prepare():
    """Write one deterministic input manifest and a non-frozen draft."""
    if (OUT / "input-manifest.json").exists() or (OUT / "registration-draft.json").exists():
        raise FileExistsError("Preparation already exists; do not substitute an input sample")
    decision, decision_hash = _record(PREVIOUS / "root-next-learning-decision.json")
    advice, advice_hash = _record(PREVIOUS / "independent-next-step.json")
    previous_registration = read(PREVIOUS / "registration.json")
    previous_acceptance_hash = sha(PREVIOUS / "root-result-acceptance.json")
    statistics = read(PREVIOUS / "holdout-statistics.json")
    sample, budgets = decision["sample"], decision["budgets"]
    if decision["task"] != "task_action_impact_diagnostic_v1" or decision["advice_sha256"] != advice_hash:
        raise ValueError("Decision/advice identity mismatch")
    if decision["previous_acceptance_sha256"] != previous_acceptance_hash:
        raise ValueError("Previous acceptance identity mismatch")
    if advice["data"] != sample:
        raise ValueError("Decision/advice sample mismatch")
    for old, new in (("test_seeds", "seed_order"), ("opponents", "opponent_order"),
                     ("seats", "seat_order"), ("arms", "arm_order")):
        if previous_registration[old] != sample[new]:
            raise ValueError("Old registration sample mismatch: " + old)
    if (sample["games"], sample["triads"], sample["comparisons"], sample["selected_file_count"]) != (96, 32, 64, 192):
        raise ValueError("Unexpected fixed denominator")
    details = {(p["seed"], p["opponent"], p["side"]): p for p in statistics["paired_detail"]}
    if len(details) != 32 or len(statistics["paired_detail"]) != 32 or not statistics["complete"]:
        raise ValueError("Incomplete or duplicate paired metadata")
    games, pairs, certificates = [], [], []
    scalar_fields = ("bank", "opponent_bank", "initial_bank", "opponent_initial_bank",
                     "net_profit", "opponent_net_profit", "margin", "score")
    expected_fields = (*scalar_fields, "banks", "rewards", "status", "states", "action_turns", "initial_banks")
    for seed in sample["seed_order"]:
        for opponent in sample["opponent_order"]:
            for side in sample["seat_order"]:
                detail = details[(seed, opponent, side)]
                pair_key = f"{seed}-{opponent}-{side}"
                first = detail["first_request_difference_step"]
                if type(first) is not int or not 0 <= first < 719:
                    raise ValueError("Missing accepted C-S first request difference")
                pairs.append({"key": pair_key, "seed": seed, "opponent": opponent, "side": side,
                              "first_candidate_baseline_request_difference": first,
                              "first_request_difference_step": first})
                for arm in sample["arm_order"]:
                    key = f"holdout-{pair_key}-{arm}"
                    row, row_hash = _record(PREVIOUS / f"job-results/{key}.row.json")
                    audit, audit_hash = _record(PREVIOUS / f"job-results/{key}.audit.json")
                    complete, complete_hash = _record(PREVIOUS / f"job-results/{key}.complete.json")
                    if not all(x.get("passed") is True and x.get("game") == key for x in (row, audit, complete)):
                        raise ValueError("Incomplete prior game certificate: " + key)
                    if (complete["row_sha256"], complete["certificate_sha256"]) != (row_hash, audit_hash):
                        raise ValueError("Prior certificate hash mismatch: " + key)
                    if (row["seed"], row["opponent"], row["side"], row["policy"], row["split"]) != (seed, opponent, side, arm, "holdout"):
                        raise ValueError("Game identity mismatch: " + key)
                    if any(row[k] != detail[arm][k] for k in scalar_fields) or detail[arm]["game"] != key:
                        raise ValueError("Stored terminal pair scalar mismatch: " + key)
                    if row["banks"][side] != row["bank"] or row["banks"][1-side] != row["opponent_bank"]:
                        raise ValueError("Seat/bank mismatch: " + key)
                    if row["margin"] != row["bank"] - row["opponent_bank"] or row["rewards"] != audit["official"]["final_rewards"]:
                        raise ValueError("Recorded terminal margin/reward mismatch: " + key)
                    if row["replay_gzip_sha256"] != audit["replay_sha256"] or row["trace_sha256"] != audit["trace_sha256"]:
                        raise ValueError("Raw certificate identity mismatch: " + key)
                    games.append({"key": key, "seed": seed, "opponent": opponent, "side": side, "arm": arm,
                                  "replay": _file_entry(f"games/{key}/replay.json.gz", row["replay_gzip_sha256"]),
                                  "trace": _file_entry(f"traces/{key}.jsonl.gz", row["trace_sha256"]),
                                  "expected": {k: row[k] for k in expected_fields}})
                    certificates.append({"key": key, "row_sha256": row_hash, "audit_sha256": audit_hash,
                                         "complete_sha256": complete_hash})
    compressed = sum(g[k]["size"] for g in games for k in ("replay", "trace"))
    if compressed != sample["selected_compressed_bytes_observed_from_metadata"]:
        raise ValueError("Fixed raw input byte total changed")
    manifest = {"schema": "task-action-impact-input-manifest-v1", "previous_root": sample["root"],
                "selection_role": sample["role"], "games": games, "pairs": pairs,
                "game_count": len(games), "pair_count": len(pairs), "file_count": 2 * len(games),
                "compressed_bytes": compressed, "metadata_certificates": certificates,
                "raw_reads_during_preparation": 0}
    write_new(OUT / "input-manifest.json", manifest)
    draft = {"schema": "task-action-impact-registration-v1", "task": decision["task"],
             "status": "draft_not_execution_authorization", "created_utc": utc_now(),
             "task_started_utc": TASK_STARTED_UTC, "task_deadline_utc": TASK_DEADLINE_UTC,
             "root_decision_sha256": decision_hash, "independent_advice_sha256": advice_hash,
             "previous_acceptance_sha256": previous_acceptance_hash,
             "sample": sample, "budgets": budgets, "analysis_method": decision["analysis_method"],
             "acceptance": decision["acceptance"], "root_adjudication": decision["root_adjudication"],
             "input_manifest_sha256": sha(OUT / "input-manifest.json"),
             "budget_interpretation": {
                 "metadata_source_scope": "The 16 MiB cap covers registration/analysis verification plus independent compact source/result review. Development source browsing, Git operations and public-content-only privacy review are excluded from this byte counter, but included in the 90 minute wall budget.",
                 "memory_enforcement": "External 0.25-second sampled working-set/CPU monitor; not an OS hard memory quota. Transient between-sample overshoot cannot be excluded.",
                 "processes": "One analysis child; no grandchildren. One CPU affinity core and one compute thread. Parent output-drain thread is infrastructure.",
                 "output_scope": "All files in the new task output directory plus temporary worker log bytes; 64 KiB reserved for terminal control receipts.",
                 "raw_accounting": "Each selected compressed file opened once by analysis, hashed during that same read, with compressed and decoded counters before each bounded read. No decoded copies on disk.",
                 "failure": "A run-started receipt consumes the only attempt, including startup, budget or extraction failure; no retries or replacement sample."},
             "preparation_meter": meter(), "raw_reads_before_registration": 0,
             "agent_workflow_enabled": False, "new_learning_authorized": False}
    write_new(OUT / "registration-draft.json", draft)
    return {"status": "prepared_not_frozen", "games": 96, "pairs": 32, "files": 192,
            "compressed_bytes": compressed, "metadata_meter": meter()}


def freeze():
    """Freeze exact final source before independent pre-review and Git commit."""
    if (OUT / "registration.json").exists() or (OUT / "freeze.json").exists():
        raise FileExistsError("Registration is immutable once frozen")
    draft = read(OUT / "registration-draft.json")
    source_paths = sorted(SOURCE.glob("*.py"))
    required = {"analysis.py", "experiment.py", "run.py"}
    if not required.issubset({p.name for p in source_paths}):
        raise ValueError("Final source is incomplete")
    source_sha = {p.relative_to(ROOT).as_posix(): sha(p) for p in source_paths}
    source_sha.update({p: sha(ROOT / p) for p in PURE_HELPERS})
    if sha(OUT / "input-manifest.json") != draft["input_manifest_sha256"]:
        raise ValueError("Prepared input manifest changed")
    registration = dict(draft, status="frozen_pending_independent_review_and_local_commit",
                        frozen_utc=utc_now(), source_sha256=source_sha,
                        scratch_ownership_helper_sha256=sha(artifact_helper_path()),
                        preparation_and_freeze_meter=meter())
    write_new(OUT / "registration.json", registration)
    frozen = {"schema": "task-action-impact-freeze-v1", "frozen_utc": utc_now(),
              "registration_sha256": sha(OUT / "registration.json"),
              "input_manifest_sha256": registration["input_manifest_sha256"],
              "source_sha256": source_sha, "raw_reads": 0}
    write_new(OUT / "freeze.json", frozen)
    return {"status": "frozen_pending_review_commit", "freeze_sha256": sha(OUT / "freeze.json"),
            "source_files": len(source_sha), "metadata_meter": meter()}


def _git_bytes(commit, relative):
    return subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=PUBLIC,
                          check=True, capture_output=True).stdout


def verify(out=OUT, commit=None):
    """Verify only compact registration, sources, metadata and raw file stats."""
    out = Path(out)
    registration, registration_hash = _record(out / "registration.json")
    frozen, frozen_hash = _record(out / "freeze.json")
    manifest, manifest_hash = _record(out / "input-manifest.json")
    if registration_hash != frozen["registration_sha256"] or manifest_hash != frozen["input_manifest_sha256"]:
        raise ValueError("Frozen registration/manifest identity changed")
    if registration["source_sha256"] != frozen["source_sha256"]:
        raise ValueError("Source freeze mappings disagree")
    if sha(artifact_helper_path()) != registration["scratch_ownership_helper_sha256"]:
        raise ValueError("The user-required scratch ownership helper changed")
    current_sources = {p.relative_to(ROOT).as_posix() for p in SOURCE.glob("*.py")}
    frozen_sources = {p for p in frozen["source_sha256"] if p.startswith("learning/task_action_impact_diagnostic_v1/")}
    if current_sources != frozen_sources:
        raise ValueError("Frozen task source set changed")
    for relative, expected_hash in frozen["source_sha256"].items():
        if sha(ROOT / relative) != expected_hash:
            raise ValueError("Frozen source changed: " + relative)
    for game in manifest["games"]:
        for kind in ("replay", "trace"):
            entry = game[kind]
            if _file_entry(entry["path"], entry["sha256"]) != entry:
                raise ValueError("Selected input file stat changed: " + entry["path"])
    if commit is not None:
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("Full local preregistration commit SHA is required")
        review, review_hash = _record(out / "pre-review.json")
        if review.get("passed") is not True or review.get("freeze_sha256") != frozen_hash:
            raise ValueError("Independent pre-review did not pass the exact freeze")
        if review.get("registration_sha256") != registration_hash:
            raise ValueError("Independent review registration binding mismatch")
        expected = {"kaggriculture/" + p: digest for p, digest in frozen["source_sha256"].items()}
        evidence = "review/evidence/" + out.name + "/"
        expected.update({evidence + "registration.json": registration_hash,
                         evidence + "input-manifest.json": manifest_hash,
                         evidence + "freeze.json": frozen_hash, evidence + "pre-review.json": review_hash})
        for path, digest in expected.items():
            if hashlib.sha256(_git_bytes(commit, path)).hexdigest() != digest:
                raise ValueError("Local preregistration commit byte mismatch: " + path)
    return {"registration": registration, "manifest": manifest,
            "registration_sha256": registration_hash, "freeze_sha256": frozen_hash,
            "metadata_meter": meter(), "raw_reads": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "freeze", "verify"))
    parser.add_argument("--commit")
    args = parser.parse_args()
    result = prepare() if args.command == "prepare" else freeze() if args.command == "freeze" else verify(commit=args.commit)
    if args.command == "verify":
        result = {k: v for k, v in result.items() if k not in ("registration", "manifest")}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
