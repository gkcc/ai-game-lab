"""Immutable two-phase registration for the bounded synthetic launcher repair."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/task-launcher-contract-repair-v1-20261027"
PUBLIC = ROOT.parent / "publication/ai-game-lab"
PREVIOUS = ROOT / "artifacts/task-action-impact-diagnostic-v1-20261027"
BASE = "741f7d10828c435761efc90cc1b2d092169dd6af"
DECISION_SHA256 = "7565c223a4b7201ce2c5c7c57509364cd311de36f461eeedb4b550f32236cbbe"
TASK_STARTED_UTC = "2026-10-27T00:19:31+00:00"
TASK_DEADLINE_UTC = "2026-10-27T00:44:31+00:00"
METADATA_LIMIT = 2 * 1024 * 1024
OUTPUT_LIMIT = 1024 * 1024
PUBLIC_RECEIPT_LIMIT = 32768
PURE_HELPERS = ("learning/task_anchor_once_v1/__init__.py",
                "learning/task_anchor_once_v1/processes.py")
PROTECTED = {
    "main.py": "74943a2e174c1fe1f9ed2e434d8aef5f13aeddc5575d06c0949ebf6bde818459",
    "bots/teacher_fertilizer_terminal_v2.py": "0844f0152e742aba684108b5914964fb85082b3c0881d0da4d41649907a9b6ff",
}
_META_BASE = int(os.environ.get("TASK_LAUNCHER_METADATA_BASE_BYTES", "0"))
_META_READ = 0


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def deadline_check():
    if datetime.now(timezone.utc) >= datetime.fromisoformat(TASK_DEADLINE_UTC):
        raise RuntimeError("Registered stage deadline reached")


def artifact_helper_path():
    value = os.environ.get("TASK_LAUNCHER_ARTIFACT_HELPER")
    if not value or not Path(value).is_absolute() or not Path(value).is_file():
        raise ValueError("TASK_LAUNCHER_ARTIFACT_HELPER must identify the existing absolute scratch ownership utility")
    return Path(value)


def meter():
    return {"inherited_prior_phase_bytes": _META_BASE, "this_process_bytes": _META_READ,
            "total_bytes": _META_BASE + _META_READ, "limit_bytes": METADATA_LIMIT}


def _bytes(path):
    global _META_READ
    path = Path(path).resolve()
    allowed = (path.is_relative_to(SOURCE) and path.suffix == ".py") or path.is_relative_to(OUT)
    allowed = allowed or path in {ROOT / p for p in (*PURE_HELPERS, *PROTECTED)}
    allowed = allowed or path in {PREVIOUS / n for n in ("root-next-learning-decision.json", "publication.json")}
    allowed = allowed or (os.environ.get("TASK_LAUNCHER_ARTIFACT_HELPER") and path == artifact_helper_path().resolve())
    allowed = allowed or path in {Path(sys.executable).resolve(),Path(sys._base_executable).resolve()}
    if not allowed or path.suffix.lower() in {".gz", ".npz", ".npy", ".pt", ".pth"}:
        raise ValueError("Read outside registered synthetic source/metadata allowlist")
    size = path.stat().st_size
    if _META_BASE < 0 or _META_BASE + _META_READ + size > METADATA_LIMIT:
        raise RuntimeError("Registered source/metadata read limit exceeded")
    with path.open("rb") as stream:
        data = stream.read(size + 1)
        ending_size = os.fstat(stream.fileno()).st_size
    _META_READ += len(data)
    if len(data) != size or ending_size != size:
        raise RuntimeError("Registered input changed while reading")
    return data


def read(path):
    return json.loads(_bytes(path).decode("utf-8-sig"))


def sha(path):
    return hashlib.sha256(_bytes(path)).hexdigest()


def write_new(path, value):
    path = Path(path)
    if not path.resolve().is_relative_to(OUT):
        raise ValueError("Persistent receipts must stay in the new output namespace")
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if len(data) > PUBLIC_RECEIPT_LIMIT:
        raise RuntimeError("Compact receipt limit exceeded")
    used = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    if used + len(data) > OUTPUT_LIMIT:
        raise RuntimeError("Owned output limit exceeded")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
    return path


def _label(label):
    if label not in ("observation", "observation-revised", "validation"):
        raise ValueError("Phase must be observation or validation")


def _sources():
    files = sorted(SOURCE.glob("*.py"))
    if not {"__init__.py", "experiment.py", "observer.py", "worker.py"}.issubset({p.name for p in files}):
        raise ValueError("Required synthetic observation source is incomplete")
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in files} | {p: sha(ROOT / p) for p in PURE_HELPERS}


def _protection():
    observed = {p: sha(ROOT / p) for p in PROTECTED}
    if observed != PROTECTED:
        raise ValueError("Default bot or teacher source changed")
    return observed


def freeze(label):
    _label(label)
    deadline_check()
    if any((OUT / f"{label}-{name}.json").exists() for name in ("registration", "freeze")):
        raise FileExistsError("This phase has already been frozen")
    if label=='observation-revised' and (OUT/'observation-started.json').exists():
        raise RuntimeError('Cannot revise after the observation attempt started')
    decision_data = _bytes(PREVIOUS / "root-next-learning-decision.json")
    if hashlib.sha256(decision_data).hexdigest() != DECISION_SHA256:
        raise ValueError("Root decision identity changed")
    decision = json.loads(decision_data)
    publication = read(PREVIOUS / "publication.json")
    if decision["task"] != "task_launcher_contract_repair_v1" or publication["commit"] != BASE or not all(
            publication.get(k) is True for k in ("remote_main_matches", "working_tree_clean", "phase_closed")):
        raise ValueError("Prior stage publication prerequisite failed")
    observation_binding = None
    if label == "validation":
        checked = verify("observation")
        observation_binding = checked["freeze_sha256"]
    sources = _sources()
    runtime_ids=(read(OUT/'observation-revised-registration.json')['intended_runtime_images'] if label=='validation' else
        {role:dict(basename=Path(path).name,sha256=sha(path)) for role,path in
         (('requested',sys.executable),('base',sys._base_executable))})
    registration = {
        "schema": "task-launcher-contract-registration-v1", "task": decision["task"], "phase": label,
        "created_utc": utc_now(), "task_started_utc": TASK_STARTED_UTC, "task_deadline_utc": TASK_DEADLINE_UTC,
        "root_decision_sha256": DECISION_SHA256, "prior_publication_sha256": sha(PREVIOUS / "publication.json"),
        "public_baseline_commit": BASE, "budgets": decision["budgets"], "scope": decision["scope"],
        "stop_conditions_and_root_adjudication": "Bound by root_decision_sha256; inherited unchanged",
        "source_sha256": sources, "protected_source_sha256": _protection(),
        "scratch_ownership_helper_sha256": sha(artifact_helper_path()),
        "intended_runtime_images": runtime_ids,
        "observation_freeze_sha256": observation_binding,
        "runtime_identity_storage": "Private process IDs, paths, prefixes and OS details only in owned scratch; public role/hash/equivalence projections only",
        "observation_command_shape": ["registered_virtual_environment_python", "-B", "-m", "learning.task_launcher_contract_repair_v1.worker", "--scratch", "owned_scratch", "--nonce", "per_launch_nonce", "--mode", "observe"],
        "source_policy": "Observation source bytes are immutable; validation may add new launcher source files and freezes the complete new namespace",
        "budget_interpretation": "Registered verification, runtime receipts and independent review share 2 MiB; implementation browsing, Git and required public-only privacy scans excluded from bytes but included in 25 minute wall limit",
        "metadata_meter": meter(), "agent_workflow_enabled": False,
        "raw_reads": 0, "neural_forwards": 0, "official_transitions": 0, "training_updates": 0,
    }
    write_new(OUT / f"{label}-registration.json", registration)
    frozen = {"schema": "task-launcher-contract-freeze-v1", "phase": label, "created_utc": utc_now(),
              "registration_sha256": sha(OUT / f"{label}-registration.json"), "source_sha256": sources}
    write_new(OUT / f"{label}-freeze.json", frozen)
    return {"phase": label, "freeze_sha256": sha(OUT / f"{label}-freeze.json"), "metadata_meter": meter()}


def verify(label):
    _label(label)
    if label=='observation' and (OUT/'observation-revised-freeze.json').exists():
        label='observation-revised'
    registration = read(OUT / f"{label}-registration.json")
    frozen = read(OUT / f"{label}-freeze.json")
    registration_hash = sha(OUT / f"{label}-registration.json")
    if frozen["registration_sha256"] != registration_hash or frozen["source_sha256"] != registration["source_sha256"]:
        raise ValueError("Phase registration/freeze mismatch")
    if sha(PREVIOUS / "root-next-learning-decision.json") != DECISION_SHA256:
        raise ValueError("Root decision changed")
    if sha(PREVIOUS / "publication.json") != registration["prior_publication_sha256"]:
        raise ValueError("Prior publication changed")
    if sha(artifact_helper_path()) != registration["scratch_ownership_helper_sha256"]:
        raise ValueError("Scratch ownership helper changed")
    _protection()
    for relative, digest in frozen["source_sha256"].items():
        if sha(ROOT / relative) != digest:
            raise ValueError("Frozen source changed: " + relative)
    if label == "validation" and set(_sources()) != set(frozen["source_sha256"]):
        raise ValueError("Validation source set changed")
    return {"registration": registration, "registration_sha256": registration_hash,
            "freeze_sha256": sha(OUT / f"{label}-freeze.json"), "metadata_meter": meter()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "verify"))
    parser.add_argument("label", choices=("observation", "observation-revised", "validation"))
    args = parser.parse_args()
    value = freeze(args.label) if args.command == "freeze" else verify(args.label)
    print(json.dumps({k: v for k, v in value.items() if k != "registration"}, indent=2))
