"""Exclusive registrations and bounded metadata for the evidence-only repair."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts/task-launcher-evidence-repair-v1-20261027'
PUBLIC = ROOT.parent / 'publication/ai-game-lab'
PREVIOUS = ROOT / 'artifacts/task-launcher-contract-repair-v1-20261027'
DECISION_SHA256 = 'bd5aa8780eb0e11afb833da4d942c14d154d4db11ed96a4c9f4228aacaa52b9f'
BASE = '0a116042e83dd6e63b2d12811931080825d93673'
TASK_STARTED_UTC = '2026-10-27T00:43:35+00:00'
TASK_DEADLINE_UTC = '2026-10-27T01:23:35+00:00'
METADATA_LIMIT = 4194304
OUTPUT_LIMIT = 2097152
PRIVATE_LIMIT = 262144
PUBLIC_RECEIPT_LIMIT = 32768
LABELS = ('d1', 'd2', 'd3', 'final')
PURE_HELPERS = ('learning/task_anchor_once_v1/__init__.py', 'learning/task_anchor_once_v1/processes.py',
    'learning/task_launcher_contract_repair_v1/__init__.py',
    'learning/task_launcher_contract_repair_v1/observer.py',
    'learning/task_launcher_contract_repair_v1/experiment.py')
PROTECTED = {'main.py': '74943a2e174c1fe1f9ed2e434d8aef5f13aeddc5575d06c0949ebf6bde818459',
             'bots/teacher_fertilizer_terminal_v2.py': '0844f0152e742aba684108b5914964fb85082b3c0881d0da4d41649907a9b6ff'}
_META_BASE = int(os.environ.get('TASK_LAUNCHER_METADATA_BASE_BYTES', '0'))
_META_READ = 0


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def deadline_check():
    if datetime.now(timezone.utc) >= datetime.fromisoformat(TASK_DEADLINE_UTC):
        raise RuntimeError('Registered task deadline reached')


def artifact_helper_path():
    path = Path(os.environ.get('TASK_LAUNCHER_ARTIFACT_HELPER', ''))
    if not path.is_absolute() or not path.is_file():
        raise ValueError('Registered ownership helper is required')
    return path.resolve()


def meter():
    return dict(inherited_prior_phase_bytes=_META_BASE, this_process_bytes=_META_READ,
                total_bytes=_META_BASE + _META_READ, limit_bytes=METADATA_LIMIT)


def _bytes(path):
    global _META_READ
    path = Path(path).resolve()
    allowed = path.is_relative_to(OUT) or (path.is_relative_to(SOURCE) and path.suffix == '.py')
    allowed = allowed or path in {ROOT / p for p in (*PURE_HELPERS, *PROTECTED)}
    allowed = allowed or path in {PREVIOUS / n for n in ('root-next-learning-decision.json', 'publication.json')}
    allowed = allowed or path in {Path(sys.executable).resolve(), Path(sys._base_executable).resolve()}
    helper = os.environ.get('TASK_LAUNCHER_ARTIFACT_HELPER')
    allowed = allowed or bool(helper and path == Path(helper).resolve())
    if not allowed or path.suffix.lower() in {'.gz', '.npz', '.npy', '.pt', '.pth'}:
        raise ValueError('Read outside synthetic source and metadata allowlist')
    size = path.stat().st_size
    if _META_BASE < 0 or _META_BASE + _META_READ + size > METADATA_LIMIT:
        raise RuntimeError('Metadata read limit exceeded')
    with path.open('rb') as stream:
        data = stream.read(size + 1)
        final_size = os.fstat(stream.fileno()).st_size
    _META_READ += len(data)
    if len(data) != size or final_size != size:
        raise RuntimeError('Metadata changed during read')
    return data


def read(path):
    return json.loads(_bytes(path).decode('utf-8-sig'))


def sha(path):
    return hashlib.sha256(_bytes(path)).hexdigest()


def output_usage():
    return sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) if OUT.exists() else 0


def ensure_output_capacity(size, private=False):
    extra = int(os.environ.get('TASK_LAUNCHER_TEMP_BYTES', '0'))
    if size < 0 or extra < 0 or output_usage() + extra + size > OUTPUT_LIMIT:
        raise RuntimeError('Owned output limit exceeded')
    if private:
        used = sum(p.stat().st_size for p in (OUT / 'private').rglob('*') if p.is_file())
        if used + size > PRIVATE_LIMIT:
            raise RuntimeError('Private output limit exceeded')


def write_bytes_new(path, data, private=False):
    path = Path(path).resolve()
    if not path.is_relative_to(OUT) or (private and not path.is_relative_to(OUT / 'private')):
        raise ValueError('Output outside registered namespace')
    ensure_output_capacity(len(data), private=private)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return path


def write_new(path, value):
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    if len(data) > PUBLIC_RECEIPT_LIMIT:
        raise RuntimeError('Public receipt limit exceeded')
    return write_bytes_new(path, data)


def _protection():
    actual = {p: sha(ROOT / p) for p in PROTECTED}
    if actual != PROTECTED:
        raise ValueError('Protected source changed')
    return actual


def _fingerprint(path):
    stat = Path(path).stat()
    return [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino]


def _runtimes(first=False):
    paths = {'requested': Path(sys.executable).resolve(), 'base': Path(sys._base_executable).resolve()}
    private_path = OUT / 'private/runtime-identities.json'
    if first:
        digests = {p: sha(p) for p in set(paths.values())}
        rows = {role: dict(path=str(p), fingerprint=_fingerprint(p), sha256=digests[p]) for role, p in paths.items()}
        write_bytes_new(private_path, (json.dumps(rows, indent=2) + '\n').encode(), private=True)
    else:
        rows = read(private_path)
        for role, path in paths.items():
            if rows[role]['path'] != str(path) or rows[role]['fingerprint'] != _fingerprint(path):
                raise ValueError('Registered runtime file identity changed')
    return {role: dict(basename=paths[role].name, sha256=rows[role]['sha256']) for role in paths}


def freeze(label):
    if label not in LABELS:
        raise ValueError('Unknown freeze label')
    deadline_check()
    if (OUT / f'{label}-freeze.json').exists() or (OUT / f'source-{label}').exists():
        raise FileExistsError('Freeze is exclusive')
    if label != 'final' and (OUT / 'final-freeze.json').exists():
        raise RuntimeError('Final source already frozen')
    decision = read(PREVIOUS / 'root-next-learning-decision.json')
    if sha(PREVIOUS / 'root-next-learning-decision.json') != DECISION_SHA256 or decision['task'] != 'task_launcher_evidence_repair_v1':
        raise ValueError('Root decision changed')
    publication = read(PREVIOUS / 'publication.json')
    if publication['commit'] != BASE or not all(publication.get(k) is True for k in ('working_tree_clean', 'stage_closed', 'both_privacy_checks_passed')):
        raise ValueError('Prior publication incomplete')
    if label != 'd1':
        initial = read(OUT / 'd1-registration.json')
        initial_freeze = read(OUT / 'd1-freeze.json')
        if sha(OUT / 'd1-registration.json') != initial_freeze['registration_sha256']:
            raise ValueError('Initial registration changed')
    runtimes = _runtimes(first=label == 'd1')
    if label != 'd1' and runtimes != initial['intended_runtime_images']:
        raise ValueError('Initial runtime binding changed')
    source_files = sorted(SOURCE.glob('*.py'))
    if not {'experiment.py', 'evidence.py', 'test_evidence.py', 'runner.py', 'worker.py'}.issubset({p.name for p in source_files}):
        raise ValueError('Incomplete new source namespace')
    sources = {}
    for path in source_files:
        data = _bytes(path)
        relative = path.relative_to(ROOT).as_posix()
        sources[relative] = hashlib.sha256(data).hexdigest()
        write_bytes_new(OUT / f'source-{label}' / path.name, data)
    for path in PURE_HELPERS:
        sources[path] = sha(ROOT / path)
    registration = dict(schema='launcher-evidence-registration-v1', task=decision['task'], phase=label,
        created_utc=utc_now(), task_started_utc=TASK_STARTED_UTC, task_deadline_utc=TASK_DEADLINE_UTC,
        root_decision_sha256=DECISION_SHA256, prior_publication_sha256=sha(PREVIOUS / 'publication.json'),
        public_baseline_commit=BASE, budgets=decision['budget'], source_sha256=sources,
        protected_source_sha256=_protection(), scratch_ownership_helper_sha256=sha(artifact_helper_path()),
        runtime_identity_private_sha256=sha(OUT / 'private/runtime-identities.json'),
        intended_runtime_images=runtimes, metadata_meter=meter(), agent_workflow_enabled=False,
        raw_reads=0, neural_forwards=0, official_transitions=0, training_updates=0)
    if label=='final':
        registration['chain_contract_sha256']=sha(OUT/'final-chain-contract.json')
    write_new(OUT / f'{label}-registration.json', registration)
    frozen = dict(schema='launcher-evidence-freeze-v1', phase=label, created_utc=utc_now(),
        registration_sha256=sha(OUT / f'{label}-registration.json'), source_sha256=sources)
    write_new(OUT / f'{label}-freeze.json', frozen)
    return dict(phase=label, freeze_sha256=sha(OUT / f'{label}-freeze.json'), metadata_meter=meter())


def verify(label):
    if label not in LABELS:
        raise ValueError('Unknown freeze label')
    deadline_check()
    registration = read(OUT / f'{label}-registration.json')
    frozen = read(OUT / f'{label}-freeze.json')
    registration_hash = sha(OUT / f'{label}-registration.json')
    if frozen['registration_sha256'] != registration_hash or frozen['source_sha256'] != registration['source_sha256']:
        raise ValueError('Freeze and registration mismatch')
    if sha(PREVIOUS / 'root-next-learning-decision.json') != DECISION_SHA256:
        raise ValueError('Root decision changed')
    if sha(PREVIOUS / 'publication.json') != registration['prior_publication_sha256']:
        raise ValueError('Prior publication changed')
    if sha(artifact_helper_path()) != registration['scratch_ownership_helper_sha256']:
        raise ValueError('Ownership helper changed')
    _protection()
    current = {p.relative_to(ROOT).as_posix() for p in SOURCE.glob('*.py')} | set(PURE_HELPERS)
    if current != set(frozen['source_sha256']):
        raise ValueError('Frozen source set changed')
    for relative, digest in frozen['source_sha256'].items():
        if sha(ROOT / relative) != digest:
            raise ValueError('Current frozen source changed')
        if relative not in PURE_HELPERS and sha(OUT / f'source-{label}' / Path(relative).name) != digest:
            raise ValueError('Frozen source snapshot changed')
    if _runtimes() != registration['intended_runtime_images']:
        raise ValueError('Registered runtime hashes changed')
    if label=='final' and sha(OUT/'final-chain-contract.json')!=registration['chain_contract_sha256']:
        raise ValueError('Accepted diagnostic chain contract changed')
    if sha(OUT / 'private/runtime-identities.json') != registration['runtime_identity_private_sha256']:
        raise ValueError('Private runtime identity binding changed')
    return dict(registration=registration, registration_sha256=registration_hash,
                freeze_sha256=sha(OUT / f'{label}-freeze.json'), metadata_meter=meter())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('freeze', 'verify'))
    parser.add_argument('label', choices=LABELS)
    args = parser.parse_args()
    result = freeze(args.label) if args.command == 'freeze' else verify(args.label)
    print(json.dumps({k: v for k, v in result.items() if k != 'registration'}, indent=2))
