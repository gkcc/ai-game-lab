"""Exact source and registration gates for a sample-only capacity experiment."""
import argparse
import hashlib
from pathlib import Path
import subprocess

from learning.bc_crop.agent import ROOT, sha
from learning.bc_crop.experiment import read, save

PUBLIC = ROOT.parent / 'publication/ai-game-lab'
PACKAGE = Path(__file__).resolve().parent


def git(*args):
    return subprocess.check_output(['git', '-C', str(PUBLIC), *args])


def public_sources(registration, files):
    """Bind old public bytes, recording only verified CRLF/LF differences."""
    digests = {}; line_endings = {}
    for rel, runtime_digest in files.items():
        runtime = (ROOT / rel).read_bytes()
        if rel in registration['fixed_runtime_sources_sha256']:
            published = git('show', registration['baseline_head'] + ':kaggriculture/' + rel)
        else:
            published = runtime
        digest = hashlib.sha256(published).hexdigest()
        assert runtime.replace(b'\r\n', b'\n') == published.replace(b'\r\n', b'\n'), rel
        digests[rel] = digest
        if digest != runtime_digest:
            line_endings[rel] = dict(runtime_sha256=runtime_digest, public_sha256=digest,
                lf_sha256=hashlib.sha256(runtime.replace(b'\r\n', b'\n')).hexdigest(),
                difference='CRLF versus LF only; both original byte hashes retained')
    return digests, line_endings


def freeze(out, phase):
    out = Path(out).resolve()
    registration = read(out / 'preparation-registration.json')
    assert phase in ('preparation', 'formal')
    target = out / (phase + '-freeze.json')
    assert not target.exists(), 'A frozen phase cannot be overwritten'
    files = registration['fixed_runtime_sources_sha256'].copy()
    for path in PACKAGE.iterdir():
        if path.is_file() and path.suffix in ('.py', '.md'):
            files[path.relative_to(ROOT).as_posix()] = sha(path)
    value = dict(phase=phase, runtime_files=files,
                 registration_sha256=sha(out / 'preparation-registration.json'),
                 protection_sha256=sha(out / 'protection-before.json'))
    value['public_source_sha256'], value['line_ending_mappings'] = public_sources(registration, files)
    if phase == 'formal':
        verify(out, 'preparation')
        tested = read(out / 'preparation-freeze.json')
        assert files == tested['runtime_files'], 'Formal source must equal the tested preparation source'
        assert value['public_source_sha256'] == tested['public_source_sha256']
        assert value['line_ending_mappings'] == tested['line_ending_mappings']
        prep = read(out / 'preparation-summary.json')
        assert prep['passed'] and prep['sample_only']
        assert prep['freeze_sha256'] == sha(out / 'preparation-freeze.json')
        value['preparation_freeze_sha256'] = sha(out / 'preparation-freeze.json')
        value['preparation_summary_sha256'] = sha(out / 'preparation-summary.json')
        value['sample_manifest_sha256'] = sha(out / 'sample-manifest.json')
        value['reference_receipt_sha256'] = sha(out / 'preparation/reference/reference-receipt.json')
        value['preparation_artifacts_sha256'] = {
            p.relative_to(out).as_posix(): sha(p)
            for p in (out / 'preparation').rglob('*') if p.is_file()
        }
        prefix = (out / 'budget-reservations.jsonl').read_bytes()
        value['preparation_budget_prefix'] = dict(bytes=len(prefix), sha256=hashlib.sha256(prefix).hexdigest())
    save(target, value)
    verify(out, phase)
    return value


def verify(out, phase):
    out = Path(out).resolve()
    cfg = read(out / 'preparation-registration.json')
    f = read(out / (phase + '-freeze.json'))
    assert sha(out / 'preparation-registration.json') == f['registration_sha256']
    assert sha(out / 'protection-before.json') == f['protection_sha256']
    assert cfg['sample_only'] and cfg['full_diagnostics_prohibited']
    assert cfg['policy_calls'] == cfg['training_runs'] == cfg['new_games'] == cfg['official_transitions'] == 0
    for rel, digest in f['runtime_files'].items():
        assert sha(ROOT / rel) == digest, rel
    assert sha(ROOT / cfg['task_model']) == cfg['task_model_sha256']
    for name, digest in cfg['source_metadata_sha256'].items():
        assert sha(ROOT / cfg['source_dataset'] / name) == digest, name
    if phase == 'formal':
        assert sha(out / 'preparation-freeze.json') == f['preparation_freeze_sha256']
        assert f['runtime_files'] == read(out / 'preparation-freeze.json')['runtime_files']
        assert f['public_source_sha256'] == read(out / 'preparation-freeze.json')['public_source_sha256']
        for field, rel in [('preparation_summary_sha256', 'preparation-summary.json'),
                           ('sample_manifest_sha256', 'sample-manifest.json'),
                           ('reference_receipt_sha256', 'preparation/reference/reference-receipt.json')]:
            assert sha(out / rel) == f[field], rel
        for rel, digest in f['preparation_artifacts_sha256'].items():
            assert sha(out / rel) == digest, rel
        prefix = f['preparation_budget_prefix']
        raw = (out / 'budget-reservations.jsonl').read_bytes()
        assert len(raw) >= prefix['bytes'] and hashlib.sha256(raw[:prefix['bytes']]).hexdigest() == prefix['sha256']
    return cfg


def ready(out):
    out = Path(out).resolve()
    cfg = verify(out, 'formal')
    r = read(out / 'pre-review.json')
    assert r['passed'] and r['freeze_sha256'] == sha(out / 'formal-freeze.json')
    commit = read(out / 'pre-execution-commit.json')['commit']
    for name in ('preparation-registration.json', 'formal-freeze.json', 'pre-review.json'):
        assert git('show', commit + ':review/evidence/' + out.name + '/' + name) == (out / name).read_bytes()
    for rel, digest in read(out / 'formal-freeze.json')['public_source_sha256'].items():
        assert hashlib.sha256(git('show', commit + ':kaggriculture/' + rel)).hexdigest() == digest, rel
    return cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('freeze', 'verify'))
    parser.add_argument('--phase', choices=('preparation', 'formal'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args.out, args.phase) if args.mode == 'freeze' else verify(args.out, args.phase)
    print(dict(passed=True, phase=args.phase, mode=args.mode))
