"""Prepare new holdouts or explicitly non-independent historical reproductions.

The original experiment.py is frozen evidence and remains byte-identical.
This entry creates registrations only; it never trains or launches games.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

from .agent import ROOT, TEACHER_SHA, sha
from .experiment import ENGINE_COMMIT, ENGINE_SHA, PANEL, read, save

SPLITS = {'train_seeds': 8, 'validation_seeds': 2, 'test_seeds': 8}
SEED_KEYS = {*SPLITS, 'parity_seed'}
KINDS = ('new-holdout', 'historical-replay')
TEXT_SUFFIXES = {'.json', '.py', '.md', '.csv', '.txt', '.yaml', '.yml'}


def seed_config(value):
    if not isinstance(value, dict) or set(value) != SEED_KEYS:
        raise ValueError('Seed config requires exactly train_seeds, validation_seeds, test_seeds, parity_seed')
    for key, count in SPLITS.items():
        if not isinstance(value[key], list) or len(value[key]) != count:
            raise ValueError(f'{key} must contain exactly {count} seeds')
    seeds = [s for key in SPLITS for s in value[key]] + [value['parity_seed']]
    if any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError('Seeds must be integers in [0, 2**32); booleans are not integers here')
    if len(set(seeds)) != len(seeds):
        raise ValueError('Seeds must be unique within and across all splits, including parity')
    return deepcopy(value)


def git(public, *args):
    return subprocess.check_output(['git', '-C', str(public), *args], encoding='utf-8').strip()


def novelty(seeds, public, tracked, history_root):
    """Conservative decimal-occurrence scan; this is not a global novelty proof."""
    candidates = {s for key in SPLITS for s in seeds[key]} | {seeds['parity_seed']}
    pattern = re.compile(r'(?<!\d)(?:' + '|'.join(str(s) for s in sorted(candidates)) + r')(?!\d)')
    files = [('public/' + name, public/name) for name in tracked
             if Path(name).suffix.lower() in TEXT_SUFFIXES]
    # Include local-only matches and experiment registrations, not privileged replays.
    if not history_root.is_dir():
        raise ValueError('Local history root must exist; cannot silently skip local history')
    local = sorted(p for p in history_root.rglob('*.json')
                   if re.search(r'(registration|manifest|ledger|config|result|summary|seed)', p.name, re.I)
                   and not p.name.lower().startswith('replay'))
    files += [('local/' + p.relative_to(history_root).as_posix(), p) for p in local]
    found = set()
    identities = []
    for label, path in files:
        raw = path.read_bytes()
        found.update(int(m.group()) for m in pattern.finditer(raw.decode('utf-8-sig', errors='strict')))
        identities.append((label, hashlib.sha256(raw).hexdigest()))
    if found:
        raise ValueError('Previously recorded seed(s): ' + ', '.join(map(str, sorted(found))))
    return dict(passed=True, public_text_files=len(files)-len(local), local_evidence_files=len(local),
                scanned_manifest_sha256=hashlib.sha256(json.dumps(identities, separators=(',', ':')).encode()).hexdigest(),
                scope='Current tracked public text plus local JSON filenames containing registration/manifest/ledger/config/result/summary/seed',
                limitation='Conservative decimal occurrences; no claim about deleted Git history, raw replay-only seeds, other machines or external experiments')


def prepare(out, public, kind, reference_registration, seeds_path=None):
    out, public = out.resolve(), public.resolve()
    if out.exists():
        raise FileExistsError('Output already exists; historical results must not be overwritten')
    if kind not in KINDS:
        raise ValueError('Choose new-holdout or historical-replay explicitly')
    if (kind == 'new-holdout') != (seeds_path is not None):
        raise ValueError('Only new-holdout requires --seed-config; historical-replay takes the reference splits unchanged')
    reference_bytes = reference_registration.read_bytes()
    cfg = json.loads(reference_bytes)
    if kind == 'new-holdout' and (cfg.get('evaluation_role') == 'historical-replay'
                                  or cfg.get('new_independent_evidence') is False):
        raise ValueError('A historical-replay registration cannot supply new-holdout acceptance criteria; use its original reference registration')
    old_splits = seed_config({key: cfg[key] for key in SEED_KEYS})
    seeds_bytes = seeds_path.read_bytes() if seeds_path else None
    splits = seed_config(json.loads(seeds_bytes)) if seeds_path is not None else old_splits
    if (cfg['engine_commit'], cfg['engine_sha256'], cfg['teacher_sha256']) != (ENGINE_COMMIT, ENGINE_SHA, TEACHER_SHA):
        raise ValueError('Reference engine/teacher identity differs from the fixed BC experiment')
    if cfg['opponents'] != list(PANEL) or cfg['seats'] != [0, 1] or cfg['formal_games'] != 144 or cfg['demonstration_games'] != 60:
        raise ValueError('Reference panel differs from the supported bounded experiment')
    # Verify all source bytes before creating any output. No copying changed policies.
    sources = {'teacher.py': ROOT/'bots/teacher_fertilizer_terminal_v2.py',
               'worker.py': ROOT/'learning/round2_worker.py',
               **{n+'.py': (ROOT/p if (ROOT/p).exists() else public/'kaggriculture'/p) for n, p in PANEL.items()}}
    source_bytes = {name: path.read_bytes() for name, path in sources.items()}
    identities = {name: hashlib.sha256(raw).hexdigest() for name, raw in source_bytes.items()}
    if identities != cfg['source_sha256'] or identities['teacher.py'] != TEACHER_SHA:
        raise ValueError('Reference policy/worker source identities do not match current bytes')
    rule_path = 'kaggle_environments/envs/kaggriculture/kaggriculture.py'
    official = subprocess.check_output(['git', '-C', str(ROOT/'upstream'), 'show', ENGINE_COMMIT+':'+rule_path])
    checkout = (ROOT/'upstream'/rule_path).read_bytes()
    # Windows Git checkout uses CRLF; the official blob and installed package
    # use LF. Record both, and accept ONLY this explicitly checked mapping.
    engines = [ROOT/'.venv/Lib/site-packages'/rule_path]
    if not engines[0].is_file():
        engines = list((ROOT/'.venv/lib').glob('python*/site-packages/'+rule_path))
    if (hashlib.sha256(official).hexdigest() != ENGINE_SHA or checkout.replace(b'\r\n', b'\n') != official
            or len(engines) != 1 or sha(engines[0]) != ENGINE_SHA):
        raise ValueError('Official rule source or installed engine hash mismatch')
    tracked = [p for p in git(public, 'ls-files', '-z').split('\0') if p]
    protected = {name: sha(public/name) for name in tracked}
    scan = novelty(splits, public, tracked, ROOT/'artifacts') if kind == 'new-holdout' else None
    # The reference registration is copied verbatim for reproducible policy and
    # training settings. Only run identity, seed splits and interpretation change.
    cfg.update(splits)
    cfg.update(created_utc=datetime.now(timezone.utc).isoformat(), baseline_head=git(public, 'rev-parse', 'HEAD'),
               initial_worktree_clean=not bool(git(public, 'status', '--porcelain')),
               preparation_schema=2, evaluation_role=kind, new_independent_evidence=kind == 'new-holdout',
               reference_registration_sha256=hashlib.sha256(reference_bytes).hexdigest(),
               seed_config_sha256=hashlib.sha256(seeds_bytes).hexdigest() if seeds_bytes else None,
               seed_novelty=scan if scan else 'Historical splits reused exactly; no new independent evidence',
               preparation_source_sha256=sha(Path(__file__)),
               official_checkout_raw_sha256=hashlib.sha256(checkout).hexdigest(),
               official_checkout_mapping='CRLF-to-LF equals pinned Git blob; installed game-engine bytes equal pinned SHA256',
               reproduction_scope='Reference seeds, policies, engine, training and bootstrap settings; does not promise identical trained weights or reproduce historical source checkout')
    if kind == 'historical-replay':
        cfg['acceptance_B'] = 'Historical reproduction only: metric gates may be recomputed, but cannot establish new independent imitation evidence'
    runtime = [ROOT/'main.py', *sorted((ROOT/'bots').glob('*.py')), *sorted((ROOT/'models').glob('*.py'))]
    protection = dict(public_files=protected, runtime_policies={p.relative_to(ROOT).as_posix(): sha(p) for p in runtime})
    # Validation failures above leave no empty/partial run. Exclusive mkdir also
    # protects against a concurrently created destination.
    out.mkdir(parents=True, exist_ok=False)
    (out/'sources').mkdir()
    for name, raw in source_bytes.items():
        (out/'sources'/name).write_bytes(raw)
    (out/'reference-registration.json').write_bytes(reference_bytes)
    save(out/'registration.json', cfg)
    save(out/'protection.json', protection)
    return dict(prepared=True, evaluation_role=kind, new_independent_evidence=cfg['new_independent_evidence'],
                registration_sha256=sha(out/'registration.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=KINDS, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--public', type=Path, required=True)
    parser.add_argument('--reference-registration', type=Path, required=True)
    parser.add_argument('--seed-config', type=Path)
    args = parser.parse_args()
    try:
        result = prepare(args.out, args.public, args.kind, args.reference_registration, args.seed_config)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f'Preparation rejected: {exc}\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
