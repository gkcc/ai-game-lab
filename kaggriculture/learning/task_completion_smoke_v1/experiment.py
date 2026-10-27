"""Freeze a single paired official-entry smoke panel; no execution here."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT.parent/'publication/ai-game-lab'
OUT = ROOT/'artifacts/task-completion-smoke-v1-20261027'
PARENT = ROOT/'artifacts/task-completion-bc-reentry-v1-20261027'
OLD = ROOT/'artifacts/task-completion-bc-v1-20261027'
SOURCE = Path(__file__).resolve().parent
BASE = '21155e496cd29fb5f2f0d3163fae61f2d1da5f47'
EXCLUDED = {1210575027, 441357442, 2026102708, 3911611436, 2328454240}
MIB = 1024**2
EVIDENCE = ('root-result-acceptance.json','registration.json','freeze.json','fit-result.json',
            'fit-transport-result.json','fit-independent-review.json','model-seal.json',
            'evaluation-result.json','evaluation-independent-review.json','independent-result-review.json')


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(MIB), b''): h.update(block)
    return h.hexdigest()


def git(*args): return subprocess.check_output(['git',*args],cwd=PUBLIC)


def write_new(path, value):
    path = Path(path).resolve()
    if not path.is_relative_to(OUT): raise ValueError('Output outside experiment')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')


def deadline_check():
    deadline = datetime.fromisoformat(read(OUT/'root-decision.json')['deadline_utc'])
    if deadline.tzinfo is None or datetime.now(timezone.utc) >= deadline:
        raise TimeoutError('Stage deadline reached or invalid')


def clean_head(expected):
    if git('rev-parse','HEAD').decode().strip() != expected or git('status','--porcelain').strip():
        raise ValueError('Unexpected or dirty public HEAD')


def select_seed(seen):
    excluded = set(seen) | EXCLUDED
    index = 0
    while True:
        value = int.from_bytes(hashlib.sha256(f'task-completion-smoke-v1:20261027:{index}'.encode()).digest()[:4], 'big')
        if value not in excluded: return value
        index += 1


def seed_inventory(excluded):
    from learning.bc_crop.prepare import TEXT_SUFFIXES
    names = {'registration.json','input-manifest.json','root-decision.json','seed-inventory.local.json','seed-inventory.json','seeds.json'}
    paths = [('public/'+n,PUBLIC/n) for n in git('ls-files','-z').decode().split('\0') if n and Path(n).suffix.lower() in TEXT_SUFFIXES]
    paths += [('local/'+p.relative_to(ROOT/'artifacts').as_posix(),p) for p in sorted((ROOT/'artifacts').rglob('*.json'))
              if not p.is_relative_to(OUT) and (p.name.lower() in names or p.parent.name == 'job-inputs')]
    seen, identities, size = set(excluded)|EXCLUDED, [], 0
    pattern = re.compile(r'(?<!\d)\d{1,10}(?!\d)')
    for label,path in paths:
        deadline_check()
        size += path.stat().st_size
        if size > 256*MIB: raise ValueError('Seed inventory budget exceeded')
        data = path.read_bytes()
        seen.update(int(m.group()) for m in pattern.finditer(data.decode('utf-8-sig')) if int(m.group()) < 2**32)
        identities.append([label,hashlib.sha256(data).hexdigest()])
    return dict(test_seeds=[select_seed(seen)],excluded_uint32=sorted(seen),identities=identities,input_bytes=size,
        scope='Tracked public text plus local registration, seed inventories, input manifests and job-input metadata',
        limitation='Deleted, unavailable, external and unlisted seed history cannot be searched')


def check_identity(path, expected):
    if sha(path) != expected: raise ValueError('Bound identity changed')


def _environment(cfg):
    check_identity(ROOT/cfg['crop_checkpoint'],cfg['crop_checkpoint_sha256'])
    if os.name == 'nt':
        from learning.task_entry_repair_v1.experiment import installed_files
        from learning.task_reward_gate_v1.experiment import versions
        installed = installed_files()
        if (versions() != cfg['runtime_versions'] or sha(installed['rules']) != cfg['engine_sha256']
                or sha(installed['loader']) != cfg['official_loader_sha256']):
            raise ValueError('Official environment changed')


def prepare(out=OUT):
    out = Path(out).resolve()
    if os.name != 'nt': raise ValueError('Registration requires Windows')
    if out != OUT or (out/'registration.json').exists() or (out/'freeze.json').exists(): raise ValueError('Registration is exclusive')
    deadline_check()
    decision = read(out/'root-decision.json')
    if decision.get('previous_commit') != BASE or decision.get('status') != 'approved_for_implementation_not_execution':
        raise ValueError('Invalid root decision')
    clean_head(BASE)
    for name in EVIDENCE:
        if git('show',BASE+':review/evidence/'+PARENT.name+'/'+name) != (PARENT/name).read_bytes():
            raise ValueError('Published parent evidence changed')
    result, cfg, freeze, fit, fit_review, seal, evaluation, eval_review, independent = (read(PARENT/n) for n in EVIDENCE if n != 'fit-transport-result.json')
    if not all(x['passed'] for x in (result,fit,seal,evaluation,fit_review,eval_review,independent)) or not result['training_accepted']:
        raise ValueError('Parent result not accepted')
    for name in EVIDENCE[1:]:
        if result['evidence_sha256'].get(name) != sha(PARENT/name): raise ValueError('Parent evidence hash chain changed')
    if (freeze['registration_sha256'] != sha(PARENT/'registration.json') or seal['freeze_sha256'] != sha(PARENT/'freeze.json')
        or seal['fit_result_sha256'] != sha(PARENT/'fit-result.json') or seal['fit_transport_sha256'] != sha(PARENT/'fit-transport-result.json')
        or fit['checkpoint_sha256'] != seal['checkpoint_sha256']):
        raise ValueError('Parent fit seal chain changed')
    for review in (fit_review,eval_review):
        for name in ('registration.json','freeze.json','fit-result.json','model-seal.json'):
            if review['identities_sha256'][name] != sha(PARENT/name): raise ValueError('Independent review identity changed')
    if eval_review['identities_sha256']['evaluation-result.json'] != sha(PARENT/'evaluation-result.json'):
        raise ValueError('Evaluation review identity changed')
    model, old = PARENT/'private/model-final.npz', OLD/'private/model-final.npz'
    check_identity(model,seal['checkpoint_sha256'])
    check_identity(old,cfg['old_model_sha256'])
    runtime = dict(freeze['runtime_files'])
    runtime.update(freeze.get('protected_source_sha256',{}))
    source = {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(SOURCE.glob('*.py'))}
    runtime.update(source)
    for rel,digest in runtime.items(): check_identity(ROOT/rel,digest)
    _environment(cfg)
    excluded = set(cfg['train_seeds']) | set(cfg['test_seeds']) | EXCLUDED
    inventory = seed_inventory(excluded)
    market = PARENT/'sources/market.py'
    check_identity(market,cfg['source_sha256']['market.py'])
    target = out/'sources/market.py'
    target.parent.mkdir(parents=True,exist_ok=True)
    with market.open('rb') as src, target.open('xb') as dst:
        for block in iter(lambda:src.read(MIB),b''): dst.write(block)
    check_identity(target,cfg['source_sha256']['market.py'])
    write_new(out/'seed-inventory.local.json',inventory)
    new = dict(experiment='task-completion-smoke-v1',baseline_head=BASE,deadline_utc=decision['deadline_utc'],
        test_seeds=inventory['test_seeds'],train_seeds=cfg['train_seeds'],excluded_seeds=sorted(excluded),
        seed_inventory_sha256=sha(out/'seed-inventory.local.json'),seed_scope=inventory['scope'],seed_limitation=inventory['limitation'],
        candidate_checkpoint=model.relative_to(ROOT).as_posix(),candidate_checkpoint_sha256=seal['checkpoint_sha256'],
        old_model_sha256=cfg['old_model_sha256'],parent_evidence={n:sha(PARENT/n) for n in EVIDENCE},
        crop_checkpoint=cfg['crop_checkpoint'],crop_checkpoint_sha256=cfg['crop_checkpoint_sha256'],
        source_sha256={'market.py':cfg['source_sha256']['market.py']},runtime_versions=cfg['runtime_versions'],
        engine_commit=cfg['engine_commit'],engine_sha256=cfg['engine_sha256'],official_loader_sha256=cfg['official_loader_sha256'],
        game_limits=cfg['game_limits'],storage=cfg['storage'],policies=['candidate','teacher_task'],opponents=['market'],seats=[0,1],
        official_limits=dict(new_games=4,live_transitions=4*719,primary_replay_transitions=4*719,
            independent_replay_transitions=719,total_transitions=9*719,per_job_seconds=cfg['official_limits']['per_job_seconds']),
        independent_selection='candidate, market, seat 0; first 20 callback identities by SHA256, independent of choice and outcome',
        independent_limits=dict(raw_files=2,decode_passes_per_file=1,compressed_bytes=128*MIB,
            decoded_bytes=1024*MIB,seconds=300,official_transitions=719,sample_count=20,attempts=1),
        analysis_limits=dict(replays=4,traces=4,decode_passes_per_file=1,
            compressed_bytes=256*MIB,decoded_bytes=4*1024*MIB,seconds=600,attempts=1,official_transitions=0),
        analysis_time_limit_semantics='Checked between file reads, trace records and official events; an individual decompression or JSON parse is not preempted.',
        acceptance=dict(complete_games=4,states=720,actions=719,zero_agent_errors=True,primary_replay_each=True,
            independent_replay=1,candidate_task_forwards_each_positive=True,teacher_task_forwards_zero=True,
            influence='At least one candidate/reference pair has a differing model choice with identical callback input at the first complete-request difference on the full history prefix',
            no_influence='Report successful execution with unobserved intervention; do not claim control of actions',
            no_game_strength_claim=True,no_promotion=True,stop_on_error=True),
        training_updates=0,retry_allowed=False,stop_on_failure=True,prior_checkpoint_bytes_equal=result['checkpoint_identical_to_older_unsealed_file'])
    write_new(out/'registration.json',new)
    write_new(out/'freeze.json',dict(registration_sha256=sha(out/'registration.json'),root_decision_sha256=sha(out/'root-decision.json'),
        runtime_files=runtime,new_source_sha256=source,parent_commit=BASE))
    return verify(out)


def verify(out=OUT):
    out = Path(out).resolve()
    if out != OUT: raise ValueError('Unexpected experiment root')
    deadline_check()
    receipt = out/'pre-execution-commit.json'
    clean_head(read(receipt)['commit'] if receipt.exists() else BASE)
    cfg,freeze = read(out/'registration.json'),read(out/'freeze.json')
    check_identity(out/'registration.json',freeze['registration_sha256'])
    check_identity(out/'root-decision.json',freeze['root_decision_sha256'])
    if {p.relative_to(ROOT).as_posix() for p in SOURCE.glob('*.py')} != set(freeze['new_source_sha256']): raise ValueError('Source set changed')
    for rel,digest in freeze['runtime_files'].items(): check_identity(ROOT/rel,digest)
    for name,digest in cfg['parent_evidence'].items(): check_identity(PARENT/name,digest)
    check_identity(PARENT/'private/model-final.npz',cfg['candidate_checkpoint_sha256'])
    if cfg['candidate_checkpoint'] != (PARENT/'private/model-final.npz').relative_to(ROOT).as_posix(): raise ValueError('Candidate path changed')
    check_identity(OLD/'private/model-final.npz',cfg['old_model_sha256'])
    check_identity(out/'sources/market.py',cfg['source_sha256']['market.py'])
    check_identity(PARENT/'sources/market.py',cfg['source_sha256']['market.py'])
    check_identity(out/'seed-inventory.local.json',cfg['seed_inventory_sha256'])
    seeds = cfg['test_seeds']
    if len(seeds) != 1 or seeds[0] in set(cfg['excluded_seeds'])|EXCLUDED: raise ValueError('Seed isolation changed')
    _environment(cfg)
    return cfg


def ready(out=OUT):
    cfg = verify(out)
    review,receipt = read(Path(out)/'pre-review.json'),read(Path(out)/'pre-execution-commit.json')
    if (not review['passed'] or review['freeze_sha256'] != sha(Path(out)/'freeze.json')
        or receipt['freeze_sha256'] != sha(Path(out)/'freeze.json') or receipt['registration_sha256'] != sha(Path(out)/'registration.json')):
        raise ValueError('Review or commitment changed')
    verify_commit(out)
    return cfg


def verify_commit(out=OUT):
    out = Path(out).resolve()
    cfg = verify(out)
    commit = read(out/'pre-execution-commit.json')['commit']
    clean_head(commit)
    if git('merge-base','--is-ancestor',BASE,commit) != b'': raise ValueError('Invalid ancestry')
    for name in ('root-decision.json','registration.json','freeze.json','pre-review.json'):
        if git('show',commit+':review/evidence/'+out.name+'/'+name) != (out/name).read_bytes(): raise ValueError('Committed evidence changed')
    for rel in read(out/'freeze.json')['new_source_sha256']:
        if git('show',commit+':kaggriculture/'+rel) != (ROOT/rel).read_bytes(): raise ValueError('Committed source changed')
    return cfg


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=('prepare','verify','ready','verify_commit'))
    args = parser.parse_args()
    result = {'prepare':prepare,'verify':verify,'ready':ready,'verify_commit':verify_commit}[args.command]()
    print(json.dumps(dict(experiment=result['experiment'],passed=True)))
