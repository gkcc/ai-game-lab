"""Freeze one post-hoc callback probe without opening trace or model content."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT.parent/'publication/ai-game-lab'
OUT = ROOT/'artifacts/task-completion-score-probe-v1-20261027'
PARENT = ROOT/'artifacts/task-completion-worst-pair-v1-20261027'
RAW_PARENT = ROOT/'artifacts/task-completion-economy-v1-20261027'
SOURCE = Path(__file__).resolve().parent
BASE = '17469e03e7aaf9db5cb0d40fbb10f818d8bde903'
EVIDENCE = ('root-result-acceptance.json','registration.json','freeze.json','analysis-result.json',
            'independent-result-assessment.json','root-next-learning-decision.json')
LIMITS = dict(raw_files=2,decode_passes_per_file=1,compressed_bytes=128*1024**2,
              decoded_bytes=1024**3,max_line_bytes=16*1024**2,max_candidates=8192,max_cpu_forwards=1,seconds=300)


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def git(*args): return subprocess.check_output(['git',*args],cwd=PUBLIC)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''): h.update(block)
    return h.hexdigest()


def check_identity(path, expected):
    if sha(path) != expected: raise ValueError('Bound identity changed')


def write_new(path,value):
    path = Path(path).resolve()
    if not path.is_relative_to(OUT): raise ValueError('Output outside experiment')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')


def deadline_check():
    deadline = datetime.fromisoformat(read(OUT/'root-decision.json')['deadline_utc'])
    if deadline.tzinfo is None or datetime.now(timezone.utc) >= deadline: raise TimeoutError('Invalid or expired deadline')


def clean_head(expected):
    if git('rev-parse','HEAD').decode().strip() != expected or git('status','--porcelain').strip():
        raise ValueError('Unexpected or dirty public HEAD')


def raw_manifest(ledger, registered):
    from learning.task_completion_bc_v1.collect import current_game
    from learning.task_action_observation_v1.analysis import raw_path
    result = {}
    for arm in ('candidate','teacher_task'):
        rows = [r for r in ledger if r['seed']==3342200684 and r['side']==0 and r['opponent']=='round2' and r['policy']==arm]
        if len(rows) != 1: raise ValueError('Fixed pair is not unique')
        game, _ = current_game(RAW_PARENT,rows[0])
        identity = game['trace']
        if identity != registered[arm]['trace']: raise ValueError('Registered trace identity changed')
        path = raw_path(RAW_PARENT,identity,'trace')
        if path.stat().st_size != identity['size']: raise ValueError('Raw size changed')
        result[arm] = dict(identity)
    return result


def _environment(cfg):
    check_identity(ROOT/cfg['candidate_checkpoint'],cfg['candidate_checkpoint_sha256'])
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
    if decision.get('previous_commit') != BASE or decision.get('status') != 'approved_for_implementation_not_execution': raise ValueError('Invalid decision')
    clean_head(BASE)
    for name in EVIDENCE:
        if git('show',BASE+':review/evidence/'+PARENT.name+'/'+name) != (PARENT/name).read_bytes():
            raise ValueError('Parent evidence differs from publication')
    result, old_cfg, old_freeze, analysis = (read(PARENT/n) for n in EVIDENCE[:4])
    if (not result['passed'] or not result['post_hoc_not_holdout'] or result['seed'] != 3342200684
        or result['side'] != 0 or result['first_request_difference_step'] != 2
        or result['model_and_teacher_operations'] != ['PLANT','PLANT'] or result['same_unit'] is not False
        or not result['same_input_choice_difference_coincides_with_request_difference']
        or result['requested_operation'] != 'NORTH' or result['official_status'] != 'executed'
        or not analysis['passed'] or analysis['first_request_difference_step'] != 2
        or analysis['linked_choice']['candidate_operation'] != 'PLANT'
        or analysis['linked_choice']['teacher_operation'] != 'PLANT'
        or analysis['linked_choice']['same_unit'] is not False
        or analysis['official_event']['requested_operation'] != 'NORTH'
        or not read(PARENT/'independent-result-assessment.json')['passed']):
        raise ValueError('Expected post-hoc diagnosis not accepted')
    for name in EVIDENCE[1:-1]: check_identity(PARENT/name,result['evidence_sha256'][name])
    check_identity(PARENT/'registration.json',old_freeze['registration_sha256'])
    if (old_cfg['seed'],old_cfg['side'],old_cfg['opponent']) != (3342200684,0,'round2'):
        raise ValueError('Parent selection changed')
    raw_ledger_sha = old_cfg['parent_evidence']['game-ledger.json']
    check_identity(RAW_PARENT/'game-ledger.json',raw_ledger_sha)
    manifest = raw_manifest(read(RAW_PARENT/'game-ledger.json'),old_cfg['raw_inputs'])
    runtime = dict(old_freeze['runtime_files'])
    source = {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(SOURCE.glob('*.py'))}
    runtime.update(source)
    for rel,digest in runtime.items(): check_identity(ROOT/rel,digest)
    _environment(old_cfg)
    cfg = dict(experiment='task-completion-score-probe-v1',baseline_head=BASE,deadline_utc=decision['deadline_utc'],
        seed=3342200684,side=0,opponent='round2',step=2,limits=dict(LIMITS),raw_inputs=manifest,
        parent_evidence={n:sha(PARENT/n) for n in EVIDENCE},post_hoc_not_holdout=True,
        selection='first matched differing callback on previously selected post-hoc worst pair',raw_ledger_sha256=raw_ledger_sha,
        candidate_checkpoint=old_cfg['candidate_checkpoint'],candidate_checkpoint_sha256=old_cfg['candidate_checkpoint_sha256'],
        crop_checkpoint=old_cfg['crop_checkpoint'],crop_checkpoint_sha256=old_cfg['crop_checkpoint_sha256'],
        engine_sha256=old_cfg['engine_sha256'],official_loader_sha256=old_cfg['official_loader_sha256'],
        runtime_versions=old_cfg['runtime_versions'],new_games=0,training_updates=0,
        max_model_inference_calls=1,model_inference_calls_before_execution=0,
        attempts=1,retry_allowed=False,official_transitions=0,gpu_forwards=0,
        trace_decoding='Both complete trace streams decoded and hash checked; only step 2 JSON parsed',
        time_limit_semantics='Checked between chunks; decompression and JSON parsing are not preempted')
    write_new(out/'registration.json',cfg)
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
    if (cfg['seed'],cfg['side'],cfg['opponent'],cfg['step']) != (3342200684,0,'round2',2) or cfg['limits'] != LIMITS:
        raise ValueError('Scope changed')
    check_identity(RAW_PARENT/'game-ledger.json',cfg['raw_ledger_sha256'])
    if raw_manifest(read(RAW_PARENT/'game-ledger.json'),read(PARENT/'registration.json')['raw_inputs']) != cfg['raw_inputs']:
        raise ValueError('Raw manifest changed')
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
    cfg = {'prepare':prepare,'verify':verify,'ready':ready,'verify_commit':verify_commit}[args.command]()
    print(json.dumps(dict(experiment=cfg['experiment'],passed=True)))
