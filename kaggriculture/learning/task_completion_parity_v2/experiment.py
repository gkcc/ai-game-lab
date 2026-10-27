"""Freeze one read-only GPU/CPU parity diagnostic on the failed BC output."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT.parent/'publication/ai-game-lab'
OUT = ROOT/'artifacts/task-completion-parity-v2-20261027'
SOURCE = Path(__file__).resolve().parent
PREVIOUS = ROOT/'artifacts/task-completion-bc-v1-20261027'
PARENT = ROOT/'artifacts/task-completion-parity-v1-20261027'
START = '2026-10-27T08:18:00+00:00'
DEADLINE = '2026-10-27T10:18:00+00:00'
MIB = 1024**2


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(MIB), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_new(path, value):
    path = Path(path).resolve()
    if not path.is_relative_to(OUT):
        raise ValueError('Output outside owned diagnostic')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')


def git(*args):
    return subprocess.check_output(['git',*args],cwd=PUBLIC)


def deadline_check():
    if datetime.now(timezone.utc) >= datetime.fromisoformat(DEADLINE):
        raise TimeoutError('Diagnostic stage deadline reached')


def validate_prior_registration(path, previous_freeze):
    if sha(path) != previous_freeze['registration_sha256']:
        raise ValueError('Previous experiment registration changed')


def prepare(out=OUT):
    out = Path(out).resolve()
    deadline_check()
    if out != OUT or (out/'registration.json').exists():
        raise ValueError('Registration is exclusive')
    decision = read(out/'root-decision.json')
    base = decision['previous_commit']
    if decision['status'] != 'approved_for_implementation_not_execution' or git('rev-parse','HEAD').decode().strip() != base or git('status','--porcelain').strip():
        raise ValueError('Unexpected public baseline or pending work')
    prefix = 'review/evidence/'+PARENT.name+'/'
    for name in ('root-result-acceptance.json','freeze.json','registration.json','run-started.json',
                 'independent-failure-review.json','start-failure-reconstruction.json','root-next-learning-decision.json'):
        if git('show',base+':'+prefix+name) != (PARENT/name).read_bytes():
            raise ValueError('Prior diagnostic failure differs from public commit')
    parent, parent_cfg, parent_freeze = (read(PARENT/name) for name in
        ('root-result-acceptance.json','registration.json','freeze.json'))
    if (not parent['verification_passed'] or parent['diagnostic_completed']
            or not parent['attempt_consumed'] or parent['gpu_transport_started']
            or parent['retry_allowed'] or sha(PARENT/'registration.json') != parent_freeze['registration_sha256']):
        raise ValueError('Expected consumed prelaunch failure is missing')
    prior, training, previous_freeze = (read(PREVIOUS/name) for name in
        ('root-result-acceptance.json','train-data-result.json','freeze.json'))
    if not prior['verification_passed'] or prior['scientific_experiment_completed'] or not training['passed']:
        raise ValueError('Expected failed fit and accepted training data are required')
    model, cache = PREVIOUS/'private/model-final.npz', PREVIOUS/'private/train.npz'
    if sha(model) != prior['unsealed_checkpoint_sha256'] or sha(cache) != training['array_sha256']:
        raise ValueError('Frozen diagnostic input identity changed')
    if sha(model) != parent_cfg['checkpoint_sha256'] or sha(cache) != parent_cfg['training_cache_sha256']:
        raise ValueError('New diagnostic inputs differ from the failed attempt')
    if cache.stat().st_size > 256*MIB or model.stat().st_size > 16*MIB:
        raise ValueError('Diagnostic inputs exceed declared limits')
    schema = PREVIOUS/'schema.json'
    if sha(schema) != previous_freeze['schema_sha256']:
        raise ValueError('Previous schema identity changed')
    previous_cfg = read(PREVIOUS/'registration.json')
    validate_prior_registration(PREVIOUS/'registration.json', previous_freeze)
    launch = PREVIOUS/'gpu-launch.local.json'
    if sha(launch) != previous_cfg['gpu_launch_config_sha256']:
        raise ValueError('Private device identity changed')
    runtime = dict(parent_freeze['runtime_files'])
    runtime.update(parent_freeze['new_source_sha256'])
    new_source = {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(SOURCE.glob('*.py'))}
    runtime.update(new_source)
    for rel, expected in runtime.items():
        if sha(ROOT/rel) != expected:
            raise ValueError('Frozen runtime changed: '+rel)
    for rel, expected in previous_freeze['protected_source_sha256'].items():
        if sha(ROOT/rel) != expected:
            raise ValueError('Protected source changed')
    cfg = dict(experiment='task-completion-parity-v2', baseline_head=base,
        created_utc=datetime.now(timezone.utc).isoformat(), task_started_utc=START,
        task_deadline_utc=DEADLINE, total_stage_seconds=7200,
        parent_failure_sha256=sha(PARENT/'root-result-acceptance.json'),
        parent_freeze_sha256=sha(PARENT/'freeze.json'),
        parent_registration_sha256=sha(PARENT/'registration.json'),
        parent_started_sha256=sha(PARENT/'run-started.json'),
        parent_review_sha256=sha(PARENT/'independent-failure-review.json'),
        parent_reconstruction_sha256=sha(PARENT/'start-failure-reconstruction.json'),
        prior_acceptance_sha256=sha(PREVIOUS/'root-result-acceptance.json'),
        prior_training_data_result_sha256=sha(PREVIOUS/'train-data-result.json'),
        prior_freeze_sha256=sha(PREVIOUS/'freeze.json'), prior_next_decision_sha256=sha(PREVIOUS/'root-next-learning-decision.json'),
        prior_registration_sha256=sha(PREVIOUS/'registration.json'),
        checkpoint_sha256=sha(model), checkpoint_bytes=model.stat().st_size,
        training_cache_sha256=sha(cache), training_cache_bytes=cache.stat().st_size,
        schema_sha256=sha(schema), private_gpu_launch_sha256=sha(launch),
        selection='First 16 completed_observed episode origins in saved training cache order; no replacement',
        max_groups=16, max_candidate_rows=8192, atol=0.0001, rtol=0.00001,
        max_cache_bytes=256*MIB, max_checkpoint_bytes=16*MIB,
        work_seconds=300, process_seconds=420, cleanup_seconds=20, log_bytes=MIB,
        training_updates=0, gradients=0, official_transitions=0, new_games=0,
        holdout_reads=0, candidate_promoted=False, retry_allowed=False,
        output_scope='aggregate errors and cleanup only; no weights, features, logits or device details',
        interpretation='Only tests present inference parity; cannot uniquely explain previous ValueError',
        prior_attempt_consumed_and_not_refunded=True,
        local_registration_not_third_party_timestamp=True)
    write_new(out/'registration.json',cfg)
    write_new(out/'freeze.json',dict(registration_sha256=sha(out/'registration.json'),
        root_decision_sha256=sha(out/'root-decision.json'), runtime_files=runtime,
        new_source_sha256=new_source, protected_source_sha256=previous_freeze['protected_source_sha256'],
        prior_acceptance_sha256=sha(PREVIOUS/'root-result-acceptance.json')))
    return verify(out)


def verify(out=OUT):
    out = Path(out).resolve()
    deadline_check()
    if out != OUT:
        raise ValueError('Unexpected diagnostic root')
    cfg, freeze = read(out/'registration.json'), read(out/'freeze.json')
    if sha(out/'registration.json') != freeze['registration_sha256'] or sha(out/'root-decision.json') != freeze['root_decision_sha256']:
        raise ValueError('Frozen registration changed')
    if {p.relative_to(ROOT).as_posix() for p in SOURCE.glob('*.py')} != set(freeze['new_source_sha256']):
        raise ValueError('Frozen diagnostic source set changed')
    for rel, expected in freeze['runtime_files'].items():
        if sha(ROOT/rel) != expected:
            raise ValueError('Frozen runtime changed: '+rel)
    for rel, expected in freeze['protected_source_sha256'].items():
        if sha(ROOT/rel) != expected:
            raise ValueError('Protected source changed')
    for name,key in (('root-result-acceptance.json','parent_failure_sha256'),
                     ('freeze.json','parent_freeze_sha256'),
                     ('registration.json','parent_registration_sha256'),
                     ('run-started.json','parent_started_sha256'),
                     ('independent-failure-review.json','parent_review_sha256'),
                     ('start-failure-reconstruction.json','parent_reconstruction_sha256')):
        if sha(PARENT/name) != cfg[key]:
            raise ValueError('Prior diagnostic failure evidence changed')
    for name,key in (('root-result-acceptance.json','prior_acceptance_sha256'),
                     ('train-data-result.json','prior_training_data_result_sha256'),
                     ('freeze.json','prior_freeze_sha256'),
                     ('registration.json','prior_registration_sha256'),
                     ('root-next-learning-decision.json','prior_next_decision_sha256'),
                     ('schema.json','schema_sha256'),
                     ('gpu-launch.local.json','private_gpu_launch_sha256'),
                     ('private/model-final.npz','checkpoint_sha256'),
                     ('private/train.npz','training_cache_sha256')):
        if sha(PREVIOUS/name) != cfg[key]:
            raise ValueError('Frozen diagnostic input changed')
    if (cfg['max_groups'] != 16 or cfg['max_candidate_rows'] != 8192
            or cfg['atol'] != 0.0001 or cfg['rtol'] != 0.00001):
        raise ValueError('Diagnostic budget or threshold changed')
    return cfg


def ready(out=OUT):
    out = Path(out).resolve()
    cfg = verify(out)
    review, receipt = read(out/'pre-review.json'), read(out/'pre-execution-commit.json')
    if (not review['passed'] or review['freeze_sha256'] != sha(out/'freeze.json')
            or receipt['freeze_sha256'] != sha(out/'freeze.json')
            or receipt['registration_sha256'] != sha(out/'registration.json')):
        raise ValueError('Review or local commitment does not match freeze')
    return cfg


def verify_commit(out=OUT):
    out = Path(out).resolve()
    receipt = read(out/'pre-execution-commit.json')
    for name in ('root-decision.json','registration.json','freeze.json','pre-review.json'):
        if git('show',receipt['commit']+':review/evidence/'+out.name+'/'+name) != (out/name).read_bytes():
            raise ValueError('Committed preregistration differs')
    for rel in read(out/'freeze.json')['new_source_sha256']:
        if git('show',receipt['commit']+':kaggriculture/'+rel) != (ROOT/rel).read_bytes():
            raise ValueError('Committed diagnostic source differs')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=('prepare','verify'))
    args = parser.parse_args()
    cfg = prepare() if args.command == 'prepare' else verify()
    print(json.dumps(dict(experiment=cfg['experiment'],passed=True)))
