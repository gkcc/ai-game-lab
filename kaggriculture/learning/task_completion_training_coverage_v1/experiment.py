"""Register a metadata-selected sample of one consumed training cache."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT.parent/'publication/ai-game-lab'
OUT = ROOT/'artifacts/task-completion-training-coverage-v1-20261027'
PARENT = ROOT/'artifacts/task-completion-score-probe-v1-20261027'
TRAIN = ROOT/'artifacts/task-completion-bc-v1-20261027'
REENTRY = ROOT/'artifacts/task-completion-bc-reentry-v1-20261027'
CACHE = TRAIN/'private/train.npz'
SOURCE = Path(__file__).resolve().parent
BASE = 'df4efb8ee7e192859ac2c3b1979c40a6e5ab82eb'
LIMITS = dict(max_archive_bytes=256*1024**2,max_uncompressed_bytes=512*1024**2,
              max_groups=16000,max_candidates=1200000,max_selected=200,seconds=300)
SUPERVISION = dict(max_rss_bytes=1536*1024**2,min_available_bytes=2*1024**3,
                   wall_seconds=360,poll_seconds=1,termination_wait_seconds=10)
SCORE_EVIDENCE = ('root-result-acceptance.json','registration.json','freeze.json','probe-result.json',
                  'independent-result-assessment.json','root-next-learning-decision.json')
TRAIN_EVIDENCE = ('registration.json','freeze.json','train-data-result.json','schema.json')


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def git(*args): return subprocess.check_output(['git',*args],cwd=PUBLIC)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''): h.update(block)
    return h.hexdigest()


def write_new(path,value):
    path = Path(path).resolve()
    if not path.is_relative_to(OUT): raise ValueError('Output outside experiment')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def deadline_check():
    deadline = datetime.fromisoformat(read(OUT/'root-decision.json')['deadline_utc'])
    if deadline.tzinfo is None or datetime.now(timezone.utc) >= deadline: raise TimeoutError('Invalid or expired deadline')


def clean_head(expected):
    if git('rev-parse','HEAD').decode().strip() != expected or git('status','--porcelain').strip(): raise ValueError('Unexpected or dirty public HEAD')


def source_segments(registration, training):
    source, games = registration['source_groups'], training['games']
    if not training['passed'] or len(source) != 8 or len(games) != 8: raise ValueError('Invalid training sources')
    offset, result = 0, []
    for original, done in zip(source,games):
        if (original['key'] != done['game'] or original['arm'] != done['arm']
            or original['labels_rows'] != done['groups'] or original['labels_sha256'] != done['labels_sha256']
            or original['trace']['sha256'] != done['trace_sha256']
            or type(done['groups']) is not int or done['groups'] <= 0
            or type(original['seed']) is not int or not 0 <= original['seed'] <= 0xffffffff):
            raise ValueError('Source order or counts differ')
        result.append(dict(game=done['game'],arm=done['arm'],seed=original['seed'],count=done['groups'],offset=offset))
        offset += done['groups']
    if offset != training['groups'] or offset > LIMITS['max_groups'] or len({r['game'] for r in result}) != 8:
        raise ValueError('Invalid source population')
    return result


def check_published(folder,names):
    for name in names:
        if git('show',BASE+':review/evidence/'+folder.name+'/'+name) != (folder/name).read_bytes():
            raise ValueError('Published evidence changed')


def prepare(out=OUT):
    out = Path(out).resolve()
    if os.name != 'nt': raise ValueError('Registration requires Windows')
    if out != OUT or (out/'registration.json').exists() or (out/'freeze.json').exists(): raise ValueError('Registration exclusive')
    deadline_check()
    decision = read(out/'root-decision.json')
    if decision.get('previous_commit') != BASE or decision.get('status') != 'approved_for_implementation_not_execution': raise ValueError('Invalid decision')
    clean_head(BASE)
    check_published(PARENT,SCORE_EVIDENCE)
    check_published(TRAIN,TRAIN_EVIDENCE)
    check_published(REENTRY,('registration.json','freeze.json','root-result-acceptance.json'))
    score,score_cfg,score_freeze = (read(PARENT/name) for name in SCORE_EVIDENCE[:3])
    if not score['passed'] or not score['post_hoc_not_holdout'] or not read(PARENT/'independent-result-assessment.json')['passed']:
        raise ValueError('Score stage not accepted')
    for name in SCORE_EVIDENCE[1:-1]:
        if sha(PARENT/name) != score['evidence_sha256'][name]: raise ValueError('Score evidence chain changed')
    if sha(PARENT/'registration.json') != score_freeze['registration_sha256']: raise ValueError('Score registration changed')
    registration,freeze,training = (read(TRAIN/name) for name in TRAIN_EVIDENCE[:3])
    reentry,reentry_result = read(REENTRY/'registration.json'),read(REENTRY/'root-result-acceptance.json')
    if (not reentry_result['passed'] or not reentry_result['training_accepted']
        or reentry_result['evidence_sha256']['registration.json'] != sha(REENTRY/'registration.json')
        or reentry_result['evidence_sha256']['freeze.json'] != sha(REENTRY/'freeze.json')
        or reentry['copied_files']['train-data-result.json'] != sha(TRAIN/'train-data-result.json')
        or reentry['copied_files']['schema.json'] != sha(TRAIN/'schema.json')
        or reentry['training_cache_sha256'] != training['array_sha256']):
        raise ValueError('Accepted training source chain changed')
    if sha(TRAIN/'registration.json') != freeze['registration_sha256'] or sha(TRAIN/'schema.json') != freeze['schema_sha256']:
        raise ValueError('Training registration or schema changed')
    if CACHE.stat().st_size != training['array_bytes'] or CACHE.stat().st_size > LIMITS['max_archive_bytes']:
        raise ValueError('Cache size changed or exceeds limit')
    if sha(CACHE) != training['array_sha256']: raise ValueError('Cache identity changed')
    segments = source_segments(registration,training)
    from .analysis import deterministic_select
    positions = deterministic_select(segments,200)
    runtime = dict(score_freeze['runtime_files'])
    source = {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(SOURCE.glob('*.py'))}
    runtime.update(source)
    for rel,digest in runtime.items():
        if sha(ROOT/rel) != digest: raise ValueError('Runtime changed')
    schema = TRAIN/'schema.json'
    target = out/'schema.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    with schema.open('rb') as src,target.open('xb') as dst: dst.write(src.read())
    if sha(target) != freeze['schema_sha256']: raise ValueError('Schema copy changed')
    write_new(out/'sample-positions.local.json',positions)
    cfg = dict(experiment='task-completion-training-coverage-v1',baseline_head=BASE,deadline_utc=decision['deadline_utc'],
        source_segments=segments,sample_positions_sha256=sha(out/'sample-positions.local.json'),schema_sha256=sha(target),
        cache_sha256=training['array_sha256'],cache_bytes=training['array_bytes'],limits=dict(LIMITS),
        supervision=dict(SUPERVISION),supervision_entry='learning.task_completion_training_coverage_v1.supervisor',
        supervision_limit='Only the registered supervisor entry is monitored; direct Python API calls are not prevented. Sampled child RSS and system available memory are not a hard memory cap',
        score_evidence={n:sha(PARENT/n) for n in SCORE_EVIDENCE},training_evidence={n:sha(TRAIN/n) for n in TRAIN_EVIDENCE},
        reentry_evidence={n:sha(REENTRY/n) for n in ('registration.json','freeze.json','root-result-acceptance.json')},
        sampling='SHA256 of UTF-8 compact JSON [game,local_ordinal], ascending digest then game then ordinal; metadata only',
        attribution='Cache has no per-row game IDs; arm/game attribution relies on prior collector ordering and counts; same seed is not an independent row identity',
        physical_reads='One registration file hash; execution performs before/after hashes, NPZ directory reads and one complete decode',
        time_limit_semantics='Checked between operations; hashing and decompression are not preempted',
        new_games=0,model_inference=0,training_updates=0,official_transitions=0,gpu_calls=0,attempts=1,retry_allowed=False)
    write_new(out/'registration.json',cfg)
    write_new(out/'freeze.json',dict(registration_sha256=sha(out/'registration.json'),root_decision_sha256=sha(out/'root-decision.json'),
        runtime_files=runtime,new_source_sha256=source,parent_commit=BASE))
    return verify(out)


def verify(out=OUT):
    out = Path(out).resolve()
    if out != OUT: raise ValueError('Unexpected output')
    deadline_check()
    receipt = out/'pre-execution-commit.json'
    clean_head(read(receipt)['commit'] if receipt.exists() else BASE)
    cfg,freeze = read(out/'registration.json'),read(out/'freeze.json')
    if sha(out/'registration.json') != freeze['registration_sha256'] or sha(out/'root-decision.json') != freeze['root_decision_sha256']:
        raise ValueError('Registration changed')
    if cfg['limits'] != LIMITS or cfg['supervision'] != SUPERVISION: raise ValueError('Limits changed')
    if {p.relative_to(ROOT).as_posix() for p in SOURCE.glob('*.py')} != set(freeze['new_source_sha256']): raise ValueError('Source set changed')
    for rel,digest in freeze['runtime_files'].items():
        if sha(ROOT/rel) != digest: raise ValueError('Runtime changed')
    for folder,key in ((PARENT,'score_evidence'),(TRAIN,'training_evidence'),(REENTRY,'reentry_evidence')):
        for name,digest in cfg[key].items():
            if sha(folder/name) != digest: raise ValueError('Source evidence changed')
    if sha(out/'schema.json') != cfg['schema_sha256'] or sha(out/'sample-positions.local.json') != cfg['sample_positions_sha256']:
        raise ValueError('Local metadata changed')
    if CACHE.stat().st_size != cfg['cache_bytes'] or cfg['cache_bytes'] > LIMITS['max_archive_bytes']:
        raise ValueError('Cache size changed')
    from .analysis import deterministic_select
    if cfg['source_segments'] != source_segments(read(TRAIN/'registration.json'),read(TRAIN/'train-data-result.json')):
        raise ValueError('Source segments changed')
    if read(out/'sample-positions.local.json') != deterministic_select(cfg['source_segments'],200): raise ValueError('Selection changed')
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
    for name in ('root-decision.json','registration.json','freeze.json','pre-review.json','schema.json'):
        if git('show',commit+':review/evidence/'+out.name+'/'+name) != (out/name).read_bytes(): raise ValueError('Committed evidence changed')
    for rel in read(out/'freeze.json')['new_source_sha256']:
        if git('show',commit+':kaggriculture/'+rel) != (ROOT/rel).read_bytes(): raise ValueError('Committed source changed')
    return cfg
