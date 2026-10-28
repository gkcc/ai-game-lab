"""Single-use supervised data collection and parent-side acceptance."""
import json
import os
from pathlib import Path
import secrets
import sys
from learning.task_job_empty_exit_v1.sampling import supervise
from learning.task_teacher_dev_deviation_v1.membership import ENV_KEY, NamedWindowsAPI, token_hash
from learning.task_probe_failure_reporting_v1.report import safe_job, error_kind
from learning.task_choice_v1.features import OPS
from .experiment import OUT, ROOT, ready, verify_commit, deadline_check, read, sha, write_new
from .campaign import validate_panel, memory_ready, job_ok
from .independent_runner import validate_completed

NAMES=('data-started.json','data-worker-started.json','data-worker-entered.json',
       'data-worker-result.json','data-worker-failure.json','data-supervision.local.json',
       'data-failure.local.json','data-result.json','data-seal.json','private/new-teacher.npz')
IDENTITIES=('freeze.json','registration.json','pre-execution-commit.json','source-manifest.json',
            'schema.json','independent-runner-result.json','independent-replay-verification.json')


def marker(out, name, value, state):
    path=(out/name).resolve()
    if not path.is_relative_to(OUT.resolve()): raise ValueError('Marker outside experiment')
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        state['created']=True
        json.dump(value,stream,allow_nan=False)
        stream.write('\n')
    state['complete']=True


def inputs(out,cfg):
    manifest=validate_panel(out,cfg)
    validate_completed(out,cfg)
    if sha(out/'schema.json')!=cfg['schema_sha256']: raise ValueError('Schema changed')
    return manifest,{name:sha(out/name) for name in IDENTITIES}


def validate_statistics(worker,cfg):
    """Cross-check bounded worker summaries; no parent-side archive decoding."""
    stats=worker.get('full_stream'); summary=worker.get('coverage'); limits=cfg['data_limits']
    if not isinstance(stats,dict) or not isinstance(summary,dict): raise ValueError('Missing statistics')
    def count(value):
        if type(value) is not int or value<0: raise ValueError('Invalid count')
        return value
    def op_counts(value):
        if not isinstance(value,dict) or set(value)!=set(OPS): raise ValueError('Invalid operation counts')
        return {op:count(value[op]) for op in OPS}
    entries=stats.get('by_seed_opponent'); seeds=cfg['train_seeds']
    expected=[(seed,opponent) for seed in seeds for opponent in ('round2','market')]
    if not isinstance(entries,list) or len(entries)!=len(expected): raise ValueError('Invalid stream partitions')
    selected={op:0 for op in OPS}; per_seed={str(seed):0 for seed in seeds}; dig_seed={str(seed):0 for seed in seeds}
    callbacks=rows=retained=0
    for item,(seed,opponent) in zip(entries,expected):
        if not isinstance(item,dict) or item.get('seed')!=seed or item.get('opponent')!=opponent: raise ValueError('Stream partition mismatch')
        chosen=op_counts(item.get('chosen')); kept=op_counts(item.get('retained_chosen'))
        candidates=op_counts(item.get('candidate_rows')); candidate_callbacks=op_counts(item.get('candidate_callbacks'))
        n=count(item.get('callbacks')); k=count(item.get('retained'))
        if (sum(chosen.values())!=n or sum(kept.values())!=k or k>n
            or any(kept[op]>chosen[op] or chosen[op]>candidate_callbacks[op] or candidate_callbacks[op]>n or candidate_callbacks[op]>candidates[op] for op in OPS)
            or any(kept[op]!=chosen[op] for op in ('DIG','FERTILIZE'))): raise ValueError('Inconsistent stream counts')
        callbacks+=n; rows+=sum(candidates.values()); retained+=k; per_seed[str(seed)]+=k; dig_seed[str(seed)]+=kept['DIG']
        for op in OPS: selected[op]+=kept[op]
    meter=stats.get('meter',{})
    if (stats.get('total_callbacks')!=callbacks or stats.get('total_candidate_rows')!=rows
        or stats.get('retained_groups')!=retained or count(stats.get('retained_candidate_rows'))<retained
        or stats['retained_candidate_rows']>rows or retained>limits['max_groups']
        or stats['retained_candidate_rows']>limits['max_candidates']
        or callbacks>cfg['collection_limits']['max_total_callbacks'] or rows>cfg['collection_limits']['max_total_candidate_rows']
        or meter.get('files_opened')!=8 or meter.get('files_completed')!=8 or meter.get('raw_decode_passes_per_file')!=1):
        raise ValueError('Invalid full stream totals')
    for field,limit in (('compressed_input_read_bytes','compressed_bytes'),('decoded_input_bytes','decoded_bytes')):
        if field not in meter or count(meter[field])>cfg['collection_limits'][limit]: raise ValueError('Missing or exceeded stream byte limit')
    if summary.get('groups')!=retained or summary.get('groups_by_seed')!=per_seed: raise ValueError('Summary mismatch')
    adequate=(retained>=limits['min_groups'] and all(per_seed[str(s)]>=limits['min_groups_per_seed'] for s in seeds)
        and sum(v>0 for v in selected.values())>=limits['min_selected_operations']
        and selected['DIG']>=limits['min_dig'] and selected['FERTILIZE']>=limits['min_fertilize']
        and all(dig_seed[str(s)]>=limits['min_dig_per_seed'] for s in seeds))
    if summary.get('retained_selected_operations')!=selected or summary.get('dig_by_seed')!=dig_seed:
        raise ValueError('Coverage mismatch')
    if worker.get('status')=='collected':
        sample=worker.get('sample_summary')
        if (not isinstance(sample,dict) or sample.get('groups')!=retained or sample.get('groups_by_seed')!=per_seed
            or sample.get('selected_operations')!={k:v for k,v in selected.items() if v}):
            raise ValueError('Sample summary mismatch')
    return adequate


def validate_sealed(out,cfg):
    """Current-stage completion gate: a pending seal alone never constitutes success.

    The worker performs numeric readback inside its Job; this validator checks
    the archive by streaming hash and size, without loading its arrays. A later
    stage must attest this result and archive under its own frozen gates; this
    stage's deadline and exact-HEAD checks are intentionally not bypassed.
    """
    out=Path(out).resolve(); ready(out); verify_commit(out); deadline_check()
    if any((out/name).exists() for name in ('data-failure.local.json','data-worker-failure.json')):
        raise ValueError('Failure evidence present')
    manifest,identities=inputs(out,cfg)
    seal=read(out/'data-seal.json'); result=read(out/'data-result.json')
    worker=read(out/'data-worker-result.json'); marker=read(out/'data-worker-started.json')
    entered=read(out/'data-worker-entered.json'); started=read(out/'data-started.json')
    supervision=read(out/'data-supervision.local.json'); archive=out/'private/new-teacher.npz'
    common=dict(freeze_sha256=identities['freeze.json'],source_manifest_sha256=identities['source-manifest.json'],
        schema_sha256=identities['schema.json'],independent_runner_result_sha256=identities['independent-runner-result.json'],
        independent_replay_sha256=identities['independent-replay-verification.json'],
        worker_result_sha256=sha(out/'data-worker-result.json'),worker_marker_sha256=sha(out/'data-worker-started.json'),
        supervision_sha256=sha(out/'data-supervision.local.json'),retry_allowed=False,training_started=False)
    if (result!=dict(passed=True,status='collected',data_seal_sha256=sha(out/'data-seal.json'),**common)
        or seal.get('passed') is not True or seal.get('completion_state')!='pending_result_confirmation'
        or any(seal.get(k)!=v for k,v in common.items())
        or seal.get('archive_numeric_readback')!='worker Job only; parent verified size and SHA256'
        or supervision.get('error_kind') is not None or supervision.get('cleanup_state')!='recorded'
        or supervision.get('retry_allowed') is not False or not job_ok(supervision.get('supervision'))
        or started!=dict(attempt=1,freeze_sha256=identities['freeze.json'],retry_allowed=False)
        or entered!=dict(attempt=1,freeze_sha256=identities['freeze.json'],retry_allowed=False)):
        raise ValueError('Incomplete data completion evidence')
    marker_fields=dict(attempt=1,freeze_sha256=identities['freeze.json'],source_manifest_sha256=identities['source-manifest.json'],
        schema_sha256=identities['schema.json'],independent_runner_result_sha256=identities['independent-runner-result.json'],
        independent_replay_sha256=identities['independent-replay-verification.json'],retry_allowed=False)
    token=marker.get('job_token_sha256')
    if (set(marker)!=set(marker_fields)|{'job_token_sha256'} or any(marker.get(k)!=v for k,v in marker_fields.items())
        or not isinstance(token,str) or len(token)!=64 or any(c not in '0123456789abcdef' for c in token)):
        raise ValueError('Worker marker invalid')
    if (worker.get('passed') is not True or worker.get('status')!='collected'
        or worker.get('freeze_sha256')!=identities['freeze.json'] or worker.get('source_manifest_sha256')!=identities['source-manifest.json']
        or worker.get('schema_sha256')!=identities['schema.json'] or worker.get('retry_allowed') is not False
        or worker.get('job_cleanup_verified_here') is not False or worker.get('archive_snapshot_reads')!=1
        or worker.get('collection_attempt')!=1 or not validate_statistics(worker,cfg)):
        raise ValueError('Worker evidence invalid')
    if (not archive.is_file() or archive.stat().st_size>cfg['data_limits']['max_archive_bytes']
        or archive.stat().st_size!=seal.get('archive_size_bytes') or sha(archive)!=seal.get('array_sha256')
        or worker.get('array_sha256')!=seal.get('array_sha256')):
        raise ValueError('Archive identity invalid')
    if inputs(out,cfg)!=(manifest,identities): raise ValueError('Completion inputs changed')
    return result


def run(out=OUT):
    out=Path(out).resolve(); cfg=ready(out); verify_commit(out)
    manifest, identities=inputs(out,cfg)
    if (cfg['host_preflight']['samples']!=3 or cfg['host_preflight']['min_available_bytes']<4*1024**3
        or cfg['data_job_supervision']['min_available_bytes']<2*1024**3):
        raise ValueError('Invalid memory limits')
    if any((out/name).exists() for name in NAMES) or ((out/'private').exists() and any((out/'private').glob('*.npz'))): raise ValueError('Data attempt already consumed')
    if not memory_ready(cfg): raise RuntimeError('Insufficient preflight memory')
    # Revalidate after the preflight delay before consuming the attempt.
    ready(out); verify_commit(out); deadline_check()
    if inputs(out,cfg)!=(manifest,identities): raise ValueError('Data inputs changed')
    state=dict(created=False,complete=False); job=None; launch_entered=False
    try:
        marker(out,'data-started.json',dict(attempt=1,freeze_sha256=identities['freeze.json'],retry_allowed=False),state)
        name='Local\\TeacherDeviation-'+secrets.token_hex(16)
        worker_marker=dict(attempt=1,freeze_sha256=identities['freeze.json'],job_token_sha256=token_hash(name),
            source_manifest_sha256=identities['source-manifest.json'],schema_sha256=identities['schema.json'],
            independent_runner_result_sha256=identities['independent-runner-result.json'],
            independent_replay_sha256=identities['independent-replay-verification.json'],retry_allowed=False)
        write_new(out/'data-worker-started.json',worker_marker)
        worker_marker_sha=sha(out/'data-worker-started.json')
        previous=os.environ.get(ENV_KEY); supervisor_error=None
        try:
            os.environ[ENV_KEY]=name; launch_entered=True
            job=supervise(NamedWindowsAPI(name),[sys.executable,'-B','-m','learning.task_teacher_refresh_v2.data_worker'],ROOT,cfg['data_job_supervision'])
        except BaseException as exc:
            supervisor_error=error_kind(exc)
        finally:
            if previous is None: os.environ.pop(ENV_KEY,None)
            else: os.environ[ENV_KEY]=previous
        safe=safe_job(job)
        write_new(out/'data-supervision.local.json',dict(supervision=safe,error_kind=supervisor_error,
            cleanup_state='recorded' if safe is not None else 'unknown',retry_allowed=False))
        if supervisor_error or safe is None or not job_ok(safe): raise RuntimeError('Job failed or cleanup unconfirmed')
        ready(out); verify_commit(out); deadline_check()
        if inputs(out,cfg)!=(manifest,identities) or sha(out/'data-worker-started.json')!=worker_marker_sha:
            raise ValueError('Data identity changed')
        if read(out/'data-worker-started.json')!=worker_marker: raise ValueError('Worker marker changed')
        entered=read(out/'data-worker-entered.json')
        if entered!=dict(attempt=1,freeze_sha256=identities['freeze.json'],retry_allowed=False): raise ValueError('Worker entry invalid')
        if (out/'data-worker-failure.json').exists(): raise ValueError('Worker failure present')
        worker=read(out/'data-worker-result.json'); archive=out/'private/new-teacher.npz'
        if (worker.get('freeze_sha256')!=identities['freeze.json'] or worker.get('retry_allowed') is not False
            or worker.get('job_cleanup_verified_here') is not False): raise ValueError('Worker result invalid')
        common=dict(freeze_sha256=identities['freeze.json'],source_manifest_sha256=identities['source-manifest.json'],
            schema_sha256=identities['schema.json'],independent_runner_result_sha256=identities['independent-runner-result.json'],
            independent_replay_sha256=identities['independent-replay-verification.json'],
            worker_result_sha256=sha(out/'data-worker-result.json'),worker_marker_sha256=worker_marker_sha,
            supervision_sha256=sha(out/'data-supervision.local.json'),retry_allowed=False,training_started=False)
        adequate=validate_statistics(worker,cfg)
        if worker.get('status')=='coverage_insufficient' and worker.get('passed') is False:
            if adequate or archive.exists(): raise ValueError('Invalid insufficient coverage result')
            result=dict(passed=False,status='coverage_insufficient',**common)
            write_new(out/'data-result.json',result); return result
        if (worker.get('status')!='collected' or worker.get('passed') is not True
            or worker.get('source_manifest_sha256')!=identities['source-manifest.json']
            or worker.get('schema_sha256')!=identities['schema.json'] or worker.get('archive_snapshot_reads')!=1
            or worker.get('collection_attempt')!=1 or not archive.is_file()
            or sha(archive)!=worker.get('array_sha256')): raise ValueError('Worker archive invalid')
        limits=cfg['data_limits']
        if not adequate or archive.stat().st_size>limits['max_archive_bytes']: raise ValueError('Archive size or coverage invalid')
        ready(out); verify_commit(out); deadline_check()
        if inputs(out,cfg)!=(manifest,identities) or sha(archive)!=worker['array_sha256'] or sha(out/'data-worker-result.json')!=common['worker_result_sha256'] or sha(out/'data-worker-started.json')!=worker_marker_sha:
            raise ValueError('Data changed before seal')
        seal=dict(passed=True,completion_state='pending_result_confirmation',array_sha256=worker['array_sha256'],archive_size_bytes=archive.stat().st_size,
                  archive_numeric_readback='worker Job only; parent verified size and SHA256',**common)
        write_new(out/'data-seal.json',seal)
        result=dict(passed=True,status='collected',data_seal_sha256=sha(out/'data-seal.json'),**common)
        write_new(out/'data-result.json',result)
        return validate_sealed(out,cfg)
    except BaseException as exc:
        if state['created']:
            try: write_new(out/'data-failure.local.json',dict(error_kind=error_kind(exc),marker=state,
                launch_entered=launch_entered,cleanup_state='recorded' if safe_job(job) is not None else 'unknown',retry_allowed=False))
            except BaseException as receipt: raise RuntimeError('Data and failure receipt errors: '+error_kind(exc)+'/'+error_kind(receipt)) from None
        raise


if __name__=='__main__':
    result=run()
    print(json.dumps(dict(passed=result['passed'],status=result['status'])))
