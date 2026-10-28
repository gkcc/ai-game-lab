"""One named-Job collection of accepted teacher demonstrations; never seals."""
import os
from pathlib import Path
import numpy as np
from learning.task_choice_v1.features import OPS
from learning.task_teacher_choice_bc_v1 import numeric
from learning.task_teacher_choice_bc_v1.data_worker import sample_summary
from learning.task_teacher_dev_deviation_v1.membership import ENV_KEY,verify_worker
from .experiment import OUT,ready,deadline_check,read,sha,write_new
from .collector import collect
from .independent_runner import validate_completed

OVERHEAD_BYTES=8*1024**2


def check_independent(out,cfg):
    return validate_completed(out,cfg)


def coverage(arrays,seeds,limits):
    required={'min_groups':256,'min_groups_per_seed':32,'min_selected_operations':3,
              'min_dig':32,'min_fertilize':100,'min_dig_per_seed':1}
    if any(type(limits.get(k)) is not int or limits[k]<minimum for k,minimum in required.items()):
        raise ValueError('Invalid coverage thresholds')
    seed_counts={s:0 for s in seeds}; op_counts={op:0 for op in OPS}; dig_by_seed={s:0 for s in seeds}
    for i,label in enumerate(arrays['teacher_choice_label']):
        seed=int(arrays['seed'][i])
        if seed not in seed_counts: raise ValueError('Unexpected data seed')
        seed_counts[seed]+=1
        op=OPS[int(arrays['ops'][int(arrays['offsets'][i])+int(label)])]
        op_counts[op]+=1
        if op=='DIG': dig_by_seed[seed]+=1
    if set(int(s) for s in arrays['seed'])!=set(seeds): raise ValueError('Training seed set differs')
    adequate=(len(arrays['shared'])>=limits['min_groups']
        and all(seed_counts[s]>=limits['min_groups_per_seed'] for s in seeds)
        and sum(v>0 for v in op_counts.values())>=limits['min_selected_operations']
        and op_counts['DIG']>=limits['min_dig'] and op_counts['FERTILIZE']>=limits['min_fertilize']
        and all(dig_by_seed[s]>=limits['min_dig_per_seed'] for s in seeds))
    return adequate,dict(groups=len(arrays['shared']),groups_by_seed={str(k):v for k,v in seed_counts.items()},
        retained_selected_operations=op_counts,dig_by_seed={str(k):v for k,v in dig_by_seed.items()})


def validate_stream_counts(stats,seeds,coverage_summary):
    entries=stats.get('by_seed_opponent') if isinstance(stats,dict) else None
    expected={(seed,opponent) for seed in seeds for opponent in ('round2','market')}
    if not isinstance(entries,list) or len(entries)!=4: raise ValueError('Missing full-stream counts')
    seen=set(); seed_counts={s:0 for s in seeds}; selected={op:0 for op in OPS}; dig={s:0 for s in seeds}
    for entry in entries:
        key=(entry.get('seed'),entry.get('opponent'))
        if key not in expected or key in seen or type(entry.get('retained')) is not int or entry['retained']<0:
            raise ValueError('Invalid full-stream identity')
        seen.add(key); chosen=entry.get('chosen'); retained=entry.get('retained_chosen')
        if (not isinstance(chosen,dict) or not isinstance(retained,dict) or set(chosen)!=set(OPS) or set(retained)!=set(OPS)
            or any(type(v) is not int or v<0 for v in (*chosen.values(),*retained.values()))
            or any(retained[op]>chosen[op] for op in OPS) or sum(retained.values())!=entry['retained']
            or any(chosen[op]!=retained[op] for op in ('DIG','FERTILIZE'))):
            raise ValueError('Inconsistent full-stream counts')
        seed_counts[key[0]]+=entry['retained']; dig[key[0]]+=retained['DIG']
        for op in OPS: selected[op]+=retained[op]
    if (seen!=expected or {str(k):v for k,v in seed_counts.items()}!=coverage_summary['groups_by_seed']
        or {str(k):v for k,v in dig.items()}!=coverage_summary['dig_by_seed']
        or selected!=coverage_summary['retained_selected_operations']
        or sum(seed_counts.values())!=coverage_summary['groups']
        or stats.get('retained_groups')!=coverage_summary['groups']):
        raise ValueError('Retained data and full-stream counts differ')


def run(out=OUT):
    out=Path(out).resolve(); cfg=ready(out); entered=False
    try:
        marker=read(out/'data-worker-started.json'); freeze=sha(out/'freeze.json')
        verify_worker(os.environ.get(ENV_KEY),marker,freeze)
        entered=True
        identity_names=('freeze.json','registration.json','pre-execution-commit.json','data-worker-started.json',
                        'source-manifest.json','schema.json','independent-runner-result.json','independent-replay-verification.json')
        identities={name:sha(out/name) for name in identity_names}
        if (marker.get('source_manifest_sha256')!=identities['source-manifest.json']
            or marker.get('schema_sha256')!=identities['schema.json'] or identities['schema.json']!=cfg['schema_sha256']
            or marker.get('independent_replay_sha256')!=identities['independent-replay-verification.json']
            or marker.get('independent_runner_result_sha256')!=identities['independent-runner-result.json']):
            raise ValueError('Collection input identity changed')
        check_independent(out,cfg)
        seeds=cfg['train_seeds']; reserved=cfg['reserved_test_seeds']
        if (not isinstance(seeds,list) or len(seeds)!=2 or not isinstance(reserved,list) or len(reserved)!=4
            or any(type(s) is not int or not 0<=s<2**32 for s in seeds+reserved)
            or len(set(seeds+reserved))!=6): raise ValueError('Seed roles invalid')
        write_new(out/'data-worker-entered.json',dict(attempt=1,freeze_sha256=freeze,retry_allowed=False))
        sources=read(out/'source-manifest.json'); schema=read(out/'schema.json')
        arrays,stats=collect(out,sources,sources,seeds,schema,cfg['collection_limits'],deadline_check)
        limits=cfg['data_limits']; numeric.validate(arrays,limits['max_groups'],limits['max_candidates'])
        adequate,summary=coverage(arrays,seeds,limits)
        validate_stream_counts(stats,seeds,summary)
        if any(sha(out/name)!=value for name,value in identities.items()): raise ValueError('Collection identity changed')
        if not adequate:
            result=dict(passed=False,status='coverage_insufficient',coverage=summary,full_stream=stats,
                freeze_sha256=freeze,retry_allowed=False,job_cleanup_verified_here=False)
            write_new(out/'data-worker-result.json',result)
            return result
        sample=sample_summary(arrays,seeds,limits)
        max_archive=limits['max_archive_bytes']
        if type(max_archive) is not int or max_archive>256*1024**2 or sum(a.nbytes for a in arrays.values())+OVERHEAD_BYTES>max_archive:
            raise ValueError('Archive memory budget exceeded')
        archive=out/'private/new-teacher.npz'; archive.parent.mkdir(parents=True,exist_ok=True)
        numeric.write_new(archive,arrays,limits['max_groups'],limits['max_candidates'])
        if archive.stat().st_size>max_archive: raise ValueError('Archive exceeds budget')
        archive_sha=sha(archive)
        loaded=numeric.read(archive,limits['max_groups'],limits['max_candidates'],max_archive,
                            limits['max_uncompressed_bytes'],archive_sha)
        if set(loaded)!=set(arrays) or any(not np.array_equal(loaded[k],arrays[k]) for k in arrays):
            raise ValueError('Archive roundtrip mismatch')
        deadline_check()
        if any(sha(out/name)!=value for name,value in identities.items()): raise ValueError('Collection identity changed')
        result=dict(passed=True,status='collected',freeze_sha256=freeze,source_manifest_sha256=identities['source-manifest.json'],
            schema_sha256=identities['schema.json'],array_sha256=archive_sha,coverage=summary,sample_summary=sample,full_stream=stats,
            archive_snapshot_reads=1,collection_attempt=1,retry_allowed=False,job_cleanup_verified_here=False,
            interpretation='Retained sample intentionally changes operation distribution; labels do not prove execution or value')
        write_new(out/'data-worker-result.json',result)
        return result
    except BaseException as exc:
        if entered:
            try: write_new(out/'data-worker-failure.json',dict(passed=False,error_kind=type(exc).__name__,retry_allowed=False,
                job_cleanup_verified_here=False,partial_archive_may_exist=True))
            except BaseException as receipt: raise RuntimeError('Worker and receipt failure: '+type(exc).__name__+'/'+type(receipt).__name__) from None
        raise


if __name__=='__main__': run()
