"""Owned one-shot collector; publishes an intermediate receipt, never a seal."""
import json
from collections import Counter
from pathlib import Path
from learning.task_choice_v1.features import OPS
from .experiment import OUT, ROOT, ready, deadline_check, read, sha, write_new


def sample_summary(arrays, train_seeds, limits):
    minimum_groups = limits['min_groups']
    minimum_seed = limits['min_groups_per_seed']
    minimum_operations = limits['min_selected_operations']
    if (any(type(v) is not int or v < 1 for v in (minimum_groups,minimum_seed,minimum_operations))
        or minimum_groups < 256 or minimum_seed < 32 or minimum_operations < 3
        or minimum_operations > len(OPS) or not isinstance(train_seeds,list) or not train_seeds
        or any(type(s) is not int or not 0 <= s <= 0xffffffff for s in train_seeds)
        or len(set(train_seeds)) != len(train_seeds)):
        raise ValueError('Invalid sample thresholds or seed roles')
    seed_counts = Counter(int(x) for x in arrays['seed'])
    selected = Counter()
    for i,label in enumerate(arrays['teacher_choice_label']):
        index = int(arrays['offsets'][i])+int(label)
        selected[OPS[int(arrays['ops'][index])]] += 1
    if (len(arrays['shared']) < minimum_groups or set(seed_counts) != set(train_seeds)
        or any(seed_counts[s] < minimum_seed for s in train_seeds) or len(selected) < minimum_operations):
        raise ValueError('Sample availability threshold failed')
    return dict(groups=len(arrays['shared']),groups_by_seed={str(s):seed_counts[s] for s in train_seeds},
                selected_operations=dict(sorted(selected.items())))


def run(out=OUT):
    from . import collect, numeric
    out = Path(out).resolve()
    cfg = ready(out)
    marker = read(out/'data-started.json')
    if marker['freeze_sha256'] != sha(out/'freeze.json') or marker['attempt'] != 1:
        raise ValueError('Collection marker changed')
    if (out/'data-worker-started.json').exists(): raise ValueError('Worker already consumed')
    write_new(out/'data-worker-started.json',dict(attempt=1,freeze_sha256=marker['freeze_sha256'],retry_allowed=False))
    try:
        if cfg['sampling_algorithm'] != collect.SAMPLING_ALGORITHM: raise ValueError('Sampling algorithm changed')
        if sha(out/'source-manifest.json') != cfg['source_manifest_sha256'] or sha(out/'schema.json') != cfg['schema_sha256']:
            raise ValueError('Frozen source metadata changed')
        sources = read(out/'source-manifest.json')
        if sources != cfg['attested_sources']: raise ValueError('Source attestations changed')
        arrays,receipt = collect.collect(ROOT/cfg['source_root'],sources,cfg['attested_sources'],
            read(out/'schema.json'),cfg['collection_limits'],deadline_check)
        limits = cfg['data_limits']
        summary = sample_summary(arrays,cfg['train_seeds'],limits)
        target = out/'private/train.npz'
        target.parent.mkdir(parents=True,exist_ok=True)
        numeric.write_new(target,arrays,limits['max_groups'],limits['max_candidates'])
        if target.stat().st_size > limits['max_archive_bytes']: raise ValueError('Archive exceeds budget')
        digest = sha(target)
        numeric.read(target,limits['max_groups'],limits['max_candidates'],limits['max_archive_bytes'],limits['max_uncompressed_bytes'],digest)
        deadline_check()
        write_new(out/'data-worker-result.json',dict(passed=True,array_sha256=digest,freeze_sha256=marker['freeze_sha256'],
            source_manifest_sha256=cfg['source_manifest_sha256'],sampling_algorithm=cfg['sampling_algorithm'],
            collection_attempt=1,raw_reads=receipt,**summary,candidate_rows=len(arrays['candidates']),retry_allowed=False))
    except BaseException as exc:
        try: write_new(out/'data-worker-failure.json',dict(passed=False,error_kind=type(exc).__name__,retry_allowed=False))
        except BaseException: pass
        raise


if __name__ == '__main__': run()
