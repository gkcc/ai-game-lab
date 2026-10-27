"""One immutable old-training snapshot and synthetic merge interface check."""
import os
from pathlib import Path, PurePosixPath
from learning.task_teacher_dev_deviation_v1.membership import ENV_KEY,verify_worker
from .experiment import OUT,ROOT,read,sha,ready,write_new,environment


def _path(relative):
    if not isinstance(relative,str) or not relative or '\\' in relative or ':' in relative:
        raise ValueError('Invalid project-relative path')
    posix=PurePosixPath(relative)
    if posix.is_absolute() or posix.as_posix()!=relative or any(part in ('.','..') for part in posix.parts):
        raise ValueError('Noncanonical project-relative path')
    path=(ROOT/relative).resolve()
    if not path.is_relative_to(ROOT.resolve()) or path==ROOT.resolve(): raise ValueError('Path outside project')
    return path


def run(out=OUT):
    from learning.task_teacher_choice_bc_v1 import numeric
    from .analysis import check
    out=Path(out).resolve(); cfg=ready(out); marked=False
    try:
        marker=read(out/'worker-started.json')
        verify_worker(os.environ.get(ENV_KEY),marker,sha(out/'freeze.json'))
        marked=True
        if (marker.get('seed_roles_sha256')!=cfg['seed_roles_sha256'] or marker.get('schema_sha256')!=cfg['schema_sha256']
            or marker.get('cache_sha256')!=cfg['cache_sha256']): raise ValueError('Worker marker identities changed')
        if environment()!=cfg['environment']: raise ValueError('Worker environment changed')
        roles_path=out/'seed-roles.local.json'
        if sha(roles_path)!=cfg['seed_roles_sha256']: raise ValueError('Seed roles changed')
        roles=read(roles_path)
        schema=_path(cfg['schema_path'])
        if sha(schema)!=cfg['schema_sha256']: raise ValueError('Schema changed')
        cache=_path(cfg['cache_path'])
        old=numeric.read(cache,cfg['max_groups'],cfg['max_candidates'],cfg['max_archive_bytes'],
                         cfg['max_uncompressed_bytes'],cfg['cache_sha256'])
        result=check(old,old_seeds=cfg['old_train_seeds'],holdout_seeds=roles['holdout_seeds'],
                     synthetic_seed=cfg['synthetic_seed'],max_groups=cfg['merge_max_groups'],max_candidates=cfg['merge_max_candidates'])
        if (not result['passed'] or result['old_groups']!=cfg['old_groups']
            or result['old_candidate_rows']!=cfg['old_candidate_rows']): raise ValueError('Old aggregate identity changed')
        write_new(out/'worker-result.json',dict(passed=True,aggregate=result,freeze_sha256=sha(out/'freeze.json'),
            seed_roles_sha256=cfg['seed_roles_sha256'],cache_sha256=cfg['cache_sha256'],schema_sha256=cfg['schema_sha256'],
            archive_snapshot_reads=1,no_training=True,retry_allowed=False))
        return result
    except BaseException as exc:
        if marked:
            try: write_new(out/'worker-failure.json',dict(error_kind=type(exc).__name__,retry_allowed=False))
            except BaseException: pass
        raise


if __name__=='__main__': run()
