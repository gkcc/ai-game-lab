"""One supervised development job from an externally authenticated context.

An outer isolated bootstrap must validate registration, sources and the
independently fixed private manifest before calling this entry.  No standalone
CLI is provided that could accept an unauthenticated private contract.
"""
import hashlib
import json
from pathlib import Path
import stat

from learning.task_pair_dev_games_v1 import worker
from .manifest import PrivateManifest


MAX_WRAPPER_BYTES = 65536


def _read_regular(path, maximum):
    path = Path(path)
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or getattr(info, 'st_file_attributes', 0) & 0x400 or info.st_size > maximum):
        raise ValueError('Unexpected private source file')
    with path.open('rb') as stream:
        data = stream.read(maximum + 1)
    if len(data) != info.st_size or len(data) > maximum:
        raise ValueError('Private source length differs')
    return data


def _write_new(path, data):
    if type(data) is not bytes or not 0 < len(data) <= worker.MAX_JOB_BYTES:
        raise ValueError('Invalid private job document')
    with Path(path).open('xb') as stream:
        if stream.write(data) != len(data):
            raise OSError('Incomplete private job document')
    if _read_regular(path, worker.MAX_JOB_BYTES) != data:
        raise ValueError('Private job readback differs')


def run_one(*, manifest, worker_source, game_root, stage, plan, check):
    """Create one wrapper/job and run the one-match Worker in this child.

    The caller owns and supervises the whole process.  A normal return still
    requires the parent to verify every receipt and Job cleanup independently.
    """
    try:
        worker._single_python_thread()
        with worker._quiet_output():
            if type(manifest) is not PrivateManifest:
                raise ValueError('Expected authenticated manifest')
            root, output = worker._root_and_stage(game_root, stage)
            plan = manifest.panel.validate_plan(plan)
            worker._registered(check)
            if (type(worker_source) is not bytes or not 0 < len(worker_source) <= worker.MAX_WORKER_BYTES
                    or hashlib.sha256(worker_source).hexdigest() != manifest.panel.worker_sha):
                raise ValueError('Worker source identity differs')
            wrapper = worker.make_wrapper(panel=manifest.panel, source=worker_source,
                                          game_root=root, stage=output, plan=plan, check=check)
            raw = _read_regular(wrapper, MAX_WRAPPER_BYTES)
            task_sha = manifest.panel.task_model(plan)[1]
            rival_sha = manifest.opponent(plan['opponent'])[1]
            job = manifest.panel.validate_plan(dict(plan, bot_sha256=hashlib.sha256(raw).hexdigest(),
                                                     rival_sha256=rival_sha,
                                                     task_checkpoint_sha256=task_sha), job=True)
            data = (json.dumps(job, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
            _write_new(output / 'job-inputs' / (plan['game'] + '.json'), data)
            worker._registered(check)
            worker.run_job(panel=manifest.panel, source=worker_source, game_root=root,
                           stage=output, plan=job, check=check)
            worker._registered(check)
    except BaseException:
        raise RuntimeError('Development child failed') from None
