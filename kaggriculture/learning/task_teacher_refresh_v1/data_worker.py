"""One bounded sample of the already accepted development trajectories."""
import os
from .experiment import OUT, deadline_check, read, ready, sha, write_new
from .collector import collect
from learning.task_teacher_dev_deviation_v1.membership import ENV_KEY, verify_worker
from learning.task_teacher_choice_bc_v1.data_worker import sample_summary
from learning.task_teacher_choice_bc_v1 import numeric


def run():
    cfg = ready(OUT)
    marker = read(OUT/'data-worker-started.json')
    freeze_sha = sha(OUT/'freeze.json')
    verify_worker(os.environ.get(ENV_KEY),marker,freeze_sha)
    if (marker.get('source_manifest_sha256') != sha(OUT/'source-manifest.json')
        or marker.get('schema_sha256') != sha(OUT/'schema.json')):
        raise ValueError('Collection input identity changed')
    write_new(OUT/'data-worker-entered.json',dict(attempt=1,freeze_sha256=freeze_sha,retry_allowed=False))
    try:
        sources, schema = read(OUT/'source-manifest.json'),read(OUT/'schema.json')
        arrays, receipt, checked = collect(OUT,sources,schema,cfg,deadline_check)
        limits = cfg['data_limits']
        summary = sample_summary(arrays,cfg['train_seeds'],limits)
        archive = OUT/'private/new-teacher.npz'
        archive.parent.mkdir(parents=True,exist_ok=True)
        numeric.write_new(archive,arrays,limits['max_groups'],limits['max_candidates'])
        if archive.stat().st_size > limits['max_archive_bytes']:
            raise ValueError('Archive exceeds budget')
        digest = sha(archive)
        numeric.read(archive,limits['max_groups'],limits['max_candidates'],
            limits['max_archive_bytes'],limits['max_uncompressed_bytes'],digest)
        deadline_check()
        write_new(OUT/'data-worker-result.json',dict(passed=True,freeze_sha256=freeze_sha,
            source_manifest_sha256=marker['source_manifest_sha256'],schema_sha256=marker['schema_sha256'],
            array_sha256=digest,raw_reads=receipt,teacher_choices_checked=checked,
            candidate_rows=len(arrays['candidates']),sampling_algorithm=cfg['sampling_algorithm'],
            collection_attempt=1,retry_allowed=False,**summary))
    except BaseException as exc:
        try:
            write_new(OUT/'data-worker-failure.json',dict(passed=False,error_kind=type(exc).__name__,retry_allowed=False))
        except BaseException:
            pass
        raise


if __name__ == '__main__':
    run()
