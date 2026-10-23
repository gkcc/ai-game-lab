"""One-shot, bounded sample capacity work. Never execute a game or train."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

from learning.bc_crop.agent import ROOT, sha
from learning.bc_crop.experiment import read, save
from .experiment import verify, ready


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_time(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Registered capacity deadline reached')


def new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def strict_json(line):
    def invalid(_):
        raise ValueError('Nonfinite JSON source')
    return json.loads(line, parse_constant=invalid)


def key_name(value):
    require(value and all(c.isalnum() or c in '-_' for c in value), 'Invalid job name')
    return value


def budget_rows(out):
    path = Path(out) / 'budget-reservations.jsonl'
    if not path.exists():
        return []
    rows = []; prior = '0' * 64
    for i, line in enumerate(path.read_bytes().splitlines()):
        value = json.loads(line)
        require(value['sequence'] == i and value['previous_line_sha256'] == prior, 'Budget chain changed')
        require(type(value['groups']) is int and type(value['candidates']) is int and
                value['groups'] >= 0 and value['candidates'] >= 0, 'Invalid reserved amount')
        rows.append(value); prior = hashlib.sha256(line).hexdigest()
    return rows


def reserve(out, cfg, phase, key, groups, candidates):
    """Reserve the entire declared job before dispatch; never refund failures."""
    path = Path(out) / 'budget-reservations.jsonl'; rows = budget_rows(out)
    require(not any(r['phase'] == phase and r['key'] == key for r in rows), 'A budget reservation cannot repeat')
    require(sum(r['groups'] for r in rows) + groups <= cfg['real_evaluated_groups_shared_max'], 'Shared group cap')
    require(sum(r['candidates'] for r in rows) + candidates <= cfg['real_evaluated_candidates_shared_max'], 'Shared candidate cap')
    prior = hashlib.sha256(path.read_bytes().splitlines()[-1]).hexdigest() if rows else '0' * 64
    row = dict(sequence=len(rows), previous_line_sha256=prior, phase=phase, key=key,
               groups=groups, candidates=candidates, reserved_before_dispatch=True,
               refunded=False, created_utc=datetime.now(timezone.utc).isoformat())
    raw = json.dumps(row, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')
    with path.open('ab') as stream:
        stream.write(raw + b'\n'); stream.flush(); os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


def resources(out, cfg):
    from .windows_tree import available_memory_bytes
    limit = cfg['resource_limits']
    require(available_memory_bytes() >= limit['minimum_available_host_memory_bytes'], 'Available memory below fixed floor')
    require(shutil.disk_usage(out).free >= limit['minimum_free_disk_bytes'], 'Free disk below fixed floor')
    size = sum(p.stat().st_size for p in Path(out).rglob('*') if p.is_file())
    require(size <= limit['new_artifact_bytes_max'], 'Owned artifact byte cap')
    return dict(owned_artifact_bytes=size)


def collect_sample(out, cfg, deadline):
    """Read fixed physical prefixes once. Never skip or replace an invalid row."""
    from learning.task_bc_v1.cache import decode
    out = Path(out); source = ROOT / cfg['source_dataset']
    require(not (out / 'sample-manifest.json').exists(), 'Sample cannot be recollected')
    sample_dir = out / 'sample'; sample_dir.mkdir()
    schema = None; groups = []; total_rows = total_candidates = source_bytes = 0
    decode_seconds = hash_seconds = 0.
    for group in cfg['selected_groups']:
        check_time(deadline); paths = {}
        tick = time.monotonic()
        for kind, info in group['source_files'].items():
            path = (source / info['path']).resolve()
            require(path.is_relative_to(source / 'data'), 'Source path outside registered data')
            require(sha(path) == info['sha256'], 'Frozen source bytes changed')
            paths[kind] = path; source_bytes += path.stat().st_size
        hash_seconds += time.monotonic() - tick
        target = sample_dir / (group['game'] + '.jsonl.gz')
        candidates = 0; previous = None; tick = time.monotonic()
        with ExitStack() as stack:
            streams = {k: stack.enter_context(gzip.open(paths[k], 'rt', encoding='utf-8'))
                       for k in ('inputs', 'labels', 'index')}
            output = stack.enter_context(gzip.open(target, 'xt', encoding='utf-8', compresslevel=1))
            for physical_row in range(128):
                check_time(deadline)
                lines = [streams[k].readline() for k in ('inputs', 'labels', 'index')]
                require(all(lines), 'Missing registered physical prefix row')
                inputs, label, meta = map(strict_json, lines)
                current = meta['step'], meta['decision']
                require(previous is None or current > previous, 'Source decision order')
                require(current[1] == (0 if previous is None or current[0] != previous[0] else previous[1] + 1),
                        'Missing source decision')
                previous = current
                schema, *_ = decode(inputs, label, meta, game=group['game'], split=group['split'],
                    expected_seed=group['seed'], expected_side=group['side'], expected_opponent=group['opponent'], schema=schema)
                n = len(inputs['candidates'])
                require(total_rows + 1 <= cfg['sample_groups_max'] and total_candidates + n <= cfg['sample_candidates_max'],
                        'Predefined sample exceeds cap; do not truncate or change sample')
                total_rows += 1; total_candidates += n; candidates += n
                output.write(json.dumps(dict(inputs=inputs, label=label, meta=meta,
                    source_row_index=physical_row), separators=(',', ':'), ensure_ascii=False, allow_nan=False) + '\n')
        decode_seconds += time.monotonic() - tick
        groups.append({k: group[k] for k in ('game', 'split', 'seed', 'side', 'opponent')} |
            dict(path=target.relative_to(out).as_posix(), rows=128, candidates=candidates, sha256=sha(target),
                 source_sha256={k: v['sha256'] for k, v in group['source_files'].items()},
                 source_row_start=0, source_row_stop=128))
    require(total_rows == 1280 and len(groups) == 10 and sum(g['rows'] for g in groups if g['split'] == 'train') == 1024,
            'Incomplete fixed sample')
    manifest = dict(schema='task-capacity-sample-v1', sample_only=True, groups=groups,
        source_manifest_sha256=cfg['source_metadata_sha256']['dataset-manifest.json'],
        registration_sha256=sha(out / 'preparation-registration.json'), rows=total_rows, candidates=total_candidates,
        train_rows=1024, validation_rows=256, original_prefix_rows_decoded=total_rows,
        source_compressed_bytes_hashed=source_bytes, source_hash_seconds=hash_seconds,
        prefix_decode_validate_write_seconds=decode_seconds)
    new_json(out / 'sample-manifest.json', manifest)
    return dict(passed=True, sample_only=True, sample_manifest_sha256=sha(out / 'sample-manifest.json'),
                groups=total_rows, candidates=total_candidates, source_hash_seconds=hash_seconds,
                prefix_decode_validate_write_seconds=decode_seconds, model_calls=0)


def compare_outputs(out, runout, deadline, pipeline_config):
    from .pipeline import compare_reference
    cfg = dict(pipeline_config, sample_root=str(out))
    result = compare_reference(runout, out / 'preparation/reference', cfg, deadline)
    new_json(runout / 'comparison.json', result)
    return result


def worker(out, phase, key):
    out = Path(out).resolve(); key_name(key); cfg = verify(out, phase)
    request_path = out / phase / 'job-inputs' / (key + '.json')
    request = read(request_path); deadline = request['deadline_monotonic']
    require(request['phase'] == phase and request['key'] == key, 'Job identity mismatch')
    require(request['freeze_sha256'] == sha(out / (phase + '-freeze.json')), 'Worker freeze mismatch')
    reserved = [r for r in budget_rows(out) if r['phase'] == phase and r['key'] == key]
    require(len(reserved) == 1, 'Missing unique pre-dispatch budget reservation')
    encoded = json.dumps(reserved[0], separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')
    require(hashlib.sha256(encoded).hexdigest() == request['budget_reservation_sha256'], 'Reservation identity mismatch')
    require(reserved[0]['groups'] == request['pipeline_config']['max_evaluated_groups'] and
            reserved[0]['candidates'] == request['pipeline_config']['max_evaluated_candidates'], 'Reserved limits mismatch')
    marker = read(out / (phase + '-started.json'))
    require(deadline <= marker['deadline_monotonic'] and deadline-time.monotonic() <= cfg['per_job_seconds'],
            'Worker deadline exceeds registered phase/job')
    check_time(deadline)
    new_json(out / phase / 'job-started' / (key + '.json'), dict(request_sha256=sha(request_path)))
    resources(out, cfg)
    if key == 'tree':
        from .verify_tree import run_regressions
        result = run_regressions(out / phase / 'tree')
    elif key == 'synthetic':
        from .verify_pipeline import run_regressions
        result = run_regressions(out / phase / 'synthetic')
        require(result['synthetic_evaluated_groups'] <= cfg['synthetic_limits']['evaluations'] and
                result['synthetic_evaluated_candidates'] <= cfg['synthetic_limits']['candidate_evaluations'],
                'Synthetic evaluation cap exceeded')
    elif key == 'sample':
        result = collect_sample(out, cfg, deadline)
    elif key == 'compare':
        result = compare_outputs(out, out / phase / 'candidate', deadline, request['pipeline_config'])
    else:
        from . import pipeline
        from learning.task_bc_v1.model import TaskNetwork
        c = request['pipeline_config']
        c['sample_root'] = str(out)
        model = None if key == 'aggregate' else TaskNetwork(ROOT / cfg['task_model'], cfg['task_model_sha256'])
        target = out / phase / ('reference' if key == 'reference' else 'candidate')
        if key == 'reference': result = pipeline.reference(target, c, deadline, model)
        elif key == 'build': result = pipeline.build_index(target, c, deadline, model)
        elif key.startswith('score-'): result = pipeline.score_shard(target, c, deadline, key[6:], model)
        elif key == 'aggregate': result = pipeline.aggregate(target, c, deadline)
        else: raise ValueError('Unknown capacity job')
    new_json(out / phase / 'job-results' / (key + '.json'), dict(key=key, result=result,
        request_sha256=sha(request_path), freeze_sha256=request['freeze_sha256']))
    require(isinstance(result, dict) and result.get('passed') is True, 'Worker returned a failed result')
    if key == 'reference' or key.startswith('score-'):
        counts = result['counts']; limits = request['pipeline_config']
        require(counts['evaluated_groups'] == counts['completed_groups'] == limits['max_evaluated_groups'] and
                counts['evaluated_candidates'] == counts['completed_candidates'] == limits['max_evaluated_candidates'],
                'Scoring receipt does not cover reserved sample population')
        require(0 <= counts['actual_score_calls'] <= counts['evaluated_groups'] and
                0 <= counts['actual_score_candidates'] <= counts['evaluated_candidates'], 'Scoring counters exceed evaluations')
    check_time(deadline); verify(out, phase); resources(out, cfg)
    new_json(out / phase / 'job-complete' / (key + '.json'), dict(passed=True, key=key, result=result,
        request_sha256=sha(request_path), freeze_sha256=request['freeze_sha256']))


def run(out, phase):
    from .windows_tree import run_owned
    started = time.monotonic(); out = Path(out).resolve()
    cfg = ready(out) if phase == 'formal' else verify(out, phase)
    if phase == 'preparation':
        review = read(out / 'preparation-source-review.json')
        require(review['passed'] and review['freeze_sha256'] == sha(out / 'preparation-freeze.json'),
                'Preparation requires an independent review of the exact frozen source')
    marker = out / (phase + '-started.json')
    deadline = started + cfg[phase + '_runtime_seconds']
    new_json(marker, dict(started_utc=datetime.now(timezone.utc).isoformat(), deadline_monotonic=deadline,
                         freeze_sha256=sha(out / (phase + '-freeze.json'))))
    attempts = []; jobs = []; failure = None
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    previous = sha(marker); pipeline_config = {}
    def persist():
        save(out / phase / 'attempts.json', attempts); save(out / phase / 'jobs-ledger.json', jobs)
    def dispatch(key, groups=0, candidates=0):
        nonlocal previous
        check_time(deadline); verify(out, phase); resources(out, cfg)
        reservation = reserve(out, cfg, phase, key, groups, candidates)
        duration = min(cfg['per_job_seconds'], deadline - time.monotonic())
        c = dict(pipeline_config, freeze_sha256=sha(out / (phase + '-freeze.json')),
                 previous_receipt_sha256=previous, max_evaluated_groups=groups, max_evaluated_candidates=candidates,
                 model_sha256=cfg['task_model_sha256'])
        request = dict(phase=phase, key=key, freeze_sha256=c['freeze_sha256'],
                       deadline_monotonic=min(deadline, time.monotonic() + duration),
                       budget_reservation_sha256=reservation, pipeline_config=c)
        new_json(out / phase / 'job-inputs' / (key + '.json'), request)
        attempt = dict(key=key, groups_reserved=groups, candidates_reserved=candidates,
                       budget_reservation_sha256=reservation, status='started', lifecycle={})
        attempts.append(attempt); persist()
        print('capacity:', phase, key, flush=True)
        path = out / phase / 'logs' / (key + '.log'); path.parent.mkdir(parents=True, exist_ok=True)
        tick = time.monotonic()
        try:
            with path.open('x', encoding='utf-8') as log:
                code = run_owned([sys.executable, '-B', '-m', 'learning.task_bc_diagnostics_capacity_v1.runner',
                                  'worker', '--out', str(out), '--phase', phase, '--key', key],
                    cwd=ROOT, env=environment, stream=log, timeout=min(duration, deadline-time.monotonic()),
                    lifecycle=attempt['lifecycle'], memory_limit_bytes=cfg['resource_limits']['process_tree_private_commit_bytes'])
            require(code == 0, 'Owned capacity worker failed')
            require(attempt['lifecycle']['tree_closed'] and attempt['lifecycle']['active_processes_after_close'] == 0,
                    'Unproven process tree closure')
            require(attempt['lifecycle']['containment_established'] and attempt['lifecycle']['job_handle_closed'] and
                    attempt['lifecycle']['all_process_handles_closed'] and not attempt['lifecycle']['termination_requested'],
                    'A successful worker must finish without forced descendant cleanup')
            receipt = out / phase / 'job-complete' / (key + '.json')
            result = read(receipt); require(result['passed'] and result['key'] == key, 'Missing complete result')
            require(result['request_sha256'] == sha(out / phase / 'job-inputs' / (key + '.json')) and
                    result['freeze_sha256'] == request['freeze_sha256'], 'Worker result request/freeze binding mismatch')
            jobs.append(dict(key=key, receipt_sha256=sha(receipt), result=result['result']))
            previous = sha(receipt); attempt['status'] = 'passed'
            check_time(deadline); return result['result']
        except BaseException:
            attempt['status'] = 'failed'; raise
        finally:
            attempt['elapsed_seconds'] = time.monotonic()-tick; persist()
    try:
        if phase == 'preparation':
            dispatch('tree'); dispatch('synthetic'); dispatch('sample')
        sample = read(out / 'sample-manifest.json')
        pipeline_config['sample_manifest_sha256'] = sha(out / 'sample-manifest.json')
        pipeline_config['reference_freeze_sha256'] = sha(out / 'preparation-freeze.json')
        if phase == 'preparation':
            reference = dispatch('reference', sample['rows'], sample['candidates'])
            pipeline_config['reference_receipt_sha256'] = reference['receipt_sha256']
        else:
            pipeline_config['reference_receipt_sha256'] = read(out / 'formal-freeze.json')['reference_receipt_sha256']
        build = dispatch('build')
        pipeline_config.update(support_index_sha256=build['support_index_sha256'],
                               support_index_receipt_sha256=build['support_index_receipt_sha256'])
        hashes = {}
        for group in sample['groups']:
            result = dispatch('score-' + group['game'], group['rows'], group['candidates'])
            hashes[group['game']] = result['receipt_sha256']
        pipeline_config['shard_receipt_sha256'] = hashes
        aggregated = dispatch('aggregate')
        pipeline_config['aggregate_receipt_sha256'] = aggregated['receipt_sha256']
        dispatch('compare')
        require(all(a['lifecycle']['tree_closed'] for a in attempts), 'Incomplete process closure')
        resources(out, cfg); verify(out, phase); check_time(deadline)
    except BaseException as exc:
        failure = dict(kind=type(exc).__name__, passed=False, retry=False)
        path = out / phase / 'logs'; path.mkdir(parents=True, exist_ok=True)
        (path / 'failure.log').write_text(traceback.format_exc(), encoding='utf-8')
        new_json(out / (phase + '-failure.json'), failure)
    finally:
        persist(); rows = budget_rows(out)
        result = dict(passed=failure is None, phase=phase, sample_only=True, elapsed_seconds=time.monotonic()-started,
            completed_jobs=len(jobs), attempts=len(attempts),
            owned_trees_closed=all(a['lifecycle'].get('tree_closed', False) for a in attempts),
            reserved_real_evaluations=sum(r['groups'] for r in rows), reserved_real_candidates=sum(r['candidates'] for r in rows),
            reservations_not_refunded_on_failure=True, budget_ledger_sha256=sha(out/'budget-reservations.jsonl') if rows else None,
            freeze_sha256=sha(out/(phase+'-freeze.json')), failure=failure, retries=0,
            training_runs=0,new_games=0,official_transitions=0,policy_calls=0,
            full_diagnostics_completed=False, original_diagnostic_stage_acceptance=False, original_policy_admission=False)
        new_json(out / (phase + '-summary.json'), result)
    print('capacity:', phase, 'passed' if failure is None else 'stopped; failure preserved', flush=True)
    return failure is None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('run', 'worker'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--phase', choices=('preparation', 'formal'), required=True)
    parser.add_argument('--key')
    args = parser.parse_args()
    if args.mode == 'worker': worker(args.out, args.phase, args.key)
    else: sys.exit(0 if run(args.out, args.phase) else 1)
