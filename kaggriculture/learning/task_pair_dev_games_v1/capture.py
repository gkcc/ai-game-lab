"""Join a private trace to the existing worker's one official replay pass."""
from contextlib import contextmanager
import gzip
import hashlib
import json
from pathlib import Path

from .panel import _JOB


TURNS = 719
MAX_COMPRESSED_BYTES = 128 * 1024 * 1024
MAX_LINE_BYTES = 8 * 1024 * 1024
MAX_DECODED_BYTES = 512 * 1024 * 1024
MAX_METRICS_BYTES = 8 * 1024 * 1024


def _unique_pairs(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError('Duplicate trace key')
        output[key] = value
    return output


def _nonfinite(_value):
    raise ValueError('Nonfinite trace value')


def _hash_stream(stream):
    stream.seek(0)
    count = 0
    digest = hashlib.sha256()
    while True:
        block = stream.read(65536)
        if not block:
            break
        count += len(block)
        if count > MAX_COMPRESSED_BYTES:
            raise ValueError('Compressed trace budget exceeded')
        digest.update(block)
    return digest.hexdigest()


def feed_trace(path, expected_sha, consumer):
    """Use one opened file descriptor, bounded hashes and bounded gzip lines."""
    with Path(path).open('rb') as raw:
        if _hash_stream(raw) != expected_sha:
            raise ValueError('Trace identity differs')
        raw.seek(0)
        total = 0
        with gzip.GzipFile(fileobj=raw, mode='rb') as trace:
            for step in range(TURNS):
                line = trace.readline(MAX_LINE_BYTES + 1)
                total += len(line)
                if not line or len(line) > MAX_LINE_BYTES or total > MAX_DECODED_BYTES:
                    raise ValueError('Trace line or decoded budget differs')
                turn = json.loads(line, object_pairs_hook=_unique_pairs,
                                  parse_constant=_nonfinite)
                if type(turn) is not dict or type(turn.get('step')) is not int or turn['step'] != step:
                    raise ValueError('Trace turn sequence differs')
                consumer.trace_turn(turn)
            if trace.read(1):
                raise ValueError('Trace has extra data')
        if _hash_stream(raw) != expected_sha:
            raise ValueError('Trace changed while consumed')


def _strict_true(check):
    if not callable(check) or check() is not True:
        raise ValueError('External registration check failed')


@contextmanager
def collect_one(module, *, stage, panel, expected, check, consumer_factory):
    """Wrap only an isolated Worker, preserving its original replay and sink."""
    old_audit, old_reexecute = module.audit, module.reexecute
    captured = []
    active = False

    def audit(out, row, cfg):
        nonlocal active
        if active or captured or Path(out).resolve(strict=False) != stage:
            raise ValueError('Unexpected audit invocation')
        if type(row) is not dict or not _JOB.issubset(row):
            raise ValueError('Incomplete audit row')
        if panel.validate_plan({k: row[k] for k in _JOB}, job=True) != expected:
            raise ValueError('Audit row identity differs')
        if cfg != panel.worker_config():
            raise ValueError('Audit configuration differs')
        _strict_true(check)
        consumer = consumer_factory(expected['side'])
        replay_results = []
        attempts = 0
        active = True

        def reexecute(replay, *, trace=False, label_sink=None):
            nonlocal attempts
            if attempts != 0 or trace is not True or not callable(label_sink):
                raise ValueError('Unexpected replay invocation')
            attempts += 1
            if type(replay) is not dict or type(replay.get('steps')) is not list or len(replay['steps']) != TURNS + 1:
                raise ValueError('Incomplete replay')
            trace_path = stage / 'traces' / (expected['game'] + '.jsonl.gz')
            feed_trace(trace_path, row['trace_sha256'], consumer)

            def both_sinks(event):
                label_sink(event)
                consumer.sink(event)

            official = old_reexecute(replay, trace=True, label_sink=both_sinks)
            replay_results.append((replay, official))
            return official

        module.reexecute = reexecute
        try:
            certificate = old_audit(out, row, cfg)
        finally:
            module.reexecute = old_reexecute
            active = False
        if (len(replay_results) != 1 or type(certificate) is not dict
                or certificate.get('passed') is not True
                or certificate.get('official') != replay_results[0][1]):
            raise ValueError('Original official audit did not pass')
        replay, official = replay_results[0]
        first, last = replay['steps'][0], replay['steps'][-1]
        initial = [module.observation(first, player) for player in (0, 1)]
        final = [module.observation(last, player) for player in (0, 1)]
        statuses = [last[player]['status'] for player in (0, 1)]
        result = consumer.finish(initial, final, statuses, official)
        if (type(result) is not dict or result.get('actor') != expected['side']
                or result.get('official_turns') != TURNS):
            raise ValueError('Consumer result differs')
        _strict_true(check)
        captured.append(dict(schema='task-pair-dev-game-metrics-v1',
                             game=expected['game'], seed_alias=expected['seed_alias'],
                             seed=expected['seed'], opponent=expected['opponent'],
                             side=expected['side'], arm=expected['policy'],
                             task_checkpoint_sha256=expected['task_checkpoint_sha256'],
                             crop_checkpoint_sha256=cfg['crop_checkpoint_sha256'],
                             trace_sha256=row['trace_sha256'],
                             replay_sha256=row['replay_gzip_sha256'], result=result))
        return certificate

    try:
        module.audit = audit
        yield captured
    finally:
        module.audit, module.reexecute = old_audit, old_reexecute


def _encoded(value):
    return (json.dumps(value, ensure_ascii=False, separators=(',', ':'),
                       allow_nan=False) + '\n').encode('utf-8')


def _write_exact(path, data):
    with path.open('xb') as stream:
        if stream.write(data) != len(data):
            raise OSError('Incomplete private output')
    with path.open('rb') as stream:
        if stream.read(MAX_METRICS_BYTES + 1) != data:
            raise ValueError('Private output readback differs')


def store_metrics(module, stage, payload):
    """Write metrics and a private receipt under a separate small budget."""
    if type(payload) is not dict or type(payload.get('game')) is not str:
        raise ValueError('Invalid private metrics')
    key = payload['game']
    if Path(key).name != key or key in ('', '.', '..'):
        raise ValueError('Invalid game alias')
    metrics = _encoded(payload)
    receipt = _encoded(dict(schema='task-pair-dev-game-metrics-receipt-v1', game=key,
                            official_replay_passes=1, metrics_bytes=len(metrics),
                            metrics_sha256=hashlib.sha256(metrics).hexdigest(),
                            budget_bytes=MAX_METRICS_BYTES))
    if len(metrics) + len(receipt) > MAX_METRICS_BYTES:
        raise ValueError('Private metrics budget exceeded')
    relative = 'job-results/' + key + '.metrics.local.json'
    receipt_relative = 'job-results/' + key + '.metrics-receipt.local.json'
    with module.owned_writes(stage, MAX_METRICS_BYTES,
                             allowed_paths=[relative, receipt_relative]):
        _write_exact(stage / relative, metrics)
        _write_exact(stage / receipt_relative, receipt)
