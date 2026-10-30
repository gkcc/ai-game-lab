"""Read the locally bound toy-failure evidence and emit only aggregate data.

This entry never runs a game, loads a model, changes a budget, or retries an
experiment. Local identity records and raw evidence are inputs, not outputs.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

from learning.task_pair_dev_games_v1.capture import _nonfinite, _unique_pairs
from . import analysis


FAILURE_COMMIT = '78f14984f32ef68028fd1a8798c51d51e027cb81'
TOY_REGISTRATION = '4d7f480fa41cadcbae8427722e600472cfd2fe8f'
MAX_RECORD_BYTES = 64 * 1024


def _read_record(path):
    with Path(path).open('rb') as stream:
        raw = stream.read(MAX_RECORD_BYTES + 1)
    if not raw or len(raw) > MAX_RECORD_BYTES:
        raise ValueError('Invalid local evidence record length')
    return raw, json.loads(raw, object_pairs_hook=_unique_pairs, parse_constant=_nonfinite)


def _path(root, name):
    if type(name) is not str or not name or '\\' in name or ':' in name:
        raise ValueError('Invalid bound artifact name')
    relative = PurePosixPath(name)
    if relative.is_absolute() or relative.as_posix() != name or any(part in ('.', '..') for part in relative.parts):
        raise ValueError('Invalid bound artifact name')
    path = root.joinpath(*relative.parts).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError('Bound artifact escapes root') from None
    return path


def _validate_files(root, files):
    if type(files) is not dict or len(files) != 16:
        raise ValueError('Unexpected failure artifact count')
    for name, identity in files.items():
        if (type(identity) is not dict or set(identity) != {'bytes', 'sha256'}
                or type(identity['bytes']) is not int or not 0 <= identity['bytes'] <= analysis.MAX_COMPRESSED_BYTES
                or type(identity['sha256']) is not str or len(identity['sha256']) != 64
                or any(char not in '0123456789abcdef' for char in identity['sha256'])):
            raise ValueError('Invalid bound artifact identity')
        count, digest = 0, hashlib.sha256()
        with _path(root, name).open('rb') as stream:
            while True:
                block = stream.read(65536)
                if not block:
                    break
                count += len(block)
                if count > analysis.MAX_COMPRESSED_BYTES:
                    raise ValueError('Failure artifact exceeds bound')
                digest.update(block)
        if count != identity['bytes'] or digest.hexdigest() != identity['sha256']:
            raise ValueError('Failure artifact identity changed')


def _validate_binding(bound, evidence_raw, source):
    if (type(bound) is not dict or bound.get('schema') != 'task-pair-capacity-v1-binding'
            or bound.get('failure_commit') != FAILURE_COMMIT
            or type(bound.get('failure_artifacts_verified')) is not int or bound['failure_artifacts_verified'] != 16
            or type(bound.get('failure_record_bytes')) is not int
            or bound['failure_record_bytes'] != len(evidence_raw)
            or bound.get('failure_record_sha256') != hashlib.sha256(evidence_raw).hexdigest()
            or type(source) is not dict or source.get('schema') != 'task-pair-toy-smoke-failure-private-evidence-v1'
            or source.get('registration_commit') != TOY_REGISTRATION
            or type(source.get('attempted_games')) is not int or source['attempted_games'] != 1
            or type(source.get('accepted_games')) is not int or source['accepted_games'] != 0
            or source.get('retry_allowed') is not False):
        raise ValueError('Failure evidence binding differs')
    files, trace = source.get('files'), bound.get('trace')
    if type(trace) is not dict or set(trace) != {'relative_name', 'sha256', 'bytes'}:
        raise ValueError('Invalid trace binding')
    name = trace['relative_name']
    if (type(files) is not dict or len(files) != 16 or any(type(key) is not str for key in files)
            or type(name) is not str or not name.endswith('.jsonl.gz')
            or [key for key in files if key.endswith('.jsonl.gz')] != [name]
            or files.get(name) != {'sha256': trace['sha256'], 'bytes': trace['bytes']}):
        raise ValueError('Trace selection differs from failure evidence')
    return files, trace


def run(*, evidence, binding, artifact_root):
    """Validate the prior local binding and summarize its one recorded trace."""
    try:
        bound_raw, bound = _read_record(binding)
        evidence_raw, source = _read_record(evidence)
        files, trace = _validate_binding(bound, evidence_raw, source)
        name = trace['relative_name']
        root = Path(artifact_root).resolve(strict=True)
        _validate_files(root, files)
        with _path(root, name).open('rb') as stream:
            result = analysis.analyze(stream, trace['sha256'], trace['bytes'])
        _validate_files(root, files)
        if _read_record(binding)[0] != bound_raw or _read_record(evidence)[0] != evidence_raw:
            raise ValueError('Local identity record changed during analysis')
        return result
    except OSError:
        raise RuntimeError('Local failure evidence unavailable') from None


class _Parser(argparse.ArgumentParser):
    def error(self, _message):
        raise SystemExit('Capacity analysis stopped; invalid arguments.')


def _result_path(result, artifact_root):
    root = Path(artifact_root).resolve(strict=True)
    path = Path(result).resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError:
        return path
    raise ValueError('Output must be outside source artifacts')


def main(argv=None):
    parser = _Parser(description=__doc__)
    parser.add_argument('--evidence', required=True)
    parser.add_argument('--binding', required=True)
    parser.add_argument('--artifact-root', required=True)
    parser.add_argument('--result', required=True)
    args = parser.parse_args(argv)
    try:
        result_path = _result_path(args.result, args.artifact_root)
        result = run(evidence=args.evidence, binding=args.binding, artifact_root=args.artifact_root)
        with result_path.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(result, stream, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
    except Exception:
        raise SystemExit('Capacity analysis stopped; no result accepted.') from None


if __name__ == '__main__':
    main()
