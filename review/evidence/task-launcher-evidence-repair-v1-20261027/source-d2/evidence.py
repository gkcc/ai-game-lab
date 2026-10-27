"""Fail-closed public projections with immediate durable checkpoint retention."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import traceback

from . import experiment as exp

PHASES = ('launch', 'ownership', 'handshake', 'runtime', 'image', 'command', 'payload', 'cleanup',
          'acquire_owner', 'handshake_read', 'schema_nonce', 'runtime_fields', 'image_identity',
          'command_roles', 'payload_capture', 'payload_exit', 'cleanup_reap', 'cleanup_readers',
          'cleanup_mark_complete', 'scratch_remove')
REASONS = frozenset({p + '_failed' for p in PHASES} | {
    'launch_failed', 'ownership_ambiguous', 'undeclared_child', 'handshake_missing', 'handshake_invalid',
    'schema_mismatch', 'nonce_mismatch', 'runtime_mismatch', 'image_mismatch', 'command_mismatch',
    'output_limit', 'nonzero_exit', 'cancelled', 'deadline', 'cleanup_failed', 'reader_join_failed',
    'scratch_cleanup_failed', 'private_limit', 'metadata_limit', 'evidence_write_failed', 'opaque_failure'})
ROLES = frozenset(('requested', 'base', 'wrapper', 'worker', 'worker_direct', 'supervisor', 'unexpected', 'unresolved'))
CHECKS = frozenset(('launched', 'owner_acquired', 'identity_matches', 'creation_matches', 'ancestry_matches',
    'handshake_present', 'schema_matches', 'nonce_matches', 'runtime_matches', 'image_matches',
    'command_matches', 'payload_matches', 'exit_matches', 'reaped', 'readers_joined', 'scratch_cleaned',
    'owner_marked_complete', 'all_owned', 'output_within_cap', 'runtime_equivalent', 'requested_matches',
    'base_matches', 'prefix_matches', 'executable_matches', 'worker_registered', 'wrapper_registered',
    'allowed_image', 'allowed_command', 'cancelled', 'deadline_reached', 'nonce_equal', 'mode_equal',
    'worker_present', 'parent_equal', 'executable_equal', 'base_executable_equal', 'prefix_equal',
    'base_prefix_equal', 'path_equal', 'requested_image_equal', 'base_image_equal', 'command_tail_equal',
    'command_argv0_allowed', 'production_guard_blocked', 'output_equal', 'exit_expected',
    'ownership_complete', 'reader_joined'))
COUNTERS = frozenset(('stdout_bytes', 'stderr_bytes', 'payload_bytes', 'process_count', 'owned_count',
    'reaped_count', 'reader_count', 'returncode', 'exit_code', 'elapsed_seconds', 'cpu_seconds',
    'peak_working_set_bytes', 'metadata_bytes', 'output_bytes', 'private_bytes', 'launches', 'sequence',
    'launched_instances', 'owned_instances', 'remaining_instances', 'output_observed_bytes',
    'output_stored_bytes', 'wall_seconds', 'sample_count'))
ERROR_TYPES = frozenset(('EvidenceError', 'RuntimeError', 'ValueError', 'OSError', 'FileNotFoundError',
    'FileExistsError', 'PermissionError', 'TimeoutError', 'AssertionError', 'KeyError', 'TypeError',
    'JSONDecodeError', 'UnicodeDecodeError', 'MemoryError', 'KeyboardInterrupt', 'SystemExit'))


class EvidenceError(RuntimeError):
    def __init__(self, reason, detail=None):
        if reason not in REASONS:
            raise ValueError('Unknown fixed reason')
        self.reason = reason
        super().__init__(reason if detail is None else detail)


def _encode(value):
    return (json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')


def _safe(value):
    if value is None:
        return {}
    if not isinstance(value, dict) or set(value) - {'checks', 'roles', 'image_hashes', 'counters'}:
        raise ValueError('Unknown public evidence field')
    clean = {}
    for key, item in value.items():
        if key == 'checks':
            if not isinstance(item, dict) or set(item) - CHECKS or any(type(v) is not bool for v in item.values()):
                raise ValueError('Unknown boolean check')
        elif key == 'roles':
            if not isinstance(item, list) or any(v not in ROLES for v in item):
                raise ValueError('Unknown normalized role')
        elif key == 'image_hashes':
            if not isinstance(item, dict) or set(item) - ROLES or any(not isinstance(v, str) or not re.fullmatch('[0-9a-f]{64}', v) for v in item.values()):
                raise ValueError('Invalid image hash projection')
        elif key == 'counters':
            if not isinstance(item, dict) or set(item) - COUNTERS or any(type(v) not in (int, float) or not math.isfinite(v) for v in item.values()):
                raise ValueError('Invalid public counter')
        clean[key] = json.loads(json.dumps(item))
    return clean


class Recorder:
    def __init__(self, out=None, name=None):
        self.phase = 'launch'
        self.events = []
        self.private_events = []
        self.primary_failure = None
        self.cleanup_failure = None
        self.evidence_failure = False
        self._out = None
        self._name = None
        self._private_bytes = 0
        if out is not None or name is not None:
            self._bind(out, name)

    def _bind(self, out, name):
        out = Path(out).resolve()
        if out != exp.OUT.resolve() or not isinstance(name, str) or not re.fullmatch('[a-z][a-z0-9_-]{0,47}', name):
            raise ValueError('Invalid recorder output binding')
        if self._out is not None:
            if (out, name) != (self._out, self._name):
                raise ValueError('Recorder binding cannot change')
            return
        self._out, self._name = out, name
        exp.write_bytes_new(out / f'{name}-events.jsonl', b'')
        exp.write_bytes_new(out / 'private' / f'{name}-events.jsonl', b'', private=True)
        for event in self.events:
            self._append(event)
        for event in self.private_events:
            self._append(event, private=True)

    def _append(self, value, private=False):
        data = _encode(value)
        if len(data) > (65536 if private else exp.PUBLIC_RECEIPT_LIMIT):
            raise EvidenceError('private_limit' if private else 'evidence_write_failed')
        if self._out is not None:
            exp.ensure_output_capacity(len(data), private=private)
            path = self._out / ('private' if private else '') / f'{self._name}-events.jsonl'
            with path.open('ab') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())

    def checkpoint(self, name, safe=None):
        if name not in PHASES:
            raise ValueError('Undeclared checkpoint phase')
        self.phase = name
        event = dict(sequence=len(self.events), phase=name, kind='checkpoint', safe=_safe(safe))
        self.events.append(event)
        self._append(event)
        return event

    def private(self, name, value):
        if not isinstance(name, str) or not re.fullmatch('[a-z][a-z0-9_-]{0,47}', name):
            raise ValueError('Invalid private record name')
        event = dict(sequence=len(self.private_events), phase=self.phase, name=name, value=value)
        data = _encode(event)
        if self._private_bytes + len(data) > exp.PRIVATE_LIMIT or len(data) > 65536:
            raise EvidenceError('private_limit')
        self.private_events.append(event)
        self._private_bytes += len(data)
        self._append(event, private=True)

    def _failure(self, exc, cleanup):
        field = 'cleanup_failure' if cleanup else 'primary_failure'
        reason = exc.reason if isinstance(exc, EvidenceError) else self.phase + '_failed'
        failure = dict(phase=self.phase, error_type=type(exc).__name__ if type(exc).__name__ in ERROR_TYPES else 'Exception', reason=reason)
        if getattr(self, field) is None:
            setattr(self, field, failure)
        event = dict(sequence=len(self.events), phase=self.phase, kind=field, failure=failure)
        self.events.append(event)
        try:
            self._append(event)
        except Exception:
            self.evidence_failure = True
        try:
            self.private(field.replace('_', '-') + '-' + str(len(self.private_events)),
                dict(exception_type=type(exc).__name__, message=str(exc), traceback=''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))))
        except Exception:
            self.evidence_failure = True
        return failure

    def fail(self, exc):
        return self._failure(exc, False)

    def cleanup_fail(self, exc):
        return self._failure(exc, True)

    def rejection(self, reason):
        if reason not in {'undeclared_child','output_limit','cancelled','nonzero_exit'}:
            raise ValueError('Unknown expected rejection')
        event=dict(sequence=len(self.events),phase=self.phase,kind='expected_rejection',reason=reason)
        self.events.append(event);self._append(event)
        return event

    def receipt(self):
        value = dict(schema='launcher-evidence-receipt-v1', phase=self.phase,
            primary_failure=self.primary_failure, cleanup_failure=self.cleanup_failure,
            evidence_failure=self.evidence_failure, events=self.events, private_records=len(self.private_events),
            private_bytes=self._private_bytes)
        if len(_encode(value)) > exp.PUBLIC_RECEIPT_LIMIT:
            raise EvidenceError('evidence_write_failed')
        return value

    def persist(self, out=None, name=None):
        if out is not None or name is not None:
            self._bind(out, name)
        value = self.receipt()
        if self._out is not None:
            exp.write_new(self._out / f'{self._name}-evidence.json', value)
        return value
