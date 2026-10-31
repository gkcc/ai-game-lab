"""Run one registered GPU worker in an owned Linux pidfd-guarded session."""
import argparse
import json
import os
from pathlib import Path
import runpy
import secrets
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True)
    parser.add_argument('--registration-commit', required=True)
    args = parser.parse_args()
    if os.name != 'posix' or not sys.flags.isolated or not sys.flags.dont_write_bytecode:
        raise ValueError('Invalid GPU supervision context')
    root = Path(__file__).resolve().parents[2]
    bootstrap = runpy.run_path(str(root / 'learning' / 'task_equiv_bc_v1' / 'gpu_bootstrap.py'))
    imports, checked_root = bootstrap['bind']()
    if checked_root != root:
        raise ValueError('Project root differs')
    from learning.task_pair_pidfd_guard_v1.guard import GATE_SHA256, supervise, _stat
    from learning.task_equiv_bc_v1.gpu_supervisor import Cancellation
    from learning.task_water_residual_v1.experiment import verify_registration, _json, _read, _write, MAX_METADATA
    stage = Path(args.stage)
    verify_registration(args.registration_commit, stage)
    attempt = _json(_read(stage / 'host-attempt.local.json', MAX_METADATA))
    if (attempt != dict(schema='water-residual-host-attempt-v1', commit=args.registration_commit,
                        retry_allowed=False) or (stage / 'gpu-cancel.local.json').exists()):
        raise ValueError('Host attempt is absent or canceled')
    identity = _stat(os.getpid())
    if identity is None or identity.state in (b'Z', b'X'):
        raise ValueError('Supervisor identity is unavailable')
    nonce = secrets.token_hex(16)
    marker = dict(schema='water-residual-supervisor-attempt-v1', registration_commit=args.registration_commit,
                  nonce=nonce, supervisor_pid=identity.pid, supervisor_starttime=identity.starttime,
                  retry_allowed=False)
    _write(stage / 'gpu-supervisor-attempt.local.json',
           (json.dumps(marker, sort_keys=True, separators=(',', ':')) + '\n').encode('utf-8'))
    imports.verify_origins(root)
    command = [sys.executable, '-I', '-B', str(Path(__file__).with_name('gpu_worker.py')),
               '--stage', str(stage), '--registration-commit', args.registration_commit, '--nonce', nonce]
    with Cancellation((stage / 'gpu-cancel.local.json').exists) as cancellation:
        report = supervise(command, 600, cancellation.requested, term_seconds=2,
                           cleanup_seconds=15, gate_seconds=10, expected_gate_sha256=GATE_SHA256)
        with (stage / 'gpu-supervision.local.json').open('xb') as stream:
            raw = (json.dumps(report, allow_nan=False) + '\n').encode('utf-8')
            if stream.write(raw) != len(raw):
                raise OSError('Incomplete GPU supervision receipt')
        if (report.get('completed') is not True or report.get('child_exit_code') != 0
                or report.get('owned_session_empty_before_reap') is not True
                or report.get('direct_child_reaped') is not True or report.get('used_pidfd') is not True
                or cancellation.requested()):
            raise RuntimeError('Owned GPU worker did not complete')
    print(json.dumps({'phase': 'gpu_supervisor', 'status': 'passed'}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        print(json.dumps({'phase': 'gpu_supervisor', 'status': 'failed'}), file=sys.stderr, flush=True)
        raise SystemExit(1) from None
