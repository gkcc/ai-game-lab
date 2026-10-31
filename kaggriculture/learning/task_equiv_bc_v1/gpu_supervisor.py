"""Use the existing Linux pidfd session guard for one private GPU child."""
import argparse
import json
import os
from pathlib import Path
import runpy
import signal
import sys


class Cancellation:
    """Convert TERM/INT to a sampled cancel until owned-session cleanup ends."""
    def __init__(self, external):
        self.external, self.signalled, self.previous = external, False, {}

    def __enter__(self):
        def on_signal(_signum, _frame):
            self.signalled = True
        self.previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        for sig in self.previous:
            signal.signal(sig, on_signal)
        return self

    def requested(self):
        return bool(self.signalled or self.external())

    def __exit__(self, *_args):
        for sig, handler in self.previous.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True)
    parser.add_argument('--registration-commit', required=True)
    args = parser.parse_args()
    stage = Path(args.stage)
    if (os.name != 'posix' or not sys.flags.isolated or not sys.flags.dont_write_bytecode
            or not stage.is_absolute() or stage.resolve(strict=True) != stage
            or stage.name != 'task-equivalent-bc-v3-20261031' or stage.parent.name != 'artifacts'):
        raise ValueError('Invalid GPU supervision context')
    bootstrap = runpy.run_path(str(Path(__file__).with_name('gpu_bootstrap.py')))
    imports, project_root = bootstrap['bind']()
    from learning.task_pair_pidfd_guard_v1.guard import GATE_SHA256, supervise
    imports.verify_origins(project_root)
    cancel = stage / 'gpu-cancel.local.json'
    command = [sys.executable, '-I', '-B', str(Path(__file__).with_name('gpu_worker.py')),
               '--stage', str(stage), '--registration-commit', args.registration_commit]
    with Cancellation(cancel.exists) as cancellation:
        report = supervise(command, 880, cancellation.requested, term_seconds=2,
                           cleanup_seconds=15, gate_seconds=10, expected_gate_sha256=GATE_SHA256)
        with (stage / 'gpu-supervision.local.json').open('xb') as stream:
            stream.write((json.dumps(report, allow_nan=False) + '\n').encode())
        if (report.get('completed') is not True or report.get('child_exit_code') != 0
                or report.get('owned_session_empty_before_reap') is not True
                or report.get('direct_child_reaped') is not True or report.get('used_pidfd') is not True
                or cancellation.requested()):
            raise RuntimeError('Owned GPU child did not complete')
    print(json.dumps({'phase': 'gpu_supervisor', 'status': 'passed'}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        print(json.dumps({'phase': 'gpu_supervisor', 'status': 'failed'}), file=sys.stderr, flush=True)
        raise SystemExit(1)
