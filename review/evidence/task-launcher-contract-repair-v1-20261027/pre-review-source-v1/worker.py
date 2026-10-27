"""Tiny handshake-gated payload for the fixed launcher acceptance cases."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scratch', required=True)
    parser.add_argument('--nonce', required=True)
    parser.add_argument('--mode', required=True, choices=('observe', 'normal', 'descendant', 'overflow', 'failure', 'cancel'))
    args = parser.parse_args()
    scratch = Path(args.scratch)

    def guard(event, values):
        if event in ('subprocess.Popen', 'os.system', 'os.fork', 'os.posix_spawn') and args.mode != 'descendant':
            raise RuntimeError('Synthetic production worker process guard')
        if event == 'import' and any(values[0] == p or values[0].startswith(p+'.')
                                    for p in ('numpy', 'torch', 'jax', 'kaggle_environments', 'bots')):
            raise RuntimeError('Model or official engine import refused')
    sys.addaudithook(guard)
    blocked = None
    if args.mode != 'descendant':
        try:
            subprocess.Popen([sys.executable, '-B', '-c', 'raise SystemExit(99)'],
                             creationflags=subprocess.CREATE_NO_WINDOW)
        except RuntimeError:
            blocked = True
        else:
            raise RuntimeError('Production process guard did not reject launch')
    handshake = dict(pid=os.getpid(), parent_pid=os.getppid(), nonce=args.nonce,
        executable=sys.executable, base_executable=sys._base_executable,
        prefix=sys.prefix, base_prefix=sys.base_prefix, path=sys.path,
        process_guard_probe_blocked=blocked, mode=args.mode)
    pending = scratch/'handshake.pending'
    with pending.open('x', encoding='utf-8') as stream:
        json.dump(handshake, stream)
    pending.replace(scratch/'handshake.json')
    deadline = time.monotonic()+5
    while not (scratch/'go.json').is_file():
        if time.monotonic() >= deadline:
            raise RuntimeError('Handshake authorization timeout')
        time.sleep(.01)
    permit = json.loads((scratch/'go.json').read_text(encoding='utf-8'))
    if permit != dict(nonce=args.nonce, worker_pid=os.getpid()):
        raise RuntimeError('Incorrect worker authorization')
    if args.mode in ('observe', 'normal'):
        print('worker-start', flush=True)
        print('worker-stderr', file=sys.stderr, flush=True)
        print('worker-finish', flush=True)
    elif args.mode == 'overflow':
        os.write(sys.stdout.fileno(), b'x'*65537)
    elif args.mode == 'failure':
        print('known-failure', file=sys.stderr, flush=True)
        raise SystemExit(7)
    elif args.mode == 'cancel':
        print('waiting', flush=True)
        time.sleep(30)
    elif args.mode == 'descendant':
        child = subprocess.Popen([sys.executable, '-B', '-c', 'import time; time.sleep(20)'],
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        with (scratch/'adversarial-child.json').open('x', encoding='utf-8') as stream:
            json.dump(dict(pid=child.pid), stream)
        time.sleep(20)


if __name__ == '__main__':
    main()
