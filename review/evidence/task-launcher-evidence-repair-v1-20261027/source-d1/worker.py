"""Fixed standard-library payload; no game, model or saved numeric data."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scratch',required=True)
    parser.add_argument('--nonce',required=True)
    parser.add_argument('--mode',required=True,choices=('normal','descendant','overflow','failure','cancel'))
    args=parser.parse_args(); scratch=Path(args.scratch)
    def guard(event,values):
        if event in ('subprocess.Popen','os.system','os.fork','os.posix_spawn','os.spawn') and args.mode!='descendant':
            raise RuntimeError('Production process guard rejected launch')
        if event=='import':
            name=values[0]
            forbidden=('numpy','torch','jax','tensorflow','kaggle_environments','bots','models','learning')
            if any(name==p or name.startswith(p+'.') for p in forbidden):
                raise RuntimeError('Unregistered model, engine or learning import')
    sys.addaudithook(guard)
    blocked=None
    if args.mode!='descendant':
        try:
            subprocess.Popen([sys.executable,'-B','-c','raise SystemExit(99)'],creationflags=subprocess.CREATE_NO_WINDOW)
        except RuntimeError:
            blocked=True
        else:
            raise RuntimeError('Process guard failed its negative check')
    value=dict(pid=os.getpid(),parent_pid=os.getppid(),nonce=args.nonce,mode=args.mode,
        executable=sys.executable,base_executable=sys._base_executable,prefix=sys.prefix,
        base_prefix=sys.base_prefix,path=sys.path,process_guard_probe_blocked=blocked)
    pending=scratch/'handshake.pending'
    with pending.open('x',encoding='utf-8') as stream:json.dump(value,stream)
    pending.replace(scratch/'handshake.json')
    end=time.monotonic()+5
    while not (scratch/'go.json').exists():
        if time.monotonic()>end:raise RuntimeError('Authorization handshake expired')
        time.sleep(.01)
    permit=json.loads((scratch/'go.json').read_text(encoding='utf-8'))
    if permit!=dict(nonce=args.nonce,worker_pid=os.getpid()):raise RuntimeError('Invalid authorization')
    if args.mode=='normal':
        os.write(sys.stdout.fileno(),b'worker-start\n')
        os.write(sys.stderr.fileno(),b'worker-stderr\n')
        os.write(sys.stdout.fileno(),b'worker-finish\n')
    elif args.mode=='overflow':os.write(sys.stdout.fileno(),b'x'*65537)
    elif args.mode=='failure':
        os.write(sys.stderr.fileno(),b'known-failure\n');raise SystemExit(7)
    elif args.mode=='cancel':
        os.write(sys.stdout.fileno(),b'waiting\n');time.sleep(30)
    else:
        child=subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(20)'],creationflags=subprocess.CREATE_NO_WINDOW)
        with (scratch/'adversarial-child.json').open('x',encoding='utf-8') as stream:json.dump(dict(pid=child.pid),stream)
        time.sleep(20)


if __name__=='__main__':main()
