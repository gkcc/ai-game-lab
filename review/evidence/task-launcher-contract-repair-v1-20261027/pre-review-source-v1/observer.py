"""One bounded synthetic observation of the existing interpreter launch shape.

Private PID, command and runtime path evidence lives only in owned scratch.
This module records the process chain; it does not whitelist that chain for a
production analysis or import the failed historical analysis entrypoint.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import uuid

import psutil
from learning.task_anchor_once_v1.processes import CapturedProcess
from .experiment import ROOT, OUT, read, write_new, verify, deadline_check

NORMAL_OUTPUT = b'worker-start\nworker-stderr\nworker-finish\n'
MODULE = 'learning.task_launcher_contract_repair_v1.worker'


def ownership_helper():
    path = Path(os.environ['TASK_LAUNCHER_ARTIFACT_HELPER'])
    spec = importlib.util.spec_from_file_location('launcher_owned_scratch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Images:
    def __init__(self):
        self.cache, self.bytes_read = {}, 0

    def identity(self, path):
        path = Path(path)
        key = (str(path.resolve()).casefold(), path.stat().st_size, path.stat().st_mtime_ns)
        if key not in self.cache:
            if self.bytes_read + key[1] > 1024*1024:
                raise RuntimeError('Executable identity read cap exceeded')
            raw = path.read_bytes(); self.bytes_read += len(raw)
            self.cache[key] = dict(basename=path.name, sha256=hashlib.sha256(raw).hexdigest())
        return self.cache[key]


def runtime():
    return dict(executable=sys.executable, base_executable=sys._base_executable,
        prefix=sys.prefix, base_prefix=sys.base_prefix, path=sys.path)


def environment_matches(handshake):
    return {k: handshake[k] == value for k, value in runtime().items()}


def observe_node(process, images):
    with process.oneshot():
        cpu = process.cpu_times()
        return dict(pid=process.pid, parent_pid=process.ppid(), created=process.create_time(),
            image=images.identity(process.exe()), cmdline=process.cmdline(),
            working_set=process.memory_info().rss, cpu=cpu.user+cpu.system)


def snapshot(launched, images, known):
    processes = [psutil.Process(launched.pid)]
    processes += processes[0].children(recursive=True)
    nodes = []
    for process in processes:
        try:
            node = observe_node(process, images)
        except psutil.NoSuchProcess:
            continue
        old = known.get(node['pid'])
        if old is not None and old['created'] != node['created']:
            raise RuntimeError('Owned PID creation identity changed')
        known[node['pid']] = node
        nodes.append(node)
    return nodes


def normalized_graph(nodes, launched_pid, worker_pid, supervisor_pid):
    roles = {supervisor_pid: 'supervisor', launched_pid: 'launched_process', worker_pid: 'worker'}
    if launched_pid == worker_pid:
        roles[worker_pid] = 'worker_direct'
    for node in nodes:
        if node['pid'] not in roles:
            roles[node['pid']] = 'undeclared_'+str(len([v for v in roles.values() if v.startswith('undeclared_')])+1)
    return [dict(role=roles[node['pid']], parent_role=roles.get(node['parent_pid'], 'outside_owned_chain'),
        image=node['image'], creation_identity_observed=True) for node in nodes]


def reap(captured, known):
    # Match creation identities before terminating, never use PID alone.
    identities = {(pid, node['created']) for pid, node in known.items()}
    if captured.process.poll() is None:
        root = psutil.Process(captured.process.pid)
        for child in root.children(recursive=True):
            try:
                identities.add((child.pid, child.create_time()))
            except psutil.NoSuchProcess:
                pass
        identities.add((root.pid, root.create_time()))
    targets = []
    for pid, created in identities:
        try:
            process = psutil.Process(pid)
            if process.create_time() != created:
                continue  # The owned instance has exited; do not touch reuse.
            targets.append(process)
        except psutil.NoSuchProcess:
            pass
    for process in reversed(targets):
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(targets, timeout=5)
    captured.stop()
    if alive:
        raise RuntimeError('Owned process instances survived cleanup')
    return dict(owned_instances_observed=len(identities), owned_instances_remaining=0,
                reader_joined=not captured.thread.is_alive(), launched_process_reaped=True)


def command(scratch, nonce, mode):
    return [sys.executable, '-B', '-m', MODULE, '--scratch', str(scratch), '--nonce', nonce, '--mode', mode]


def wait_handshake(captured, scratch, images, known, start):
    while time.monotonic()-start < 5:
        deadline_check()
        captured.check()
        if captured.process.poll() is not None:
            raise RuntimeError('Worker exited before handshake')
        nodes = snapshot(captured.process, images, known)
        if len(nodes)+1 > 5:
            raise RuntimeError('Owned process count exceeded')
        total = sum(n['working_set'] for n in nodes)+psutil.Process().memory_info().rss
        if total > 512*1024*1024:
            raise RuntimeError('Sampled memory cap exceeded')
        if (scratch/'handshake.json').is_file():
            path=scratch/'handshake.json'
            if path.stat().st_size>16384:
                raise RuntimeError('Handshake metadata cap exceeded')
            return json.loads(path.read_text(encoding='utf-8')), nodes, total
        time.sleep(.01)
    raise RuntimeError('Handshake deadline exceeded')


def observe():
    verify('observation'); deadline_check()
    write_new(OUT/'observation-started.json', dict(attempt_consumed=True, launch_number=1, mode='synthetic_only'))
    started = time.monotonic(); images = Images(); known = {}
    helper = ownership_helper(); result = None; error = None
    parent = psutil.Process(); parent.cpu_affinity([parent.cpu_affinity()[0]])
    nonce = uuid.uuid4().hex; captured = None; scratch_path = None; cleanup = None
    try:
        with helper.scratch_directory('task-launcher-contract-repair-v1', root=Path(tempfile.gettempdir())/'task-launcher-contract-repair-v1') as scratch:
            scratch_path = scratch
            captured = CapturedProcess(command(scratch,nonce,'observe'),cwd=ROOT,log=scratch/'capture.log',limit_bytes=65536)
            try:
                handshake,nodes,total = wait_handshake(captured,scratch,images,known,started)
                if handshake['nonce'] != nonce or handshake['mode'] != 'observe':
                    raise RuntimeError('Wrong handshake identity')
                checks = environment_matches(handshake)
                if not all(checks.values()):
                    raise RuntimeError('Worker runtime environment differs')
                worker = next(n for n in nodes if n['pid']==handshake['pid'])
                if worker['parent_pid'] != handshake['parent_pid']:
                    raise RuntimeError('Reported parent differs from observed parent')
                requested = images.identity(sys.executable)
                base = images.identity(sys._base_executable)
                argv_checks=[dict(tail_equal=n['cmdline'][1:]==command(scratch,nonce,'observe')[1:],
                    first_is_requested=n['cmdline'][0]==sys.executable,
                    first_is_base=n['cmdline'][0]==sys._base_executable) for n in nodes]
                observed = dict(role_graph=normalized_graph(nodes,captured.process.pid,handshake['pid'],os.getpid()),
                    direct_parent_matches_old_contract=handshake['parent_pid']==os.getpid(),
                    launched_pid_equals_worker_pid=captured.process.pid==handshake['pid'],
                    requested_executable_image=requested, base_executable_image=base,
                    runtime_environment_equal=checks, process_guard_probe_blocked=handshake['process_guard_probe_blocked'],
                    command_roles=argv_checks,
                    command_roles_match=all(c['tail_equal'] and (c['first_is_requested'] or c['first_is_base']) for c in argv_checks),
                    process_creation_identities_observed=len(nodes), sampled_total_working_set_bytes=total)
                with (scratch/'go.json').open('x',encoding='utf-8') as stream:
                    json.dump(dict(nonce=nonce,worker_pid=handshake['pid']),stream)
                captured.process.wait(timeout=5); capture = captured.finish()
                data=(scratch/'capture.log').read_bytes()
                if captured.process.returncode != 0 or data != NORMAL_OUTPUT:
                    raise RuntimeError('Synthetic observation payload failed')
                result=dict(observed,observed_stdout_stderr_sha256=hashlib.sha256(data).hexdigest(),
                    output_bytes=len(data),capture=capture,passed=True)
            finally:
                cleanup=reap(captured,known)
        result['scratch_cleaned']=not scratch_path.exists()
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc))
    finally:
        value=dict(schema='launcher-contract-observation-v1',passed=error is None,error=error,
            result=result,cleanup=cleanup,scratch_cleaned=scratch_path is not None and not scratch_path.exists(),
            wall_seconds=time.monotonic()-started,image_identity_read_bytes=images.bytes_read,
            launch_attempts=1,raw_or_weight_reads=0,neural_forwards=0,new_games=0,official_transitions=0)
        write_new(OUT/'observation-result.json',value)
    if error:
        raise RuntimeError('Observation closed; no retry: '+error['message'])
    return value


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(observe()))
