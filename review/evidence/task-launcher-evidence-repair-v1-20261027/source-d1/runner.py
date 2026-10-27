"""Real Windows launch path, durable phase evidence, and fixed synthetic cases."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
import uuid

import psutil
from learning.task_launcher_contract_repair_v1.observer import Images, observe_node, snapshot, reap, ownership_helper
from .experiment import ROOT,OUT,read,write_new,write_bytes_new,verify,deadline_check,meter,ensure_output_capacity
from .evidence import Recorder, EvidenceError
from .capture import OwnedCapture as Capture,OutputLimitExceeded

MODULE='learning.task_launcher_evidence_repair_v1.worker'
NORMAL=b'worker-start\nworker-stderr\nworker-finish\n'
APPROVED_IMAGE_HASHES=set()


def command(scratch,nonce,mode):
    return [sys.executable,'-B','-m',MODULE,'--scratch',str(scratch),'--nonce',nonce,'--mode',mode]


def expected_runtime():
    return dict(executable=sys.executable,base_executable=sys._base_executable,
                prefix=sys.prefix,base_prefix=sys.base_prefix,path=sys.path)


def graph(nodes,launch_pid,worker_pid):
    roles={os.getpid():'supervisor',launch_pid:'wrapper'}
    if worker_pid is not None:roles[worker_pid]='worker_direct' if worker_pid==launch_pid else 'worker'
    result=[]
    for node in nodes:
        image=node['image']
        row=dict(role=roles.get(node['pid'],'unexpected'),parent_role=roles.get(node['parent_pid'],'unresolved'),
            image_is_registered=image['sha256'] in APPROVED_IMAGE_HASHES,creation_identity_observed=True)
        if row['image_is_registered']:row['image_sha256']=image['sha256']
        result.append(row)
    return result


def safe_roles(nodes,launch_pid,worker_pid):
    return [n['role'] for n in graph(nodes,launch_pid,worker_pid)]


def fixture(name,mode,images,*,enforce_chain=False):
    # Reserve more than the bounded merged log + handshake + ownership marker.
    os.environ['TASK_LAUNCHER_TEMP_BYTES']='131072'
    ensure_output_capacity(0)
    deadline_check();rec=Recorder(OUT,name);start=time.monotonic();parent_cpu=time.process_time()
    private=dict(mode=mode);known={};nodes=[];captured=None;child_pid=None;scratch_path=None
    primary=None;cleanup_error=None;reason='normal';worker_pid=None;summary={};peak=0;cpu=0;samples=0
    result=dict(name=name,mode=mode,partial_role_graph=[],runtime_checks=None,command_checks=None)
    helper=None;scratch_context=None;scratch=None
    try:
        rec.checkpoint('ownership')
        helper=ownership_helper();scratch_context=helper.scratch_directory('task-launcher-evidence-repair-v1',root=Path(tempfile.gettempdir())/'task-launcher-evidence-repair-v1')
        scratch=scratch_context.__enter__();scratch_path=scratch
        nonce=uuid.uuid4().hex;cmd=command(scratch,nonce,mode);private['command']=cmd
        rec.checkpoint('launch')
        captured=Capture(log=scratch/'capture.log',limit_bytes=65536)
        captured.start(cmd,cwd=ROOT)
        child_pid=captured.process.pid
        rec.checkpoint('acquire_owner',{'roles':['unresolved']})
        helper.protect_children(scratch,[{'pid':child_pid,'process_identity':helper.process_identity(child_pid)[1]}],root=scratch.parent,complete=False)
        rec.checkpoint('handshake_read',{'checks':{'handshake_present':False}})
        handshake=None;last_pids=None
        while time.monotonic()-start<5:
            deadline_check()
            if captured.process.poll() is not None:
                result['early_exit_code']=captured.process.returncode
                raise RuntimeError('Worker exited before its handshake')
            nodes=snapshot(captured.process,images,known)
            private['observed_nodes']=list(known.values())
            peak=max(peak,sum(n['working_set'] for n in nodes)+psutil.Process().memory_info().rss)
            cpu=max(cpu,sum(n['cpu'] for n in nodes));samples+=1
            if len(nodes)+1>5 or peak>512*1024*1024:raise RuntimeError('Registered process or memory cap exceeded')
            pids=tuple(n['pid'] for n in nodes)
            if pids!=last_pids:
                result['partial_role_graph']=graph(nodes,child_pid,None)
                rec.checkpoint('handshake_read',{'roles':safe_roles(nodes,child_pid,None),'counters':{'owned_instances':len(nodes)}})
                rec.private('partial_processes',private['observed_nodes']);last_pids=pids
            path=scratch/'handshake.json'
            if path.exists():
                if path.stat().st_size>16384:raise RuntimeError('Handshake exceeds metadata cap')
                handshake=json.loads(path.read_text(encoding='utf-8'));break
            time.sleep(.01)
        if handshake is None:raise RuntimeError('Handshake timed out')
        private['handshake']=handshake;rec.private('handshake',handshake)
        worker_pid=handshake.get('pid');result['partial_role_graph']=graph(nodes,child_pid,worker_pid)
        rec.checkpoint('schema_nonce',{'checks':{'handshake_present':True,
            'nonce_equal':handshake.get('nonce')==nonce,'mode_equal':handshake.get('mode')==mode},
            'roles':safe_roles(nodes,child_pid,worker_pid)})
        if handshake.get('nonce')!=nonce or handshake.get('mode')!=mode:raise RuntimeError('Handshake token/mode mismatch')
        rec.checkpoint('runtime_fields')
        checks={k+'_equal':handshake.get(k)==v for k,v in expected_runtime().items()}
        result['runtime_checks']=checks;private['expected_runtime']=expected_runtime()
        rec.checkpoint('runtime_fields',{'checks':checks})
        if not all(checks.values()):raise RuntimeError('Actual runtime environment mismatch')
        # Refresh after handshake: child may have appeared after an earlier snapshot.
        nodes=snapshot(captured.process,images,known);private['observed_nodes']=list(known.values())
        result['partial_role_graph']=graph(nodes,child_pid,worker_pid)
        rec.checkpoint('image_identity',{'roles':safe_roles(nodes,child_pid,worker_pid)})
        worker=next((n for n in nodes if n['pid']==worker_pid),None)
        result['worker_present']=worker is not None
        rec.checkpoint('image_identity',{'checks':{'worker_present':worker is not None}})
        if worker is None:raise RuntimeError('Handshaking worker absent from owned process snapshot')
        requested=images.identity(sys.executable);base=images.identity(sys._base_executable)
        result['runtime_images']=dict(requested=requested,base=base)
        rec.checkpoint('image_identity',{'image_hashes':{'requested':requested['sha256'],'base':base['sha256']},
             'checks':{'parent_equal':worker['parent_pid']==handshake['parent_pid']}})
        if worker['parent_pid']!=handshake['parent_pid']:raise RuntimeError('Observed parent mismatch')
        rec.checkpoint('command_roles')
        command_checks=[dict(tail_equal=n['cmdline'][1:]==cmd[1:],
            first_allowed=n['cmdline'][0] in (sys.executable,sys._base_executable)) for n in nodes]
        result['command_checks']=command_checks
        rec.checkpoint('command_roles',{'checks':{'command_tail_equal':all(n['tail_equal'] for n in command_checks),
            'command_argv0_allowed':all(n['first_allowed'] for n in command_checks)}})
        if not all(n['tail_equal'] and n['first_allowed'] for n in command_checks):raise RuntimeError('Unexpected interpreter command role')
        # Permit only this measured direct or two-node launch shape, never a Python-name exception.
        launched=next(n for n in nodes if n['pid']==child_pid)
        narrow=(len(nodes)==1 and child_pid==worker_pid and worker['parent_pid']==os.getpid()
                and worker['image']==base) or (len(nodes)==2 and child_pid!=worker_pid
                and launched['parent_pid']==os.getpid() and worker['parent_pid']==child_pid
                and launched['image']==requested and worker['image']==base)
        result['narrow_chain_identified']=narrow
        if not narrow:raise RuntimeError('Unidentified owned interpreter chain')
        if enforce_chain:
            contract=read(OUT/'final-chain-contract.json')
            actual=sorted(graph(nodes,child_pid,worker_pid),key=lambda x:x['role'])
            if actual!=contract['role_graph']:raise RuntimeError('Measured chain differs from accepted diagnostic contract')
        if mode!='descendant' and handshake.get('process_guard_probe_blocked') is not True:raise RuntimeError('Production process guard not confirmed')
        allowed={(n['pid'],n['created']) for n in nodes}
        result['old_direct_parent_contract_matches']=handshake['parent_pid']==os.getpid()
        rec.checkpoint('payload_capture',{'checks':{'production_guard_blocked':handshake.get('process_guard_probe_blocked') is True}})
        with (scratch/'go.json').open('x',encoding='utf-8') as stream:json.dump(dict(nonce=nonce,worker_pid=worker_pid),stream)
        payload_started=time.monotonic()
        while captured.process.poll() is None:
            deadline_check()
            try:
                nodes=snapshot(captured.process,images,known)
            except psutil.NoSuchProcess:
                if captured.process.poll() is not None:break
                raise
            private['observed_nodes']=list(known.values())
            peak=max(peak,sum(n['working_set'] for n in nodes)+psutil.Process().memory_info().rss)
            cpu=max(cpu,sum(n['cpu'] for n in nodes));samples+=1
            if len(nodes)+1>5 or peak>512*1024*1024:raise RuntimeError('Registered process or memory cap exceeded')
            if any((n['pid'],n['created']) not in allowed for n in nodes):
                result['rejected_role_graph']=graph(nodes,child_pid,worker_pid)
                reason='unexpected_descendant';break
            if captured.error is not None:
                if isinstance(captured.error,OutputLimitExceeded):reason='output_overflow';break
                captured.check()
            if mode=='cancel' and time.monotonic()-payload_started>=.3:
                reason='cancelled';break
            if time.monotonic()-start>10:raise RuntimeError('Fixture wall limit exceeded')
            if cpu+time.process_time()-parent_cpu>20:raise RuntimeError('Fixture CPU limit exceeded')
            time.sleep(.01)
        rec.checkpoint('payload_exit')
        if reason=='normal':
            captured.process.wait(timeout=5)
            captured.thread.join(timeout=5)
            if captured.thread.is_alive():raise RuntimeError('Output reader did not finish')
            if captured.error is not None:
                if isinstance(captured.error,OutputLimitExceeded) and captured.seen>captured.limit_bytes and captured.written==captured.limit_bytes:
                    reason='output_overflow'
                else:captured.check()
        if reason=='normal':
            capture=captured.finish();result['capture']=capture
            data=(scratch/'capture.log').read_bytes()
            private['payload_sha256']=hashlib.sha256(data).hexdigest()
            result['payload_bytes']=len(data);result['exit_code']=captured.process.returncode
            if captured.process.returncode!=0:reason='worker_failure'
            if mode=='normal' and (captured.process.returncode!=0 or data!=NORMAL):raise RuntimeError('Normal payload bytes or exit mismatch')
            if mode=='failure' and (captured.process.returncode!=7 or data!=b'known-failure\n'):raise RuntimeError('Known failure fixture mismatch')
        result['outcome']=reason
        if reason!='normal':
            rec.rejection({'unexpected_descendant':'undeclared_child','output_overflow':'output_limit','worker_failure':'nonzero_exit','cancelled':'cancelled'}[reason])
    except BaseException as exc:
        primary=exc;rec.fail(exc);private['primary_traceback']=traceback.format_exc()
        result['outcome']='diagnostic_failure'
    finally:
        if captured is not None and captured.process is not None:
            try:
                rec.checkpoint('cleanup_reap')
                cleanup=reap(captured,known);result['cleanup']=cleanup
                rec.checkpoint('cleanup_readers',{'checks':{'reader_joined':not captured.thread.is_alive()},
                    'counters':{'remaining_instances':cleanup['owned_instances_remaining']}})
                result['output_observed_bytes']=captured.seen;result['output_stored_bytes']=captured.written
                if scratch is not None and (scratch/'capture.log').exists():
                    data=(scratch/'capture.log').read_bytes()
                    write_bytes_new(OUT/'private'/(name+'-capture.bin'),data,private=True)
                    private['captured_output_file']=name+'-capture.bin'
                    private['captured_output_bytes']=len(data)
                    private['captured_output_sha256']=hashlib.sha256(data).hexdigest()
                rec.checkpoint('cleanup_mark_complete')
                helper.protect_children(scratch,[],root=scratch.parent,complete=True)
            except BaseException as exc:
                cleanup_error=exc;rec.cleanup_fail(exc);private['cleanup_traceback']=traceback.format_exc()
        elif captured is not None:
            try:captured.stop()
            except BaseException as exc:
                cleanup_error=exc;rec.cleanup_fail(exc)
        try:
            rec.private('final_detail',private)
        except BaseException as exc:
            cleanup_error=cleanup_error or exc;rec.cleanup_fail(exc)
        try:
            rec.checkpoint('scratch_remove')
            if scratch is not None:scratch_context.__exit__(None,None,None)
            result['scratch_cleaned']=scratch_path is not None and not scratch_path.exists()
        except BaseException as exc:
            cleanup_error=cleanup_error or exc;rec.cleanup_fail(exc)
            result['scratch_cleaned']=False
        os.environ['TASK_LAUNCHER_TEMP_BYTES']='0' if result.get('scratch_cleaned') else '131072'
        result.update(wall_seconds=time.monotonic()-start,sampled_peak_working_set_bytes=peak,
            sampled_worker_cpu_seconds=cpu,parent_cpu_seconds=time.process_time()-parent_cpu,
            monitor_samples=samples,primary_failed=primary is not None,cleanup_failed=cleanup_error is not None,
            executable_identity_read_bytes=images.bytes_read,os_launch_started=captured is not None)
        receipt=rec.persist()
        result['evidence_phase']=receipt.get('phase')
        result['evidence_failure']=receipt['evidence_failure']
        write_new(OUT/(name+'-result.json'),result)
    return result


def run(label):
    checked=verify(label);deadline_check()
    pre=read(OUT/(label+'-pre-review.json'))
    if not pre.get('passed') or pre.get('freeze_sha256')!=checked['freeze_sha256']:
        raise RuntimeError('Exact frozen source lacks independent pre-review')
    if label in ('d2','d3'):
        cause=read(OUT/(label+'-cause-decision.json'))
        if not cause.get('approved') or not cause.get('actionable_evidence'):
            raise RuntimeError('No cause-linked revision approval')
    previous_summaries=[read(OUT/(p+'-summary.json')) for p in ('d1','d2','d3') if p!=label and (OUT/(p+'-summary.json')).is_file()]
    prior_wall=sum(r['total_wall_seconds'] for r in previous_summaries)
    prior_cpu=sum(r.get('total_cpu_seconds',0) for r in previous_summaries)
    if label=='final':
        successful=[read(OUT/(p+'-summary.json')) for p in ('d1','d2','d3') if (OUT/(p+'-summary.json')).is_file()]
        if not any(r['passed'] for r in successful):raise RuntimeError('No normal diagnostic passed')
        names=[('accept-'+m,m) for m in ('normal','descendant','overflow','failure','cancel')]
    elif label in ('d1','d2','d3'):names=[(label,'normal')]
    else:raise ValueError('Invalid frozen launch phase')
    write_new(OUT/(label+'-started.json'),dict(label=label,freeze_sha256=checked['freeze_sha256'],attempt_consumed=True))
    images=Images();results=[]
    try:
        parent=psutil.Process();parent.cpu_affinity([parent.cpu_affinity()[0]])
        actual_runtime=dict(requested=images.identity(sys.executable),base=images.identity(sys._base_executable))
        if actual_runtime!=checked['registration']['intended_runtime_images']:
            raise RuntimeError('Runtime images differ from registration')
        APPROVED_IMAGE_HASHES.update(v['sha256'] for v in actual_runtime.values())
    except BaseException as exc:
        # Attempt was already consumed; retain an explicit no-launch failure.
        failed=dict(label=label,passed=False,launched_fixtures=0,planned_fixtures=len(names),
            outcome='runtime_preflight_failure',total_wall_seconds=0,total_cpu_seconds=0,
            image_identity_read_bytes=images.bytes_read,source_metadata_meter=meter())
        write_new(OUT/(label+'-summary.json'),failed)
        rec=Recorder(OUT,label+'-preflight');rec.checkpoint('runtime');rec.fail(exc);rec.persist()
        return failed
    expected={'normal':'normal','descendant':'unexpected_descendant','overflow':'output_overflow','failure':'worker_failure','cancel':'cancelled'}
    for name,mode in names:
        try:
            result=fixture(name,mode,images,enforce_chain=label=='final')
        except BaseException as exc:
            # A recorder bind/write failure still leaves a consumed phase receipt.
            result=dict(name=name,mode=mode,outcome='evidence_setup_failure',primary_failed=True,
                cleanup_failed=True,evidence_failure=True,scratch_cleaned=False,
                wall_seconds=0,sampled_worker_cpu_seconds=0,parent_cpu_seconds=0,
                failure_type=type(exc).__name__ if type(exc).__name__ in ('OSError','RuntimeError','ValueError') else 'Exception')
            write_new(OUT/(name+'-setup-failure.json'),result)
        results.append(result)
        passed=(not result['primary_failed'] and not result['cleanup_failed'] and not result['evidence_failure']
            and result['scratch_cleaned'] and result.get('outcome')==expected[mode]
            and result.get('cleanup',{}).get('reader_joined') is True
            and result.get('cleanup',{}).get('owned_instances_remaining')==0)
        if mode=='overflow':passed=passed and result['output_observed_bytes']==65537 and result['output_stored_bytes']==65536
        if not passed:break
        if prior_wall+sum(r['wall_seconds'] for r in results)>180 or prior_cpu+sum(r['sampled_worker_cpu_seconds']+r['parent_cpu_seconds'] for r in results)>90:
            raise RuntimeError('Synthetic execution budget exceeded')
    complete=len(results)==len(names) and passed
    summary=dict(label=label,passed=complete,launched_fixtures=len(results),planned_fixtures=len(names),
        results=[dict(name=r['name'],outcome=r.get('outcome'),primary_failed=r['primary_failed'],cleanup_failed=r['cleanup_failed']) for r in results],
        source_metadata_meter=meter(),image_identity_read_bytes=images.bytes_read,
        total_wall_seconds=sum(r['wall_seconds'] for r in results),
        total_cpu_seconds=sum(r['sampled_worker_cpu_seconds']+r['parent_cpu_seconds'] for r in results),
        prior_wall_seconds=prior_wall,prior_cpu_seconds=prior_cpu,
        raw_or_weight_reads=0,neural_forwards=0,official_transitions=0,new_games=0)
    write_new(OUT/(label+'-summary.json'),summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('label',choices=('d1','d2','d3','final'))
    print(json.dumps(run(parser.parse_args().label)))
