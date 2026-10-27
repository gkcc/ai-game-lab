"""Two preregistered owned synthetic Job scenarios; no activity at import."""
import json
from pathlib import Path
import sys
from .job import WindowsAPI, supervise

SCENARIOS = {
    'normal': dict(worker_mib=32,worker_seconds=5,max_rss_bytes=512*1024**2,max_commit_bytes=1024**3,
                   min_available_bytes=2*1024**3,wall_seconds=30,poll_seconds=0.25,cleanup_seconds=10),
    'rss_stop': dict(worker_mib=128,worker_seconds=20,max_rss_bytes=80*1024**2,max_commit_bytes=1024**3,
                     min_available_bytes=2*1024**3,wall_seconds=30,poll_seconds=0.25,cleanup_seconds=10),
}


class ReceiptFailure(RuntimeError):
    def __init__(self,primary_error_kind,receipt_error_kind):
        super().__init__('Scenario receipt failed')
        self.primary_error_kind,self.receipt_error_kind = primary_error_kind,receipt_error_kind


def command_for(name, executable):
    config = SCENARIOS[name]
    worker = ('import time; data=bytearray('+str(config['worker_mib'])+'*1024*1024); '
              'time.sleep('+str(config['worker_seconds'])+')')
    launcher = ('import subprocess,sys; p=subprocess.Popen([sys.executable,"-B","-c",'+repr(worker)+']); '
                'raise SystemExit(p.wait())')
    return [str(executable),'-B','-c',launcher]


def accepted(name, result):
    clean = (result['assigned'] and result['resumed'] and result['launcher_reaped']
             and result['job_members_zero'] and result['error_kind'] is None and result['cleanup_error_kind'] is None
             and result['max_sampled_members'] >= 2)
    if name == 'normal': return bool(clean and result['passed'] and result['exit_code'] == 0)
    return bool(clean and result['stop_reason'] == 'rss_limit' and result['passed'] is False
                and result['rss_threshold_sample_members'] >= 2)


def run_scenario(name, api_factory=WindowsAPI):
    from .experiment import OUT, ROOT, ready, verify_commit, sha, read, write_new
    if name not in SCENARIOS: raise ValueError('Unknown scenario')
    cfg = ready(OUT)
    verify_commit(OUT)
    if cfg['scenarios'] != SCENARIOS: raise ValueError('Frozen scenarios changed')
    out = Path(OUT).resolve()
    if name == 'rss_stop':
        normal_marker = read(out/'normal-started.json')
        normal = read(out/'normal-result.json')
        if (normal_marker.get('attempt') != 1 or normal_marker.get('retry_allowed') is not False
            or normal_marker.get('freeze_sha256') != sha(out/'freeze.json')
            or normal.get('scenario') != 'normal' or normal.get('passed') is not True
            or normal.get('freeze_sha256') != sha(out/'freeze.json')
            or normal.get('marker_sha256') != sha(out/'normal-started.json')
            or normal.get('marker_created') is not True or normal.get('marker_complete') is not True
            or normal.get('error_kind') is not None or normal.get('retry_allowed') is not False
            or not isinstance(normal.get('result'),dict) or not accepted('normal',normal['result'])):
            raise ValueError('Normal scenario acceptance missing or changed')
    marker_path = (out/(name+'-started.json')).resolve()
    if not marker_path.is_relative_to(out): raise ValueError('Marker outside experiment')
    if marker_path.exists(): raise ValueError('Scenario already consumed')
    marker = dict(created=False,complete=False)
    result,error = None,None
    try:
        with marker_path.open('x',encoding='utf-8',newline='\n') as stream:
            marker['created'] = True
            json.dump(dict(attempt=1,retry_allowed=False,freeze_sha256=sha(out/'freeze.json')),stream,indent=2)
            stream.write('\n')
        marker['complete'] = True
        settings = SCENARIOS[name]
        limits = {k:settings[k] for k in ('max_rss_bytes','max_commit_bytes','min_available_bytes','wall_seconds','poll_seconds','cleanup_seconds')}
        result = supervise(api_factory(),command_for(name,sys.executable),ROOT,limits)
    except BaseException as exc:
        error = type(exc).__name__
    identity_error = None
    freeze_digest = marker_digest = None
    if marker['created']:
        try:
            freeze_digest = sha(out/'freeze.json')
            if marker['complete']: marker_digest = sha(marker_path)
        except BaseException as exc:
            identity_error = type(exc).__name__
    passed = bool(error is None and identity_error is None and result is not None and accepted(name,result))
    receipt = dict(passed=passed,scenario=name,marker_created=marker['created'],marker_complete=marker['complete'],
        error_kind=error,identity_error_kind=identity_error,result=result,retry_allowed=False,partial_work_not_refunded=True,
        freeze_sha256=freeze_digest,marker_sha256=marker_digest,
        member_count_scope='A sampled count of Job members does not identify the worker; RSS is total sampled membership, not an individual allocation')
    if marker['created']:
        try: write_new(out/(name+'-result.json'),receipt)
        except BaseException as exc: raise ReceiptFailure(error or identity_error or (result or {}).get('error_kind'),type(exc).__name__) from None
    if not passed: raise RuntimeError('Scenario not accepted; no retry')
    return receipt


def run_all():
    # A failure or uncertain cleanup stops before the second scenario.
    return [run_scenario('normal'),run_scenario('rss_stop')]
