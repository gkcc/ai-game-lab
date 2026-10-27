"""Bounded probe result classification; callers supply owned Job receipts."""

JOB_FIELDS=('passed','assigned','resumed','launcher_reaped','job_members_zero','exit_code',
            'error_kind','cleanup_error_kind','stop_reason','peak_sampled_rss_bytes',
            'peak_sampled_job_commit_bytes','rss_sampling_retries','rss_complete_samples')
STOP_REASONS={None,'rss_limit','commit_limit','available_memory_limit','wall_limit'}
ERROR_KINDS={'OSError','PermissionError','FileNotFoundError','TimeoutError','RuntimeError',
             'ValueError','TypeError','KeyError','IndexError','AttributeError','AssertionError',
             'OverflowError','MemoryError','UnicodeError','JSONDecodeError','NoSuchProcess',
             'ZombieProcess','AccessDenied','SamplingExhausted','EmptySnapshot',
             'SystemExit','KeyboardInterrupt','Exception','OtherError'}


def error_kind(exc):
    name=type(exc).__name__
    return name if name in ERROR_KINDS else 'OtherError'


def safe_job(job):
    """Keep only typed, bounded aggregate fields; reject arbitrary messages."""
    if not isinstance(job,dict) or not set(JOB_FIELDS[:9]).issubset(job): return None
    result={key:job.get(key) for key in JOB_FIELDS}
    if any(type(result[key]) is not bool for key in ('passed','assigned','resumed','launcher_reaped','job_members_zero')):
        return None
    reason=result['stop_reason']
    if reason is not None and (not isinstance(reason,str) or reason not in STOP_REASONS): return None
    for key in ('error_kind','cleanup_error_kind'):
        value=result[key]
        if value is not None:
            if not isinstance(value,str): return None
            result[key]=value if value in ERROR_KINDS else 'OtherError'
    code=result['exit_code']
    if code is not None and (type(code) is not int or not 0<=code<=0xffffffff): return None
    for key in ('peak_sampled_rss_bytes','peak_sampled_job_commit_bytes','rss_sampling_retries','rss_complete_samples'):
        value=result[key]
        if value is not None and (type(value) is not int or value<0 or value>2**63-1): return None
    return result

def classify(job,read_probe,expected_freeze_sha256,expected_marker_sha256):
    """Only read the probe receipt after execution and cleanup are confirmed."""
    def valid_sha(value): return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
    if (not isinstance(job,dict) or not callable(read_probe) or not valid_sha(expected_freeze_sha256)
        or not valid_sha(expected_marker_sha256)): raise ValueError('Invalid probe inputs')
    summary=safe_job(job)
    if summary is None:
        return dict(passed=False,primary_kind='job_receipt_invalid',stop_reason=None,job=None,
                    probe=None,worker_allowed=False,retry_allowed=False)
    clean=(job.get('passed') is True and job.get('assigned') is True and job.get('resumed') is True
        and job.get('launcher_reaped') is True and job.get('job_members_zero') is True
        and type(job.get('exit_code')) is int and job['exit_code']==0
        and job.get('error_kind') is None and job.get('cleanup_error_kind') is None and job.get('stop_reason') is None)
    if not clean:
        if job.get('cleanup_error_kind') is not None or job.get('launcher_reaped') is not True or job.get('job_members_zero') is not True:
            primary='job_cleanup_unconfirmed'
        elif job.get('stop_reason') is not None:
            primary='job_stopped'
        else:
            primary='job_execution_failed'
        return dict(passed=False,primary_kind=primary,stop_reason=job.get('stop_reason'),job=summary,
                    probe=None,worker_allowed=False,retry_allowed=False)
    try:
        probe=read_probe()
    except FileNotFoundError:
        return dict(passed=False,primary_kind='probe_receipt_missing',stop_reason=None,job=summary,
                    probe=None,worker_allowed=False,retry_allowed=False)
    except Exception as exc:
        return dict(passed=False,primary_kind='probe_receipt_unreadable',receipt_error_kind=error_kind(exc),
                    stop_reason=None,job=summary,probe=None,worker_allowed=False,retry_allowed=False)
    receipt_fields={'passed','venv_verified','job_membership_verified','training_inputs_read',
                    'retry_allowed','freeze_sha256','marker_sha256'}
    if (not isinstance(probe,dict) or set(probe)!=receipt_fields
        or probe.get('passed') is not True or probe.get('venv_verified') is not True
        or probe.get('job_membership_verified') is not True or probe.get('training_inputs_read') is not False
        or probe.get('retry_allowed') is not False or probe.get('freeze_sha256')!=expected_freeze_sha256
        or probe.get('marker_sha256')!=expected_marker_sha256):
        return dict(passed=False,primary_kind='probe_receipt_invalid',stop_reason=None,job=summary,
                    probe=None,worker_allowed=False,retry_allowed=False)
    return dict(passed=True,primary_kind=None,stop_reason=None,job=summary,probe=probe,
                worker_allowed=True,retry_allowed=False)


def run_after_probe(job,read_probe,expected_freeze_sha256,expected_marker_sha256,start_worker):
    """Invoke a worker Job only after the caller-bound probe is accepted.

    A launch exception may occur before or after a child begins; the caller must
    preserve its Job cleanup receipt and must not infer that work never started.
    A successful result confirms Job status only; callers must separately check
    the worker's semantic result and bind its evidence before accepting a task.
    """
    if not callable(start_worker): raise ValueError('Invalid worker launcher')
    verdict=classify(job,read_probe,expected_freeze_sha256,expected_marker_sha256)
    if not verdict['worker_allowed']:
        return dict(passed=False,probe=verdict,worker_started=False,worker_result=None,retry_allowed=False)
    try:
        result=start_worker()
    except Exception as exc:
        return dict(passed=False,probe=verdict,worker_started='unknown_after_launch',
                    worker_error_kind=error_kind(exc),worker_result=None,retry_allowed=False)
    worker_summary=safe_job(result)
    good=(worker_summary is not None and result.get('passed') is True and result.get('assigned') is True
          and result.get('resumed') is True and result.get('launcher_reaped') is True
          and result.get('job_members_zero') is True and result.get('error_kind') is None
          and result.get('cleanup_error_kind') is None and result.get('stop_reason') is None
          and type(result.get('exit_code')) is int and result['exit_code']==0)
    return dict(passed=good,probe=verdict,worker_started=True,worker_result=worker_summary,
                worker_semantic_result_checked=False,retry_allowed=False)
