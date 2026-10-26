"""Keep artifact guards around receipt writes, never around spawn IPC.

The target owns its own artifact guard. The caller must retain the existing
Linux whole-session supervisor: a failed Process.start may have created a child
before Python returns its handle. This module never weakens storage.py.
"""
import json
import multiprocessing
from pathlib import Path
import time

from learning.task_anchor_once_v1.storage import owned_writes, OutputBudgetExceeded


class Receipts:
    def __init__(self, out, limit=65536):
        if type(limit) is not int or limit<=0:raise ValueError('Positive receipt allowance required')
        self.out,self.limit,self.used,self.refused=Path(out),limit,0,False

    def write(self,name,value):
        if name not in ('launch-started.json','launch-result.json'):
            raise ValueError('Unregistered parent receipt')
        if self.refused or self.used>=self.limit:
            raise OutputBudgetExceeded('Cumulative parent receipt allowance exhausted')
        budget=owned_writes(self.out,self.limit-self.used,allowed_paths=[name])
        try:
            with budget:
                self.out.mkdir(parents=True,exist_ok=True)
                with (self.out/name).open('x',encoding='utf-8',newline='\n') as stream:
                    json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
                    stream.write('\n')
        finally:
            # No refund for failed/partial writes; a refusal cannot be cleared
            # by opening a fresh narrow context for the second receipt.
            self.used+=budget.used_bytes
            self.refused=self.refused or bool(budget.snapshot()['refused'])


def launch(out, target, args, *, seconds, receipt_limit=65536):
    """One real spawn. Both parent receipts share one cumulative allowance."""
    if not isinstance(seconds,(int,float)) or not 0<seconds<=1200:
        raise ValueError('Bounded process deadline required')
    receipts=Receipts(out,receipt_limit)
    receipts.write('launch-started.json',dict(attempt=1,method='spawn',process_seconds=seconds,
        receipt_limit_bytes=receipt_limit,retry_allowed=False))
    started=time.monotonic();process=None;error=None;exit_code=None;started_ok=False
    try:
        # No artifact context surrounds construction, serialization, pipes,
        # start, join, terminate, kill or close.
        process=multiprocessing.get_context('spawn').Process(target=target,args=args)
        process.start();started_ok=True
        process.join(max(0.,seconds-(time.monotonic()-started)))
        if process.is_alive():raise TimeoutError('Owned spawn target exceeded deadline')
        if process.exitcode!=0:raise RuntimeError('Owned spawn target failed')
    except BaseException as exc:
        error=exc
    finally:
        if process is not None:
            try:
                if process.is_alive():
                    process.terminate();process.join(5)
                    if process.is_alive():process.kill();process.join(5)
                if process.is_alive():raise RuntimeError('Owned spawn target remains alive')
                exit_code=process.exitcode
            finally:
                if not process.is_alive():process.close()
    result=dict(passed=error is None,process_start_returned=started_ok,child_exit_code=exit_code,
        error_kind=type(error).__name__ if error else None,elapsed_seconds=time.monotonic()-started,
        target_handle_closed=process is not None,outer_session_supervisor_still_required=True,
        parent_receipt_bytes_before_final=receipts.used,retry_allowed=False)
    receipts.write('launch-result.json',result)
    result['parent_receipt_bytes_total']=receipts.used
    result['parent_receipt_refused']=receipts.refused
    if error is not None:raise error
    return result
