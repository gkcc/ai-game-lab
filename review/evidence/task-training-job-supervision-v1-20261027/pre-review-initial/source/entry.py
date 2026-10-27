"""Receipt wrapper; caller supplies the frozen configuration and exclusive writes."""
from .job import supervise


class ReceiptFailure(RuntimeError):
    def __init__(self,primary_error_kind,receipt_error_kind):
        super().__init__('Supervision receipt failed')
        self.primary_error_kind,self.receipt_error_kind = primary_error_kind,receipt_error_kind


def run(api, command, cwd, limits, create_marker, write_receipt):
    """create_marker(state) must exclusively create and track partial writes."""
    marker = dict(created=False,complete=False)
    result = None
    error = None
    try:
        create_marker(marker)
        if not marker['created'] or not marker['complete']: raise ValueError('Incomplete start marker')
        result = supervise(api,command,cwd,limits)
    except BaseException as exc:
        error = type(exc).__name__
    if marker['created']:
        receipt = dict(marker_created=marker['created'],marker_complete=marker['complete'],
                       error_kind=error,result=result,retry_allowed=False,partial_work_not_refunded=True)
        try: write_receipt(receipt)
        except BaseException as exc: raise ReceiptFailure(error or (result or {}).get('error_kind'),type(exc).__name__) from None
    if error: raise RuntimeError('Supervision setup failed: '+error)
    if not result['passed']: raise RuntimeError('Supervised attempt did not pass')
    return result
