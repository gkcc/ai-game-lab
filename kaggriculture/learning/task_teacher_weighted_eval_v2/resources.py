"""Read-only host headroom preflight; sampling is not a hard memory cap."""
import time
import psutil
from .experiment import deadline_check


def memory_ready(cfg,available=lambda:psutil.virtual_memory().available,sleep=time.sleep):
    limits=cfg['host_preflight']
    for index in range(limits['samples']):
        deadline_check(); value=available()
        if type(value) is not int or value<limits['min_available_bytes']: return False
        if index+1<limits['samples']: sleep(limits['interval_seconds'])
    return True
