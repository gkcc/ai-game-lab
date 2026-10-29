"""Temporarily bind frozen workers to one verified panel context.

Bindings are process-local and serialized. Callers must pass ``context.OUT``
explicitly to every legacy entry point: their default arguments remain bound
to the historical panel. Never use this adapter with concurrent panel threads.
"""
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

from learning.task_teacher_merged_eval_v1 import campaign, games, independent_method, independent_runner

_LOCK = RLock()
_METHODS = ('ready', 'verify', 'verify_commit', 'deadline_check', 'load_seeds', 'write_new', 'trusted_out')
_TARGETS = (
    (campaign, ('ready', 'verify_commit', 'deadline_check', 'load_seeds', 'write_new', 'trusted_out')),
    (independent_runner, ('ready', 'verify_commit', 'load_seeds', 'write_new', 'trusted_out')),
    (independent_method, ('ready', 'verify_commit', 'deadline_check', 'load_seeds', 'write_new', 'trusted_out')),
)


@contextmanager
def bind_context(context, *, memory_ready=None, supervise_game=None, supervise_replay=None,
                 make_wrapper=None, accept_job=None):
    """Bind one context; restore every touched reference even after an error."""
    if (not __debug__ or not all(callable(getattr(context, name, None)) for name in _METHODS)
        or Path(context.ROOT) != Path(campaign.ROOT) or not isinstance(context.OUT, Path)):
        raise ValueError('Invalid or optimized panel context')
    hooks = ((campaign, 'memory_ready', memory_ready), (campaign, 'supervise', supervise_game),
             (independent_runner, 'supervise', supervise_replay),
             (campaign, 'make_wrapper', make_wrapper), (campaign, 'accept_job', accept_job))
    if any(value is not None and not callable(value) for _, _, value in hooks):
        raise ValueError('Invalid panel adapter hook')
    with _LOCK:
        saved = []
        def replace(module, name, value):
            saved.append((module, name, getattr(module, name)))
            setattr(module, name, value)
        try:
            replace(games, 'experiment', context)
            for module, names in _TARGETS:
                for name in names:
                    replace(module, name, getattr(context, name))
            for module, name, value in hooks:
                if value is not None:
                    replace(module, name, value)
            yield
        finally:
            for module, name, original in reversed(saved):
                setattr(module, name, original)
