"""Reuse checked identities only inside an explicitly bounded process epoch.

The full context still scans recognized history at every game/replay/evaluation
boundary. Legacy nested checks within that boundary verify a private snapshot
of the static inputs rather than repeatedly walking unrelated seed history.
This is an operational consistency check, not a defense against a same-user
writer changing and restoring files between checks.
"""
import copy
import os
from pathlib import Path, PurePosixPath

from learning.panel_safety_v2 import registered_rescan as registered
from learning.task_teacher_merged_eval_v1 import experiment as old
from . import bootstrap, experiment, failed_history, prior_v2_guard


def _project_path(value):
    if not isinstance(value, str) or not value or '\\' in value or ':' in value:
        raise ValueError('Invalid frozen project path')
    rel = PurePosixPath(value)
    if rel.is_absolute() or rel.as_posix() != value or any(p in ('.', '..') for p in rel.parts):
        raise ValueError('Invalid frozen project path')
    return experiment.ROOT.joinpath(*rel.parts)


class EpochContext:
    """Provide old worker APIs without letting an epoch cross a process."""
    def __init__(self, context):
        self.context = context
        self.ROOT, self.OUT = context.ROOT, context.OUT
        self.prereg_commit = context.prereg_commit
        self.pid = os.getpid()
        self._cfg = self._paths = self._hashes = None

    def __getattr__(self, name):
        return getattr(self.context, name)

    def _identity_paths(self):
        cfg = experiment._read(self.OUT, 'registration.json')
        if not isinstance(cfg, dict):
            raise ValueError('Invalid registered configuration')
        reference = experiment._reference_freeze()
        paths = {self.OUT/name for name in (*registered.PUBLIC_FILES, registered.SAFETY,
                  registered.MARKER, registered.PRIVATE, registered.SNAPSHOT, registered.EXECUTION)}
        paths.update(_project_path(name) for name in reference['runtime_files'])
        paths.update(_project_path(name) for name in reference['new_source_sha256'])
        paths.update(_project_path(name) for name in experiment.SUPPLEMENTAL_FILES)
        # Registration is not yet trusted here.  Resolve model paths from the
        # previously validated historical identities, never from its fields.
        _, models, _, _ = old.historical_inputs()
        paths.update(models.values())
        paths.update(self.OUT/'sources'/(name+'.py') for name in ('round2','market','teacher'))
        paths.update((old.ACTION/'sources'/'round2.py', old.ACTION/'sources'/'market.py',
                      self.ROOT/'bots'/'teacher_fertilizer_terminal_v2.py',
                      self.ROOT/'.venv'/'Scripts'/'python.exe'))
        paths.update(experiment._installed_files().values())
        prior_stage = self.ROOT/'artifacts'/failed_history.STAGE_NAME
        paths.update(prior_stage/rel for rel in failed_history.RECEIPT_PATHS.values())
        paths.add(prior_stage/'private-seeds.local.json')
        return cfg, frozenset(paths)

    @staticmethod
    def _fingerprints(paths):
        try:
            return {path: old.sha(path) for path in paths}
        except (OSError, TypeError):
            raise ValueError('Frozen epoch input unavailable') from None

    def refresh(self):
        """Start a new epoch only after a full, stable history check."""
        if os.getpid() != self.pid:
            raise ValueError('Verification epoch cannot cross processes')
        self._cfg = self._paths = self._hashes = None
        prior_v2_guard.require_v2_consumed_chain(self.prereg_commit)
        bootstrap.verify_initializers(self.ROOT)
        before_cfg, before_paths = self._identity_paths()
        before = self._fingerprints(before_paths)
        cfg = self.context.ready()
        after_cfg, after_paths = self._identity_paths()
        after = self._fingerprints(after_paths)
        if (not experiment._same(cfg, before_cfg) or not experiment._same(cfg, after_cfg)
            or before_paths != after_paths or before != after):
            raise ValueError('Frozen inputs changed during full verification')
        self._cfg, self._paths, self._hashes = copy.deepcopy(cfg), after_paths, after
        return copy.deepcopy(cfg)

    def ready(self, out=None):
        """Check static identities inside the current explicit epoch."""
        if os.getpid() != self.pid or self._cfg is None:
            raise ValueError('No current process verification epoch')
        bootstrap.verify_initializers(self.ROOT)
        self.context.trusted_out(self.OUT if out is None else out)
        self.context.deadline_check()
        old.clean_head(self.prereg_commit)
        if self._fingerprints(self._paths) != self._hashes:
            raise ValueError('Frozen epoch input changed')
        experiment._check_official_runtime(self._cfg['official_loader_sha256'])
        self.context.deadline_check()
        return copy.deepcopy(self._cfg)

    def verify(self, out=None):
        return self.ready(out)

    def verify_commit(self, out=None):
        return self.ready(out)
