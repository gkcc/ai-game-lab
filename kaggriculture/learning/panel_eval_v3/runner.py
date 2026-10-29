"""One bounded official panel using the frozen v1 workers and evaluator.

The trusted parent must start this module with ``-B -X pycache_prefix=...``
before importing project code and own the supplied OS-temporary scratch until
all children exit. A started coordinator cannot be resumed or retried. Raw
results, seeds and job receipts stay local; this module prints no economics.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import re
import sys
import time

import psutil

from learning.panel_eval_v3 import adapter, isolation
from learning.panel_eval_v3.epoch import EpochContext
from learning.panel_eval_v3.experiment import Context, ROOT, _same, FORMAL, require_run_kind
from learning.panel_safety_v1 import memory_gate, seed_inventory
from learning.task_job_empty_exit_v1.sampling import supervise as supervise_job
from learning.task_teacher_merged_eval_v1 import campaign, independent_runner


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _monotonic_deadline(cfg):
    try:
        end = datetime.fromisoformat(cfg['deadline_utc'])
        if end.tzinfo is None or end.utcoffset() is None:
            raise ValueError('Missing timezone')
        remaining = (end.astimezone(timezone.utc) - datetime.now(timezone.utc)).total_seconds()
    except (KeyError, TypeError, ValueError, OverflowError):
        raise ValueError('Invalid panel deadline') from None
    if remaining <= 0:
        raise TimeoutError('Panel deadline reached')
    return time.monotonic() + remaining


class Coordinator:
    """Bridge the unchanged campaign to bounded, isolated child processes."""

    def __init__(self, context, scratch, budget):
        self.context = context
        self.scratch = Path(scratch)
        self.budget = budget
        self.wait_count = 0
        self.child_count = 0
        self.active = False
        self.cleanup_state = 'not_started'
        self.canonical = {}

    def _prefix(self, kind):
        self.child_count += 1
        return self.scratch/('cache-' + kind + '-' + str(self.child_count))

    def _command(self, kind, module, *arguments):
        prefix = self._prefix(kind)
        command = isolation.bootstrap_command(sys.executable, self.scratch, prefix, ROOT, module)
        return prefix, command + list(arguments)

    def _available(self):
        self.context.deadline_check()
        return psutil.virtual_memory().available

    def memory_ready(self, cfg, first_sample=None):
        if self.active:
            raise RuntimeError('A child is already active')
        self.wait_count += 1
        name = 'wait-' + str(self.wait_count)
        self.context.write_new(self.context.OUT/'results'/(name+'-start.json'),
            dict(attempt=self.wait_count, budget_terminal=self.budget.terminal,
                 prior_blocked_seconds=self.budget.blocked_seconds, retry_allowed=False))
        first = [] if first_sample is None else [first_sample]
        def available():
            if first:
                self.context.deadline_check()
                return first.pop()
            return self._available()
        result = memory_gate.wait_for_preflight(budget=self.budget, deadline=_monotonic_deadline(cfg),
            available=available, job_not_started=True)
        self.context.write_new(self.context.OUT/'results'/(name+'-result.json'),
            dict(passed=result.passed, reason=result.reason, attempts=result.attempts, samples=result.samples,
                 shortages=result.shortages, last_shortage_sample=result.last_shortage_sample,
                 blocked_seconds=result.blocked_seconds, total_blocked_seconds=result.total_blocked_seconds,
                 duration_known=self.budget.duration_known, retry_allowed=False))
        return result.passed

    def _supervise(self, api, command, cwd, limits, prefix):
        if self.active:
            raise RuntimeError('A child is already active')
        self.active = True
        self.cleanup_state = 'unknown_after_spawn_attempt'
        try:
            result = supervise_job(api, command, cwd, limits)
            if (isinstance(result, dict) and result.get('launcher_reaped') is True
                and result.get('job_members_zero') is True and result.get('cleanup_error_kind') is None):
                self.cleanup_state = 'confirmed_empty'
        finally:
            self.active = False
        isolation.validate_prefix(self.scratch, prefix, ROOT)
        return result

    def supervise_game(self, api, command, cwd, limits):
        if (not isinstance(command, list) or len(command) != 8
            or command[:7] != [sys.executable, '-B', '-m',
                'learning.task_teacher_merged_eval_v1.games', '--out', str(self.context.OUT), '--key']
            or Path(cwd) != ROOT or not _same(limits, self.context.ready()['official_job_supervision'])):
            raise ValueError('Unexpected official worker command')
        key = command[7]
        if not isinstance(key, str) or Path(key).name != key:
            raise ValueError('Invalid game key')
        self.context.refresh()
        stage = self.context.OUT
        item = seed_inventory._json(seed_inventory._read_regular(stage/'job-inputs'/(key+'.json'), stage))
        if not isinstance(item, dict) or item.get('game') != key:
            raise ValueError('Game input identity changed')
        wrapper = stage/'agents'/(key+'.py')
        expected = item.get('bot_sha256')
        if not isinstance(expected, str) or _sha(wrapper) != expected:
            raise ValueError('Wrapper identity changed')
        prefix, canonical_cmd = self._command('canonical', 'learning.panel_eval_v3.worker',
            '--out', str(stage), '--key', key, '--prereg', self.context.prereg_commit,
            '--scratch', str(self.scratch), '--mode', 'canonical')
        canonical_result = self._supervise(api, canonical_cmd, cwd, limits, prefix)
        if not campaign.job_ok(canonical_result) or _sha(wrapper) != expected:
            raise RuntimeError('Canonical wrapper or cleanup failed')
        self.context.write_new(stage/'results'/'canonical'/(key+'.json'),
            dict(game=key, wrapper_sha256=expected, passed=True, launcher_reaped=True,
                 job_members_zero=True, isolated_cache_prefix_absent=True, retry_allowed=False))
        cfg = self.context.refresh()
        # The legacy three-sample check precedes canonical verification.  A
        # final 4 GiB sample prevents launching on a visibly stale check.  If
        # it is low, the same bounded budget requires another full preflight;
        # recheck identity afterwards because waiting leaves time for changes.
        while True:
            current = self._available()
            if type(current) is not int or current < 0:
                raise ValueError('Invalid available-memory sample')
            if current >= memory_gate.MIN_AVAILABLE_BYTES:
                break
            if not self.memory_ready(cfg, first_sample=current):
                raise RuntimeError('Insufficient preflight memory before official launch')
            refreshed = self.context.refresh()
            if not _same(refreshed, cfg):
                raise ValueError('Configuration changed while waiting')
        self.context.deadline_check()
        if _sha(wrapper) != expected:
            raise ValueError('Wrapper changed before official launch')
        self.context.write_new(stage/'results'/'job-start'/(key+'.json'),
            dict(game=key, wrapper_sha256=expected, retry_allowed=False))
        self.canonical[key] = expected
        prefix, game_cmd = self._command('game', 'learning.panel_eval_v3.worker',
            '--out', str(stage), '--key', key, '--prereg', self.context.prereg_commit,
            '--scratch', str(self.scratch), '--mode', 'game')
        result = self._supervise(api, game_cmd, cwd, limits, prefix)
        if campaign.job_ok(result):
            self.context.refresh()
            if _sha(wrapper) != expected:
                raise ValueError('Wrapper changed during official game')
            self.context.write_new(stage/'results'/'canonical-post'/(key+'.json'),
                dict(game=key, wrapper_sha256=expected, passed=True, retry_allowed=False))
        return result

    def accept_job(self, out, item, cfg):
        expected = self.canonical.get(item.get('game'))
        if (expected is None or expected != item.get('bot_sha256')
            or _sha(Path(out)/'agents'/(item['game']+'.py')) != expected):
            raise ValueError('Canonical wrapper evidence missing')
        return accept_persistent_job(out, item, cfg)

    def supervise_replay(self, api, command, cwd, limits):
        if (command != [sys.executable, '-B', '-m',
                         'learning.task_teacher_merged_eval_v1.independent_method']
            or Path(cwd) != ROOT or not _same(limits, self.context.ready()['independent_job_supervision'])):
            raise ValueError('Unexpected independent replay command')
        self.context.refresh()
        prefix, child = self._command('replay', 'learning.panel_eval_v3.independent_worker',
            '--out', str(self.context.OUT), '--prereg', self.context.prereg_commit,
            '--scratch', str(self.scratch))
        result = self._supervise(api, child, cwd, limits, prefix)
        if campaign.job_ok(result):
            self.context.refresh()
        return result


_original_accept_job = campaign.accept_job
_GAME = re.compile(r'holdout-S0[1-4]-(?:round2|market|teacher)-[01]-(?:merged|unweighted|teacher_task)\Z')


def canonical_receipts(out, item):
    """Bind the three local wrapper receipts to one input and current wrapper.

    This proves consistency of the retained records, not resistance to a
    same-user writer replacing all records or changing files between checks.
    """
    out = Path(out)
    key = item.get('game') if isinstance(item, dict) else None
    expected = item.get('bot_sha256') if isinstance(item, dict) else None
    if not isinstance(key, str) or not _GAME.fullmatch(key) or not seed_inventory.old.digest(expected):
        raise ValueError('Invalid canonical receipt identity')
    input_path = out/'job-inputs'/(key+'.json')
    input_raw = seed_inventory._read_regular(input_path, out)
    if not _same(seed_inventory._json(input_raw), item) or _sha(out/'agents'/(key+'.py')) != expected:
        raise ValueError('Canonical receipt input changed')
    names = {
        'canonical': dict(game=key, wrapper_sha256=expected, passed=True, launcher_reaped=True,
                          job_members_zero=True, isolated_cache_prefix_absent=True, retry_allowed=False),
        'job-start': dict(game=key, wrapper_sha256=expected, retry_allowed=False),
        'canonical-post': dict(game=key, wrapper_sha256=expected, passed=True, retry_allowed=False),
    }
    hashes = {}
    for name, expected_doc in names.items():
        raw = seed_inventory._read_regular(out/'results'/name/(key+'.json'), out)
        if not _same(seed_inventory._json(raw), expected_doc):
            raise ValueError('Canonical receipt changed or missing')
        hashes[name] = hashlib.sha256(raw).hexdigest()
    return hashes


def accept_persistent_job(out, item, cfg):
    """Apply the same wrapper-evidence checks in coordinator and replay child."""
    hashes = canonical_receipts(out, item)
    row, audit = _original_accept_job(out, item, cfg)
    if canonical_receipts(out, item) != hashes:
        raise ValueError('Canonical receipts changed during acceptance')
    audit['canonical_receipt_sha256'] = hashes
    return row, audit


def run(stage, prereg_commit, scratch):
    """Run one registered panel; no partial economic evaluation on failure."""
    isolation.validate_runtime(scratch, sys.pycache_prefix, ROOT)
    context = EpochContext(Context(stage, prereg_commit))
    cfg = context.refresh()
    require_run_kind(cfg, FORMAL)
    results = context.OUT/'results'
    if results.exists():
        raise ValueError('Coordinator attempt already consumed')
    # This short initial check happens before any coordinator marker or game.
    # The outer launcher has its own stricter one-shot marker and may not be
    # invoked again after this check fails.
    preflight = memory_gate.wait_for_preflight(budget=memory_gate.WaitBudget(),
        deadline=min(_monotonic_deadline(cfg), time.monotonic()+25),
        available=lambda: psutil.virtual_memory().available, job_not_started=True)
    if not preflight.passed:
        raise RuntimeError('Initial preflight did not pass')
    context.refresh()
    context.write_new(results/'coordinator-started.json',
        dict(attempt=1, prereg_commit=prereg_commit, retry_allowed=False))
    controller = Coordinator(context, scratch, memory_gate.WaitBudget())
    try:
        with adapter.bind_context(context, memory_ready=controller.memory_ready,
                                  supervise_game=controller.supervise_game,
                                  supervise_replay=controller.supervise_replay,
                                  accept_job=controller.accept_job):
            campaign.run(context.OUT)
            context.refresh()
            independent_runner.run(context.OUT)
            context.refresh()
            isolation.validate_runtime(scratch, sys.pycache_prefix, ROOT)
            independent_runner.finalize(context.OUT)
            context.refresh()
        isolation.validate_runtime(scratch, sys.pycache_prefix, ROOT)
        context.write_new(results/'coordinator-complete.json',
            dict(passed=True, retry_allowed=False, economic_evaluation_completed=True))
    except BaseException as exc:
        try:
            context.write_new(results/'coordinator-failure.json',
                dict(error_kind=type(exc).__name__, retry_allowed=False,
                     cleanup_state=controller.cleanup_state))
        except BaseException:
            pass
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--prereg', required=True)
    parser.add_argument('--scratch', required=True)
    args = parser.parse_args()
    run(args.out, args.prereg, args.scratch)


if __name__ == '__main__':
    try:
        with open(os.devnull, 'w', encoding='utf-8') as sink:
            from contextlib import redirect_stdout, redirect_stderr
            with redirect_stdout(sink), redirect_stderr(sink):
                main()
    except BaseException:
        raise SystemExit(1) from None
    raise SystemExit(0)
