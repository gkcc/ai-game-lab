"""Exercise Git checks through the real worker budget without creating files."""
import hashlib
import os
from pathlib import Path
import unittest
from unittest import mock

from learning.task_anchor_once_v1.storage import UnsafeWriteError, owned_writes
from learning.task_pair_dev_games_v1 import worker
from learning.task_pair_dev_games_v1.panel import Panel
from learning.task_pair_dev_games_v1.test_panel import synthetic_inputs
from . import bootstrap, identity


REPO = Path(__file__).resolve().parents[3]


class _PlanReached(Exception):
    """Stop before reading any plan or entering the official game."""


class GitBudgetCompatibilityTests(unittest.TestCase):
    def test_real_worker_budget_allows_git_checks_before_plan_read(self):
        checks, receipts = [], []
        cfg = {'storage': {'game_write_bytes': 1024, 'small_receipt_bytes': 1024}}

        def verify(path):
            self.assertEqual(Path(path), REPO)
            head = identity._git(REPO, 'rev-parse', 'HEAD')
            identity._git(REPO, 'status', '--porcelain', '--untracked-files=all')
            self.assertEqual(len(head.strip()), 40)
            checks.append(True)
            return cfg

        def read_plan(_path):
            raise _PlanReached()

        def receipt(_path, value):
            receipts.append(value)

        # The old worker imports the installed engine and normally captures its
        # output. Suppress that output, without starting an environment or game.
        with worker._quiet_output():
            from learning.task_anchor_once_v1 import games
            with mock.patch.dict(os.environ, {'GIT_OPTIONAL_LOCKS': '0'}), \
                 mock.patch.object(games, 'verify', side_effect=verify), \
                 mock.patch.object(games, 'read', side_effect=read_plan), \
                 mock.patch.object(games, 'write_new', side_effect=receipt):
                with self.assertRaises(_PlanReached):
                    games.job(REPO, 'synthetic')

        # The first check is outside the worker budget; the second is inside it.
        self.assertEqual(len(checks), 2)
        self.assertEqual(len(receipts), 1)
        result = receipts[0]
        self.assertEqual(result['error_kind'], '_PlanReached')
        self.assertFalse(result['passed'])
        self.assertEqual((result['output']['used_bytes'],
                          result['output']['created_count'],
                          result['output']['refused']), (0, 0, 0))

    def test_bound_worker_recheck_and_real_git_inside_budget(self):
        # Mock only private files/state and the final writes; the worker module,
        # recheck logic, Git subprocesses and storage context are real.
        with worker._quiet_output():
            from learning.bc_crop.agent import ROOT as game_root

            game_root = Path(game_root).resolve()
            stage = game_root / 'artifacts' / '__synthetic_guard_integration_not_created__'
            source = (game_root / worker.WORKER_RELATIVE).read_bytes()
            inputs = synthetic_inputs()
            inputs['worker_sha'] = hashlib.sha256(source).hexdigest()
            panel = Panel.from_private_contract(**inputs)
            plan = panel.plans()[0]
            manifest = b'synthetic-manifest'
            binding = {'synthetic': True}
            real_git = identity._git
            with mock.patch.dict(os.environ, {'GIT_OPTIONAL_LOCKS': '0'}):
                commit = real_git(REPO, 'rev-parse', 'HEAD').decode().strip()

            state = dict(state='running', registration_commit=commit,
                         stage_name=bootstrap.STAGE_NAME,
                         manifest_sha256=hashlib.sha256(manifest).hexdigest())
            in_budget, git_calls, receipts = [False], [], []

            def git_observer(repo, *args):
                result = real_git(repo, *args)
                git_calls.append((args, in_budget[0]))
                # The worktree may contain this very test; only the clean result
                # is simulated, after the real status command has executed.
                if args == ('status', '--porcelain', '--untracked-files=all'):
                    return b''
                return result

            def check():
                return bootstrap._recheck(identity=identity, game_root=game_root,
                    registration_commit=commit, manifest_raw=manifest,
                    binding=binding, required_state='running')

            def read_manifest(path):
                self.assertEqual(path, bootstrap.PUBLIC_ROOT.parent / 'private' / bootstrap.MANIFEST)
                return manifest

            def read_plan(path):
                self.assertEqual(path, stage / 'job-inputs' / (plan['game'] + '.json'))
                raise _PlanReached()

            def save(_path, value):
                receipts.append(value)

            original_load = worker._load_snapshot

            def load_and_observe(*args):
                module = original_load(*args)
                original_budget = module.owned_writes

                class ObservedBudget:
                    def __init__(self, inner):
                        self.inner = inner

                    def __enter__(self):
                        entered = self.inner.__enter__()
                        in_budget[0] = True
                        return entered

                    def __exit__(self, *exc):
                        try:
                            return self.inner.__exit__(*exc)
                        finally:
                            in_budget[0] = False

                    def snapshot(self):
                        return self.inner.snapshot()

                module.owned_writes = lambda *a, **k: ObservedBudget(original_budget(*a, **k))
                module.write_new = save
                return module

            with mock.patch.dict(os.environ, {'GIT_OPTIONAL_LOCKS': '0'}), \
                 mock.patch.object(identity, '_git', side_effect=git_observer), \
                 mock.patch.object(bootstrap, '_private_status', return_value=(None, {}, state, binding)), \
                 mock.patch.object(bootstrap, '_read', side_effect=read_manifest), \
                 mock.patch.object(worker, '_load_snapshot', side_effect=load_and_observe), \
                 mock.patch.object(worker, '_read_job', side_effect=read_plan):
                with worker._bound_worker(source, panel, game_root, stage, plan, check) as module:
                    with self.assertRaises(_PlanReached):
                        module.job(stage, plan['game'])

        # Both actual Git commands must succeed inside the real write guard.
        self.assertIn((('rev-parse', 'HEAD'), True), git_calls)
        self.assertIn((('status', '--porcelain', '--untracked-files=all'), True), git_calls)
        self.assertEqual(len(receipts), 1)
        self.assertEqual(receipts[0]['error_kind'], '_PlanReached')
        self.assertEqual((receipts[0]['output']['used_bytes'],
                          receipts[0]['output']['created_count'],
                          receipts[0]['output']['refused']), (0, 0, 0))

    def test_writable_descriptors_and_nonallowlisted_files_are_still_rejected(self):
        # Even a broken guard cannot open or overwrite any real target here.
        with mock.patch.object(os, 'open', side_effect=AssertionError('OS open reached')), \
             mock.patch('builtins.open', side_effect=AssertionError('File open reached')):
            with owned_writes(REPO, 1024, allowed_paths=[]) as budget:
                with self.assertRaises(UnsafeWriteError):
                    os.open(os.devnull, os.O_RDWR)
                with self.assertRaises(UnsafeWriteError):
                    os.open(REPO / 'synthetic-never-created', os.O_CREAT | os.O_WRONLY)
                with self.assertRaises(UnsafeWriteError):
                    open(REPO / 'synthetic-never-created', 'wb')
                result = budget.snapshot()
        self.assertEqual((result['used_bytes'], result['created_count'],
                          result['refused']), (0, 0, 0))


if __name__ == '__main__':
    unittest.main()
