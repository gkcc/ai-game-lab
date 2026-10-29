"""Synthetic old-ledger integration; no games, models or real seeds."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from learning.task_teacher_merged_eval_v1 import campaign, evaluation, independent_runner
from . import runner


class LedgerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.stage = Path(self.temp.name)
        self.seeds = [101, 202, 303, 404]
        self.cfg = {'seed_commitment':'a'*64}
        self.identities = {'freeze.json':'b'*64}
        self.ok = dict(passed=True, assigned=True, resumed=True, launcher_reaped=True, job_members_zero=True,
                       error_kind=None, cleanup_error_kind=None, stop_reason=None, exit_code=0)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(runner, '_original_accept_job', side_effect=self.original))
        self.stack.enter_context(patch.object(campaign, 'accept_job', side_effect=runner.accept_persistent_job))
        self.stack.enter_context(patch.object(campaign, 'load_seeds', return_value=self.seeds))
        self.stack.enter_context(patch.object(campaign, 'check_identity'))
        self.rows, self.audits = self.make_ledger()
        self.stack.enter_context(patch.object(campaign, 'validate_panel', side_effect=self.validate_panel))
        self.stack.enter_context(patch.object(campaign, 'storage_ready'))
        self.stack.enter_context(patch.object(independent_runner, 'trusted_out', side_effect=lambda p:Path(p)))
        self.stack.enter_context(patch.object(independent_runner, 'ready', return_value=self.cfg))
        self.stack.enter_context(patch.object(independent_runner, 'verify_commit', return_value=self.cfg))
        self.stack.enter_context(patch.object(independent_runner, 'load_seeds', return_value=self.seeds))
        self.stack.enter_context(patch.object(independent_runner, 'validate_completed', return_value={}))
        self.stack.enter_context(patch.object(independent_runner, 'write_new', side_effect=self.write))

    def write(self, path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            json.dump(value, stream)

    def original(self, out, item, cfg):
        return {'game':item['game']}, {'game':item['game']}

    def validate_panel(self, out, cfg):
        return campaign.verify_ledger(out, cfg, self.rows, self.audits, self.identities)

    def make_ledger(self):
        rows, audits = [], []
        for plan in evaluation.plans(self.seeds):
            key = plan['game']
            wrapper = self.stage/'agents'/(key+'.py')
            wrapper.parent.mkdir(exist_ok=True)
            wrapper.write_bytes(('synthetic '+key).encode())
            digest = runner._sha(wrapper)
            item = dict(plan, bot_sha256=digest, rival_sha256='c'*64, task_checkpoint_sha256='d'*64)
            input_path = self.stage/'job-inputs'/(key+'.json')
            self.write(input_path, item)
            docs = {
                'canonical':dict(game=key, wrapper_sha256=digest, passed=True, launcher_reaped=True,
                                 job_members_zero=True, isolated_cache_prefix_absent=True, retry_allowed=False),
                'job-start':dict(game=key, wrapper_sha256=digest, retry_allowed=False),
                'canonical-post':dict(game=key, wrapper_sha256=digest, passed=True, retry_allowed=False),
            }
            for name, document in docs.items():
                self.write(self.stage/'results'/name/(key+'.json'), document)
            row, audit = runner.accept_persistent_job(self.stage, item, self.cfg)
            supervision_path = self.stage/'job-supervision'/(key+'.json')
            self.write(supervision_path, dict(game=key, job_input_sha256=runner._sha(input_path),
                freeze_sha256=self.identities['freeze.json'], supervision=self.ok, error_kind=None,
                cleanup_state='recorded', retry_allowed=False))
            audit['job_supervision'] = self.ok
            acceptance_path = self.stage/'job-acceptance'/(key+'.json')
            self.write(acceptance_path, dict(passed=True, game=key, freeze_sha256=self.identities['freeze.json'],
                job_input_sha256=runner._sha(input_path), supervision_sha256=runner._sha(supervision_path),
                evidence=copy.deepcopy(audit), retry_allowed=False))
            audit['supervision_receipt_sha256'] = runner._sha(supervision_path)
            audit['acceptance_receipt_sha256'] = runner._sha(acceptance_path)
            rows.append(row)
            audits.append(audit)
        self.write(self.stage/'game-ledger.json', rows)
        self.write(self.stage/'game-audit-ledger.json', audits)
        return rows, audits

    def test_complete_synthetic_ledger_reaches_finalize(self):
        self.assertEqual(self.validate_panel(self.stage, self.cfg), self.rows)
        names = independent_runner.RUN_IDENTITIES + ('independent-runner-result.json',
            'independent-replay-verification.json','independent-attempt.local.json','independent-result-review.json')
        for name in names:
            path = self.stage/name
            if not path.exists(): path.write_text('{}', encoding='utf-8')
        with patch.object(evaluation, 'evaluate', return_value={'synthetic_only':True}) as evaluate:
            result = independent_runner.finalize(self.stage)
        self.assertTrue(result['passed'])
        self.assertTrue(result['synthetic_only'])
        evaluate.assert_called_once()

    def test_missing_receipt_stops_replay_and_evaluation_before_start(self):
        game = evaluation.plans(self.seeds)[37]['game']
        (self.stage/'results'/'canonical-post'/(game+'.json')).unlink()
        with patch.object(campaign, 'memory_ready', return_value=True), \
             patch.object(independent_runner, 'supervise') as supervise, \
             patch.object(evaluation, 'evaluate') as evaluate:
            with self.assertRaises((ValueError, OSError)): independent_runner.run(self.stage)
            with self.assertRaises((ValueError, OSError)): independent_runner.finalize(self.stage)
            supervise.assert_not_called()
            evaluate.assert_not_called()
        self.assertFalse((self.stage/'independent-runner-started.json').exists())
        self.assertFalse((self.stage/'evaluation-started.json').exists())


if __name__ == '__main__':
    unittest.main()
