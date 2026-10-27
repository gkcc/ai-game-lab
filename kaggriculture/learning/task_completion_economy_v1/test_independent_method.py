"""Pure synthetic checks, with an isolated stand-in for the unbuilt experiment."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from learning.task_choice_v1.features import OPS  # Load native dependencies before the isolated module patch.


def load_method():
    fake = types.ModuleType('learning.task_completion_economy_v1.experiment')
    fake.OUT = Path('synthetic-out').resolve()
    fake.ROOT = Path('.').resolve()
    for name in ('deadline_check', 'verify_commit'):
        setattr(fake, name, lambda *args: None)
    fake.read = fake.ready = fake.sha = fake.write_new = lambda *args: None
    path = Path(__file__).with_name('independent_method.py')
    spec = importlib.util.spec_from_file_location('learning.task_completion_economy_v1._synthetic_independent', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'learning.task_completion_economy_v1.experiment': fake}):
        spec.loader.exec_module(module)
    return module


class IndependentTests(unittest.TestCase):
    def test_identity_sample_and_operation_coverage(self):
        m = load_method()
        ids = [('g', i, 0) for i in range(30)]
        selected = m.sample_identities(ids)
        self.assertEqual(selected, m.sample_identities(list(reversed(ids))))
        counts = m.sample_operation_counts(selected, {i:'WATER' for i in ids})
        self.assertEqual((sum(counts.values()),counts['WATER'],counts['FERTILIZE']), (20,20,0))
        for bad in (ids[:19], ids + ids[:1]):
            with self.assertRaises(ValueError): m.sample_identities(bad)

    def test_partial_marker_and_receipt_error_preserve_primary(self):
        m = load_method()
        from learning.task_completion_bc_v1 import collect
        seeds = [7,8]
        cfg = dict(test_seeds=seeds,candidate_checkpoint_sha256='model',independent_limits=dict(seconds=1,compressed_bytes=1,decoded_bytes=1))
        rows = [dict(game=f'holdout-{seed}-round2-{seat}-{arm}',seed=seed,opponent='round2',side=seat,policy=arm,task_checkpoint_sha256='model') for seed in seeds for seat in (0,1) for arm in ('candidate','teacher_task')]
        audits = [dict(game=x['game'],certificate_sha256='hash',replay_sha256='r',trace_sha256='t',official_transitions=719) for x in rows]
        data = {'games-result.json':dict(passed=True,all_owned_processes_reaped=True,game_ledger_sha256='hash',audit_ledger_sha256='hash'),'game-ledger.json':rows,'game-audit-ledger.json':audits}
        game = dict(replay=dict(sha256='r'),trace=dict(sha256='t'))
        def marker(out,state):
            state['created'] = True
            raise OSError('synthetic partial')
        with patch.object(m,'ready',return_value=cfg), patch.object(m,'verify_commit'), \
             patch.object(m,'read',side_effect=lambda p:data[Path(p).name]), patch.object(m,'sha',return_value='hash'), \
             patch.object(collect,'current_game',return_value=(game,{})), patch.object(Path,'exists',return_value=False), \
             patch.object(m,'create_marker',side_effect=marker), patch.object(m,'write_new',side_effect=PermissionError('synthetic')):
            with self.assertRaises(m.ReceiptFailure) as caught: m.run()
        self.assertEqual((caught.exception.original_error_kind,caught.exception.receipt_error_kind),('OSError','PermissionError'))

if __name__ == '__main__': unittest.main()
