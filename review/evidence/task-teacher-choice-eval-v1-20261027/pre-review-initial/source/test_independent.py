import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

exp=types.ModuleType('learning.task_teacher_choice_eval_v1.experiment')
exp.ROOT=Path(__file__).resolve().parents[2]; exp.OUT=exp.ROOT/'synthetic'
for n in ('ready','verify_commit','deadline_check','read','sha','write_new'): setattr(exp,n,lambda *a:None)
def load(name):
    spec=importlib.util.spec_from_file_location('learning.task_teacher_choice_eval_v1._test_'+name,Path(__file__).with_name(name+'.py'))
    module=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,{'learning.task_teacher_choice_eval_v1.experiment':exp}): spec.loader.exec_module(module)
    return module
method=load('independent_method'); runner=load('independent_runner')


class IndependentTest(unittest.TestCase):
    def test_fixed_sample(self):
        identities=[('game',i,0) for i in range(25)]
        self.assertEqual(len(method.sample_identities(identities)),20)
        self.assertEqual(method.sample_identities(identities),method.sample_identities(list(reversed(identities))))
        with self.assertRaises(ValueError): method.sample_identities(identities+[identities[0]])

    def test_finalization_requires_independent_receipt(self):
        with patch.object(runner,'ready',return_value={'test_seeds':[1,2]}),patch.object(runner,'verify_commit'), \
             patch.object(Path,'exists',return_value=False),patch.object(runner,'read',side_effect=[dict(passed=True),dict(passed=False),dict(passed=True)]), \
             patch.object(runner,'write_new') as write:
            with self.assertRaises(ValueError): runner.finalize(exp.OUT)
            write.assert_not_called()

    def test_changed_panel_rejected_despite_updated_panel_receipt(self):
        cfg=dict(test_seeds=[1,2],candidate_checkpoint_sha256='model')
        hashes={'freeze.json':'freeze','registration.json':'registration','games-result.json':'new-result',
            'game-ledger.json':'new-ledger','game-audit-ledger.json':'audit','independent_method.py':'method',
            'independent-replay-verification.json':'verification'}
        def sha(path): return hashes[Path(path).name]
        panel=dict(passed=True,game_ledger_sha256='new-ledger',audit_ledger_sha256='audit')
        job=dict(passed=True,assigned=True,resumed=True,launcher_reaped=True,job_members_zero=True,
                 error_kind=None,cleanup_error_kind=None,exit_code=0)
        receipt=dict(passed=True,supervision=job,freeze_sha256='freeze',verification_sha256='verification')
        verification=dict(passed=True,freeze_sha256='freeze',registration_sha256='registration',
            games_result_sha256='old-result',game_ledger_sha256='old-ledger',audit_ledger_sha256='audit',
            method_sha256='method',checkpoint_sha256='model',game='holdout-1-round2-0-candidate',
            official_transitions=719,sample_count=20,matched=20,
            raw_reads=dict(files_opened=2,files_completed=2,raw_decode_passes_per_file=1))
        with patch.object(runner,'ready',return_value=cfg),patch.object(runner,'verify_commit'), \
             patch.object(Path,'exists',return_value=False),patch.object(runner,'sha',side_effect=sha), \
             patch.object(runner,'read',side_effect=[panel,receipt,verification]),patch.object(runner,'write_new') as write:
            with self.assertRaises(ValueError): runner.finalize(exp.OUT)
            write.assert_not_called()

    def test_verification_and_job_boundaries(self):
        cfg=dict(test_seeds=[1,2],candidate_checkpoint_sha256='model')
        verification=dict(passed=True,freeze_sha256='h',registration_sha256='h',games_result_sha256='h',
            game_ledger_sha256='h',audit_ledger_sha256='h',method_sha256='h',checkpoint_sha256='model',
            game='holdout-1-round2-0-candidate',official_transitions=719,sample_count=20,matched=20,
            raw_reads=dict(files_opened=2,files_completed=2,raw_decode_passes_per_file=1))
        with patch.object(runner,'sha',return_value='h'):
            runner.validate_verification(exp.OUT,cfg,verification)
            for change in (dict(game='other'),dict(matched=19),dict(method_sha256='changed'),dict(raw_reads={'files_opened':1})):
                with self.assertRaises(ValueError): runner.validate_verification(exp.OUT,cfg,{**verification,**change})
        self.assertFalse(runner.validate_job(dict(passed=True,launcher_reaped=True,job_members_zero=False)))


if __name__ == '__main__': unittest.main()
