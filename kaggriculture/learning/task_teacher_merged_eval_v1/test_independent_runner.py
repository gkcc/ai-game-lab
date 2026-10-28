import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from learning.task_choice_v1.features import OPS
from . import campaign, experiment, independent_runner as runner
from .test_evaluation import SEEDS, panel


class IndependentRunnerTests(unittest.TestCase):
    def setUp(self):
        self.cfg=dict(seed_document_sha256='a'*64,merged_checkpoint_sha256='b'*64,
            crop_checkpoint_sha256='c'*64,engine_sha256='9'*64,independent_selection='fixed selection',
            seed_commitment='d'*64,independent_job_supervision={})
        boundary=patch.object(runner,'trusted_out',side_effect=lambda out:Path(out).resolve())
        boundary.start(); self.addCleanup(boundary.stop)

    def test_low_memory_stops_before_marker_and_job(self):
        with patch.object(runner,'ready',return_value=self.cfg),patch.object(runner,'verify_commit',return_value=self.cfg), \
             patch.object(campaign,'validate_panel'),patch.object(campaign,'memory_ready',return_value=False), \
             patch.object(runner,'load_seeds',return_value=SEEDS),patch.object(runner,'sha',return_value='a'*64), \
             patch.object(Path,'exists',return_value=False),patch.object(Path,'open') as marker, \
             patch.object(runner,'supervise') as job,patch.object(runner,'write_new') as write:
            with self.assertRaisesRegex(RuntimeError,'preflight memory'): runner.run()
            marker.assert_not_called(); job.assert_not_called(); write.assert_not_called()

    def test_drift_during_preflight_stops_before_marker_and_job(self):
        changed={**self.cfg,'seed_commitment':'f'*64}
        with patch.object(runner,'ready',side_effect=[self.cfg,changed]),patch.object(runner,'verify_commit',return_value=self.cfg), \
             patch.object(campaign,'validate_panel'),patch.object(campaign,'memory_ready',return_value=True), \
             patch.object(campaign,'storage_ready'), \
             patch.object(runner,'load_seeds',return_value=SEEDS),patch.object(runner,'sha',return_value='a'*64), \
             patch.object(Path,'exists',return_value=False),patch.object(Path,'open') as marker, \
             patch.object(runner,'supervise') as job:
            with self.assertRaisesRegex(ValueError,'preflight'): runner.run()
            marker.assert_not_called(); job.assert_not_called()

    def test_panel_drift_during_preflight_stops_before_marker(self):
        with patch.object(runner,'ready',return_value=self.cfg),patch.object(runner,'verify_commit',return_value=self.cfg), \
             patch.object(campaign,'validate_panel',side_effect=[[],ValueError('panel drift')]), \
             patch.object(campaign,'memory_ready',return_value=True),patch.object(runner,'load_seeds',return_value=SEEDS), \
             patch.object(campaign,'storage_ready'), \
             patch.object(runner,'sha',return_value='a'*64),patch.object(Path,'exists',return_value=False), \
             patch.object(Path,'open') as marker,patch.object(runner,'supervise') as job:
            with self.assertRaises(ValueError): runner.run()
            marker.assert_not_called(); job.assert_not_called()

    def test_storage_failure_stops_before_marker_and_job(self):
        with patch.object(runner,'ready',return_value=self.cfg),patch.object(runner,'verify_commit',return_value=self.cfg), \
             patch.object(campaign,'validate_panel'),patch.object(campaign,'memory_ready',return_value=True), \
             patch.object(campaign,'storage_ready',side_effect=RuntimeError('budget')), \
             patch.object(runner,'load_seeds',return_value=SEEDS),patch.object(runner,'sha',return_value='a'*64), \
             patch.object(Path,'exists',return_value=False),patch.object(Path,'open') as marker, \
             patch.object(runner,'supervise') as job:
            with self.assertRaises(RuntimeError): runner.run()
            marker.assert_not_called(); job.assert_not_called()

    def test_verification_binds_alias_counts_and_sources(self):
        counts={op:(20 if op==OPS[0] else 0) for op in OPS}
        verification=dict(passed=True,freeze_sha256='e'*64,registration_sha256='e'*64,
            games_result_sha256='e'*64,game_ledger_sha256='e'*64,audit_ledger_sha256='e'*64,method_sha256='e'*64,
            checkpoint_sha256='b'*64,crop_checkpoint_sha256='c'*64,seed_document_sha256='a'*64,
            game='holdout-S01-round2-0-merged',selection_rule='fixed selection',official_transitions=719,
            engine_sha256='9'*64,semantic_sha256='1'*64,semantic_sha256_v2='2'*64,
            game_row_sha256='e'*64,game_audit_sha256='e'*64,
            both_public_and_private_states_match_each_step=True,inventory_order_checked_each_step=True,
            final_rewards_match=True,terminal_status_done=True,
            sample_count=20,matched=20,sample_operation_counts=counts,sample_identity_sha256='f'*64,
            raw_reads=dict(files_opened=2,files_completed=2,raw_decode_passes_per_file=1),
            no_model_inference=True,no_new_games=True,retry_allowed=False)
        row=dict(semantic_replay_sha256='1'*64,semantic_replay_sha256_v2='2'*64)
        with patch.object(runner,'sha',return_value='e'*64),patch.object(runner,'read',return_value=row):
            runner.validate_verification(Path('synthetic'),self.cfg,verification)
            for change in (dict(game='holdout-S02-round2-0-merged'),dict(matched=19),
                           dict(checkpoint_sha256='9'*64),dict(no_model_inference=False),dict(semantic_sha256_v2='3'*64),
                           dict(inventory_order_checked_each_step=False),dict(game_row_sha256='f'*64),
                           dict(sample_operation_counts={op:0 for op in OPS})):
                with self.assertRaises(ValueError):
                    runner.validate_verification(Path('synthetic'),self.cfg,{**verification,**change})

    def test_job_requires_zero_exit_and_cleanup(self):
        job=dict(passed=True,assigned=True,resumed=True,launcher_reaped=True,job_members_zero=True,
                 error_kind=None,cleanup_error_kind=None,stop_reason=None,exit_code=0)
        self.assertTrue(runner.validate_job(job))
        for change in (dict(job_members_zero=False),dict(exit_code=True),dict(stop_reason='deadline')):
            self.assertFalse(runner.validate_job({**job,**change}))

    def test_completion_binds_method_marker_review_and_job(self):
        job=dict(passed=True,assigned=True,resumed=True,launcher_reaped=True,job_members_zero=True,
                 error_kind=None,cleanup_error_kind=None,stop_reason=None,exit_code=0)
        identities={name:'a'*64 for name in runner.RUN_IDENTITIES}
        receipt=dict(passed=True,supervision=job,freeze_sha256='a'*64,retry_allowed=False,
                     verification_sha256='a'*64,attempt_sha256='a'*64,review_sha256='a'*64,identities=identities)
        attempt=dict(attempt=1,retry_allowed=False,freeze_sha256='a'*64)
        review=dict(passed=True,selected_game='holdout-S01-round2-0-merged',checkpoint_sha256='b'*64,
            verification_sha256='a'*64,method_sha256='a'*64,official_transitions=719,sample_count=20,sample_matched=20,
            freeze_sha256='a'*64,registration_sha256='a'*64,games_result_sha256='a'*64,
            game_ledger_sha256='a'*64,audit_ledger_sha256='a'*64)
        records={'independent-runner-result.json':receipt,'independent-replay-verification.json':{},
                 'independent-attempt.local.json':attempt,'independent-result-review.json':review}
        def read(path): return records[Path(path).name]
        with patch.object(runner,'verify_commit',return_value=self.cfg),patch.object(runner,'read',side_effect=read), \
             patch.object(runner,'sha',return_value='a'*64),patch.object(runner,'validate_verification'), \
             patch.object(campaign,'validate_panel'),patch.object(Path,'exists',return_value=False):
            self.assertEqual(runner.validate_completed(Path('synthetic'),self.cfg),{})
            records['independent-attempt.local.json']={**attempt,'retry_allowed':True}
            with self.assertRaises(ValueError): runner.validate_completed(Path('synthetic'),self.cfg)
            records['independent-attempt.local.json']=attempt
            records['independent-result-review.json']={**review,'sample_matched':19}
            with self.assertRaises(ValueError): runner.validate_completed(Path('synthetic'),self.cfg)

    def test_supervised_replay_receipt_chain_with_owned_fake_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'project'; out=root/'artifacts'/experiment.STAGE_NAME; out.mkdir(parents=True)
            for name in runner.BASE_IDENTITIES:
                (out/name).write_bytes(('synthetic-'+name).encode())
            jobdir=out/'job-results'; jobdir.mkdir()
            key='holdout-S01-round2-0-merged'
            row=dict(semantic_replay_sha256='1'*64,semantic_replay_sha256_v2='2'*64)
            (jobdir/(key+'.row.json')).write_text(json.dumps(row),encoding='utf-8')
            (jobdir/(key+'.audit.json')).write_text('{}',encoding='utf-8')
            cfg={**self.cfg,'seed_document_sha256':runner.sha(out/'private-seeds.local.json')}
            job=dict(passed=True,assigned=True,resumed=True,launcher_reaped=True,job_members_zero=True,
                     error_kind=None,cleanup_error_kind=None,stop_reason=None,exit_code=0)
            def write(name,value):
                with (out/name).open('x',encoding='utf-8') as stream: json.dump(value,stream)
            def fake_supervise(api,command,cwd,limits):
                marker=json.loads((out/'independent-runner-started.json').read_text())
                self.assertEqual(marker,dict(attempt=1,freeze_sha256=runner.sha(out/'freeze.json'),retry_allowed=False))
                write('independent-attempt.local.json',marker)
                counts={op:(20 if op==OPS[0] else 0) for op in OPS}
                verification=dict(passed=True,game=key,official_transitions=719,sample_count=20,matched=20,
                    selection_rule=cfg['independent_selection'],sample_identity_sha256='f'*64,sample_operation_counts=counts,
                    raw_reads=dict(files_opened=2,files_completed=2,raw_decode_passes_per_file=1),
                    engine_sha256=cfg['engine_sha256'],semantic_sha256='1'*64,semantic_sha256_v2='2'*64,
                    both_public_and_private_states_match_each_step=True,inventory_order_checked_each_step=True,
                    final_rewards_match=True,terminal_status_done=True,checkpoint_sha256=cfg['merged_checkpoint_sha256'],
                    crop_checkpoint_sha256=cfg['crop_checkpoint_sha256'],seed_document_sha256=cfg['seed_document_sha256'],
                    freeze_sha256=runner.sha(out/'freeze.json'),registration_sha256=runner.sha(out/'registration.json'),
                    games_result_sha256=runner.sha(out/'games-result.json'),game_ledger_sha256=runner.sha(out/'game-ledger.json'),
                    audit_ledger_sha256=runner.sha(out/'game-audit-ledger.json'),game_row_sha256=runner.sha(jobdir/(key+'.row.json')),
                    game_audit_sha256=runner.sha(jobdir/(key+'.audit.json')),
                    method_sha256=runner.sha(Path(runner.__file__).with_name('independent_method.py')),
                    no_model_inference=True,no_new_games=True,retry_allowed=False)
                write('independent-replay-verification.json',verification)
                review=dict(passed=True,selected_game=key,checkpoint_sha256=cfg['merged_checkpoint_sha256'],
                    verification_sha256=runner.sha(out/'independent-replay-verification.json'),method_sha256=verification['method_sha256'],
                    official_transitions=719,sample_count=20,sample_matched=20,freeze_sha256=verification['freeze_sha256'],
                    registration_sha256=verification['registration_sha256'],games_result_sha256=verification['games_result_sha256'],
                    game_ledger_sha256=verification['game_ledger_sha256'],audit_ledger_sha256=verification['audit_ledger_sha256'])
                write('independent-result-review.json',review)
                return job
            with patch.object(runner,'OUT',out),patch.object(experiment,'ROOT',root),patch.object(experiment,'OUT',out), \
                 patch.object(runner,'trusted_out',side_effect=experiment.trusted_out),patch.object(runner,'ready',return_value=cfg), \
                 patch.object(runner,'verify_commit',return_value=cfg),patch.object(runner,'load_seeds',return_value=SEEDS), \
                 patch.object(campaign,'validate_panel'),patch.object(campaign,'memory_ready',return_value=True), \
                 patch.object(campaign,'storage_ready'),patch.object(runner,'WindowsAPI',return_value=object()), \
                 patch.object(runner,'supervise',side_effect=fake_supervise):
                result=runner.run(out)
                self.assertTrue(result['passed'])
                self.assertTrue(runner.validate_completed(out,cfg)['passed'])
                self.assertEqual(result['attempt_sha256'],runner.sha(out/'independent-attempt.local.json'))
                self.assertEqual(result['review_sha256'],runner.sha(out/'independent-result-review.json'))

    def test_finalize_requires_complete_replay_before_marker(self):
        with patch.object(runner,'ready',return_value=self.cfg),patch.object(runner,'verify_commit',return_value=self.cfg), \
             patch.object(Path,'exists',return_value=False),patch.object(campaign,'validate_panel',return_value=[]), \
             patch.object(runner,'validate_completed',side_effect=ValueError('missing replay')), \
             patch.object(runner,'write_new') as write:
            with self.assertRaises(ValueError): runner.finalize()
            write.assert_not_called()

    def test_finalize_writes_only_aliases_after_all_checks(self):
        writes={}
        def write(path,value): writes[Path(path).name]=value
        with patch.object(runner,'ready',return_value=self.cfg),patch.object(runner,'verify_commit',return_value=self.cfg), \
             patch.object(Path,'exists',return_value=False),patch.object(campaign,'validate_panel',return_value=panel()), \
             patch.object(campaign,'storage_ready'), \
             patch.object(runner,'validate_completed'),patch.object(runner,'load_seeds',return_value=SEEDS), \
             patch.object(runner,'sha',return_value='a'*64),patch.object(runner,'write_new',side_effect=write):
            result=runner.finalize()
        self.assertTrue(result['passed']); self.assertIn('evaluation-started.json',writes)
        self.assertEqual(writes['evaluation-result.json'],result)
        self.assertTrue(all(str(seed) not in json.dumps(result) for seed in SEEDS))


if __name__=='__main__': unittest.main()
