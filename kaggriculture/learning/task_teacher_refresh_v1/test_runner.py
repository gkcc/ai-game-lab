import unittest
from unittest.mock import patch
from . import experiment, runner


class BrokenMarker:
    def __enter__(self): return self
    def __exit__(self,*unused): return False
    def write(self,_): raise OSError('synthetic failed marker write')


class RunnerSafetyTest(unittest.TestCase):
    def test_private_metadata_is_excluded_case_insensitively(self):
        self.assertTrue(experiment.metadata_path_allowed('review/evidence/registration.json'))
        self.assertFalse(experiment.metadata_path_allowed('review/Private/registration.json'))
        self.assertFalse(experiment.metadata_path_allowed('stage/PRIVATE/registration.json'))

    def test_partial_exclusive_marker_records_consumed_attempt(self):
        for func,receipt_name in ((runner.run_games,'games-failure.json'),(runner.run_data,'data-result.json')):
            receipts=[]
            with self.subTest(stage=receipt_name), \
                patch.object(runner,'ready',return_value={}), \
                patch.object(runner,'verify_commit'), \
                patch.object(runner,'validate_panel'), \
                patch.object(runner.games,'plans',return_value=[]), \
                patch.object(runner.Path,'exists',return_value=False), \
                patch.object(runner.Path,'open',return_value=BrokenMarker()), \
                patch.object(runner,'sha',return_value='a'*64), \
                patch.object(runner,'write_new',side_effect=lambda p,v:receipts.append((p.name,v))):
                with self.assertRaises(OSError): func()
            self.assertEqual(len(receipts),1)
            self.assertEqual(receipts[0][0],receipt_name)
            self.assertEqual(receipts[0][1]['marker'],dict(created=True,complete=False))
            self.assertEqual(receipts[0][1]['error_kind'],'OSError')
            self.assertFalse(receipts[0][1]['retry_allowed'])

    def test_changed_game_audit_stops_before_probe_or_data(self):
        cfg=dict(train_seeds=[1,2],reserved_test_seeds=[3,4],opponents=['round2'],seats=[0,1])
        rows=[]; audits=[]; sources=[]
        for plan in runner.games.plans(cfg):
            rows.append(dict(plan,bot_sha256='a'*64,rival_sha256='b'*64,task_checkpoint_sha256=None,
                             replay_gzip_sha256='c'*64,trace_sha256='d'*64))
            audits.append(dict(game=plan['game'],row_sha256='1'*64,audit_sha256='2'*64,
                               complete_sha256='3'*64,storage_sha256='4'*64,trace_size=1))
            sources.append(dict(game=plan['game'],seed=plan['seed'],side=plan['side'],
                trace=dict(path='traces/'+plan['game']+'.jsonl.gz',size=1,sha256='d'*64),
                replay_sha256='c'*64,audit_sha256='2'*64))
        result=dict(passed=True,complete_games=4,live_official_transitions=4*719,primary_replay_transitions=4*719,
                    game_ledger_sha256='5'*64,audit_ledger_sha256='6'*64,source_manifest_sha256='7'*64)
        objects={'games-result.json':result,'game-ledger.json':rows,
                 'game-audit-ledger.json':audits,'source-manifest.json':sources}
        for missing in (False,True):
            def fake_hash(path):
                name=path.name
                if name in ('game-ledger.json','game-audit-ledger.json','source-manifest.json'):
                    return {'game-ledger.json':'5','game-audit-ledger.json':'6','source-manifest.json':'7'}[name]*64
                if name.endswith('.row.json'): return '1'*64
                if name.endswith('.audit.json'):
                    if missing: raise FileNotFoundError('synthetic missing receipt')
                    return 'f'*64  # changed against the sealed '2'
                raise AssertionError('Unexpected file read')
            with self.subTest(missing=missing), \
                patch.object(runner,'ready',return_value=cfg), patch.object(runner,'verify_commit'), \
                patch.object(runner,'read',side_effect=lambda p:objects[p.name]), \
                patch.object(runner,'sha',side_effect=fake_hash), \
                patch.object(runner,'launch_named') as launch, patch.object(runner,'marker') as marker:
                with self.assertRaises(FileNotFoundError if missing else ValueError): runner.run_data()
                launch.assert_not_called()
                marker.assert_not_called()

    def test_stopped_probe_never_reads_receipt_or_launches_data(self):
        job=dict(passed=False,assigned=True,resumed=True,launcher_reaped=True,job_members_zero=True,
                 exit_code=1,error_kind=None,cleanup_error_kind=None,stop_reason='available_memory_limit')
        receipts=[]
        def make_marker(out,name,value,state):
            state.update(created=True,complete=True)
        with patch.object(runner,'ready',return_value={'probe_job_supervision':{}}), \
            patch.object(runner,'verify_commit'), patch.object(runner,'validate_panel'), \
            patch.object(runner.Path,'exists',return_value=False), patch.object(runner,'sha',return_value='a'*64), \
            patch.object(runner,'marker',side_effect=make_marker), \
            patch.object(runner,'write_new',side_effect=lambda p,v:receipts.append((p.name,v))), \
            patch.object(runner,'read',side_effect=AssertionError('probe receipt must not be read')), \
            patch.object(runner,'launch_named',return_value=job) as launch:
            with self.assertRaisesRegex(RuntimeError,'Probe rejected'): runner.run_data()
        self.assertEqual(launch.call_count,1)
        self.assertEqual(launch.call_args.args[1],'learning.task_teacher_refresh_v1.member_probe')
        receipt=dict(receipts)['data-result.json']
        self.assertEqual(receipt['probe']['primary_kind'],'job_stopped')
        self.assertEqual(receipt['probe']['stop_reason'],'available_memory_limit')
        self.assertFalse(receipt['worker_launch_entered'])
        self.assertEqual(receipt['raw_trace_reads'],'not_started_by_runner')


if __name__ == '__main__':
    unittest.main()
