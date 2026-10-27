import unittest
from pathlib import Path
from unittest.mock import patch,mock_open
from . import experiment,runner


class GateTest(unittest.TestCase):
    def test_source_audit_binding(self):
        train=[dict(game=f'train{i}',audit_sha256='audit') for i in range(8)]
        games=[]; reports=[]
        for game,seed in zip(experiment.GAMES,(3911611436,2328454240)):
            games.append(dict(key=game,seed=seed,side=0,opponent='market',arm='teacher_task',
                trace=dict(path='traces/x',size=1,sha256='t'),replay=dict(sha256='r')))
            reports.append(dict(game=game,passed=True,errors=0,actor_actions_reproduced=719,
                rival_actions_reproduced=719,trace_turns_reproduced=719,trace_sha256='t',replay_sha256='r'))
        self.assertEqual(len(experiment.bind_sources(train,dict(games=games),reports,'audit')),2)
        with self.assertRaises(ValueError): experiment.bind_sources(train,dict(games=games),reports,'other')
        with self.assertRaises(ValueError): experiment.bind_sources(train[:-1]+[dict(game=experiment.GAMES[0],audit_sha256='audit')],dict(games=games),reports,'audit')
        with self.assertRaises(ValueError): experiment.bind_sources(train,dict(games=games),[{**reports[0],'trace_sha256':'bad'},reports[1]],'audit')

    def test_probe_failure_blocks_worker(self):
        records={}; cfg=dict(probe_job_limits={},job_limits={})
        def read(path):
            if Path(path).name=='pre-execution-commit.json': return dict(commit='c')
            if Path(path).name=='probe-result.json': return dict(passed=False)
            raise AssertionError(Path(path).name)
        with patch.object(runner,'ready',return_value=cfg),patch.object(runner,'verify_commit'), \
             patch.object(Path,'exists',return_value=False),patch.object(Path,'open',mock_open()), \
             patch.object(runner,'sha',return_value='h'),patch.object(runner,'read',side_effect=read), \
             patch.object(runner,'launch_named',return_value={'passed':False}) as launch, \
             patch.object(runner,'progress_snapshot',return_value={}), \
             patch.object(runner,'write_new',side_effect=lambda p,v:records.update({Path(p).name:v})):
            with self.assertRaises(RuntimeError): runner.run()
            self.assertEqual(launch.call_count,1); self.assertFalse(records['result.json']['passed'])

    def test_probe_identity_mismatch(self):
        job=dict(passed=True,assigned=True,resumed=True,launcher_reaped=True,job_members_zero=True,error_kind=None,cleanup_error_kind=None,exit_code=0)
        probe=dict(passed=True,venv_verified=True,job_membership_verified=True,training_inputs_read=False,freeze_sha256='f',marker_sha256='bad')
        with patch.object(runner,'sha',return_value='m'),self.assertRaises(ValueError):
            runner.validate_probe(Path('unused'),dict(freeze_sha256='f',probe_marker_sha256='m'),job,probe)

    def test_partial_marker_stops_without_probe(self):
        records={}
        with patch.object(runner,'ready',return_value={}),patch.object(runner,'verify_commit'), \
             patch.object(Path,'exists',return_value=False),patch.object(Path,'open',mock_open()), \
             patch.object(runner,'sha',return_value='h'),patch.object(runner.json,'dump',side_effect=OSError('partial')), \
             patch.object(runner,'launch_named') as launch,patch.object(runner,'progress_snapshot',return_value={}), \
             patch.object(runner,'write_new',side_effect=lambda p,v:records.update({Path(p).name:v})):
            with self.assertRaises(RuntimeError): runner.run()
            launch.assert_not_called()
            self.assertFalse(records['result.json']['marker']['complete'])


if __name__=='__main__': unittest.main()
