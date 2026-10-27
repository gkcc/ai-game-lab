"""Games entry lifecycle checks with no files or game processes."""
from contextlib import ExitStack
from pathlib import Path
import sys
import types
import unittest
import uuid
from unittest.mock import patch
from .test_entry import c


class GamesEntryTest(unittest.TestCase):
    def run_case(self, failure):
        out = c.ROOT/('synthetic-games-'+uuid.uuid4().hex)
        records = {}
        queue = [dict(game=f'g{i}', opponent='rival') for i in range(8)]
        cfg = dict(storage=dict(game_write_bytes=1, process_log_bytes=1),
                   official_limits=dict(per_job_seconds=100), engine_sha256='hash')
        clean = dict(owned_instances_remaining=0, reader_joined=True, launched_process_reaped=True)
        class Process:
            def poll(self):
                if failure == 'poll': raise OSError('poll')
                return 0
        class Child:
            process = Process()
            def finish(self): return dict(child_exit_code=1 if failure == 'worker' else 0)
        def marker(out, value, state):
            state['created'] = True
            records['games-started.json'] = value
            if failure == 'marker': raise OSError('partial')
            state['complete'] = True
        def write(path, value):
            name = Path(path).name
            if failure == 'receipt' and name == 'games-failure.local.json': raise PermissionError('receipt')
            if name in records: raise FileExistsError(name)
            records[name] = value
        def read(path):
            name = Path(path).name
            key, kind = name.split('.')[:2]
            plan = next(p for p in queue if p['game'] == key)
            row = dict(plan, bot_sha256='hash', rival_sha256='hash', task_checkpoint_sha256=None,
                       passed=True, states=720, action_turns=719, status=['DONE','DONE'], agent_errors=0,
                       banks=[1,2], replay_gzip_sha256='hash', trace_sha256='hash')
            values = dict(row=row, complete=dict(passed=True, game=key, row_sha256='hash', certificate_sha256='hash'),
                storage=dict(passed=True, error_kind=None, output=dict(refused=0)),
                audit=dict(passed=True, errors=0, actor_actions_reproduced=719, rival_actions_reproduced=719,
                    trace_turns_reproduced=719, official=dict(official_transitions_reexecuted=719,
                    engine_sha256='hash', final_rewards=[1,2])))
            return values[kind]
        fake_games = types.ModuleType('learning.task_completion_bc_reentry_v1.games')
        fake_games.make_wrapper = lambda *args: out/'bot.py'
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {'learning.task_completion_bc_reentry_v1.games': fake_games}))
            for name, result in (('ready',cfg),('verify_commit',None),('verify_model_seal',({},{})),
                                 ('plans',queue),('sha','hash'),('deadline_check',None),
                                 ('storage_ready',None),('observe_processes',None)):
                stack.enter_context(patch.object(c,name,return_value=result))
            stack.enter_context(patch.object(c,'write_games_marker',side_effect=marker))
            stack.enter_context(patch.object(c,'write_new',side_effect=write))
            stack.enter_context(patch.object(c,'read',side_effect=read))
            stack.enter_context(patch.object(Path,'exists',autospec=True,side_effect=lambda p: Path(p).name in records))
            stack.enter_context(patch.object(c,'CapturedProcess',return_value=Child(),
                side_effect=OSError('spawn') if failure in ('spawn','receipt') else None))
            stack.enter_context(patch.object(c,'reap',return_value=({**clean,'owned_instances_remaining':1} if failure == 'incomplete' else clean),
                side_effect=RuntimeError('reap') if failure == 'cleanup' else None))
            if failure is None:
                self.assertTrue(c.games(out)['passed'])
            else:
                with self.assertRaises(c.FitFailure) as caught: c.games(out)
                if failure == 'receipt': self.assertEqual(caught.exception.receipt_error_kind,'PermissionError')
            if failure and failure != 'receipt':
                r = records['games-failure.local.json']
                self.assertFalse(r['retry_allowed'])
                if failure == 'marker': self.assertFalse(r['marker_complete'])
                if failure == 'spawn': self.assertEqual(r['child_lifecycle'],'unknown_after_spawn_attempt')
                if failure == 'incomplete':
                    self.assertEqual(r['last_cleanup']['owned_instances_remaining'],1)
                    self.assertEqual(r['child_lifecycle'], 'cleanup_recorded')
                if failure == 'cleanup': self.assertEqual(r['cleanup_error_kind'],'RuntimeError')
            with self.assertRaises(ValueError): c.games(out)

    def test_games_paths(self):
        for failure in (None,'marker','spawn','poll','worker','incomplete','cleanup','receipt'):
            with self.subTest(failure=failure): self.run_case(failure)


if __name__ == '__main__': unittest.main()
