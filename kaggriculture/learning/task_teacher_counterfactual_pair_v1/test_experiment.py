"""Synthetic coordinator checks; no private candidates, models or games."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import os
from pathlib import Path
import stat
import subprocess
from types import SimpleNamespace
import unittest
from unittest import mock

from . import experiment as e


def fake_plan():
    rows = []
    for role, slot in (('A05', 'T01'), ('A06', 'T02')):
        for opponent, side in (('market', 0), ('round2', 1)):
            candidate = dict(role=role, slot=slot, opponent=opponent, side=side,
                window=0, rank=0, step=0, ordinal=0, input_sha256='a'*64,
                original_choice='synthetic-student', teacher_choice='synthetic-teacher')
            rows.append(dict(candidate=candidate))
    return dict(cases=rows)


def fake_binding():
    old = dict(roles={'A05': 1, 'A06': 2}, baseline_sha256='b'*64,
               crop_sha256='c'*64, software={'synthetic': 'test'})
    return e._binding(fake_plan(), old, {'synthetic': 'identity'}, salt='d'*64, default_sha='e'*64)


def account(delta=1):
    before = dict(own=10, opponent=10, margin=0, score=0.5)
    after = dict(own=10+delta, opponent=10, margin=delta,
                 score=1 if delta > 0 else 0.5 if delta == 0 else 0)
    diff = {key: after[key]-before[key] for key in before}
    return dict(control=before, intervention=after, delta=diff)


def phase_result(phase, *, control_ok=True):
    detail = {}
    if phase == 'prepare_case':
        detail = dict(prepared=True)
    elif phase in ('control', 'intervene'):
        detail = dict(states=720, action_turns=719, both_done=True, agent_errors=0)
    elif phase in ('control_audit', 'intervene_audit'):
        detail = dict(action_turns=719, shed_order_frames_checked=720,
                      semantic_sha256='c'*64, semantic_sha256_v2='d'*64)
    elif phase == 'control_verify':
        detail = dict(control_verified=control_ok)
    elif phase == 'pair_verify':
        detail = dict(pair_verified=True, account=account())
    return dict(passed=True, phase=phase,
                outputs={name: 'f'*64 for name in e._required_paths(phase)}, details=detail)


class PrivateBindingTests(unittest.TestCase):
    def test_all_qualified_candidates_and_actual_order_are_bound(self):
        binding = fake_binding()
        self.assertEqual(len(binding['cases']), 4)
        self.assertEqual(binding['execution_order'],
                         [dict(case_index=i, phase=p) for i in range(4) for p in e.PHASES])
        changed = fake_plan()
        changed['cases'][0]['candidate']['ordinal'] = 2
        old = dict(roles={'A05': 1, 'A06': 2}, baseline_sha256='b'*64,
                   crop_sha256='c'*64, software={'synthetic': 'test'})
        other = e._binding(changed, old, {'synthetic': 'identity'}, salt='d'*64, default_sha='e'*64)
        self.assertNotEqual(e._sha(e.base._encoded(binding)), e._sha(e.base._encoded(other)))
        self.assertNotIn('cases', e._registration({}, 'f'*64,
            dict(python='3.11.5', numpy='1.26.0', psutil='6.0.0', **{'kaggle-environments':'1.33.0'})))

    def test_screen_snapshot_hashes_exact_bytes_and_detects_changed_seal(self):
        result = {'gate': {'status': 'sufficient_for_separate_design'}}
        plan = {'cases': []}
        plan_raw, result_raw = e.base._encoded(plan), e.base._encoded(result)
        closed = dict(registration_commit=e.SCREEN_COMMIT,
                      receipts={'private-plan':e._sha(plan_raw), 'worker-result':e._sha(result_raw)})
        closed_raw = e.base._encoded(closed)
        files = {'closed.local.json':closed_raw, 'private-plan.local.json':plan_raw,
                 'worker-result.local.json':result_raw}
        old = {'synthetic': 'binding'}
        with mock.patch.object(e, '_read', side_effect=lambda p, **_k:files[p.name]), \
             mock.patch.object(e.screen, 'load_completed', return_value=result), \
             mock.patch.object(e.screen, '_validate_documents', return_value=result), \
             mock.patch.object(e.screen, '_registered_binding', return_value=(old, 'a'*64)):
            actual, actual_old, identity = e._screen_snapshot(lambda:None)
            self.assertEqual((actual, actual_old), (plan, old))
            self.assertEqual(identity['screen_plan_sha256'], e._sha(plan_raw))
            files['private-plan.local.json'] = e.base._encoded({'cases':['changed']})
            with self.assertRaises(ValueError):
                e._screen_snapshot(lambda:None)

    def test_descriptive_signal_requires_own_and_both_guardrails(self):
        self.assertTrue(e._signal([account() for _ in range(4)])['descriptive_signal'])
        self.assertFalse(e._signal([account(0) for _ in range(4)])['descriptive_signal'])
        with self.assertRaises(ValueError):
            e._signal([account() for _ in range(3)])
        bad = account(); bad['delta']['own'] = float('nan')
        with self.assertRaises(ValueError):
            e._validate_account(bad)
        binding = fake_binding()
        roles = e._role_summary([account(), account(), account(0), account(0)], binding['cases'])
        self.assertEqual(roles['A05']['own'], 1)
        self.assertEqual(roles['A06']['own'], 0)

    def test_required_game_and_audit_outcomes_fail_closed(self):
        e._phase_result(phase_result('control'), 'control')
        e._phase_result(phase_result('control_audit'), 'control_audit')
        bad = phase_result('control'); bad['outputs'].pop('games/control/replay.json')
        with self.assertRaises(ValueError): e._phase_result(bad, 'control')
        bad = phase_result('control_audit'); bad['details']['shed_order_frames_checked'] = 719
        with self.assertRaises(ValueError): e._phase_result(bad, 'control_audit')
        with self.assertRaises(ValueError): e._phase_result(phase_result('control_verify', control_ok=False), 'control_verify')


class OwnedWriteIntegrationTests(unittest.TestCase):
    def test_controlled_git_source_with_devnull_and_other_writes_blocked(self):
        root = os.environ.get('TEACHER_PAIR_TEST_OWNED_SCRATCH')
        if os.name != 'nt' or not root:
            self.skipTest('Requires Windows and an externally owned scratch directory')
        root = Path(root)
        self.assertTrue(root.is_absolute() and (root/'.agent-workflow-owner.json').is_file())
        repo, stage = root/'synthetic-repo', root/'synthetic-output'
        repo.mkdir(); stage.mkdir()
        flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        def git(*args):
            return subprocess.run(['git', *args], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, check=True, timeout=30, creationflags=flags)
        git('init', '--quiet', str(repo))
        (repo/'synthetic').mkdir()
        expected = b'{"synthetic":true}\n'
        (repo/'synthetic'/'ledger.json').write_bytes(expected)
        git('-C', str(repo), 'add', 'synthetic/ledger.json')
        git('-C', str(repo), '-c', 'user.name=fixture', '-c', 'user.email=fixture@invalid',
            'commit', '--quiet', '-m', 'synthetic')
        # Git marks its new object files read-only on Windows. This is an
        # exclusively owned synthetic repository; make its objects removable
        # before the external scratch owner performs normal cleanup.
        for path in (repo/'.git'/'objects').rglob('*'):
            if path.is_file():
                path.chmod(stat.S_IREAD | stat.S_IWRITE)
        from learning.task_student_anchor_collection_v1 import sources
        from learning.task_anchor_once_v1.storage import UnsafeWriteError
        original_open = os.open
        with mock.patch.object(sources, 'ANCHOR_COMMIT', 'HEAD'), \
             mock.patch.object(sources, 'ANCHOR_PREFIX', 'synthetic/'), \
             mock.patch.object(sources.old.fresh, 'REPO', repo):
            with self.assertRaisesRegex(RuntimeError, 'synthetic stop'):
                with e._owned_writes_with_subprocess_null(stage, 1024) as budget:
                    raw = sources._git_blob('HEAD', 'synthetic/ledger.json', lambda: None)
                    self.assertEqual(e._sha(raw), e._sha(expected))
                    with self.assertRaises(UnsafeWriteError):
                        os.open(str(stage/'blocked'), os.O_CREAT | os.O_WRONLY)
                    with self.assertRaises(UnsafeWriteError):
                        os.open(os.devnull, os.O_WRONLY)
                    with mock.patch.object(os, 'devnull', str(stage/'ordinary-file')):
                        with self.assertRaises(UnsafeWriteError):
                            os.open(os.devnull, os.O_RDWR)
                    (stage/'charged').write_bytes(b'123')
                    self.assertEqual(budget.used_bytes, 3)
                    raise RuntimeError('synthetic stop')
        self.assertIs(os.open, original_open)
        self.assertFalse((stage/'blocked').exists())


class CoordinatorTests(unittest.TestCase):
    def _simulate(self, *, fail_phase=None):
        binding = fake_binding()
        files, calls = {}, []
        head, digest = 'a'*40, 'b'*64
        def write(path, value):
            key = str(path)
            if key in files: raise FileExistsError('synthetic duplicate')
            raw = e.base._encoded(value); files[key] = raw
            return e._sha(raw)
        def read(path, **_kwargs): return files[str(path)]
        def launch(command, _cwd, _limits, _name):
            seq = int(command[-1]); unit = binding['execution_order'][seq]; phase = unit['phase']
            calls.append(phase)
            claim = e.base._document(files[str(e._claim_path(seq))])
            marker = dict(schema='teacher-pair-child-attempt-v1', commit=head, sequence=seq, retry_allowed=False)
            write(e.PRIVATE/('child-attempt-'+e._unit_name(seq)+'.local.json'), marker)
            result = phase_result(phase, control_ok=phase != fail_phase)
            receipt = dict(schema='teacher-pair-child-result-v1', commit=head, sequence=seq, unit=unit,
                           result=result, storage=dict(limit_bytes=claim['limit_bytes'],used_bytes=1,refused=0),
                           retry_allowed=False)
            write(e._child_result_path(seq), receipt)
            return {'synthetic':'job'}
        from learning.task_pair_fresh_multi_v1 import entry
        with mock.patch.object(e.os, 'name', 'nt'), \
             mock.patch.object(e.base, 'physical_available', return_value=8*1024**3), \
             mock.patch.object(e, '_guard'), mock.patch.object(e, '_disk_gate'), \
             mock.patch.object(e, '_runtime_binding', return_value=(head, binding, digest)), \
             mock.patch.object(e, '_write', side_effect=write), mock.patch.object(e, '_read', side_effect=read), \
             mock.patch.object(e, '_verify_outputs'), mock.patch.object(Path, 'mkdir'), \
             mock.patch.object(Path, 'exists', return_value=False), \
             mock.patch.object(entry, 'owner_marker', side_effect=lambda **kw:e.base._encoded(dict(
                 schema='task-pair-fresh-multi-owner-v1', registration_commit=kw['commit'],
                 binding_sha256=kw['binding_sha'], pid=kw['pid'], created_us=kw['created_us'],
                 deadline=kw['deadline']))), \
             mock.patch('psutil.Process', return_value=SimpleNamespace(create_time=lambda:1)), \
             mock.patch.object(e.supervisor, '_supervise_owned', side_effect=launch), \
             mock.patch.object(e.transfer, '_job_ok', return_value=True), redirect_stdout(io.StringIO()) as out:
            if fail_phase:
                with self.assertRaises(ValueError): e.run()
            else:
                e.run()
        return files, calls, out.getvalue()

    def test_control_audit_and_history_verification_precede_each_intervention(self):
        files, calls, output = self._simulate()
        self.assertEqual(calls, list(e.PHASES)*4)
        self.assertIn(str(e.PRIVATE/'closed.local.json'), files)
        self.assertNotIn('synthetic-student', output)
        self.assertNotIn('synthetic-teacher', output)

    def test_failed_control_verification_never_starts_intervention_or_seals(self):
        files, calls, _ = self._simulate(fail_phase='control_verify')
        self.assertEqual(calls, list(e.PHASES[:4]))
        self.assertNotIn('intervene', calls)
        self.assertNotIn(str(e.PRIVATE/'closed.local.json'), files)
        self.assertIn(str(e.PRIVATE/'failed.local.json'), files)

    def test_completed_consumer_requires_chain_and_detects_mutation(self):
        files, _calls, _output = self._simulate()
        binding, head, digest = fake_binding(), 'a'*40, 'b'*64
        def read(path, **_kwargs): return files[str(path)]
        with mock.patch.object(e, '_read', side_effect=read), \
             mock.patch.object(e, '_registered_binding', return_value=(binding, digest)), \
             mock.patch.object(e, '_verify_outputs'), \
             mock.patch.object(e.transfer, '_job_ok', return_value=True), \
             mock.patch.object(Path, 'exists', return_value=False):
            result = e.load_completed()
            self.assertTrue(result['technical_pass'])
            self.assertTrue(result['signal']['descriptive_signal'])
            path = str(e._child_result_path(0))
            altered = e.base._document(files[path])
            altered['storage']['used_bytes'] = 2
            files[path] = e.base._encoded(altered)
            with self.assertRaises(ValueError): e.load_completed()


if __name__ == '__main__': unittest.main()
