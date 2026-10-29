import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from learning.task_teacher_merged_eval_v1 import games as merged
from learning.task_teacher_merged_eval_v1.evaluation import plans
from . import wrapper_verify as verify


class WrapperVerifyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root/'project'
        self.project.mkdir()
        (self.project/'artifacts').mkdir()
        self.stage = self.project/'artifacts'/'fresh-panel'
        self.stage.mkdir()
        self.scratch = self.root/'scratch'
        self.scratch.mkdir()
        self.root_patch = patch.object(merged.experiment, 'ROOT', self.project)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.cfg = dict(crop_checkpoint='models/crop.npz', crop_checkpoint_sha256='a'*64,
            merged_checkpoint='models/merged.npz', merged_checkpoint_sha256='b'*64,
            unweighted_checkpoint='models/unweighted.npz', unweighted_checkpoint_sha256='c'*64,
            game_limits={'controller_calls':719, 'task_forwards':700})

    def plan(self, policy='merged', side=0):
        return next(p for p in plans([101,202,303,404])
                    if p['seed_alias']=='S01' and p['opponent']=='round2'
                    and p['side']==side and p['policy']==policy)

    def make_wrapper(self, plan, cfg=None):
        cfg = self.cfg if cfg is None else cfg
        core = merged.core
        with merged._LOCK:
            old_root, old_identity = core.ROOT, core.task_identity
            try:
                core.ROOT, core.task_identity = self.project, merged.task_identity
                path = core.wrapper(self.stage, plan, cfg)
            finally:
                core.ROOT, core.task_identity = old_root, old_identity
        _, task_sha = merged.task_identity(self.stage, plan, cfg)
        return dict(plan, bot_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    task_checkpoint_sha256=task_sha)

    def check(self, item, cfg=None):
        return verify.verify_wrapper(self.stage, item, self.cfg if cfg is None else cfg, self.scratch)

    def assert_scratch_empty(self):
        self.assertEqual(list(self.scratch.iterdir()), [])

    def test_real_generator_matches_three_policies_and_both_seats(self):
        for policy in ('merged','unweighted','teacher_task'):
            for side in (0,1):
                with self.subTest(policy=policy, side=side):
                    item = self.make_wrapper(self.plan(policy,side))
                    self.assertEqual(self.check(item),item['bot_sha256'])
                    self.assert_scratch_empty()

    def test_tampered_wrapper_with_matching_self_reported_hash_is_rejected(self):
        item = self.make_wrapper(self.plan())
        path = self.stage/'agents'/(item['game']+'.py')
        path.write_bytes(b'def agent(observation, configuration): return {}\n')
        item['bot_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaises(ValueError) as raised:
            self.check(item)
        self.assertNotIn(str(self.stage),str(raised.exception))
        self.assert_scratch_empty()

    def test_self_reported_hash_and_task_digest_must_match(self):
        item = self.make_wrapper(self.plan())
        with self.assertRaises(ValueError): self.check({**item, 'bot_sha256':'d'*64})
        with self.assertRaises(ValueError): self.check({**item, 'task_checkpoint_sha256':'d'*64})
        self.assert_scratch_empty()

    def test_teacher_task_requires_explicit_null_digest(self):
        item = self.make_wrapper(self.plan('teacher_task'))
        del item['task_checkpoint_sha256']
        with self.assertRaises(ValueError): self.check(item)
        self.assert_scratch_empty()

    def test_limit_insertion_order_must_match_the_registered_generator_input(self):
        item = self.make_wrapper(self.plan())
        changed = {**self.cfg, 'game_limits':dict(reversed(list(self.cfg['game_limits'].items())))}
        with self.assertRaises(ValueError): self.check(item,changed)
        item2 = self.make_wrapper(self.plan('merged',1),changed)
        self.assertEqual(self.check(item2,changed),item2['bot_sha256'])
        self.assert_scratch_empty()

    def test_rejects_path_escape_invalid_identity_and_nonowned_scratch(self):
        item = self.make_wrapper(self.plan())
        for key,value in (('merged_checkpoint','../outside.npz'),
                          ('crop_checkpoint','../outside.npz'),
                          ('crop_checkpoint','X'+':/outside.npz')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check(item,{**self.cfg,key:value})
        for bad in ({**item,'game':'../other'}, {**item,'side':False},
                    {**item,'policy':'unknown'}, {**item,'seed_alias':'S05'}):
            with self.assertRaises(ValueError): self.check(bad)
        with self.assertRaises(ValueError):
            verify.verify_wrapper(self.stage,item,self.cfg,self.project)
        self.assert_scratch_empty()

    def test_generator_exception_restores_globals_and_cleans_sandbox(self):
        item = self.make_wrapper(self.plan())
        original_root, original_identity = merged.core.ROOT, merged.core.task_identity
        with patch.object(merged.core, 'wrapper', side_effect=RuntimeError('sensitive details')):
            with self.assertRaises(ValueError) as raised:
                self.check(item)
        self.assertIs(merged.core.ROOT,original_root)
        self.assertIs(merged.core.task_identity,original_identity)
        self.assertNotIn('sensitive',str(raised.exception))
        self.assert_scratch_empty()

    def test_generator_system_exit_and_interrupt_do_not_expose_text(self):
        item = self.make_wrapper(self.plan())
        original_root, original_identity = merged.core.ROOT, merged.core.task_identity
        for failure, expected in ((SystemExit('sensitive details'), ValueError),
                                  (KeyboardInterrupt('sensitive details'), KeyboardInterrupt)):
            with self.subTest(kind=type(failure).__name__), patch.object(merged.core, 'wrapper', side_effect=failure):
                with self.assertRaises(expected) as raised:
                    self.check(item)
                self.assertNotIn('sensitive', str(raised.exception))
                self.assertIs(merged.core.ROOT, original_root)
                self.assertIs(merged.core.task_identity, original_identity)
                self.assert_scratch_empty()

    def test_optimized_interpreter_fails_before_touching_inputs(self):
        item = self.make_wrapper(self.plan())
        source = ('import json, sys\nfrom pathlib import Path\n'
                  'from learning.panel_eval_v2.wrapper_verify import verify_wrapper\n'
                  'from learning.task_teacher_merged_eval_v1 import games\n'
                  'games.experiment.ROOT=Path(sys.argv[1])\n'
                  'try:\n verify_wrapper(sys.argv[2],json.loads(sys.argv[4]),json.loads(sys.argv[5]),sys.argv[3])\n'
                  'except ValueError: print("refused")\n')
        result = subprocess.run([sys.executable,'-O','-B','-c',source,str(self.project),str(self.stage),
                                 str(self.scratch),json.dumps(item),json.dumps(self.cfg)],
                                capture_output=True,text=True,check=False)
        self.assertEqual(result.returncode,0)
        self.assertEqual(result.stdout.strip(),'refused')
        self.assertEqual(result.stderr,'')


if __name__ == '__main__':
    unittest.main()
