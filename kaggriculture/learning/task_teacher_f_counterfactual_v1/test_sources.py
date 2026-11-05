"""Synthetic F-source scope, stream and real screening-orchestration checks."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import FrozenInstanceError
import gzip
import hashlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from learning.task_student_collection_v1.test_sampling import trace, choice
from learning.task_student_continuation_v1 import compare, official_step
from learning.task_student_continuation_v1.compare import PrivateComparison
from learning.task_teacher_counterfactual_screen_v1 import screen as original_screen
from . import contract, sources


def completed_fixture(trace_sha='f'*64):
    """Public synthetic coordinates; no source file, replay or model is read."""
    manifests, rows, identities = {}, {}, {'games': {}}
    for index, (cohort, mapping) in enumerate(contract.COHORTS.items()):
        panel = SimpleNamespace(seeds=(100+index*2, 101+index*2),
            models=(('artifacts/synthetic/base.npz','a'*64),
                    ('artifacts/synthetic/control.npz','8'*64),
                    ('artifacts/synthetic/aux.npz','9'*64)),
            crop=('artifacts/synthetic/crop.npz','b'*64))
        manifests[cohort] = SimpleNamespace(panel=panel,
            opponent=lambda role:('bots/'+role+'.py', 'c'*64 if role=='market' else 'd'*64))
        for alias, role in mapping.items():
            for source in contract.SOURCES:
                if source['role'] != role: continue
                name = f"dev-{alias}-{source['opponent']}-{source['side']}-baseline"
                rows[(cohort,name)] = dict(game=name,policy='baseline',seed_alias=alias,
                    seed=panel.seeds[('S02','S03').index(alias)],side=source['side'],
                    opponent=source['opponent'],task_checkpoint_sha256='a'*64,
                    crop_checkpoint_sha256='b'*64,
                    rival_sha256='c'*64 if source['opponent']=='market' else 'd'*64,
                    replay_gzip_sha256='e'*64,trace_sha256=trace_sha)
                identities['games'][cohort+'/'+name] = {'receipt':'synthetic-'+cohort+'-'+name}
    return SimpleNamespace(manifests=manifests,rows=rows,identities=identities)


def context_fixture(trace_sha='f'*64):
    return sources.FContext(completed_fixture(trace_sha))


@contextmanager
def in_memory_source(zipped):
    @contextmanager
    def checked(*_args):
        with io.BytesIO(zipped) as stream: yield stream
    replay = dict(configuration={},info={'seed':100},
                  steps=[[{'action':{}},{'action':{}}] for _ in range(720)])
    with patch.object(sources.old,'_open_checked',checked), \
         patch.object(sources.old,'_read_replay',return_value=replay):
        yield


class SourceTests(unittest.TestCase):
    def test_fixed_eight_sources_and_immutable_full_identity(self):
        context = context_fixture()
        plans = context.private_source_plans()
        self.assertEqual([p.coordinate() for p in plans],list(contract.SOURCES))
        self.assertEqual({p.policy for p in plans},{'baseline'})
        self.assertEqual({p.role for p in plans},{'F01','F02','F03','F04'})
        self.assertNotIn('batch',plans[0].document())
        self.assertNotIn('seed',repr(plans[0]))
        document = plans[0].document(); document['seed'] = 999
        self.assertEqual(plans[0].seed,100)
        with self.assertRaises(FrozenInstanceError): plans[0].seed = 999
        for changed in (dict(contract.SOURCES[0],role='A05'),
                        dict(contract.SOURCES[0],policy='control'),
                        dict(contract.SOURCES[0],opponent='round2'),
                        dict(contract.SOURCES[0],side=True)):
            with self.subTest(changed=changed),self.assertRaises(ValueError): context.plan(changed)

    def test_context_authentication_uses_completed_chain_without_inventory_entrypoints(self):
        fake = completed_fixture()
        with patch.object(sources.completed,'load',return_value=fake), \
             patch.object(sources.FContext,'verify_assets',return_value=True), \
             patch.object(sources.old,'open_context',side_effect=AssertionError('forbidden old source')), \
             patch.object(sources.completed.fresh,'verify_binding',side_effect=AssertionError('forbidden inventory')), \
             patch.object(sources.completed.fresh,'load_verified_market_binding',side_effect=AssertionError('forbidden inventory')):
            value = sources.open_context(guard=lambda:None,expected_identities=fake.identities)
            self.assertEqual(len(value.private_source_plans()),8)
            with self.assertRaises(ValueError):
                sources.open_context(guard=lambda:None,expected_identities={'changed':True})
            value.close()
            with self.assertRaises(ValueError): value.private_source_plans()

    def test_full_compressed_trace_scans_and_one_open_replay_limit(self):
        zipped = gzip.compress(trace({1:[choice()],125:[choice()]}),mtime=0)
        context = context_fixture(hashlib.sha256(zipped).hexdigest())
        with in_memory_source(zipped):
            with sources.read_game(context,dict(contract.SOURCES[0]),guard=lambda:None) as game:
                with self.assertRaises(ValueError):
                    sources.read_game(context,dict(contract.SOURCES[1]),guard=lambda:None)
                accounting, windows = game.scan_candidates(guard=lambda:None)
                self.assertEqual(accounting['lines'],719)
                self.assertTrue(accounting['absolute_budget_checked'])
                self.assertEqual([len(w) for w in windows],[1,1,0,0,0,0])
                with self.assertRaises(ValueError): game.scan_candidates(guard=lambda:None)
            with self.assertRaises(ValueError): _ = game.replay
        self.assertIsNone(context._active)

    def test_source_hash_or_per_turn_budget_failure_stops_without_candidate(self):
        zipped = gzip.compress(trace({1:[choice()]}),mtime=0)
        with in_memory_source(zipped),self.assertRaises(ValueError):
            sources.read_game(context_fixture(),dict(contract.SOURCES[0]),guard=lambda:None)
        zipped = gzip.compress(trace({1:[choice()]*33}),mtime=0)
        context = context_fixture(hashlib.sha256(zipped).hexdigest())
        with in_memory_source(zipped):
            with sources.read_game(context,dict(contract.SOURCES[0]),guard=lambda:None) as game:
                with self.assertRaises(ValueError): game.scan_candidates(guard=lambda:None)

    def test_screen_source_uses_same_meter_snapshot_and_stops_at_first_official_hit(self):
        zipped = gzip.compress(trace({1:[choice()],3:[choice()],125:[choice()]}),mtime=0)
        context = context_fixture(hashlib.sha256(zipped).hexdigest())
        original_roles, original_slots = original_screen.ROLES, deepcopy(original_screen.SLOTS)
        module = sources.screening_module()
        self.assertIs(module,sources.screening_module())
        meter = module.Meter()
        result = PrivateComparison('different','synthetic',{'request_equal':False},
            {'farmer':['PASS'],'hands':[],'market':[]},{'farmer':['DIG'],'hands':[],'market':[]})
        effect = dict(factual_reproduced=True,request_changed=True,state_changed=True,
            scope='one_official_transition',complete_game=False,future_return_evaluated=False)
        with in_memory_source(zipped), \
             patch.object(module,'observation',return_value={}), \
             patch.object(module,'observe',return_value=SimpleNamespace(observation={},configuration={})), \
             patch.object(compare,'compare_turn',return_value=result), \
             patch.object(official_step,'compare_step',return_value=effect):
            coverage, candidate, plan = sources.screen_source(context,dict(contract.SOURCES[0]),lambda:None,meter=meter)
        self.assertEqual(coverage['source_trace_lines'],719)
        self.assertEqual(coverage['comparisons'],1)
        self.assertEqual(coverage['official_transitions_reserved'],2)
        self.assertEqual(coverage['meter'],meter.snapshot())
        self.assertEqual(coverage['status_by_window'][1][0],'unexamined_after_hit')
        self.assertEqual(candidate['source_identity_sha256'],plan.identity_sha256)
        self.assertEqual((candidate['role'],candidate['cohort'],candidate['alias']),('F01','C01','S02'))
        self.assertNotIn('batch',candidate)
        self.assertEqual((original_screen.ROLES,original_screen.SLOTS),(original_roles,original_slots))

    def test_empty_source_is_preserved_without_comparison_or_replacement(self):
        zipped = gzip.compress(trace(),mtime=0)
        context = context_fixture(hashlib.sha256(zipped).hexdigest())
        with in_memory_source(zipped), \
             patch.object(compare,'compare_turn',side_effect=AssertionError('no comparison')), \
             patch.object(official_step,'compare_step',side_effect=AssertionError('no official step')):
            coverage, candidate, plan = sources.screen_source(context,dict(contract.SOURCES[0]),lambda:None)
        self.assertIsNone(candidate)
        self.assertEqual(coverage['comparisons'],0)
        self.assertEqual(coverage['source_trace_lines'],719)
        self.assertEqual(plan.role,'F01')
        with self.assertRaises(ValueError):
            sources.screen_source(context,dict(contract.SOURCES[0]),lambda:None)


if __name__ == '__main__':
    unittest.main()
