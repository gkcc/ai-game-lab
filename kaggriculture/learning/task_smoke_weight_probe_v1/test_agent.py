"""Real controller class with exclusively synthetic checkpoints and inputs."""
import hashlib
import json
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from decision.api import CROPS, PRODUCTS
from learning.bc_crop import agent as crop_agent
from learning.bc_crop import features as crop_features
from learning.bc_crop.audit import stable
from learning.task_bc_v1.model import DIMENSIONS
from learning.task_choice_v1.agent import TaskAgent
from learning.task_choice_v1 import features as task_features
from learning.task_dagger_once_v1.policy import BoundedAgent, LIMITS
from . import core
from .test_core import observation


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SyntheticAgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.crop, self.task = root/'crop.npz', root/'task.npz'
        crop_names = ['synthetic_crop_feature']
        crop_meta = {'schema_version': crop_features.SCHEMA_VERSION, 'crops': list(crop_features.CROPS),
                     'teacher_sha256': crop_agent.TEACHER_SHA,
                     'schema_sha256': crop_features.schema_hash(crop_names), 'features': crop_names}
        np.savez(self.crop, metadata=np.asarray(json.dumps(crop_meta)), mean=np.zeros(1), scale=np.ones(1),
                 w0=np.zeros((1, 128)), b0=np.zeros(128), w1=np.zeros((128, 64)), b1=np.zeros(64),
                 w2=np.zeros((64, 5)), b2=np.zeros(5))
        self.shared_names = ['synthetic_shared_'+str(i) for i in range(275)]
        self.local_names = ['synthetic_local_'+str(i) for i in range(98)]
        self.schema = task_features.schema_hash(self.local_names, self.shared_names)
        task_meta = {'schema_version': task_features.SCHEMA_VERSION, 'ops': list(task_features.OPS),
                     'dimensions': list(DIMENSIONS), 'shared_feature_names': self.shared_names,
                     'feature_names': self.local_names, 'schema_sha256': self.schema}
        w0, w1, w2 = np.zeros((373, 128)), np.zeros((128, 64)), np.zeros((64, 1))
        w0[275, 0], w1[0, 0], w2[0, 0] = 1.0, 1.0, 1.0
        np.savez(self.task, metadata=np.asarray(json.dumps(task_meta)), mean=np.zeros(373), scale=np.ones(373),
                 w0=w0, b0=np.zeros(128), w1=w1, b1=np.zeros(64), w2=w2, b2=np.asarray([0.25]))
        self.task_sha, self.crop_sha = sha(self.task), sha(self.crop)

    def make_agent(self):
        return BoundedAgent(self.crop, self.crop_sha, self.task, self.task_sha, limits=LIMITS)

    def test_separate_real_bounded_agents_restore_prior_state_without_tasks(self):
        agents = [self.make_agent() for _ in range(3)]
        core._prepare_three(agents, self.task_sha)
        captures = [core._Capture(agent.network) for agent in agents]
        obs = observation(718, 0)
        obs['day'], obs['hour'] = 30, 22
        cfg = {'episodeSteps': 720, 'boardSize': 1, 'turnsPerDay': 24,
               'shedCapacity': 80, 'townShopSellInterval': 24, 'maxMarketOrdersPerTurn': 8,
               'marketParams': {}}
        view, config = core.legal_view([{'observation': obs}, {'observation': observation(718, 1)}], 0, cfg)
        prior = {name: 0 for name in core.COUNT_KEYS}
        prior['controller_calls'] = 718
        results = [core._invoke(agent, capture, view, config, 718, 0, prior)
                   for agent, capture in zip(agents, captures)]
        self.assertEqual([request for request, _ in results],
                         [{'farmer': ['PASS'], 'hands': [], 'market': []}] * 3)
        self.assertTrue(all(not capture.calls for capture in captures))
        self.assertTrue(all(trace['model_counts']['controller_calls'] == 719 for _, trace in results))

    def test_actual_task_network_flip_uses_independent_synthetic_weights(self):
        agents = [self.make_agent() for _ in range(3)]
        original, derived = core._prepare_three(agents, self.task_sha)
        self.assertNotEqual(original, derived)
        captures = [core._Capture(agent.network) for agent in agents]
        candidate_a, candidate_b = [0.0] * 98, [0.0] * 98
        candidate_a[0], candidate_b[0] = 1.0, 2.0
        inputs = {'schema_version': task_features.SCHEMA_VERSION, 'schema_sha256': self.schema,
                  'shared_feature_names': self.shared_names, 'feature_names': self.local_names,
                  'shared_x': [0.0] * 275,
                  'candidates': [{'id': 'a', 'x': candidate_a}, {'id': 'b', 'x': candidate_b}]}
        choices = [capture(inputs) for capture in captures]
        self.assertEqual(choices, ['b', 'b', 'a'])
        self.assertTrue(np.allclose(captures[2].calls[0][2], -captures[0].calls[0][2],
                                    rtol=core.LOGIT_RTOL, atol=core.LOGIT_ATOL))
        self.assertEqual(agents[2].task_sha, derived)
        self.assertNotEqual(agents[2].task_sha, self.task_sha)

    def test_real_actor_bindings_accept_fresh_and_reject_deepcopy(self):
        fresh = [self.make_agent() for _ in range(3)]
        for agent in fresh:
            core._verify_agent_bindings(agent)
        clone = deepcopy(fresh[0])
        with self.assertRaises(core.ProbeError):
            core._verify_agent_bindings(clone)
        with self.assertRaises(core.ProbeError):
            core._prepare_three([fresh[0], fresh[1], clone], self.task_sha)

    def test_ast_callbacks_bound_to_other_actor_are_rejected(self):
        for name in ('_task_select', '_crop_select', '_task_finish'):
            agent, other = self.make_agent(), self.make_agent()
            agent.controller.controller.__globals__[name] = other.controller.controller.__globals__[name]
            with self.subTest(callback=name), self.assertRaises(core.ProbeError):
                core._verify_agent_bindings(agent)
        for name in ('_bc_select', '_bc_finish'):
            agent, other = self.make_agent(), self.make_agent()
            agent.controller.crop.controller.__globals__[name] = other.controller.crop.controller.__globals__[name]
            with self.subTest(callback=name), self.assertRaises(core.ProbeError):
                core._verify_agent_bindings(agent)

    def test_real_task_and_crop_closures_from_other_actor_are_rejected(self):
        agent, other = self.make_agent(), self.make_agent()
        agent.controller.task_selector = other.controller.task_selector
        with self.assertRaises(core.ProbeError):
            core._verify_agent_bindings(agent)
        agent, other = self.make_agent(), self.make_agent()
        original_crop = core._closure(other.controller.crop.network.logits)['original_crop']
        def bind_crop(self, original_crop):
            def crop(x, names):
                self.budget.forward('crop', 1)
                return original_crop(x, names)
            return crop
        agent.controller.crop.network.logits = bind_crop(agent, original_crop)
        with self.assertRaises(core.ProbeError):
            core._verify_agent_bindings(agent)

    def test_synthetic_multicandidate_controller_reaches_trace_and_request(self):
        # No environment is created: these dictionaries are synthetic public
        # observations, and the reference selector has no neural checkpoint.
        farms = [dict(money=1000, tiles=[[None, None], [None, None]], farmer=[0, 0], hands=[],
                      unlocked_quadrants=[0], hires_today=0) for _ in range(2)]
        obs = dict(player=0, step=24, day=1, hour=0, farms=farms,
                   private=dict(shed={}, seeds={c: 1 for c in CROPS}, inventories=[{}]),
                   market=dict(inventory={p: 10000 for p in PRODUCTS},
                               prices={p: 50 for p in PRODUCTS}, params={}),
                   town=dict(unlocked_shops=[]))
        cfg = dict(episodeSteps=720, boardSize=2, turnsPerDay=24, shedCapacity=80,
                   townShopSellInterval=24, maxMarketOrdersPerTurn=8, farmHandCostMult=1,
                   marketParams={})
        seen_inputs, seen_crop_names = [], []
        def first(inputs):
            seen_inputs.append(deepcopy(inputs))
            return inputs['candidates'][0]['id']
        reference = TaskAgent(first, 'teacher')
        original_crop = reference.controller.__globals__['_crop_select']
        def capture_crop(context):
            _, names = crop_features.encode(context)
            seen_crop_names[:] = names
            return original_crop(context)
        reference.controller.__globals__['_crop_select'] = capture_crop
        reference(obs, cfg)
        self.assertTrue(seen_inputs and len(seen_inputs[0]['candidates']) > 1 and seen_crop_names)
        item = seen_inputs[0]
        shared, local = item['shared_feature_names'], item['feature_names']
        self.assertEqual((len(shared), len(local)), (275, 98))
        crop_meta = dict(schema_version=crop_features.SCHEMA_VERSION, crops=list(crop_features.CROPS),
                         teacher_sha256=crop_agent.TEACHER_SHA,
                         schema_sha256=crop_features.schema_hash(seen_crop_names), features=seen_crop_names)
        np.savez(self.crop, metadata=np.asarray(json.dumps(crop_meta)), mean=np.zeros(len(seen_crop_names)),
                 scale=np.ones(len(seen_crop_names)), w0=np.zeros((len(seen_crop_names), 128)), b0=np.zeros(128),
                 w1=np.zeros((128, 64)), b1=np.zeros(64), w2=np.zeros((64, 5)), b2=np.zeros(5))
        task_meta = dict(schema_version=task_features.SCHEMA_VERSION, ops=list(task_features.OPS),
                         dimensions=list(DIMENSIONS), shared_feature_names=shared, feature_names=local,
                         schema_sha256=task_features.schema_hash(local, shared))
        w0, w1, w2 = np.zeros((373, 128)), np.zeros((128, 64)), np.zeros((64, 1))
        w0[275 + local.index('target.x'), 0], w1[0, 0], w2[0, 0] = 1, 1, 1
        np.savez(self.task, metadata=np.asarray(json.dumps(task_meta)), mean=np.zeros(373), scale=np.ones(373),
                 w0=w0, b0=np.zeros(128), w1=w1, b1=np.zeros(64), w2=w2, b2=np.zeros(1))
        self.task_sha, self.crop_sha = sha(self.task), sha(self.crop)
        agents = [self.make_agent() for _ in range(3)]
        core._prepare_three(agents, self.task_sha)
        captures = [core._Capture(agent.network) for agent in agents]
        prior = {name: 0 for name in core.COUNT_KEYS}
        prior['controller_calls'] = 24
        view, config = core.legal_view([{'observation': obs}, {'observation': dict(obs, player=1)}], 0, cfg)
        outcomes = [core._invoke(agent, capture, view, config, 24, 0, prior)
                    for agent, capture in zip(agents, captures)]
        self.assertEqual(outcomes[0][0], outcomes[1][0])
        self.assertEqual(stable(outcomes[0][1]), stable(outcomes[1][1]))
        self.assertNotEqual(outcomes[0][0], outcomes[2][0])
        self.assertEqual(captures[0].calls[0][0], captures[2].calls[0][0])
        self.assertNotEqual(captures[0].calls[0][1], captures[2].calls[0][1])
        self.assertTrue(np.allclose(captures[2].calls[0][2], -captures[0].calls[0][2],
                                    rtol=core.LOGIT_RTOL, atol=core.LOGIT_ATOL))
        for _request, trace in outcomes:
            self.assertEqual(len(trace['records']), trace['turn_model_counts']['task_forwards'])
            self.assertEqual(len(trace['crop_records']), trace['turn_model_counts']['crop_forwards'])


if __name__ == '__main__':
    unittest.main()
