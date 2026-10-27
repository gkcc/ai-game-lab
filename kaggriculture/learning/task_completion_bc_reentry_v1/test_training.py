import sys
import types
import unittest
from unittest.mock import patch
import numpy as np

from learning.task_completion_bc_reentry_v1.training import _parity_forward, fit
from learning.task_completion_parity_v1.test_compare import group
from learning.task_choice_v1.features import schema_hash


class ReentryTest(unittest.TestCase):
    def test_precision_scope(self):
        state = {'precision': 'default', 'operations': []}
        class Context:
            def __enter__(self): state['precision'] = 'highest'
            def __exit__(self, *args): state['precision'] = 'default'
        class Array:
            def __matmul__(self, other):
                state['operations'].append(state['precision']); return self
            def __add__(self, other): return self
            def block_until_ready(self): state['operations'].append(state['precision'])
        fake = types.SimpleNamespace(default_matmul_precision=lambda p: Context() if p == 'highest' else self.fail(),
            device_put=lambda x, d: Array(), nn=types.SimpleNamespace(relu=lambda x: x))
        params = {f'{p}{i}': None for i in range(3) for p in ('w', 'b')}
        _parity_forward(fake, None, params, None)
        self.assertEqual(state['operations'], ['highest'] * 4)
        self.assertEqual(state['precision'], 'default')

    def test_update_progress_and_default_precision(self):
        class Device:
            platform, device_kind = 'gpu', 'synthetic'
            def __str__(self): return 'synthetic'
        device = Device()
        def jit(_):
            def fake_step(*args):
                self.assertEqual(precision['value'], 'default')
                raise RuntimeError('synthetic stop before gradient')
            return fake_step
        precision = {'value': 'default'}
        jax = types.ModuleType('jax')
        jax.devices = lambda _: [device]
        jax.device_put = lambda x, _: x
        jax.jit = jit
        jnp = types.ModuleType('jax.numpy')
        jax.numpy = jnp
        optax = types.ModuleType('optax')
        optax.clip_by_global_norm = lambda x: None
        optax.adam = lambda x: None
        optax.chain = lambda *args: types.SimpleNamespace(init=lambda params: None)
        names = [f'f{i}' for i in range(98)]
        shared = [f's{i}' for i in range(275)]
        schema = dict(feature_names=names, shared_feature_names=shared, schema_sha256=schema_hash(names, shared))
        events = []
        with patch.dict(sys.modules, {'jax': jax, 'jax.numpy': jnp, 'optax': optax}):
            with self.assertRaisesRegex(RuntimeError, 'synthetic stop'):
                fit([group()], schema, 'unused.npz', 1, 'synthetic', 'synthetic', events.append)
        self.assertEqual(events[-1], dict(phase='update_started', updates_started=1, updates_completed=0,
            parity_started=0, parity_completed=0, checkpoint_written=False, device_validated=True))


if __name__ == '__main__': unittest.main()
