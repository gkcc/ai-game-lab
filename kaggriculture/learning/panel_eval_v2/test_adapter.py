import unittest
from pathlib import Path

from . import adapter


class FakeContext:
    ROOT = Path(adapter.campaign.ROOT)
    OUT = ROOT/'artifacts'/'synthetic-panel'

    def ready(self, *_): return 'ready'
    def verify(self, *_): return 'verify'
    def verify_commit(self, *_): return 'commit'
    def deadline_check(self): return None
    def load_seeds(self, *_): return []
    def write_new(self, *_): return None
    def trusted_out(self, *_): return self.OUT


class AdapterTests(unittest.TestCase):
    def test_bindings_apply_and_restore_after_exception(self):
        context = FakeContext()
        before = [(module, name, getattr(module, name)) for module, names in adapter._TARGETS for name in names]
        old_experiment = adapter.games.experiment
        old_memory = adapter.campaign.memory_ready
        hook = lambda *_: True
        with self.assertRaisesRegex(RuntimeError, 'synthetic'):
            with adapter.bind_context(context, memory_ready=hook):
                self.assertIs(adapter.games.experiment, context)
                self.assertEqual(adapter.campaign.ready(context.OUT), 'ready')
                self.assertEqual(adapter.independent_method.verify_commit(context.OUT), 'commit')
                self.assertIs(adapter.campaign.memory_ready, hook)
                raise RuntimeError('synthetic')
        self.assertIs(adapter.games.experiment, old_experiment)
        self.assertIs(adapter.campaign.memory_ready, old_memory)
        for module, name, original in before:
            self.assertIs(getattr(module, name), original)

    def test_invalid_context_or_hook_fails_before_binding(self):
        before = adapter.games.experiment
        with self.assertRaises(ValueError):
            with adapter.bind_context(FakeContext(), memory_ready=False): pass
        with self.assertRaises(ValueError):
            with adapter.bind_context(object()): pass
        self.assertIs(adapter.games.experiment, before)


if __name__ == '__main__':
    unittest.main()
