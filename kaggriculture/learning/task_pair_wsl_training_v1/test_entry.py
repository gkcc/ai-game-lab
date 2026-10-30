"""Synthetic entry checks; no real data, checkpoint, GPU or temporary files."""

from contextlib import nullcontext, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


os.environ['JAX_PLATFORMS'] = 'cpu'
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from learning.task_pair_wsl_training_v1 import entry


def synthetic_limits():
    return {'work_seconds': 1000, 'guard_with_cleanup_seconds': 1010,
            'load_seconds': 200, 'fit_seconds': 300, 'evaluation_seconds': 400,
            'gate_seconds': 5, 'term_seconds': 2, 'cleanup_seconds': 5, 'outer_total_seconds': 1020,
            'memory_sample_every': 256, 'minimum_initial_available_bytes': 4 * 1024**3,
            'minimum_sampled_available_bytes': 2 * 1024**3}


def synthetic_registration():
    return {'schema': entry.SCHEMA, 'runtime': dict(entry.RUNTIME), 'work': dict(entry.WORK),
            'evaluation': dict(entry.EVALUATION), 'limits': synthetic_limits(),
            'source_sha256': {'kaggriculture/example.py': 'a' * 64}}


def synthetic_runtime():
    root = Path.cwd().resolve()
    venv = root / 'synthetic-venv'
    base = root / 'synthetic-base'
    site = venv / 'lib' / 'site-packages'
    return {'venv_executable': str(venv / 'bin' / 'python'),
            'interpreter': str(base / 'bin' / 'python'), 'venv_prefix': str(venv),
            'base_prefix': str(base), 'search_path': [str(base / 'lib'), str(site)],
            'allowed_origins': [str(site)]}


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode('utf-8')


def synthetic_metrics():
    def supported(value):
        return {'support': 'supported', 'value': value}
    return {arm: {source: {subset: {
        'plant_same_target_unit': supported(0.52 if arm == 'auxiliary' else 0.5),
        'full_top1': supported(0.5), 'full_target_operation': supported(0.5),
    } for subset in entry.EVALUATION['subsets']} for source in entry.EVALUATION['sources']}
            for arm in entry.EVALUATION['arms']}


def synthetic_guards(extra, initializers=None):
    smoke = SimpleNamespace(bind=Mock(return_value=Path.cwd()), BOOTSTRAP_SHA='e' * 64,
                            NEW_INITIALIZERS={})
    historical = SimpleNamespace(INITIALIZERS=initializers or {})
    def module(relative, digest, label):
        if relative == entry.SMOKE_REL: return smoke
        if relative == entry.BOOTSTRAP_REL: return historical
        return extra
    sources = {entry.SMOKE_REL: entry.SMOKE_SHA, entry.EXTRA_REL: 'a' * 64,
               entry.BOOTSTRAP_REL: smoke.BOOTSTRAP_SHA}
    return smoke, module, sources


class EntryTests(unittest.TestCase):
    def test_missing_registration_precedes_private_or_project_access(self):
        output = io.StringIO()
        args = ['--registration-commit', 'a' * 40, '--game-root', 'synthetic-unopened-root',
                '--private-contract', 'synthetic-unopened-private',
                '--expected-private-contract-sha256', 'b' * 64, '--receipt-token', 'c' * 32]
        with patch.object(entry, 'registration_gate', side_effect=ValueError('synthetic-private-marker')), \
             patch.object(entry, 'verify_registered_sources') as sources, \
             patch.object(entry, 'prebind') as prebind, \
             patch.object(entry, 'finish_bootstrap') as finish, \
             patch.object(entry, '_private_root') as private, redirect_stdout(output):
            code = entry.main(args)
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(output.getvalue()),
                         {'passed': False, 'stage': 'registration', 'error_type': 'ValueError'})
        self.assertNotIn('synthetic-private-marker', output.getvalue())
        sources.assert_not_called()
        prebind.assert_not_called()
        finish.assert_not_called()
        private.assert_not_called()

    def test_base_exception_and_generator_exit_never_expose_diagnostics(self):
        args = ['--registration-commit', 'a' * 40, '--game-root', 'synthetic-unopened',
                '--private-contract', 'synthetic-unopened',
                '--expected-private-contract-sha256', 'b' * 64, '--receipt-token', 'c' * 32]
        for kind in (BaseException, GeneratorExit):
            output = io.StringIO()
            with self.subTest(kind=kind.__name__), \
                 patch.object(entry, 'registration_gate', side_effect=kind('synthetic-private-marker')), \
                 redirect_stdout(output):
                code = entry.main(args)
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(output.getvalue()),
                {'passed': False, 'stage': 'registration', 'error_type': 'OtherError'})
            self.assertNotIn('synthetic-private-marker', output.getvalue())

    def test_registration_and_private_digest_are_strict(self):
        registration = synthetic_registration()
        self.assertEqual(entry.parse_registration(encode(registration)), registration)
        for mutate in ({'work': dict(entry.WORK, updates=63)},
                       {'runtime': dict(entry.RUNTIME, platform='win32')},
                       {'evaluation': dict(entry.EVALUATION, primary_each_source_minimum=-0.03)},
                       {'limits': dict(synthetic_limits(), memory_sample_every=True)},
                       {'limits': dict(synthetic_limits(), gate_seconds=1001)},
                       {'limits': dict(synthetic_limits(), term_seconds=5)},
                       {'limits': dict(synthetic_limits(), cleanup_seconds=9)},
                       {'limits': dict(synthetic_limits(), outer_total_seconds=1019)},
                       {'source_sha256': {'../outside.py': 'a' * 64}}):
            with self.subTest(keys=tuple(mutate)), self.assertRaises(ValueError):
                entry.parse_registration(encode(registration | mutate))
        private = {'schema': entry.PRIVATE_SCHEMA, 'history_receipt_sha256': 'a' * 64,
                   'target_device_kind': 'synthetic target', 'batch_rng_seed': 3,
                   'runtime_identity': synthetic_runtime()}
        raw = encode(private)
        fixed = hashlib.sha256(raw).hexdigest()
        self.assertEqual(entry.parse_private_contract(raw, fixed), private)
        with self.assertRaises(ValueError):
            entry.parse_private_contract(raw + b' ', fixed)
        with self.assertRaises(ValueError):
            entry.parse_private_contract(raw, 'b' * 64)

    def test_import_guard_requires_package_and_module_coverage_before_imports(self):
        extra = SimpleNamespace(_PACKAGES=entry.REQUIRED_PACKAGES, TRAIN_MODULES=frozenset(),
                                verify_extra_sources=Mock())
        smoke, module, sources = synthetic_guards(extra)
        with patch.object(entry, '_module', side_effect=module), \
             patch.object(entry.importlib, 'import_module') as imported, self.assertRaises(ValueError):
            entry.bootstrap(sources)
        extra.verify_extra_sources.assert_not_called()
        imported.assert_not_called()

    def test_missing_legacy_source_is_rejected_before_project_import(self):
        required = frozenset(f'learning.{name}.{module}' for name, modules in entry.REQUIRED_PACKAGES.items()
                             for module in modules) - entry.SNAPSHOT_ONLY_MODULES
        legacy = 'kaggriculture/learning/task_pair_training_v1/targets.py'
        extra = SimpleNamespace(_PACKAGES=entry.REQUIRED_PACKAGES, TRAIN_MODULES=required,
            EXTRA_SOURCE_PATHS=frozenset(), PREIMPORT_SOURCE_PATHS=frozenset((legacy,)),
            verify_preimport_sources=Mock(), verify_extra_sources=Mock())
        smoke, module, sources = synthetic_guards(extra)
        with patch.object(entry, '_module', side_effect=module), \
             patch.object(entry.importlib, 'import_module') as imported, self.assertRaises(ValueError):
            entry.prebind(sources)
        extra.verify_extra_sources.assert_not_called()
        extra.verify_preimport_sources.assert_not_called()
        smoke.bind.assert_not_called()
        imported.assert_not_called()

    def test_bad_preimport_digest_stops_before_project_import(self):
        required = frozenset(f'learning.{name}.{module}' for name, modules in entry.REQUIRED_PACKAGES.items()
                             for module in modules) - entry.SNAPSHOT_ONLY_MODULES
        legacy = 'kaggriculture/learning/task_pair_training_v1/targets.py'
        extra = SimpleNamespace(_PACKAGES=entry.REQUIRED_PACKAGES, TRAIN_MODULES=required,
            EXTRA_SOURCE_PATHS=frozenset(), PREIMPORT_SOURCE_PATHS=frozenset((legacy,)),
            verify_preimport_sources=Mock(side_effect=ValueError('Source digest differs')),
            verify_extra_sources=Mock())
        smoke, module, sources = synthetic_guards(extra)
        sources[legacy] = 'b' * 64
        with patch.object(entry, '_module', side_effect=module), \
             patch.object(entry.importlib, 'import_module') as imported, self.assertRaises(ValueError):
            entry.prebind(sources)
        extra.verify_preimport_sources.assert_called_once_with(smoke, entry.GAME_SOURCE, {legacy: 'b' * 64})
        extra.verify_extra_sources.assert_not_called()
        smoke.bind.assert_not_called()
        imported.assert_not_called()

    def test_missing_initializer_is_rejected_before_binding_decision(self):
        extra = SimpleNamespace()
        smoke, module, sources = synthetic_guards(extra, {'decision/__init__.py': 'f' * 64})
        with patch.object(entry, '_module', side_effect=module), \
             patch.object(entry.importlib, 'import_module') as imported, self.assertRaises(ValueError):
            entry.prebind(sources)
        smoke.bind.assert_not_called()
        imported.assert_not_called()

    def test_runtime_paths_are_bound_and_ordered_before_dependency_import(self):
        runtime = synthetic_runtime()
        original_resolve = Path.resolve
        def resolve(path, *, strict=False):
            if str(path) == runtime['venv_executable']:
                return Path(runtime['interpreter'])  # Synthetic venv symlink, no file is created.
            return original_resolve(path, strict=strict)
        with patch.object(sys, 'executable', runtime['venv_executable']), \
             patch.object(sys, 'prefix', runtime['venv_prefix']), \
             patch.object(sys, 'base_prefix', runtime['base_prefix']), \
             patch.object(sys, 'path', runtime['search_path'].copy()), \
             patch.object(Path, 'resolve', resolve), \
             patch.object(Path, 'is_dir', return_value=True):
            self.assertEqual(entry._runtime_identity(runtime), (Path(runtime['allowed_origins'][0]),))
            self.assertEqual(entry.child_interpreter({'runtime_identity': runtime}), runtime['venv_executable'])
            with patch.object(sys, 'executable', runtime['interpreter']), self.assertRaises(ValueError):
                entry.child_interpreter({'runtime_identity': runtime})  # Same target, wrong argv[0].
            sys.path.reverse()
            with self.assertRaises(ValueError):
                entry._runtime_identity(runtime)

    def test_shadow_specs_and_preloaded_shadow_fail_before_versions(self):
        root = Path(synthetic_runtime()['allowed_origins'][0])
        outside = Path(synthetic_runtime()['base_prefix']) / 'shadow'
        def spec(name, parent):
            return SimpleNamespace(origin=str(parent / name / '__init__.py'),
                                   submodule_search_locations=[str(parent / name)])
        def direct(name, path):
            return spec(name, root) if name in entry.CORE_DEPENDENCIES else None
        def shadow(name):
            return spec(name, outside if name == 'numpy' else root) if name in entry.CORE_DEPENDENCIES else None
        with self.assertRaises(ValueError):
            entry._candidate_specs((root,), loaded={}, path_finder=direct, finder=shadow)

        # A listed optional dependency must be checked even if both finders
        # agree on its unapproved location; absence itself remains allowed.
        def optional_shadow(name):
            if name == 'scipy':
                return spec(name, outside)
            return spec(name, root) if name in entry.CORE_DEPENDENCIES else None
        with self.assertRaises(ValueError):
            entry._candidate_specs((root,), loaded={},
                path_finder=lambda name, path: optional_shadow(name), finder=optional_shadow)

        # Every location of a discoverable optional namespace must be allowed.
        def mixed_namespace(name):
            if name == 'nvidia':
                return SimpleNamespace(origin=None, submodule_search_locations=[
                    str(root / name), str(outside / name)])
            return spec(name, root) if name in entry.CORE_DEPENDENCIES else None
        with self.assertRaises(ValueError):
            entry._candidate_specs((root,), loaded={},
                path_finder=lambda name, path: mixed_namespace(name), finder=mixed_namespace)
        entry._candidate_specs((root,), loaded={}, path_finder=direct,
                               finder=lambda name: direct(name, []))
        finder = Mock()
        with self.assertRaises(ValueError):
            entry._candidate_specs((root,), loaded={'numpy.linalg': object()}, path_finder=finder, finder=finder)
        finder.assert_not_called()
        with patch.object(entry, '_runtime_identity', return_value=(root,)), \
             patch.object(entry, '_loaded_dependencies'), \
             patch.object(entry, '_candidate_specs', side_effect=ValueError), \
             patch.object(entry, '_runtime_versions') as versions, self.assertRaises(ValueError):
            entry._runtime_pre({'runtime_identity': synthetic_runtime()})
        versions.assert_not_called()

    def test_namespace_and_generated_native_children_require_bound_origins(self):
        root = Path(synthetic_runtime()['allowed_origins'][0])
        outside = Path(synthetic_runtime()['base_prefix']) / 'shadow'
        loader = object()
        loc = str(root / 'jax_plugins')
        namespace_spec = SimpleNamespace(origin=None, submodule_search_locations=[loc], loader=loader)
        namespace = SimpleNamespace(__spec__=namespace_spec, __loader__=loader, __file__=None,
                                    __path__=[loc])
        entry._loaded_dependencies((root,), loaded={'jax_plugins': namespace})
        namespace_spec.submodule_search_locations.append(str(outside))
        namespace.__path__.append(str(outside))
        with self.assertRaises(ValueError):
            entry._loaded_dependencies((root,), loaded={'jax_plugins': namespace})
        native = str(root / 'jaxlib' / ('extension' + entry.importlib.machinery.EXTENSION_SUFFIXES[-1]))
        parent_spec = SimpleNamespace(origin=native, submodule_search_locations=None, loader=loader)
        child = SimpleNamespace(__spec__=None, __file__=None, __path__=None)
        parent = SimpleNamespace(__spec__=parent_spec, __loader__=loader, __file__=native, config=child)
        loaded = {'jaxlib.xla_extension': parent, 'jaxlib.xla_extension.config': child}
        entry._loaded_dependencies((root,), loaded=loaded)
        parent.config = object()
        with self.assertRaises(ValueError):
            entry._loaded_dependencies((root,), loaded=loaded)

    def test_memory_samples_and_sampled_limits_fail_closed(self):
        good = b'MemTotal: 9999999 kB\nMemAvailable: 8388608 kB\n'
        self.assertEqual(entry._mem_available(good), 8 * 1024**3)
        for raw in (b'', b'MemAvailable: 5 MB\n', b'MemAvailable: 0 kB\n',
                    b'MemAvailable: 5 kB\nMemAvailable: 6 kB\n', b'MemAvailable:\n5 kB\n'):
            with self.subTest(length=len(raw)), self.assertRaises(ValueError):
                entry._mem_available(raw)
        now = [0.0]
        budget = entry.Budget(synthetic_limits(), clock=lambda: now[0], read_memory=lambda: good)
        budget.phase('fit')
        now[0] = 301
        with self.assertRaises(TimeoutError):
            budget.check()
        with self.assertRaises(ValueError):
            entry.Budget(synthetic_limits(), clock=lambda: 0.0,
                         read_memory=lambda: b'MemAvailable: 1 kB\n')

    def test_old_new_views_and_predeclared_roles_cannot_be_swapped(self):
        def view(clusters):
            return SimpleNamespace(private_clusters=np.asarray(clusters, dtype=np.int64))
        original = {'train_seeds': list(range(1, 9)), 'validation_seeds': [11, 12]}
        old, new = [21, 22], [31, 32]
        roles = {'train': list(range(1, 9)), 'teacher_dev': [11, 12], 'conditional_dev': old + new}
        views = {'train': [view(roles['train'])], 'teacher_dev': [view(roles['teacher_dev'])],
                 'conditional_dev': [view(old), view(new)]}
        ordered, expected = entry.bind_views(views, roles, original, old, new, np)
        self.assertIs(ordered['conditional_old'], views['conditional_dev'][0])
        self.assertEqual(expected['conditional_new'], tuple(new))
        swapped = dict(views, conditional_dev=list(reversed(views['conditional_dev'])))
        with self.assertRaises(ValueError):
            entry.bind_views(swapped, roles, original, old, new, np)
        swapped_roles = dict(roles, conditional_dev=new + old)
        with self.assertRaises(ValueError):
            entry.bind_views(swapped, swapped_roles, original, old, new, np)

    def test_development_screen_separates_pass_failure_and_unknown(self):
        def supported(value):
            return {'support': 'supported', 'value': value}
        good = synthetic_metrics()
        self.assertEqual(entry.assess_development(good)['status'], 'passed')
        # A positive three-source mean cannot hide a large drop in one source.
        blocked = synthetic_metrics()
        for source, number in zip(entry.EVALUATION['sources'], (0.55, 0.55, 0.47), strict=True):
            blocked['auxiliary'][source]['unseen_train']['plant_same_target_unit'] = supported(number)
        self.assertEqual(entry.assess_development(blocked)['status'], 'failed')
        guarded = synthetic_metrics()
        guarded['auxiliary']['teacher_dev']['all']['full_top1'] = supported(0.47)
        self.assertEqual(entry.assess_development(guarded)['status'], 'failed')
        guarded['baseline']['conditional_new']['unseen_train']['full_target_operation'] = {
            'support': 'unknown', 'value': None}
        self.assertEqual(entry.assess_development(guarded)['status'], 'unknown')
        guarded['baseline']['conditional_new']['unseen_train']['full_target_operation']['value'] = 0
        with self.assertRaises(ValueError):
            entry.assess_development(guarded)

    def test_real_synthetic_evaluator_metrics_feed_the_registered_screen(self):
        from learning.task_pair_memory_pipeline_v1 import pipeline
        from learning.task_pair_memory_pipeline_v1.test_pipeline import synthetic_sources
        from learning.task_pair_eval_adapter_v1.test_adapter import SCHEMA
        from learning.task_pair_numpy_scorer_v1.test_scorer import synthetic_network
        networks = {name: synthetic_network(favor_high_unit=True) for name in entry.EVALUATION['arms']}
        result = pipeline.evaluate_models(networks, synthetic_sources(),
            expected_schema_sha256=SCHEMA['schema_sha256'], check=lambda: None)
        self.assertEqual(entry.assess_development(result.metrics)['status'], 'failed')

    def test_existing_output_is_not_modified_after_exclusive_create_fails(self):
        args = ['--registration-commit', 'a' * 40, '--game-root', 'synthetic-root',
                '--private-contract', 'synthetic-private',
                '--expected-private-contract-sha256', 'b' * 64, '--receipt-token', 'c' * 32]
        registration = synthetic_registration()
        output = io.StringIO()
        fake_root = Path.cwd().resolve() / 'synthetic-root'
        fake_artifacts = fake_root / 'artifacts'
        phases = []
        def phase(name, result=None):
            def call(*args, **kwargs):
                phases.append(name)
                return result
            return call
        with patch.object(entry, 'registration_gate', return_value=(b'{}', registration)), \
             patch.object(entry, 'verify_registered_sources', side_effect=phase('sources')), \
             patch.object(entry, 'prebind', side_effect=phase('prebind', object())), \
             patch.object(entry, '_runtime_pre', side_effect=phase('runtime_pre')), \
             patch.object(entry, '_runtime_after', side_effect=phase('runtime_after')), \
             patch.object(entry, 'finish_bootstrap', side_effect=phase('finish', object())), \
             patch.object(entry, '_private_root', side_effect=phase('private_root', (fake_root, fake_artifacts))), \
             patch.object(entry, '_private_file', return_value=b'{}'), \
             patch.object(entry, 'parse_private_contract', return_value={}), \
             patch.object(Path, 'mkdir', side_effect=FileExistsError) as mkdir, \
             patch.object(Path, 'is_dir', return_value=True), \
             patch.object(entry, '_write_json') as write, redirect_stdout(output):
            code = entry.main(args)
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(output.getvalue())['stage'], 'output')
        mkdir.assert_called_once()
        write.assert_not_called()
        self.assertEqual(phases, ['sources', 'prebind', 'private_root', 'runtime_pre', 'finish', 'runtime_after'])

    def test_registered_stage_order_and_swapped_sources_with_injected_io(self):
        schema = 'a' * 64
        def view(clusters):
            return SimpleNamespace(private_clusters=np.asarray(clusters, dtype=np.int64),
                                   labels=np.zeros(len(clusters), dtype=np.int64), schema_digest=schema)
        original = {'train_seeds': list(range(1, 9)), 'validation_seeds': [11, 12]}
        old, new = [21, 22], [31, 32]
        roles = {'train': list(range(1, 9)), 'teacher_dev': [11, 12], 'conditional_dev': old + new}
        views = {'train': [view(roles['train'])], 'teacher_dev': [view(roles['teacher_dev'])],
                 'conditional_dev': [view(old), view(new)]}
        original_default = b'synthetic-default'
        source_sha = hashlib.sha256(original_default).hexdigest()
        registration = {'source_sha256': {'kaggriculture/main.py': source_sha}}
        private = {'history_receipt_sha256': 'b' * 64, 'target_device_kind': 'synthetic-gpu',
                   'batch_rng_seed': 3}
        mask = lambda: {'seen_in_train': np.zeros(2, bool), 'label_conflict': np.zeros(2, bool),
                        'qualifying_plant': np.ones(2, bool)}
        masks = {'teacher_dev': [mask()], 'conditional_dev': [mask(), mask()]}
        events, written = [], {}
        class Bound:
            def load_masks(self, *args, **kwargs):
                events.append('masks')
                return masks
        class Frozen:
            def read_json(self, path):
                if path == entry.REPO / entry.ORIGINAL_REL: return original
                if path == entry.REPO / entry.CONDITIONAL_REL: return {'synthetic': True}
                if path == entry.REPO / entry.CACHE_REL: return {'numeric_feature_schema_sha256': schema}
                raise AssertionError('Unexpected synthetic metadata')
            def source_roles(self, a, b): return old, new, original['train_seeds'] + original['validation_seeds']
        frozen = Frozen()
        def load_data(root, source, check):
            events.append('data')
            return views, roles
        class Dev:
            def __init__(self, source_view, source_masks, clusters):
                self.view, self.masks, self.expected_clusters = source_view, source_masks, clusters
        def train(*args, **kwargs):
            events.append('train')
            self.assertEqual((kwargs['updates'], kwargs['batch_size']), (64, 256))
            return SimpleNamespace(completed_updates=64, auxiliary_nonzero_updates=32,
                                   control_bytes=b'control', auxiliary_bytes=b'auxiliary')
        def evaluate(models, dev, **kwargs):
            events.append('evaluate')
            self.assertIs(dev['conditional_old'].view, views['conditional_dev'][0])
            self.assertIs(dev['conditional_new'].view, views['conditional_dev'][1])
            return SimpleNamespace(metrics=synthetic_metrics())
        pipeline = SimpleNamespace(train_and_export=Mock(side_effect=train), DevSource=Dev,
                                   evaluate_models=Mock(side_effect=evaluate))
        recheck = SimpleNamespace(NEW_OUTPUT_NAME='synthetic-masks',
                                  load_frozen_v1_bounded=lambda _: frozen,
                                  read_local_mask=lambda *args: b'synthetic-mask-metadata')
        preflight = SimpleNamespace(load_data=Mock(side_effect=load_data))
        modules = SimpleNamespace(np=np, recheck=recheck, v2=object(), preflight=preflight,
            adapter=SimpleNamespace(adapt_r8=lambda *args, **kwargs: {'selected_ops': np.arange(5)}),
            train=SimpleNamespace(class_weights=lambda _: (None, np.ones(5, np.float32))),
            pipeline=pipeline, model=SimpleNamespace(TaskNetwork=Mock(side_effect=lambda *args: object())),
            snapshot=SimpleNamespace(MAX_ARCHIVE_BYTES=100, load_checkpoint=lambda *args, **kwargs: object()),
            smoke=object(), root=Path('/synthetic'),
            extra=SimpleNamespace(TRAIN_MODULES=frozenset(), verify_loaded_origins=Mock()))
        gpu = SimpleNamespace(platform='gpu', device_kind='synthetic-gpu')
        fake_jax = SimpleNamespace(devices=lambda name: [gpu], default_device=lambda device: nullcontext())
        budget = SimpleNamespace(phase=lambda name: events.append('phase-' + name), check=lambda: None)
        bound = Bound()
        old_registration = {'source_sha256': dict(registration['source_sha256'])}
        def read_file(path, maximum):
            name = Path(path).name
            if name == 'main.py': return original_default
            if name in written: return written[name]
            raise AssertionError('Unexpected synthetic file read')
        def write_file(path, raw):
            events.append('write-' + Path(path).name)
            written[Path(path).name] = raw
        def write_json(path, value):
            events.append('write-' + Path(path).name)
            written[Path(path).name] = encode(value)
        checks = Mock()
        with patch.object(entry, '_history', return_value=(bound, old_registration)), \
             patch.object(entry, '_bounded_file', side_effect=read_file), \
             patch.object(entry, '_mask_directory', return_value=Path('/synthetic/masks')), \
             patch.object(entry, '_baseline_bytes', return_value=(b'baseline', object())), \
             patch.object(entry, '_write_new', side_effect=write_file), \
             patch.object(entry, '_write_json', side_effect=write_json), \
             patch.object(entry, '_verify_reloaded'), \
             patch.object(entry, '_check_direct_scores', side_effect=lambda *args: events.append('score-check')), \
             patch.object(entry, '_verify_data_after', side_effect=lambda *args: events.append('data-recheck')), \
             patch.object(entry, 'verify_registered_sources'), \
             patch.object(entry, 'registration_gate', return_value=(b'{}', registration)), \
             patch.object(entry, '_load_jax', return_value=fake_jax), \
             patch.object(entry, '_runtime_after'):
            receipt = entry.run_registered('a' * 40, registration, private, Path('/synthetic'),
                    Path('/synthetic/out'), modules, budget, 'c' * 32, private_check=checks)
        self.assertEqual(receipt['completed_updates'], 64)
        self.assertEqual(receipt['auxiliary_nonzero_updates'], 32)
        self.assertLess(events.index('data'), events.index('masks'))
        self.assertLess(events.index('masks'), events.index('train'))
        self.assertLess(events.index('score-check'), events.index('evaluate'))
        self.assertLess(events.index('evaluate'), events.index('data-recheck'))
        self.assertEqual(json.loads(written['evaluation.local.json'])['development_screen']['status'], 'passed')
        self.assertNotIn(b'c' * 32, b''.join(written.values()))
        checks.assert_called_once()

        # Source order is checked before the private masks, checkpoint or fit.
        swapped = dict(views, conditional_dev=list(reversed(views['conditional_dev'])))
        preflight.load_data.return_value = (swapped, roles)
        preflight.load_data.side_effect = None
        pipeline.train_and_export.reset_mock()
        with patch.object(entry, '_history', return_value=(bound, old_registration)), \
             patch.object(entry, '_bounded_file', return_value=original_default), \
             patch.object(entry, '_mask_directory') as mask_dir:
            with self.assertRaises(ValueError):
                entry.run_registered('a' * 40, registration, private, Path('/synthetic'),
                    Path('/synthetic/out'), modules, budget, 'c' * 32, private_check=checks)
        mask_dir.assert_not_called()
        pipeline.train_and_export.assert_not_called()

    def test_default_scores_are_compared_to_actual_unmodified_method(self):
        from learning.task_pair_eval_adapter_v1.test_adapter import sample_view, SCHEMA
        from learning.task_pair_numpy_scorer_v1.test_scorer import synthetic_network
        from learning.task_pair_numpy_scorer_v1 import scorer
        from learning.task_bc_v1 import model
        view, network = sample_view(sizes=(10, 10)), synthetic_network(favor_high_unit=True)
        modules = SimpleNamespace(np=np, scorer=scorer, model=model)
        entry._check_direct_scores({'baseline': network}, {'teacher_dev': SimpleNamespace(view=view)},
                                   modules, SCHEMA['schema_sha256'], lambda: None)


if __name__ == '__main__':
    unittest.main()
