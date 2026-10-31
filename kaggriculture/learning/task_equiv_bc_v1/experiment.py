"""One preregistered private teacher-continuation and optional GPU experiment.

No official game, rule engine, remote submission or default Bot is run or
modified. The caller supplies a privately prebound identity file. Public output
is a small aggregate; observations, models, seeds and logs stay in the stage.
"""
import argparse
from contextlib import ExitStack
import gc
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import runpy
import subprocess
import sys
import time
import traceback


ENTRY_STARTED = time.monotonic()
REPO = Path(__file__).resolve().parents[3]
BASE = 'f7c0c026c7e538a9b70b257ba13f51e94755b673'
PUBLIC = 'review/evidence/task-equivalent-bc-v2-20261031/'
MAP = PUBLIC + 'source-map.json'
NOTE = PUBLIC + 'REGISTRATION.zh-CN.md'
OLD_PUBLIC = 'review/evidence/task-choice-v1-20261023/'
STAGE = 'task-equivalent-bc-v2-20261031'
OLD_STAGE = 'task-pair-round2-compare-v1-20261031'
GPU_CONFIG = 'artifacts/task-dagger-once-v1-20261027/gpu-launch.local.json'
HEX = re.compile(r'[0-9a-f]{64}\Z')
COMMIT = re.compile(r'[0-9a-f]{40}\Z')


class CleanupUnconfirmed(RuntimeError):
    pass


class DiskFloorError(OSError):
    pass


def require(value):
    if not value:
        raise ValueError('Experiment input validation failed')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _safe_relative(path):
    require(type(path) is str and path and '\\' not in path and ':' not in path)
    value = PurePosixPath(path)
    require(not value.is_absolute() and value.as_posix() == path
            and all(part not in ('.', '..') for part in value.parts))
    return value


def _read(root, path, limit, driver):
    return driver._read(root, path, limit)


def _json(raw):
    def pairs(items):
        result = {}
        for k, v in items:
            require(k not in result)
            result[k] = v
        return result
    def invalid(_):
        raise ValueError('Nonfinite JSON')
    value = json.loads(raw.decode('utf-8-sig'), object_pairs_hook=pairs, parse_constant=invalid)
    require(type(value) is dict)
    return value


def _source_snapshot(commit, flow, driver, check):
    require(type(commit) is str and COMMIT.fullmatch(commit) is not None)
    check()
    flow.clean_head(commit)
    flow.git('merge-base', '--is-ancestor', BASE, commit)
    note = flow.public_file(NOTE, commit)
    raw = flow.public_file(MAP, commit)
    document = _json(raw)
    require(set(document) == {'schema', 'base_commit', 'public_sources'}
            and document['schema'] == 'task-equiv-bc-source-map-v1'
            and document['base_commit'] == BASE and type(document['public_sources']) is dict
            and MAP not in document['public_sources'])
    required = {'kaggriculture/learning/task_equiv_bc_v1/experiment.py',
                'kaggriculture/learning/task_equiv_bc_v1/dataset.py',
                'kaggriculture/learning/task_equiv_bc_v1/probe.py',
                'kaggriculture/learning/task_equiv_bc_v1/train.py',
                'kaggriculture/learning/task_equiv_bc_v1/loss.py',
                'kaggriculture/learning/task_equiv_bc_v1/gpu_bootstrap.py',
                'kaggriculture/learning/task_equiv_bc_v1/gpu_worker.py',
                'kaggriculture/learning/task_equiv_bc_v1/gpu_supervisor.py',
                'kaggriculture/learning/task_equiv_bc_v1/host_gpu.py', NOTE,
                OLD_PUBLIC + 'acceptance.json'}
    require(required.issubset(document['public_sources']))
    for path, digest in document['public_sources'].items():
        check()
        _safe_relative(path)
        require(type(digest) is str and HEX.fullmatch(digest) and sha(flow.public_file(path, commit)) == digest)
    for name, module in tuple(sys.modules.items()):
        if name == 'learning' or name.startswith('learning.') or name == 'decision' or name.startswith('decision.'):
            filename = getattr(module, '__file__', None)
            if filename is None:
                continue
            path = Path(filename).resolve(strict=True)
            require(path.is_relative_to(REPO) and path.relative_to(REPO).as_posix() in document['public_sources'])
    check()
    return raw, note


class Budget:
    def __init__(self, available, disk_free, clock=time.monotonic):
        self.available, self.disk_free, self.clock = available, disk_free, clock
        self.start, self.phase_start, self.phase_limit = ENTRY_STARTED, ENTRY_STARTED, 2400

    def initial(self):
        available, free = self.available(), self.disk_free()
        require(type(available) is int and type(free) is int and available >= 0 and free >= 0)
        if available < 5 * 1024**3:
            raise MemoryError('Initial memory floor')
        if free < 4 * 1024**3:
            raise DiskFloorError('Initial disk floor')
        return self.check()

    def phase(self, seconds):
        self.phase_start, self.phase_limit = self.clock(), seconds

    def check(self):
        now = self.clock()
        if now - self.start >= 4200 or now - self.phase_start >= self.phase_limit:
            raise TimeoutError('Experiment deadline')
        if self.available() < 4 * 1024**3:
            raise MemoryError('Experiment memory floor')
        return True


def _write(path, raw):
    require(type(raw) is bytes and raw)
    with path.open('xb') as stream:
        require(stream.write(raw) == len(raw))


def _write_json(path, value):
    _write(path, (json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n').encode('utf8'))


def _eligibility(group):
    import numpy as np
    p, u = group['positive_mask'], group['verified_mask']
    return int(np.count_nonzero(p) >= 2 and np.any(u & ~p))


def _summarize(groups, roles):
    counts = {'groups': len(groups), 'eligible': 0, 'eligible_per_role': [0, 0, 0],
              'additional_positive': 0, 'different_request': 0, 'attempted_unknown': 0,
              'attempted_alternatives': 0, 'untried_candidates': 0}
    for group, role in zip(groups, roles):
        require(role in (0, 1, 2) and group.get('status') == 'ok')
        eligible = _eligibility(group)
        counts['eligible'] += eligible
        counts['eligible_per_role'][role] += eligible
        counts['additional_positive'] += len(group['positive_ids']) - 1
        counts['different_request'] += len(group['different_request_ids'])
        counts['attempted_unknown'] += len(group['unknown_ids'])
        counts['attempted_alternatives'] += len(group['alternatives'])
        counts['untried_candidates'] += len(group['features']) - 1 - len(group['alternatives'])
    return counts


def _pack_training(groups, roles):
    import numpy as np
    require(len(groups) == len(roles) == 48 and roles.count(0) == roles.count(1) == 24)
    offsets, features, positive, verified, labels = [0], [], [], [], []
    for group in groups:
        x = group['features']
        require(isinstance(x, np.ndarray) and x.dtype == np.float32 and x.ndim == 2
                and x.shape[1] == 373 and 2 <= len(x) <= 256)
        features.append(x)
        positive.append(group['positive_mask'])
        verified.append(group['verified_mask'])
        labels.append(group['teacher_index'])
        offsets.append(offsets[-1] + len(x))
    out = io.BytesIO()
    np.savez_compressed(out, features=np.concatenate(features), offsets=np.asarray(offsets, np.int32),
                        teacher=np.asarray(labels, np.int32), positive=np.concatenate(positive),
                        verified=np.concatenate(verified), roles=np.asarray(roles, np.int32))
    raw = out.getvalue()
    require(len(raw) <= 32 * 1024**2)
    return raw


def _wsl_executable():
    system_root = os.environ.get('SystemRoot')
    require(type(system_root) is str and system_root and Path(system_root).is_absolute())
    executable = Path(system_root) / 'System32' / 'wsl.exe'
    require(executable.is_file())
    return str(executable)


def _wsl_path(distro, path):
    process = subprocess.run([_wsl_executable(), '-d', distro, '--exec', 'wslpath', '-a', '-u', str(path)],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    require(process.returncode == 0 and 0 < len(process.stdout) < 4096)
    value = process.stdout.decode('utf8').strip()
    require(value.startswith('/') and '\n' not in value and '\x00' not in value)
    return value


def _gpu_failure(stage):
    """Project an unauthenticated, allowlisted observation from shared stderr."""
    path = stage / 'gpu-host.stderr.local.log'
    try:
        if not path.is_file() or path.stat().st_size > 65536:
            return None
        for line in path.read_bytes().splitlines():
            try:
                item = _json(line)
            except (ValueError, UnicodeError, TypeError):
                continue
            if (item.get('status') == 'failed' and item.get('phase') in
                    ('arguments', 'binding', 'device', 'training', 'output')
                    and item.get('category') in ('timeout', 'memory', 'invalid_or_runtime')):
                return {'phase': item['phase'], 'category': item['category']}
    except OSError:
        pass
    return None


def _wait_unconfirmed(host):
    """Keep observing an owned transport through interruptions during cleanup."""
    try:
        if host.child is not None:
            while True:
                try:
                    host.child.wait(timeout=1)
                    break
                except (subprocess.TimeoutExpired, KeyboardInterrupt):
                    # Cancellation has already been requested. A second signal
                    # must not abandon a child we can still observe.
                    continue
    except BaseException as exc:
        raise CleanupUnconfirmed('GPU cleanup was not confirmed') from exc
    finally:
        try:
            exited = host.child is None or host.child.poll() is not None
        except BaseException:
            exited = False
        if exited:
            for stream in (host.stdout_stream, host.stderr_stream):
                if stream is not None:
                    try: stream.close()
                    except BaseException: pass
    raise CleanupUnconfirmed('GPU cleanup was not confirmed')


def _evaluate(cases, networks, schema, probe, check):
    import numpy as np
    require(len(cases) == 24 and set(networks) == {'baseline', 'control', 'experiment'})
    stats = {name: {'match': 0, 'different': 0, 'unknown': 0, 'teacher_top1': 0,
                    'positive_probability_sum': 0.0, 'controller_calls': 0}
             for name in networks}
    for sample, group in cases:
        inputs, expected = sample['inputs'], sample['label']['teacher_label']
        for name, network in networks.items():
            check()
            selected = network(inputs)
            logits = np.asarray(network.last['logits'], dtype=np.float64)
            require(logits.shape == group['positive_mask'].shape and np.isfinite(logits).all())
            exp = np.exp(logits - logits.max())
            probability = float(exp[group['positive_mask']].sum() / exp.sum())
            require(np.isfinite(probability) and 0 <= probability <= 1)
            outcome = probe.evaluate_choice(inputs['observation'], inputs['configuration'], sample['decision'],
                expected_inputs=inputs, expected_label=expected,
                expected_teacher_prefix=sample['teacher_prefix'], reference=group['reference'],
                chosen_id=selected, guard=check)
            require(set(outcome) == {'outcome', 'controller_called'}
                    and outcome['outcome'] in ('match', 'different', 'unknown')
                    and type(outcome['controller_called']) is bool)
            row = stats[name]
            row[outcome['outcome']] += 1
            row['teacher_top1'] += int(selected == expected)
            row['positive_probability_sum'] += probability
            row['controller_calls'] += int(outcome['controller_called'])
    for row in stats.values():
        require(row['match'] + row['different'] + row['unknown'] == 24)
        row['positive_probability_mean'] = row.pop('positive_probability_sum') / 24
    require(sum(row['controller_calls'] for row in stats.values()) <= 72)
    unknown = any(row['unknown'] for row in stats.values())
    screen = ('unknown' if unknown else 'passed' if
              stats['experiment']['match'] >= stats['control']['match'] + 1 and
              stats['experiment']['teacher_top1'] >= stats['control']['teacher_top1'] - 2 else 'failed')
    return stats, screen


def _category(exc):
    if isinstance(exc, CleanupUnconfirmed): return 'gpu_cleanup_unconfirmed'
    if isinstance(exc, DiskFloorError): return 'disk_gate'
    # The probe wraps guard failures to prevent them being mistaken for an
    # unknown candidate; retain their precise, non-sensitive stop category.
    cause, seen = exc, set()
    while cause is not None and id(cause) not in seen and len(seen) < 8:
        seen.add(id(cause))
        if isinstance(cause, MemoryError): return 'memory_gate'
        if isinstance(cause, TimeoutError): return 'time_gate'
        cause = cause.__cause__
    if isinstance(exc, FileNotFoundError): return 'missing_file'
    if isinstance(exc, OSError): return 'io_error'
    if isinstance(exc, (ValueError, TypeError, KeyError, IndexError)): return 'invalid_input'
    return 'runtime_error'


def main(argv=None):
    phase, stage, claimed, budget, read_budget, host_reason, worker_failure = 'arguments', None, False, None, None, None, None
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--game-root', required=True)
        parser.add_argument('--private-root', required=True)
        parser.add_argument('--registration-commit', required=True)
        parser.add_argument('--binding-sha256', required=True)
        args = parser.parse_args(argv)
        require(os.name == 'nt' and sys.flags.isolated and sys.flags.dont_write_bytecode and __debug__
                and type(args.binding_sha256) is str and HEX.fullmatch(args.binding_sha256))
        bootstrap = runpy.run_path(str(Path(__file__).with_name('gpu_bootstrap.py')))
        imports, project_root = bootstrap['bind']()
        # The legacy read-only comparison runner inserts the checkout into
        # sys.path on import. Suppress that one historical insertion so later
        # NumPy/JAX and other dependency imports remain isolated.
        with imports.protected_search_path(project_root):
            import psutil
            import numpy as np
            from learning.task_equiv_bc_v1 import dataset, probe, host_gpu
            from learning.task_pair_action_flow_v1 import runner as flow
            from learning.task_pair_dev_campaign_v1 import driver
            from learning.task_choice_v1.features import OPS
            from learning.task_bc_v1.model import TaskNetwork
            from learning.task_pair_checkpoint_v1.snapshot import load_checkpoint
        imports.verify_search_path(project_root)

        root, private = Path(args.game_root), Path(args.private_root)
        require(root.is_absolute() and root.resolve(strict=True) == root
                and root == (REPO.parent.parent / 'kaggriculture').resolve(strict=True)
                and private.is_absolute() and private.resolve(strict=True) == private
                and private == (REPO.parent / 'private').resolve(strict=True))
        budget = Budget(lambda: psutil.virtual_memory().available,
                        lambda: __import__('shutil').disk_usage(root).free)
        budget.initial()
        phase = 'source_snapshot'
        source_raw, note_raw = _source_snapshot(args.registration_commit, flow, driver, budget.check)
        binding_path = private / 'task-equivalent-bc-v1-binding-v2.local.json'
        binding_raw = _read(private, binding_path, 65536, driver)
        require(sha(binding_raw) == args.binding_sha256)
        binding = _json(binding_raw)
        require(set(binding) == {'schema', 'old_manifest_sha256', 'old_result_sha256',
                                 'gpu_config_sha256', 'acceptance_sha256', 'baseline_model',
                                 'expected_device_kind'}
                and binding['schema'] == 'task-equiv-bc-private-binding-v2'
                and type(binding['expected_device_kind']) is str
                and 1 <= len(binding['expected_device_kind']) <= 128)
        stage = root / 'artifacts' / STAGE
        driver._no_links(root, stage)
        stage.mkdir(exist_ok=False)
        claimed = True
        phase = 'historical_binding'
        old = root / 'artifacts' / OLD_STAGE
        old_manifest_raw = _read(old, old / 'comparison-manifest.local.json', 8 * 1024**2, driver)
        old_result_raw = _read(old, old / 'comparison-result.local.json', 8 * 1024**2, driver)
        gpu_config_raw = _read(root, root / GPU_CONFIG, 65536, driver)
        require(sha(old_manifest_raw) == binding['old_manifest_sha256']
                and sha(old_result_raw) == binding['old_result_sha256']
                and sha(gpu_config_raw) == binding['gpu_config_sha256'])
        old_manifest = _json(old_manifest_raw)
        require(old_manifest['models']['baseline'] == binding['baseline_model'])
        local, _, accepted, _ = flow.read_context(root, private / (OLD_STAGE + '-campaign-exit.local.txt'), budget.check)
        excluded = sorted({value['row']['seed'] for value in accepted.values()})
        require(len(excluded) == 2)
        choice_root = root / 'artifacts' / 'task-choice-v1-20261023'
        evidence_names = ('registration.json', 'dataset-manifest.json', 'reconstruction-ledger.json',
                          'independent-review.json', 'schema.json', 'acceptance.json')
        evidence = {}
        for name in evidence_names:
            budget.check()
            raw = _read(choice_root, choice_root / name, 8 * 1024**2, driver)
            require(raw == flow.public_file(OLD_PUBLIC + name, BASE))
            evidence[name] = raw
        require(sha(evidence['acceptance.json']) == binding['acceptance_sha256'])
        plans = dataset.bind_sources(registration_raw=evidence['registration.json'],
            manifest_raw=evidence['dataset-manifest.json'], ledger_raw=evidence['reconstruction-ledger.json'],
            review_raw=evidence['independent-review.json'], schema_raw=evidence['schema.json'],
            acceptance_raw=evidence['acceptance.json'], expected_acceptance_sha256=binding['acceptance_sha256'],
            excluded_seeds=excluded)
        require(len(plans) == 12 and len({(p.seed, p.opponent, p.side) for p in plans}) == 12)
        seeds_train = list(dict.fromkeys(p.seed for p in plans if p.split == 'train'))
        seeds_dev = list(dict.fromkeys(p.seed for p in plans if p.split == 'dev'))
        require(len(seeds_train) == 2 and len(seeds_dev) == 1
                and not set(seeds_train + seeds_dev) & set(excluded)
                and not set(seeds_train) & set(seeds_dev))
        cfg, schema_doc = _json(evidence['registration.json']), _json(evidence['schema.json'])
        schema = {key: schema_doc[key] for key in ('schema_version', 'schema_sha256',
                                                    'feature_names', 'shared_feature_names')}
        schema['ops'] = list(OPS)
        demonstration = root.joinpath(*_safe_relative(cfg['demonstration_root']).parts)
        driver._no_links(root, demonstration)
        require(demonstration.is_dir())
        default_path = root / 'main.py'
        default_before = sha(_read(root, default_path, 8 * 1024**2, driver))
        phase = 'sampling'
        read_budget = dataset.ReadBudget(compressed_limit=1024**3, decoded_limit=16 * 1024**3,
                                         check=budget.check)
        train_samples, dev_samples = [], []
        for plan in plans:
            budget.check()
            replay = demonstration / 'games' / plan.game / 'replay.json.gz'
            paths = {kind: choice_root.joinpath(*_safe_relative(plan.file(kind).path).parts)
                     for kind in ('inputs', 'labels', 'index')}
            for path in [replay, *paths.values()]:
                driver._no_links(root, path)
            with ExitStack() as stack:
                streams = {kind: stack.enter_context(path.open('rb')) for kind, path in paths.items()}
                replay_stream = stack.enter_context(replay.open('rb'))
                samples = dataset.sample_game(plan, inputs_stream=streams['inputs'],
                    labels_stream=streams['labels'], index_stream=streams['index'],
                    replay_stream=replay_stream, budget=read_budget)
            role = seeds_train.index(plan.seed) if plan.split == 'train' else 2
            (train_samples if plan.split == 'train' else dev_samples).extend((sample, role) for sample in samples)
            print(json.dumps({'phase': 'sampling', 'games_consumed': len(train_samples + dev_samples) // 6}), flush=True)
        require(len(train_samples) == 48 and len(dev_samples) == 24)
        phase = 'teacher_continuations'
        groups_train, groups_dev, train_roles, dev_roles = [], [], [], []
        dev_cases = []
        for samples, output, roles in ((train_samples, groups_train, train_roles),
                                       (dev_samples, groups_dev, dev_roles)):
            for sample, role in samples:
                budget.check()
                inputs = sample['inputs']
                group = probe.probe_group(inputs['observation'], inputs['configuration'], sample['decision'],
                    expected_inputs=inputs, expected_label=sample['label']['teacher_label'],
                    expected_teacher_prefix=sample['teacher_prefix'], expected_request=sample['recorded_actor_request'],
                    expected_schema=schema, guard=budget.check, retain_reference=(role == 2))
                require(group.get('status') == 'ok')
                output.append(group)
                roles.append(role)
                if role == 2:
                    dev_cases.append((sample, group))
        train_stats, dev_stats = _summarize(groups_train, train_roles), _summarize(groups_dev, dev_roles)
        gate = (train_stats['eligible'] >= 4 and min(train_stats['eligible_per_role'][:2]) >= 1
                and dev_stats['eligible'] >= 2)
        read_summary = {'compressed_bytes': read_budget.compressed_bytes, 'decoded_bytes': read_budget.decoded_bytes,
                        'parsed_groups': read_budget.parsed_groups, 'replay_json_attempts': read_budget.replay_json_attempts,
                        'scope': 'selected_teacher_inputs_labels_indexes_and_replays',
                        'old_panel_identity_hash_io': 'separate_not_counted'}
        print(json.dumps({'phase': 'teacher_continuations', 'groups': 72, 'training_gate': gate}), flush=True)
        model_meta = binding['baseline_model']
        require(type(model_meta) is dict and set(model_meta) == {'path', 'sha256'}
                and HEX.fullmatch(model_meta['sha256']))
        model_path = root.joinpath(*_safe_relative(model_meta['path']).parts)
        result = {'schema': 'task-equivalent-bc-v2', 'registration_commit': args.registration_commit,
                  'public_source_map_sha256': sha(source_raw), 'sample_scope': 'consumed_teacher_data',
                  'train': train_stats, 'development': dev_stats, 'read_budget': read_summary,
                  'training_gate': gate, 'gpu_training': False, 'new_games': 0,
                  'independent_improvement': False, 'promotion': False}
        if gate:
            phase = 'gpu_training'
            model_raw = _read(root, model_path, 8 * 1024**2, driver)
            require(sha(model_raw) == model_meta['sha256'])
            load_checkpoint(model_raw, expected_sha256=model_meta['sha256'], expected_schema_sha256=schema['schema_sha256'])
            launch = _json(gpu_config_raw)
            require(set(launch) == {'distro', 'python', 'workdir', 'expected_device_substring'}
                    and all(type(value) is str and value for value in launch.values())
                    and launch['python'].startswith('/')
                    and launch['expected_device_substring'] in binding['expected_device_kind'])
            _write(stage / 'training-groups.local.npz', _pack_training(groups_train, train_roles))
            _write(stage / 'initial.local.npz', model_raw)
            gpu_contract = {'schema': 'task-equiv-bc-gpu-contract-v1', 'registration_commit': args.registration_commit,
                'groups_sha256': sha((stage / 'training-groups.local.npz').read_bytes()),
                'checkpoint_sha256': model_meta['sha256'], 'schema_sha256': schema['schema_sha256'],
                'device_kind': binding['expected_device_kind']}
            _write_json(stage / 'gpu-contract.local.json', gpu_contract)
            del train_samples, groups_train
            gc.collect()
            budget.phase(900)
            linux_stage = _wsl_path(launch['distro'], stage)
            linux_script = _wsl_path(launch['distro'], Path(__file__).with_name('gpu_supervisor.py'))
            argv_gpu = [_wsl_executable(), '-d', launch['distro'], '--exec', 'env', 'JAX_PLATFORMS=cuda',
                        'XLA_PYTHON_CLIENT_PREALLOCATE=false', 'PYTHONDONTWRITEBYTECODE=1', 'PYTHONNOUSERSITE=1',
                        launch['python'], '-I', '-B', linux_script, '--stage', linux_stage,
                        '--registration-commit', args.registration_commit]
            host = host_gpu.supervise(stage, argv_gpu, args.registration_commit)
            allowed_host_reasons = {'completed', 'initial_memory_gate', 'initial_memory_sample_error',
                'spawn_uncertain', 'canceled', 'time_gate', 'memory_gate', 'host_error',
                'cancel_write_failed', 'linux_cleanup_unconfirmed', 'receipt_unavailable', 'child_failed'}
            host_reason = host.reason if host.reason in allowed_host_reasons else 'other'
            if host.status == 'cleanup_unconfirmed':
                print(json.dumps({'phase': 'gpu_cleanup', 'status': 'waiting'}), flush=True)
                _wait_unconfirmed(host)
            if host.reason in ('initial_memory_gate', 'memory_gate'):
                raise MemoryError('Host GPU memory floor')
            if host.reason == 'time_gate':
                raise TimeoutError('Host GPU deadline')
            if not (host.status == 'passed' and host.cleanup_confirmed and host.exit_code == 0):
                worker_failure = _gpu_failure(stage)
                raise RuntimeError('GPU child failed')
            budget.check()
            gpu = _json(_read(stage, stage / 'gpu-result.local.json', 1024**2, driver))
            require(gpu.get('schema') == 'task-equiv-bc-gpu-result-v1' and gpu.get('status') == 'passed'
                    and gpu.get('device_platform') == 'gpu' and type(gpu.get('device_kind')) is str
                    and binding['expected_device_kind'] == gpu['device_kind']
                    and gpu.get('updates_per_arm') == 32 and gpu.get('both_arms_changed') is True
                    and gpu.get('arms_differ') is True and gpu.get('eligible_training_groups') == train_stats['eligible'])
            networks = {'baseline': TaskNetwork(stage / 'initial.local.npz', model_meta['sha256'])}
            for name in ('control', 'experiment'):
                networks[name] = TaskNetwork(stage / (name + '.local.npz'), gpu[name + '_sha256'])
                require(np.array_equal(networks[name].mean, networks['baseline'].mean)
                        and np.array_equal(networks[name].scale, networks['baseline'].scale)
                        and networks[name].metadata['schema_sha256'] == schema['schema_sha256'])
            result['gpu_training'] = True
            result['updates_per_arm'] = 32
            result['registered_gpu_device_verified'] = True
            result['first_gradient_gap'] = gpu['records'][0]['gradient_gap_at_current_arm_parameters']
            phase = 'development_evaluation'
            budget.phase(600)
            result['development_metrics'], result['exploratory_screen'] = _evaluate(dev_cases, networks, schema, probe, budget.check)
        else:
            result['exploratory_screen'] = 'not_run_training_gate_failed'
        phase = 'final_checks'
        budget.check()
        imports.verify_search_path(project_root)
        require(sha(_read(root, default_path, 8 * 1024**2, driver)) == default_before
                and sha(_read(root, model_path, 8 * 1024**2, driver)) == model_meta['sha256']
                and _source_snapshot(args.registration_commit, flow, driver, budget.check) == (source_raw, note_raw))
        result['status'] = 'passed'
        flow.write_summary(stage, result, budget.check)
        print(json.dumps({'phase': 'complete', 'gpu_training': result['gpu_training'],
                          'exploratory_screen': result['exploratory_screen']}), flush=True)
        return 0
    except BaseException as exc:
        failure = {'schema': 'task-equivalent-bc-failure-v2', 'status': 'failed',
                   'phase': phase, 'category': _category(exc), 'retry_allowed': False}
        if host_reason is not None:
            failure['gpu_host_reason'] = host_reason
        if worker_failure is not None:
            failure['unverified_gpu_log_observation'] = worker_failure
        if read_budget is not None:
            failure['consumption'] = {'compressed_bytes': read_budget.compressed_bytes,
                'decoded_bytes': read_budget.decoded_bytes, 'started_groups': read_budget.started_groups,
                'parsed_groups': read_budget.parsed_groups, 'replay_json_attempts': read_budget.replay_json_attempts}
        if claimed and stage is not None:
            try:
                _write_json(stage / 'failure.local.json', failure)
                detail = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                _write(stage / 'failure-detail.local.txt', detail.encode('utf8', errors='replace')[:65536] or b'failure\n')
            except BaseException: pass
        print(json.dumps(failure), file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
