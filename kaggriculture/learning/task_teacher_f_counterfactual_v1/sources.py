"""Bounded F-source access and screening; no old Anchor/inventory entrypoint.

Only an explicitly authenticated completed 32-game panel can construct this
context. The eight fixed baseline sources are addressed without reading their
returns for selection. Source plans, replays, traces and candidates are private.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
from pathlib import Path
import sys
import threading
from types import MappingProxyType, ModuleType

from decision.api import observe
from decision.audit_market_teacher import observation
from learning.task_bc_error_diagnostics_v1.shadow import first_difference, ordered_hash
from learning.task_pair_dev_games_v1.panel import GAME_LIMITS
from learning.task_student_continuation_v1 import sources as old
from learning.task_student_dagger_collection_v1 import sources as dagger, sampling
from learning.task_teacher_fresh_panel_v1 import sources as completed
from . import contract


GAME = completed.fresh.GAME
ARTIFACTS = completed.fresh.ARTIFACTS
SCREEN_RELATIVE = 'kaggriculture/learning/task_teacher_counterfactual_screen_v1/screen.py'
SCREEN_SHA256 = '61216fd17eb99b076e4ef4124016748258ccffb992c62f7580540f0dae92e7ad'
_SCREEN = None


def require(value):
    if not value:
        raise ValueError('Completed F source contract differs')


def source_key(source):
    require(type(source) is dict and set(source) == {'role', 'opponent', 'side', 'policy'}
        and type(source['side']) is int and source['side'] in (0, 1)
        and source['policy'] == 'baseline'
        and any(source == expected for expected in contract.SOURCES))
    return source['role'], source['opponent'], source['side']


@dataclass(frozen=True, slots=True, repr=False)
class SourcePlan:
    role: str
    opponent: str
    side: int
    policy: str
    cohort: str
    alias: str
    source_game: str
    slot: str
    seed: int
    task_sha256: str
    crop_sha256: str
    replay_sha256: str
    trace_sha256: str
    rival_sha256: str
    source_receipts_sha256: str

    def __repr__(self):
        return '<PrivateCompletedFSourcePlan>'

    def document(self):
        return asdict(self)

    @property
    def identity_sha256(self):
        return ordered_hash(self.document())

    def coordinate(self):
        return dict(role=self.role, opponent=self.opponent, side=self.side, policy=self.policy)


class FContext:
    """A fixed source snapshot; at most one decoded game can remain open."""
    def __init__(self, authenticated):
        self._identities = deepcopy(authenticated.identities)
        self._active = None
        self._opening = False
        self._screen_attempts = set()
        self._closed = False
        plans, models = {}, set()
        role_map = {role:(cohort, alias) for cohort, aliases in contract.COHORTS.items()
                    for alias, role in aliases.items()}
        require(len(role_map) == 4 and len(contract.SOURCES) == 8)
        for source in contract.SOURCES:
            coordinate = source_key(dict(source))
            cohort, alias = role_map[source['role']]
            manifest = authenticated.manifests[cohort]
            panel = manifest.panel
            name = f"dev-{alias}-{source['opponent']}-{source['side']}-baseline"
            row = authenticated.rows[(cohort, name)]
            task, task_sha = panel.models[0]
            crop, crop_sha = panel.crop
            seed = panel.seeds[('S02', 'S03').index(alias)]
            rival_sha = manifest.opponent(source['opponent'])[1]
            require(row.get('game') == name and row.get('policy') == 'baseline'
                and type(row.get('seed')) is int and row['seed'] == seed
                and type(row.get('side')) is int and row['side'] == source['side']
                and row.get('opponent') == source['opponent'] and row.get('seed_alias') == alias
                and row.get('task_checkpoint_sha256') == task_sha
                and row.get('crop_checkpoint_sha256') == crop_sha and row.get('rival_sha256') == rival_sha)
            values = dict(source, cohort=cohort, alias=alias, source_game=name,
                slot=contract.SLOTS[source['role']], seed=seed, task_sha256=task_sha,
                crop_sha256=crop_sha, replay_sha256=row['replay_gzip_sha256'],
                trace_sha256=row['trace_sha256'], rival_sha256=rival_sha,
                source_receipts_sha256=ordered_hash(self._identities['games'][cohort + '/' + name]))
            require(coordinate not in plans)
            plans[coordinate] = SourcePlan(**values)
            models.add((task, task_sha, crop, crop_sha))
        require(len(plans) == 8 and len(models) == 1)
        self._plans = MappingProxyType(plans)
        self._task_relative, self.task_sha256, self._crop_relative, self.crop_sha256 = models.pop()
        self._task_path = GAME / self._task_relative
        self._crop_path = GAME / self._crop_relative

    def __repr__(self):
        return '<PrivateCompletedFSourceContext>'

    @property
    def identities(self):
        require(not self._closed)
        return deepcopy(self._identities)

    def plan(self, source):
        require(not self._closed)
        if type(source) is SourcePlan:
            result = self._plans[source_key(source.coordinate())]
            require(source == result)
            return result
        return self._plans[source_key(source)]

    def private_source_plans(self):
        require(not self._closed)
        return tuple(self.plan(dict(source)) for source in contract.SOURCES)

    def verify_assets(self, guard):
        require(not self._closed and callable(guard))
        require(old._model_file(self._task_relative, self.task_sha256, guard) == str(self._task_path)
            and old._model_file(self._crop_relative, self.crop_sha256, guard) == str(self._crop_path))
        return True

    def close(self):
        require(not self._opening)
        if self._active is not None:
            self._active.close()
        self._closed = True
        self._identities = None


def open_context(*, guard, expected_identities=None, verify_game_files=True):
    require(callable(guard) and type(verify_game_files) is bool
        and (verify_game_files or type(expected_identities) is dict))
    value = completed.load(guard=guard, verify_game_files=verify_game_files)
    if expected_identities is not None:
        require(type(expected_identities) is dict and value.identities == expected_identities)
    context = FContext(value)
    context.verify_assets(guard)
    verify_limits()
    return context


def verify_limits():
    """The F Worker, BoundedAgent and independent trace verifier agree."""
    from learning.task_dagger_once_v1.policy import LIMITS
    from learning.task_smoke_weight_probe_v1 import core
    require(dict(GAME_LIMITS) == dict(dagger.GAME_LIMITS) == dict(LIMITS)
        and core._limits(dict(GAME_LIMITS)) == dict(GAME_LIMITS))
    return dict(GAME_LIMITS)


class FGame:
    def __init__(self, context, plan, replay):
        self._context, self.plan, self._replay = context, plan, replay
        self._trace_open = False
        self._trace_reads = 0
        self._scan_attempted = False

    def __repr__(self):
        return '<PrivateCompletedFGame>'

    def _open(self):
        require(self._context is not None and self._context._active is self)

    @property
    def replay(self):
        self._open()
        return self._replay

    @property
    def identity(self):
        self._open()
        return dict(source='completed_f_panel', role=self.plan.role, cohort=self.plan.cohort,
            alias=self.plan.alias, source_game=self.plan.source_game, opponent=self.plan.opponent,
            side=self.plan.side, policy='baseline', source_identity_sha256=self.plan.identity_sha256)

    @property
    def game_limits(self):
        self._open()
        return verify_limits()

    @property
    def task_sha256(self):
        self._open()
        return self.plan.task_sha256

    @property
    def crop_sha256(self):
        self._open()
        return self.plan.crop_sha256

    def make_agent(self):
        self._open()
        from learning.task_dagger_once_v1.policy import BoundedAgent
        return BoundedAgent(self._context._crop_path, self.crop_sha256,
            self._context._task_path, self.task_sha256, limits=self.game_limits)

    @contextmanager
    def open_trace(self, *, guard):
        """Require a bounded complete scan and the same compressed identity."""
        self._open()
        require(callable(guard) and not self._trace_open and self._trace_reads < 2)
        self._trace_reads += 1
        self._trace_open = True
        path = source_stage(self.plan) / 'traces' / (self.plan.source_game + '.jsonl.gz')
        try:
            with old._open_checked(ARTIFACTS, path, old.COMPRESSED_MAX, guard) as raw:
                before, size = old._hash_stream(raw, old.COMPRESSED_MAX, guard)
                require(before == self.plan.trace_sha256)
                old._rewind(raw)
                with gzip.GzipFile(fileobj=raw, mode='rb') as zipped:
                    reader = dagger._TraceReader(zipped, guard)
                    yield reader
                    require(reader._complete and reader.lines == 719)
                after, again = old._hash_stream(raw, old.COMPRESSED_MAX, guard)
                require((after, again) == (before, size))
        finally:
            self._trace_open = False

    def scan_candidates(self, *, guard):
        self._open()
        require(not self._scan_attempted)
        self._scan_attempted = True
        with self.open_trace(guard=guard) as trace:
            selector = sampling.RankedStream(trace, role=self.plan.slot, opponent=self.plan.opponent,
                side=self.plan.side, guard=guard)
            accounting = selector.scan(task_sha256=self.task_sha256, crop_sha256=self.crop_sha256)
            windows = selector.candidates()
        # Independent absolute/per-turn limits, without retaining diagnostic rows.
        from learning.task_teacher_counterfactual_pair_v1 import worker
        with self.open_trace(guard=guard) as trace:
            worker._validate_trace_budget(trace,
                dict(task_sha256=self.task_sha256, crop_sha256=self.crop_sha256), guard)
        accounting['absolute_budget_checked'] = True
        return accounting, windows

    def opponent_bytes(self, guard):
        self._open()
        path = source_stage(self.plan) / 'sources' / (self.plan.opponent + '.py')
        raw = old._small(ARTIFACTS, path, old.RESULT_MAX, guard)
        require(old._sha(raw) == self.plan.rival_sha256)
        return raw

    def close(self):
        if self._context is not None:
            require(not self._trace_open)
            self._replay = None
            self._context._active = None
            self._context = None

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *_exc):
        self.close()


def source_stage(plan):
    require(type(plan) is SourcePlan and plan.cohort in completed.fresh_entry.STAGES)
    return ARTIFACTS / completed.fresh_entry.STAGES[plan.cohort]


def read_game(context, source, *, guard):
    require(type(context) is FContext and not context._closed and context._active is None
        and not context._opening and callable(guard))
    plan = context.plan(source)
    stage = source_stage(plan)
    trace = stage / 'traces' / (plan.source_game + '.jsonl.gz')
    context._opening = True
    try:
        with old._open_checked(ARTIFACTS, trace, old.COMPRESSED_MAX, guard) as raw:
            digest, _ = old._hash_stream(raw, old.COMPRESSED_MAX, guard)
            require(digest == plan.trace_sha256)
        # Existing reader hashes the same gzip handle before/after its bounded
        # 128 MiB decode and validates the complete official replay envelope.
        replay = old._read_replay(stage, plan.source_game, plan.replay_sha256, plan.seed, guard)
        result = FGame(context, plan, replay)
        context._active = result
        return result
    finally:
        context._opening = False


def screening_module():
    """An authenticated independent copy; old module globals never change."""
    global _SCREEN
    if _SCREEN is None:
        require(threading.current_thread() is threading.main_thread() and threading.active_count() == 1)
        path = completed.fresh.REPO / SCREEN_RELATIVE
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == SCREEN_SHA256)
        name = '_completed_f_counterfactual_screen_snapshot'
        require(name not in sys.modules)
        module = ModuleType(name)
        module.__file__ = str(path)
        sys.modules[name] = module
        try:
            exec(compile(raw, str(path), 'exec', dont_inherit=True), module.__dict__)
        finally:
            del sys.modules[name]
        module.ROLES = tuple(contract.SLOTS)
        module.SLOTS = dict(contract.SLOTS)
        require(module.WINDOWS == contract.WINDOWS and module.MAX_CASE_COMPARISONS == 12)
        _SCREEN = module
    return _SCREEN


class _ScreenCompatibility:
    """Only the old pure screen's identity shape, never a true source record."""
    def __init__(self, game):
        self._game = game
        self.accounting = None

    @property
    def identity(self):
        plan = self._game.plan
        return dict(source_role=plan.role, batch=2, role=plan.slot,
                    opponent=plan.opponent, side=plan.side)

    def __getattr__(self, name):
        return getattr(self._game, name)

    def scan_candidates(self, *, guard):
        require(self.accounting is None)
        accounting, windows = self._game.scan_candidates(guard=guard)
        require(type(accounting) is dict and type(accounting.get('lines')) is int)
        self.accounting = dict(accounting)
        return accounting, windows


def screen_source(context, source, guard, *, meter=None):
    """Freeze ranked coordinates, then stop at the first official-frame change."""
    from learning.task_student_continuation_v1 import compare, official_step
    module = screening_module()
    if meter is None:
        meter = module.Meter()
    require(type(meter) is module.Meter)
    require(type(context) is FContext and callable(guard)
        and meter.comparisons_reserved == meter.comparisons_begun == meter.comparisons_completed == 0
        and meter.official_transitions_reserved == meter.official_helpers_begun == meter.official_helpers_completed == 0)
    plan = context.plan(source)
    coordinate = source_key(plan.coordinate())
    require(coordinate not in context._screen_attempts and len(context._screen_attempts) < 8)
    # Reserve the source before opening it. Empty/failed sources are consumed;
    # the outer single-attempt coordinator binds this across child processes.
    context._screen_attempts.add(coordinate)
    with read_game(context, source, guard=guard) as game:
        # compare_turn independently matches the recorded callback input against
        # this legal observation. The engine helper receives private state only
        # to reexecute it; it does not feed that state to the actor.
        proxy = _ScreenCompatibility(game)
        item = module.run_case(proxy, role=plan.role,
            opponent=plan.opponent, side=plan.side, guard=guard,
            compare_turn=compare.compare_turn, compare_step=official_step.compare_step, meter=meter)
    candidate = None if item.candidate is None else dict(item.candidate,
        seed=plan.seed, task_sha256=plan.task_sha256, crop_sha256=plan.crop_sha256,
        cohort=plan.cohort, alias=plan.alias, source_game=plan.source_game,
        source_identity_sha256=plan.identity_sha256)
    require(proxy.accounting['lines'] == 719)
    coverage = dict(item.coverage, source_trace_lines=proxy.accounting['lines'],
                    selection_plan=item.plan, selection_plan_sha256=ordered_hash(item.plan),
                    meter=meter.snapshot(), source_identity_sha256=plan.identity_sha256)
    return coverage, candidate, plan


def target_record(game, case, guard):
    """Recheck the frozen ranked target and its deployed public input exactly."""
    plan = game.plan
    require(case['source_identity_sha256'] == plan.identity_sha256)
    accounting, windows = game.scan_candidates(guard=guard)
    require(accounting.get('lines') == 719 and accounting.get('absolute_budget_checked') is True)
    selected, _ = screening_module()._select(windows, slot=plan.slot, opponent=plan.opponent, side=plan.side)
    require(case['rank'] < len(selected[case['window']]))
    chosen = selected[case['window']][case['rank']]
    require(chosen['step'] == case['step'] and chosen['ordinal'] == case['ordinal'])
    record = chosen['document']['records'][case['ordinal']]
    inputs = record['inputs']
    view = observe(observation(game.replay['steps'][case['step']], plan.side), game.replay['configuration'])
    require(first_difference(inputs.get('observation'), view.observation) is None
        and first_difference(inputs.get('configuration'), view.configuration) is None
        and ordered_hash(inputs) == case['input_sha256'] and record.get('source') == 'callback'
        and record.get('chosen') == case['original_choice'] and record.get('teacher_label') == case['teacher_choice']
        and game.replay['info']['seed'] == plan.seed)
    return record
