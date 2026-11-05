"""F-source adapters for the frozen official counterfactual pair Worker.

This module is not a paired-experiment coordinator. Its caller must separately
freeze the screened cases, authenticate every preceding phase receipt and own
the process/resource budget. No old pair experiment or private path is used.
"""
from contextlib import contextmanager
from pathlib import Path
import re
import threading

from learning.task_teacher_counterfactual_pair_v1 import worker as legacy
from . import contract, sources


CASE_KEYS = legacy.CASE_KEYS | frozenset(('cohort', 'alias', 'source_game', 'source_identity_sha256'))
_LOCK = threading.Lock()


def require(value):
    if not value:
        raise ValueError('F counterfactual phase contract differs')


def source_for_case(case):
    require(type(case) is dict)
    return dict(role=case.get('role'), opponent=case.get('opponent'), side=case.get('side'), policy='baseline')


def validate_case(case, context):
    require(type(case) is dict and set(case) == CASE_KEYS and type(context) is sources.FContext)
    plan = context.plan(source_for_case(case))
    require(all(type(case.get(key)) is type(expected) and case[key] == expected for key, expected in
        dict(role=plan.role, slot=plan.slot, opponent=plan.opponent, side=plan.side,
             cohort=plan.cohort, alias=plan.alias, source_game=plan.source_game,
             source_identity_sha256=plan.identity_sha256, seed=plan.seed,
             task_sha256=plan.task_sha256, crop_sha256=plan.crop_sha256).items())
        and type(case['window']) is int and 0 <= case['window'] < len(contract.WINDOWS)
        and type(case['rank']) is int and 0 <= case['rank'] < 2
        and type(case['step']) is int and contract.WINDOWS[case['window']][0]
            <= case['step'] <= contract.WINDOWS[case['window']][1]
        and type(case['ordinal']) is int and 0 <= case['ordinal'] < 32
        and type(case['input_sha256']) is str and re.fullmatch(r'[0-9a-f]{64}', case['input_sha256']) is not None
        and all(type(case[key]) is str and case[key] for key in ('original_choice', 'teacher_choice'))
        and case['original_choice'] != case['teacher_choice'])
    return dict(case)


def verify_bundle(stage, case, guard, context, original):
    """Keep the old actor contract while binding F models and the opponent."""
    validate_case(case, context)
    value = original(stage, case, guard)
    plan = context.plan(source_for_case(case))
    require(value['limits'] == sources.verify_limits()
        and value['task_checkpoint'] == legacy._relative_model(context._task_path)
        and value['crop_checkpoint'] == legacy._relative_model(context._crop_path))
    for name, module in (('actor', legacy.actor), ('gate', legacy.gate)):
        require(value[name + '_source_sha256'] == legacy._sha_file(Path(module.__file__), guard, legacy.MAX_SMALL))
    require(legacy._sha_file(stage / 'opponents/opponent.py', guard, legacy.MAX_SMALL) == plan.rival_sha256)
    return value


def require_products(stage, phase):
    """Data prerequisites only; the coordinator verifies the phase-receipt chain."""
    modes = []
    if phase in ('control_audit', 'control_verify', 'intervene', 'intervene_audit', 'pair_verify'):
        modes.append('control')
    if phase in ('intervene_audit', 'pair_verify'):
        modes.append('intervene')
    for mode in modes:
        for relative in (f'games/{mode}/replay.json', f'games/{mode}/result.json',
                         f'traces/{mode}.jsonl.gz', f'events/{mode}.target.local.json',
                         f'events/{mode}.complete.local.json'):
            path = stage / relative
            require(path.is_file() and not path.is_symlink() and path.resolve(strict=True) == path)


@contextmanager
def adapted_worker(context):
    """Bind only source-related seams, serially, and restore every exit path."""
    require(type(context) is sources.FContext
        and threading.current_thread() is threading.main_thread() and threading.active_count() == 1
        and _LOCK.acquire(blocking=False))
    names = ('_case', '_source_context', '_game', '_scan_target', '_opponent', '_verify_bundle')
    saved = {name:getattr(legacy, name) for name in names}
    try:
        require(sources.verify_limits() == legacy.sources.dagger.GAME_LIMITS
            and tuple(legacy.PHASES) == tuple(contract.PHASES))
        def source_context(case, guard):
            validate_case(case, context)
            require(context.verify_assets(guard))
            return context
        def game(actual, case, guard):
            require(actual is context)
            validate_case(case, context)
            return sources.read_game(context, source_for_case(case), guard=guard)
        def target(actual, case, guard):
            validate_case(case, context)
            require(actual.plan == context.plan(source_for_case(case)))
            return sources.target_record(actual, case, guard)
        def opponent(actual, case, guard):
            validate_case(case, context)
            require(actual.plan == context.plan(source_for_case(case)))
            return actual.opponent_bytes(guard)
        legacy._case = lambda case: validate_case(case, context)
        legacy._source_context, legacy._game = source_context, game
        legacy._scan_target, legacy._opponent = target, opponent
        legacy._verify_bundle = lambda stage, case, guard: verify_bundle(
            stage, case, guard, context, saved['_verify_bundle'])
        yield legacy
    finally:
        for name, value in saved.items():
            setattr(legacy, name, value)
        _LOCK.release()


def run_phase(*, context, stage, phase, case, guard):
    require(__debug__ and phase in contract.PHASES and callable(guard))
    value = validate_case(case, context)
    stage = Path(stage).resolve(strict=True)
    require(stage.is_dir() and stage.is_relative_to((sources.GAME / 'artifacts').resolve(strict=True)))
    require_products(stage, phase)
    with adapted_worker(context) as worker:
        return worker.run_phase(stage=stage, phase=phase, case=value, guard=guard)
