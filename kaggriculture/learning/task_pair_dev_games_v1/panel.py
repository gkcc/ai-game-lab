"""Pure plan and identity bindings for a fixed historical-development panel.

The caller must authenticate the private seed roles, model identities and
source snapshot against an independently frozen registration before use.
"""
from dataclasses import dataclass
from pathlib import PurePosixPath
import re


ALIASES = ('S02', 'S03')
OPPONENTS = ('market', 'round2')
POLICIES = ('baseline', 'control', 'auxiliary')
SIDES = (0, 1)
ENGINE_COMMIT = 'f994f2050d58d84f649329c1e35ce09ada7007c5'
ENGINE_SHA = 'bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e'
GAME_LIMITS = {
    'task_forwards': 4000, 'crop_forwards': 8000, 'neural_input_rows': 160000,
    'gate_evaluations': 1, 'controller_calls': 719, 'teacher_proposals': 1,
    'max_task_forwards_per_turn': 32, 'max_crop_forwards_per_turn': 64,
    'max_neural_input_rows_per_turn': 4096, 'max_controller_calls_per_turn': 1,
}
STORAGE = {'game_write_bytes': 128 * 1024 * 1024, 'small_receipt_bytes': 16384}
_HEX = re.compile(r'[0-9a-f]{64}\Z')
_PLAN = frozenset(('game', 'seed_alias', 'seed', 'opponent', 'side', 'policy',
                   'split', 'parity_kind'))
_JOB = _PLAN | frozenset(('bot_sha256', 'rival_sha256', 'task_checkpoint_sha256'))


def _digest(value):
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise ValueError('Invalid expected identity')
    return value


def _model(value):
    if type(value) is not dict or set(value) != {'path', 'sha256'}:
        raise ValueError('Invalid model binding')
    path = value['path']
    if (type(path) is not str or not path or len(path) > 4096 or '\\' in path
            or ':' in path or any(ord(char) < 32 for char in path)):
        raise ValueError('Invalid model path')
    parsed = PurePosixPath(path)
    if (parsed.is_absolute() or parsed.as_posix() != path or len(parsed.parts) < 2
            or parsed.parts[0] != 'artifacts' or any(p in ('.', '..') for p in parsed.parts)
            or parsed.suffix != '.npz'):
        raise ValueError('Invalid model path')
    return path, _digest(value['sha256'])


@dataclass(frozen=True, repr=False)
class Panel:
    """Immutable snapshots; its plans contain private values and stay local."""
    seeds: tuple
    models: tuple
    crop: tuple
    rivals: tuple
    worker_sha: str

    def __post_init__(self):
        if (type(self.seeds) is not tuple or len(self.seeds) != len(ALIASES)
                or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in self.seeds)
                or len(set(self.seeds)) != len(ALIASES)):
            raise ValueError('Invalid historical seed roles')
        if type(self.models) is not tuple or len(self.models) != len(POLICIES):
            raise ValueError('Invalid task model roles')
        for model in (*self.models, self.crop):
            if type(model) is not tuple or len(model) != 2:
                raise ValueError('Invalid model binding')
            _model({'path': model[0], 'sha256': model[1]})
        if (len({model[0] for model in self.models}) != len(POLICIES)
                or len({model[1] for model in self.models}) != len(POLICIES)):
            raise ValueError('Task model roles must be distinct')
        if type(self.rivals) is not tuple or len(self.rivals) != len(OPPONENTS):
            raise ValueError('Invalid frozen opponent identities')
        for digest in self.rivals:
            _digest(digest)
        _digest(self.worker_sha)

    @classmethod
    def from_private_contract(cls, *, seeds, models, crop, rivals, worker_sha):
        if type(seeds) is not dict or set(seeds) != set(ALIASES):
            raise ValueError('Invalid historical seed roles')
        ordered_seeds = tuple(seeds[alias] for alias in ALIASES)
        if (any(type(seed) is not int or not 0 <= seed < 2**32 for seed in ordered_seeds)
                or len(set(ordered_seeds)) != len(ALIASES)):
            raise ValueError('Invalid historical seed roles')
        if type(models) is not dict or set(models) != set(POLICIES):
            raise ValueError('Invalid task model roles')
        ordered_models = tuple(_model(models[role]) for role in POLICIES)
        if (len({model[0] for model in ordered_models}) != len(POLICIES)
                or len({model[1] for model in ordered_models}) != len(POLICIES)):
            raise ValueError('Task model roles must be distinct')
        if type(rivals) is not dict or set(rivals) != set(OPPONENTS):
            raise ValueError('Invalid frozen opponent identities')
        ordered_rivals = tuple(_digest(rivals[role]) for role in OPPONENTS)
        return cls(ordered_seeds, ordered_models, _model(crop), ordered_rivals,
                   _digest(worker_sha))

    def plans(self):
        return tuple(dict(game=f'dev-{alias}-{opponent}-{side}-{policy}',
                          seed_alias=alias, seed=self.seeds[ALIASES.index(alias)],
                          opponent=opponent, side=side, policy=policy,
                          split='dev', parity_kind=None)
                     for alias in ALIASES for opponent in OPPONENTS
                     for side in SIDES for policy in POLICIES)

    def validate_plan(self, plan, *, job=False):
        if type(plan) is not dict or set(plan) != (_JOB if job else _PLAN):
            raise ValueError('Invalid development plan')
        if (type(plan['seed']) is not int or type(plan['side']) is not int
                or type(plan['game']) is not str or type(plan['seed_alias']) is not str
                or type(plan['opponent']) is not str or type(plan['policy']) is not str
                or type(plan['split']) is not str or plan['parity_kind'] is not None):
            raise ValueError('Invalid development plan')
        requested = {key: plan[key] for key in _PLAN}
        if requested not in self.plans():
            raise ValueError('Plan is outside the development panel')
        if job:
            _digest(plan['bot_sha256'])
            if (plan['rival_sha256'] != self.rivals[OPPONENTS.index(plan['opponent'])]
                    or plan['task_checkpoint_sha256'] != self.models[POLICIES.index(plan['policy'])][1]):
                raise ValueError('Plan identity differs')
        return dict(plan)

    def worker_config(self):
        """A fresh private config for the old worker's development branch only."""
        return dict(validation_seeds=list(self.seeds),
                    crop_checkpoint=self.crop[0], crop_checkpoint_sha256=self.crop[1],
                    engine_commit=ENGINE_COMMIT, engine_sha256=ENGINE_SHA,
                    source_sha256={name + '.py': self.rivals[i]
                                   for i, name in enumerate(OPPONENTS)},
                    game_limits=dict(GAME_LIMITS), storage=dict(STORAGE))

    def task_model(self, plan):
        self.validate_plan({key: plan[key] for key in _PLAN})
        return self.models[POLICIES.index(plan['policy'])]
