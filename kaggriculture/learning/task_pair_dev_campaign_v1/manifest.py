"""Private, immutable campaign identity snapshot; no IO or authentication.

The caller must first authenticate the registration and the independently
fixed private contract/source identities, including duplicate-key rejection in
any serialized document.  Do not print or publish this object or its contents.
"""
from dataclasses import dataclass
from pathlib import PurePosixPath
import re

from learning.task_pair_dev_games_v1.panel import ALIASES, OPPONENTS, POLICIES, Panel, _digest


_RESERVED = re.compile(r'(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])\Z', re.I)


def _relative_path(path, suffix, *, artifacts=False):
    if (type(path) is not str or not path or len(path) > 4096
            or any(ord(char) < 32 or ord(char) == 127 or char in '\\:<>"|?*' for char in path)):
        raise ValueError('Invalid relative path')
    parsed = PurePosixPath(path)
    if (parsed.is_absolute() or parsed.as_posix() != path or len(parsed.parts) < 2
            or any(part in ('.', '..') or part.rstrip(' .') != part
                   or _RESERVED.fullmatch(part.split('.', 1)[0]) for part in parsed.parts)
            or parsed.suffix != suffix or (artifacts and parsed.parts[0] != 'artifacts')):
        raise ValueError('Invalid relative path')
    return path


def _source(value):
    if type(value) is not dict or set(value) != {'path', 'sha256'}:
        raise ValueError('Invalid source identity')
    path = _relative_path(value['path'], '.py')
    return path, _digest(value['sha256'])


@dataclass(frozen=True, repr=False)
class PrivateManifest:
    """A small private snapshot. Repr deliberately reveals no attributes."""
    panel: Panel
    opponent_sources: tuple

    def __post_init__(self):
        if type(self.panel) is not Panel or type(self.opponent_sources) is not tuple:
            raise ValueError('Invalid private manifest')
        if len(self.opponent_sources) != len(OPPONENTS):
            raise ValueError('Invalid opponent roles')
        sources = []
        for i, item in enumerate(self.opponent_sources):
            if type(item) is not tuple or len(item) != 2:
                raise ValueError('Invalid source identity')
            source = _source({'path': item[0], 'sha256': item[1]})
            if source[1] != self.panel.rivals[i]:
                raise ValueError('Opponent identity differs from panel')
            sources.append(source)
        if len({path.casefold() for path, _ in sources}) != len(sources):
            raise ValueError('Duplicate opponent source paths')
        if len({digest for _, digest in sources}) != len(sources):
            raise ValueError('Duplicate opponent source identities')
        all_models = (*self.panel.models, self.panel.crop)
        for path, _ in all_models:
            _relative_path(path, '.npz', artifacts=True)
        if len({path.casefold() for path, _ in all_models}) != len(all_models):
            raise ValueError('Duplicate model paths')
        if len({digest for _, digest in all_models}) != len(all_models):
            raise ValueError('Duplicate model identities')

    def opponent(self, role):
        if type(role) is not str or role not in OPPONENTS:
            raise ValueError('Invalid opponent role')
        return self.opponent_sources[OPPONENTS.index(role)]


def build_manifest(*, seeds, models, crop, opponents, worker_sha256):
    """Create a snapshot from externally authenticated, prebound identities.

    ``seeds``: exact S02/S03 role->integer mapping; ``models``: exact
    baseline/control/auxiliary-> {path,sha256}; ``crop``: {path,sha256};
    ``opponents``: exact market/round2-> {path,sha256}; ``worker_sha256``:
    independently fixed old worker digest.  The returned object and all plan
    identities are private; source paths are relative to the game project.

    This validates structure, not actual file contents, Git registration,
    model schema, symlink targets or whether the expected SHA came from an
    independent precommitment.  Those remain the caller's responsibility.
    """
    if type(seeds) is not dict or set(seeds) != set(ALIASES):
        raise ValueError('Invalid seed roles')
    if type(models) is not dict or set(models) != set(POLICIES):
        raise ValueError('Invalid task model roles')
    if type(opponents) is not dict or set(opponents) != set(OPPONENTS):
        raise ValueError('Invalid opponent roles')
    sources = tuple(_source(opponents[role]) for role in OPPONENTS)
    panel = Panel.from_private_contract(seeds=seeds, models=models, crop=crop,
        rivals={role: sources[i][1] for i, role in enumerate(OPPONENTS)},
        worker_sha=worker_sha256)
    return PrivateManifest(panel=panel, opponent_sources=sources)
