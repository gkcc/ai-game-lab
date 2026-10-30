"""Map a caller-authenticated training view to the existing batch API.

This module does not authenticate any source file or registration, load data,
or establish that a view really came from R8. The caller must do those things
before passing the view, independently fix the schema and training clusters,
and keep the authenticated data immutable for the entire use of this mapping.
"""

from types import MappingProxyType
import re

import numpy as np

from learning.task_pair_data_preflight_v1 import core
from learning.task_choice_v1.features import OPS


R8_GROUPS = 108310
R8_CANDIDATES = 5546854
R8_CLUSTERS = 8
MAX_GROUP_WIDTH = 256
MAX_CLUSTER = 2**32 - 1
SHA256 = re.compile(r'[0-9a-f]{64}\Z')

_MAPPING = (
    ('shared', 'shared', 'float32'),
    ('candidates', 'candidates', 'float32'),
    ('offsets', 'offsets', 'int64'),
    ('labels', 'labels', 'int64'),
    ('selected_ops', 'selected_ops', 'int8'),
    ('candidate_ops', 'ops', 'int8'),
    ('candidate_identity', 'candidate_identity', 'int16'),
    ('seeds', 'private_clusters', 'int64'),
)
_VIEW_FIELDS = frozenset(('shared', 'candidates', 'offsets', 'labels', 'ops',
                          'private_clusters', 'shared_names', 'feature_names',
                          'schema_digest', 'selected_ops', 'candidate_identity'))


def _adapt(view, *, role, expected_clusters, expected_schema_sha256,
           expected_groups, expected_candidates):
    """Internal common implementation; production sizes are fixed by adapt_r8."""
    if (type(role) is not str or role != 'train'
            or type(view) is not core.RaggedView
            or set(vars(view)) != _VIEW_FIELDS
            or type(expected_schema_sha256) is not str
            or SHA256.fullmatch(expected_schema_sha256) is None
            or view.schema_digest != expected_schema_sha256
            or type(expected_clusters) not in (list, tuple)
            or len(expected_clusters) != R8_CLUSTERS
            or any(type(item) is not int or not 0 <= item <= MAX_CLUSTER
                   for item in expected_clusters)
            or len(set(expected_clusters)) != R8_CLUSTERS
            or type(expected_groups) is not int or expected_groups <= 0
            or type(expected_candidates) is not int or expected_candidates < expected_groups):
        raise ValueError('Training view contract differs')

    # This checks the frozen schema and core ragged shape without copying the
    # large feature arrays. Source/hash/role authentication is still upstream.
    core._structure(view)
    data = {}
    for key, attribute, dtype in _MAPPING:
        array = getattr(view, attribute)
        if (not isinstance(array, np.ndarray) or array.dtype != np.dtype(dtype)
                or array.flags.writeable):
            raise ValueError('Training array type or read-only status differs')
        data[key] = array

    n, m = expected_groups, expected_candidates
    shapes = {'shared': (n, 275), 'candidates': (m, 98), 'offsets': (n + 1,),
              'labels': (n,), 'selected_ops': (n,), 'candidate_ops': (m,),
              'candidate_identity': (m, 3), 'seeds': (n,)}
    if any(data[name].shape != shape for name, shape in shapes.items()):
        raise ValueError('Training array lengths differ')
    if not np.array_equal(np.unique(data['seeds']), np.asarray(sorted(expected_clusters), dtype=np.int64)):
        raise ValueError('Training cluster membership differs')

    offsets, labels = data['offsets'], data['labels']
    widths = np.diff(offsets)
    if (int(offsets[0]) != 0 or int(offsets[-1]) != m
            or np.any(widths < 1) or np.any(widths > MAX_GROUP_WIDTH)
            or np.any(labels < 0) or np.any(labels >= widths)):
        raise ValueError('Training group boundaries differ')
    if (np.any(data['candidate_ops'] < 0) or np.any(data['candidate_ops'] >= len(OPS))
            or np.any(data['selected_ops'] < 0) or np.any(data['selected_ops'] >= len(OPS))
            or not np.array_equal(data['candidate_ops'][offsets[:-1] + labels], data['selected_ops'])):
        raise ValueError('Training selected operations differ')
    # Values retain their exact source object, storage and order. The proxy
    # prevents rebinding keys, but is not a lock against a separate owner.
    return MappingProxyType(data)


def adapt_r8(view, *, role, expected_clusters, expected_schema_sha256):
    """Return a zero-copy, read-only mapping for the fixed authenticated R8 split.

    ``expected_clusters`` and the schema hash must come from separately
    authenticated registration, not from the candidate view itself. The
    caller must authenticate all arrays, their content and their ordering;
    structural validation alone cannot prove provenance or exclude substitution.
    Do not log the returned mapping, cluster values, labels or source features.
    """
    return _adapt(view, role=role, expected_clusters=expected_clusters,
                  expected_schema_sha256=expected_schema_sha256,
                  expected_groups=R8_GROUPS, expected_candidates=R8_CANDIDATES)
