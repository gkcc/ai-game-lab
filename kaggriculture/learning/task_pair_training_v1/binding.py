"""Authenticate the local recheck binding and its private mask snapshots.

The caller must independently fix the expected receipt SHA before this call;
computing an expected value from the same receipt at call time provides no
historical identity. Likewise, the expected public-result SHA must come from an
independently authenticated committed result. This helper does not authenticate
the registration chain, source arrays, source order, or the original process
exit code: the caller must verify each of those separately before use. Equal
mask lengths or values do not independently prove their source or order.

Only supplied bytes and a bounded reader are used. Private masks are returned
in memory; no hashes, identifiers, masks, or features are printed or written.
"""

import hashlib
import json
import re

from learning.task_pair_training_v1 import masks


_MAX_METADATA_BYTES = 65536
_SHA256 = re.compile(r'[0-9a-f]{64}\Z')
_COMMIT = re.compile(r'[0-9a-f]{40}\Z')
_RECEIPT_FIELDS = frozenset((
    'schema', 'registration_commit', 'previous_registration_commit',
    'previous_result_commit', 'fresh_manifest_sha256', 'conditional_source_order',
    'old_snapshot_matches_recomputation', 'fresh_masks_match_recomputation',
    'old_creation_time_proven',
))
_RESULT_FIELDS = frozenset((
    'schema', 'registration_commit', 'previous_registration_commit',
    'old_snapshot_matches_recomputation', 'fresh_masks_match_recomputation',
    'historical_overlap_matches', 'previous_public_projection_matches',
    'old_creation_time_proven', 'checkpoint_weights_loaded', 'model_inference',
    'training', 'new_games',
))


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError('Invalid JSON numeric constant')


def _json_object(raw):
    try:
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique_object,
                           parse_constant=_reject_constant)
    except (UnicodeError, ValueError, TypeError):
        raise ValueError('Invalid JSON metadata') from None
    if type(value) is not dict:
        raise ValueError('Expected JSON object')
    return value


def _identity(value, pattern):
    return type(value) is str and pattern.fullmatch(value) is not None


def _checked_bytes(value):
    return type(value) is bytes and len(value) <= _MAX_METADATA_BYTES


def load_bound_masks(receipt_bytes, manifest_bytes, public_result_bytes, read_file, *,
                     expected_receipt_sha256, expected_public_result_sha256,
                     registration_commit, previous_registration_commit,
                     previous_result_commit, role_lengths):
    """Validate a fixed receipt, committed result and nine local private masks.

    Both expected SHA values and all three commit IDs must come from the
    caller's independently authenticated, previously fixed evidence, not from
    the bytes passed here. ``role_lengths`` must come from authenticated views
    in the independently verified teacher / old / new source order.

    ``read_file(name, maximum_bytes)`` must itself enforce a regular file under
    the authorized directory and read no more than the given cap. The caller
    must separately authenticate registration and array provenance and observe
    the original successful process exit. SHA checks are not filesystem locks.
    """
    if (not all(_checked_bytes(value) for value in
                (receipt_bytes, manifest_bytes, public_result_bytes))
            or not callable(read_file)
            or not _identity(expected_receipt_sha256, _SHA256)
            or not _identity(expected_public_result_sha256, _SHA256)
            or not all(_identity(value, _COMMIT) for value in
                       (registration_commit, previous_registration_commit, previous_result_commit))):
        raise ValueError('Invalid binding inputs')
    if (hashlib.sha256(receipt_bytes).hexdigest() != expected_receipt_sha256
            or hashlib.sha256(public_result_bytes).hexdigest() != expected_public_result_sha256):
        raise ValueError('Fixed receipt or public result identity differs')
    receipt = _json_object(receipt_bytes)
    public = _json_object(public_result_bytes)
    if (set(receipt) != _RECEIPT_FIELDS
            or receipt['schema'] != 'task-pair-mask-recheck-binding-v1'
            or receipt['registration_commit'] != registration_commit
            or receipt['previous_registration_commit'] != previous_registration_commit
            or receipt['previous_result_commit'] != previous_result_commit
            or type(receipt['conditional_source_order']) is not list
            or receipt['conditional_source_order'] != ['old', 'new']
            or not _identity(receipt['fresh_manifest_sha256'], _SHA256)
            or receipt['old_snapshot_matches_recomputation'] is not True
            or receipt['fresh_masks_match_recomputation'] is not True
            or receipt['old_creation_time_proven'] is not False):
        raise ValueError('Private binding contract differs')
    if (set(public) != _RESULT_FIELDS
            or public['schema'] != 'task-pair-mask-recheck-result-v1'
            or public['registration_commit'] != registration_commit
            or public['previous_registration_commit'] != previous_registration_commit
            or any(public[name] is not True for name in (
                'old_snapshot_matches_recomputation', 'fresh_masks_match_recomputation',
                'historical_overlap_matches', 'previous_public_projection_matches'))
            or any(public[name] is not False for name in (
                'old_creation_time_proven', 'checkpoint_weights_loaded', 'model_inference',
                'training', 'new_games'))):
        raise ValueError('Public result contract differs')
    if hashlib.sha256(manifest_bytes).hexdigest() != receipt['fresh_manifest_sha256']:
        raise ValueError('Manifest identity differs from fixed binding')
    result = masks.load_masks(manifest_bytes, read_file,
                              manifest_sha256=receipt['fresh_manifest_sha256'],
                              registration_commit=registration_commit,
                              role_lengths=role_lengths)
    # Avoid accidental changes between source binding and later evaluation;
    # NumPy read-only flags are not a security boundary against the caller.
    for shards in result.values():
        for shard in shards:
            for mask in shard.values():
                mask.setflags(write=False)
    return result
