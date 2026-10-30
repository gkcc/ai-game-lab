"""Verify public historical Git bytes before binding private masks.

This module is not a training entry. The caller must independently fix the
future registration HEAD and private receipt digest, authenticate the complete
future registration and all project sources, and confirm the original recheck
exit separately. Git and filesystem comparisons are not locks or timestamps.
"""

from dataclasses import dataclass
import hashlib
import re
import subprocess

from learning.task_pair_data_preflight_v1 import run as preflight
from learning.task_pair_mask_recheck_v1 import run as recheck
from learning.task_pair_training_v1 import binding
from learning.task_worker_ranking_v2 import run as v2


MASK_REGISTRATION_COMMIT = '8b96b695192e64f48ead122961d03bd818bae612'
MASK_RESULT_COMMIT = 'd5044c89a90c017975886959031bcf17f6724a15'
MASK_RESULT_REL = 'review/evidence/task-pair-mask-recheck-v1-result-20261030/result.json'
_PATHS = (recheck.OLD_REG_REL, recheck.OLD_RESULT_REL, recheck.NEW_REG_REL, MASK_RESULT_REL)
_HEX_COMMIT = re.compile(r'[0-9a-f]{40}\Z')
_HEX_SHA = re.compile(r'[0-9a-f]{64}\Z')
_OLD_RESULT_KEYS = frozenset(('schema', 'registration_commit', 'coverage', 'unmatched_coverage',
                              'conditional_sources', 'overlap', 'label_conflict',
                              'all_groups_primary_ready', 'unmatched_primary_ready'))


@dataclass(frozen=True, repr=False)
class BoundHistory:
    """Private in-memory binding; never print the receipt identity or masks."""
    _registration_commit: str
    _previous_registration_commit: str
    _previous_result_commit: str
    _receipt_sha256: str
    _public_result_sha256: str
    _public_result_bytes: bytes

    def __repr__(self):
        return '<PrivateHistoricalBinding>'

    def load_masks(self, receipt_bytes, manifest_bytes, read_file, *, role_lengths):
        """Bind private bytes after the caller has authenticated source lengths.

        The receipt hash was fixed independently before the history check;
        lengths and old/new order must come from freshly authenticated sources.
        The callback must enforce its own direct-file and read-size limits.
        """
        return binding.load_bound_masks(
            receipt_bytes, manifest_bytes, self._public_result_bytes, read_file,
            expected_receipt_sha256=self._receipt_sha256,
            expected_public_result_sha256=self._public_result_sha256,
            registration_commit=self._registration_commit,
            previous_registration_commit=self._previous_registration_commit,
            previous_result_commit=self._previous_result_commit, role_lengths=role_lengths)


def current_head():
    """Read the current Git HEAD, without accepting abbreviations."""
    raw = subprocess.check_output(['git', '-C', str(recheck.REPO), 'rev-parse', '--verify', 'HEAD'],
                                  stderr=subprocess.DEVNULL, timeout=15)
    if len(raw) != 41 or raw[-1:] != b'\n':
        raise ValueError('Invalid current Git HEAD')
    try:
        value = raw[:-1].decode('ascii')
    except UnicodeError:
        raise ValueError('Invalid current Git HEAD') from None
    if _HEX_COMMIT.fullmatch(value) is None:
        raise ValueError('Invalid current Git HEAD')
    return value


def _sources(value):
    if (type(value) is not dict or not 1 <= len(value) <= 256
            or any(type(name) is not str or not name or len(name) > 512
                   or name.startswith(('/', '-')) or '\\' in name or ':' in name
                   or '..' in name.split('/') or type(digest) is not str
                   or _HEX_SHA.fullmatch(digest) is None for name, digest in value.items())):
        raise ValueError('Invalid historical source identities')
    return value


def _registered_contract(value, contract):
    return (type(value) is dict and set(value) == set(contract) | {'source_sha256'}
            and all(v2.strict_contract_equal(value[name], expected)
                    for name, expected in contract.items()))


def _validate_public(old_registration, old_result, mask_registration, mask_result,
                     *, mask_registration_commit):
    old = v2.parse_registration(old_registration)
    previous = v2.parse_registration(old_result)
    current = v2.parse_registration(mask_registration)
    # Use the same strict JSON parser that the actual private binding will use.
    result = binding._json_object(mask_result)
    if not _registered_contract(old, preflight.CONTRACT) or not _registered_contract(current, recheck.CONTRACT):
        raise ValueError('Historical registration contract differs')
    old_sources, current_sources = _sources(old['source_sha256']), _sources(current['source_sha256'])
    if (not set(old_sources).isdisjoint(recheck.EXTRA_SOURCE_PATHS)
            or set(current_sources) != set(old_sources) | set(recheck.EXTRA_SOURCE_PATHS)
            or any(current_sources[name] != digest for name, digest in old_sources.items())):
        raise ValueError('Historical source registration differs')
    if (type(previous) is not dict or set(previous) != _OLD_RESULT_KEYS
            or previous['schema'] != 'task-pair-data-preflight-result-v1'
            or previous['registration_commit'] != recheck.OLD_COMMIT
            or previous['label_conflict'] != 'none'
            or previous['all_groups_primary_ready'] is not True
            or previous['unmatched_primary_ready'] is not True
            or any(type(previous[key]) is not dict for key in
                   ('coverage', 'unmatched_coverage', 'conditional_sources', 'overlap'))):
        raise ValueError('Historical preflight result differs')
    if (type(result) is not dict or set(result) != binding._RESULT_FIELDS
            or result['schema'] != 'task-pair-mask-recheck-result-v1'
            or result['registration_commit'] != mask_registration_commit
            or result['previous_registration_commit'] != recheck.OLD_COMMIT
            or any(result[name] is not True for name in (
                'old_snapshot_matches_recomputation', 'fresh_masks_match_recomputation',
                'historical_overlap_matches', 'previous_public_projection_matches'))
            or any(result[name] is not False for name in (
                'old_creation_time_proven', 'checkpoint_weights_loaded', 'model_inference',
                'training', 'new_games'))):
        raise ValueError('Historical recheck result differs')


def bind_history(*, expected_head, preflight_registration_commit, preflight_result_commit,
                 mask_registration_commit, mask_result_commit, public_bytes,
                 expected_receipt_sha256):
    """Bind four fixed public historical files to a separately fixed HEAD.

    ``public_bytes`` must be an exact mapping of the four public relative
    paths to bytes, independently acquired from authenticated prior evidence.
    The private receipt SHA must already have been fixed separately; this
    function never reads or computes it from the receipt. The supplied HEAD
    must be independently identified as the intended future registration.
    Source files, data, masks, checkpoint and process exit are not authenticated
    here. No evidence of old file creation time or statistical independence is
    implied. Use the returned object's ``load_masks`` only after authenticating
    source identities and role lengths.
    """
    commits = (preflight_registration_commit, preflight_result_commit,
               mask_registration_commit, mask_result_commit, expected_head)
    if (any(type(c) is not str or _HEX_COMMIT.fullmatch(c) is None for c in commits)
            or commits[:4] != (recheck.OLD_COMMIT, recheck.PUBLISHED_COMMIT,
                               MASK_REGISTRATION_COMMIT, MASK_RESULT_COMMIT)
            or len(set(commits)) != len(commits)
            or type(expected_receipt_sha256) is not str
            or _HEX_SHA.fullmatch(expected_receipt_sha256) is None
            or type(public_bytes) is not dict or set(public_bytes) != set(_PATHS)
            or any(type(raw) is not bytes or len(raw) > recheck.METADATA_LIMIT
                   for raw in public_bytes.values())
            or len(public_bytes[MASK_RESULT_REL]) > binding._MAX_METADATA_BYTES):
        raise ValueError('Invalid fixed historical inputs')
    # Snapshot the four immutable byte references so a caller's later dict
    # mutation cannot substitute unchecked bytes after the Git comparisons.
    public = {path: public_bytes[path] for path in _PATHS}
    if current_head() != expected_head:
        raise ValueError('Expected registration is not HEAD')
    if any(not recheck.is_ancestor(parent, child) for parent, child in zip(commits, commits[1:])):
        raise ValueError('Historical commit order differs')
    allowed = frozenset(_PATHS)
    for commit, path in zip(commits[:4], _PATHS):
        recorded = recheck.git_blob(commit, path, allowed=allowed)
        at_head = recheck.git_blob(expected_head, path, allowed=allowed)
        workspace = recheck.bounded_file(recheck.REPO, path, recheck.METADATA_LIMIT)
        if not public[path] == recorded == at_head == workspace:
            raise ValueError('Historical Git and worktree bytes differ')
    _validate_public(*(public[path] for path in _PATHS),
                     mask_registration_commit=mask_registration_commit)
    if current_head() != expected_head:
        raise ValueError('Git HEAD changed during verification')
    result = public[MASK_RESULT_REL]
    return BoundHistory(mask_registration_commit, preflight_registration_commit,
                        preflight_result_commit, expected_receipt_sha256,
                        hashlib.sha256(result).hexdigest(), result)
