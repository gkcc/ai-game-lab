"""Select baseline games from the previously authenticated 32-game campaign.

This adapter reuses the original private evidence chain and official replay
reader. It only changes the offline source coordinate and six-window sampler;
it never plays a game, trains a model, or exposes private source identities.
"""
import gzip

from learning.task_student_continuation_v1 import samples as original_samples
from learning.task_student_continuation_v1 import sources as original
from . import sampling


def _ranked_trace(stage, name, expected, task_sha, crop_sha, *, role, opponent, side, guard):
    path = stage / 'traces' / (name + '.jsonl.gz')
    with original._open_checked(original.fresh.ARTIFACTS, path, original.COMPRESSED_MAX, guard) as raw:
        before, compressed = original._hash_stream(raw, original.COMPRESSED_MAX, guard)
        original._require(before == expected, 'trace digest differs')
        raw.seek(0)
        with gzip.GzipFile(fileobj=raw, mode='rb') as stream:
            ranked = sampling.RankedStream(stream, role=role, opponent=opponent, side=side, guard=guard)
            checked = ranked.scan(task_sha256=task_sha, crop_sha256=crop_sha)
            selected = ranked.selected()
        after, again = original._hash_stream(raw, original.COMPRESSED_MAX, guard)
        original._require((after, again) == (before, compressed), 'trace changed during read')
    return selected, checked, compressed


def read_game(context, *, role, opponent, side, guard):
    """Read one fixed baseline role/opponent/side game, releasing it on close."""
    original._require(isinstance(context, original.PrivateContext) and context.active is False
        and role in original_samples.ROLES and opponent in original_samples.OPPONENTS
        and type(side) is int and side in (0, 1) and callable(guard), 'invalid source coordinate')
    cohort, alias, seed = context.role_map[role]
    manifest = context.manifests[cohort]
    plan = next((p for p in manifest.panel.plans() if p['seed_alias'] == alias
                 and p['opponent'] == opponent and p['side'] == side and p['policy'] == 'baseline'), None)
    original._require(plan is not None and plan['seed'] == seed, 'baseline source differs')
    name = plan['game']
    stage = original.fresh.ARTIFACTS / original.entry.STAGES[cohort]
    context.active = True
    try:
        guard()
        hashes = original._verified_receipt(stage, name, context.acceptances[(cohort, name)], guard)
        row, row_sha = original._receipt(stage, name, 'row', hashes, guard)
        audit, audit_sha = original._receipt(stage, name, 'audit', hashes, guard)
        complete, _ = original._receipt(stage, name, 'complete', hashes, guard)
        original._validate_row_audit(row, audit, complete, row_sha, audit_sha, manifest, plan)
        task_relative, task_sha = manifest.panel.task_model(plan)
        crop_relative, crop_sha = manifest.panel.crop
        task_path = original._model_file(task_relative, task_sha, guard)
        crop_path = original._model_file(crop_relative, crop_sha, guard)
        replay = original._read_replay(stage, name, row['replay_gzip_sha256'], seed, guard)
        selected, checked, compressed = _ranked_trace(stage, name, row['trace_sha256'], task_sha,
            crop_sha, role=role, opponent=opponent, side=side, guard=guard)
        identity = dict(role=role, opponent=opponent, side=side, arm='baseline',
            task_path=task_path, task_sha256=task_sha, crop_path=crop_path, crop_sha256=crop_sha,
            game_limits=manifest.panel.worker_config()['game_limits'],
            trace_decoded_bytes=checked['decoded_bytes'], trace_compressed_bytes=compressed)
        guard()
        return original.PrivateGame(context, replay, selected, identity)
    except BaseException:
        context.active = False
        raise
