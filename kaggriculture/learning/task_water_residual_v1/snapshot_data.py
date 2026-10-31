"""Decode one authenticated transfer split without unbounded NPZ allocation."""
import hashlib
import io
import zipfile

import numpy as np

from learning.task_teacher_transfer_dataset_v1 import dataset, experiment as transfer


def read_split(raw, expected_sha, split, *, guard):
    """Return immutable validated groups from the exact byte snapshot supplied.

    Expected digests must come from a separately authenticated private binding;
    a hash computed from ``raw`` by the caller is not an independent identity.
    """
    if (type(raw) is not bytes or not 0 < len(raw) <= transfer.FINAL_BYTES
            or type(expected_sha) is not str or len(expected_sha) != 64
            or hashlib.sha256(raw).hexdigest() != expected_sha
            or split not in ('train', 'dev') or not callable(guard)):
        raise ValueError('Transfer snapshot identity differs')
    guard()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        if ({item.filename for item in infos} != {name + '.npy' for name in transfer.SPLIT_FIELDS}
                or len(infos) != len(transfer.SPLIT_FIELDS)
                or sum(item.file_size for item in infos) > transfer.MAX_GROUPS * 256 * 373 * 4 + 2 * 1024**2):
            raise ValueError('Transfer snapshot archive differs')
        transfer._validate_array_headers(archive, infos, transfer._array_dtypes(split=True), transfer.MAX_GROUPS)
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        if set(archive.files) != transfer.SPLIT_FIELDS:
            raise ValueError('Transfer snapshot fields differ')
        arrays = {key: archive[key] for key in transfer.SPLIT_FIELDS}
    offsets, features = arrays['offsets'], arrays['features']
    n = len(arrays['target'])
    if (offsets[0] != 0 or offsets[-1] != len(features) or not np.isfinite(features).all()):
        raise ValueError('Transfer snapshot offsets or features differ')
    result, seen = [], set()
    for i in range(n):
        guard()
        start, end = int(offsets[i]), int(offsets[i + 1])
        role, opponent, side, operation = (int(arrays[k][i]) for k in ('role', 'opponent', 'side', 'operation'))
        step, teacher, terminal, target = (int(arrays[k][i]) for k in
                                            ('step', 'teacher_ordinal', 'terminal_ordinal', 'target'))
        if (not 0 <= role < len(transfer.ROLES) or not 0 <= opponent < len(transfer.OPPONENTS)
                or side not in (0, 1) or not 0 <= operation < len(transfer.OPS)
                or not 0 <= start < end <= len(features) or end - start > 256
                or not 0 <= target < end - start):
            raise ValueError('Transfer snapshot group coordinates differ')
        role = transfer.ROLES[role]
        if role not in (dataset.TRAIN_ROLES if split == 'train' else dataset.DEV_ROLES):
            raise ValueError('Transfer snapshot split role differs')
        raw_group = np.frombuffer(features[start:end].tobytes(order='C'), dtype=np.float32).reshape(end - start, 373)
        item = dataset.Group(role, transfer.OPPONENTS[opponent], side, step, teacher, terminal,
                             transfer.OPS[operation], raw_group, target)
        item = dataset._checked_copy(item)
        coordinate = (role, item.opponent, side, step, teacher)
        if coordinate in seen:
            raise ValueError('Transfer snapshot group repeated')
        seen.add(coordinate)
        result.append(item)
    return result
