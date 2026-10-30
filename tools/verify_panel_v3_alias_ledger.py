"""Recompute the public alias-only panel statistics without private inputs."""
import hashlib
import json
from pathlib import Path
import types


REPO = Path(__file__).resolve().parents[1]
RESULT = REPO/'review/evidence/task-teacher-merged-eval-v3-formal-result-20261030'
REGISTRATION = REPO/'review/evidence/task-teacher-merged-eval-v3-formal-20261030'
PREREG_COMMIT = 'e64f3bad589268a7b1b7c57844a602b595cca2d4'
PARENT_COMMIT = '3a3194707edc5a6f23f8c645b24f396a8fd2b96c'
EVALUATOR_SHA256 = '574a4d9a2f612e624ef48659272d684e4c0b812fae22f15b65e6b2f45b75e6dc'
PROJECTOR_SHA256 = '6409fe65dfa5f332edc52ee1d2edccf8694a08f27be20de9a3d460b0ef88a9b3'
FREEZE_SHA256 = '249d72b0770bb97cb4a1cc66fc14586dd674c778185f6aebf8c50c73fad9cb4d'
REGISTRATION_SHA256 = '293478899d30f47830add190e699b3077d0466806d2276cee0a4828b822bb245'
SAFETY_SHA256 = '7ba98816d4eccb04d563d878aa4606b358612958c1619b0ccedbaecc0fd15c50'
CERTIFICATE_SHA256 = '3f331d0692b5373ad639b7e5f6a01fc942ff96fca15ae02bb13620c706d6b92a'
EVALUATOR_REL = 'learning/task_teacher_merged_eval_v1/evaluation.py'


def load_module(name, path, raw):
    """Execute the exact bytes that were checked, without reopening the path."""
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec', dont_inherit=True), module.__dict__)
    return module


def parse(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('Duplicate JSON field')
            value[key] = item
        return value
    return json.loads(raw.decode('utf-8-sig'), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Invalid number')))


def read(path):
    raw = path.read_bytes()
    return raw, parse(raw)


def same(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


def digest(raw):
    """Accept Git checkouts that convert LF to CRLF, while pinning LF bytes."""
    return hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest()


def require_frozen_sources(cert, documents, safety_raw, evaluator_raw, projector_raw):
    """Check source and registration identities before importing either module.

    This establishes local consistency with pinned hashes; it does not protect
    against replacement of this verifier itself or prove raw game execution.
    """
    names = ('root-decision.json', 'registration.json', 'freeze.json', 'pre-review.json')
    if not isinstance(documents, dict) or set(documents) != set(names):
        raise ValueError('Registration files differ')
    freeze = parse(documents['freeze.json'])
    safety = parse(safety_raw)
    if (not isinstance(cert, dict) or not isinstance(freeze, dict) or not isinstance(safety, dict)
        or cert.get('preregistration_commit') != PREREG_COMMIT or digest(safety_raw) != SAFETY_SHA256
        or freeze.get('parent_commit') != PARENT_COMMIT
        or safety.get('parent_commit') != PARENT_COMMIT
        or safety.get('stage') != 'task-teacher-merged-eval-v3-formal-20261030'
        or not isinstance(safety.get('evidence_sha256'), dict)
        or set(safety['evidence_sha256']) != set(names)
        or any(digest(documents[name]) != safety['evidence_sha256'][name] for name in names)
        or digest(documents['freeze.json']) != FREEZE_SHA256
        or digest(documents['registration.json']) != REGISTRATION_SHA256
        or freeze.get('registration_sha256') != digest(documents['registration.json'])
        or freeze.get('root_decision_sha256') != digest(documents['root-decision.json'])
        or freeze.get('supplemental_runtime_sha256', {}).get(EVALUATOR_REL) !=
           {'local_sha256': EVALUATOR_SHA256, 'public_sha256': EVALUATOR_SHA256}
        or digest(evaluator_raw) != EVALUATOR_SHA256 or digest(projector_raw) != PROJECTOR_SHA256):
        raise ValueError('Registration or source identity differs')


def require_projection_hashes(cert_raw, ledger_raw, stats_raw):
    """Pin the post-run certificate and its two published projections."""
    cert = parse(cert_raw)
    if (not isinstance(cert, dict) or digest(cert_raw) != CERTIFICATE_SHA256
        or cert.get('schema') != 'panel-v3-formal-alias-projection-v1'
        or cert.get('local_ledger_sha256') != digest(ledger_raw)
        or cert.get('local_statistics_sha256') != digest(stats_raw)):
        raise ValueError('Projection hashes differ')
    return cert


def main():
    ledger_raw, ledger = read(RESULT/'alias-ledger.json')
    stats_raw, expected = read(RESULT/'statistics.json')
    cert = require_projection_hashes((RESULT/'certificate.json').read_bytes(), ledger_raw, stats_raw)
    names = ('root-decision.json', 'registration.json', 'freeze.json', 'pre-review.json')
    loaded = {name: read(REGISTRATION/name) for name in names}
    documents = {name: item[0] for name, item in loaded.items()}
    safety_raw, _ = read(REGISTRATION/'safety-registration.json')
    evaluator_path = REPO/'kaggriculture'/EVALUATOR_REL
    projector_path = REPO/'tools/panel_v3_alias_ledger.py'
    evaluator_raw, projector_raw = evaluator_path.read_bytes(), projector_path.read_bytes()
    require_frozen_sources(cert, documents, safety_raw, evaluator_raw, projector_raw)
    projector = load_module('public_alias_projector', projector_path, projector_raw)
    evaluation = load_module('frozen_panel_evaluation', evaluator_path, evaluator_raw)
    projected, recomputed = projector.project(ledger, evaluation.evaluate)
    if not same(projected, ledger) or not same(recomputed, expected):
        raise ValueError('Public statistics differ')
    print('72 alias rows, pinned LF source/registration hashes and statistics match; raw game evidence was not replayed.')


if __name__ == '__main__':
    main()
