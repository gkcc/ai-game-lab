"""Check public evidence bindings and synthetic ledger rejection cases."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest


REPO = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


projector = load_module('alias_projector', REPO/'tools/panel_v3_alias_ledger.py')
evaluation = load_module('frozen_evaluation', REPO/'kaggriculture/learning/task_teacher_merged_eval_v1/evaluation.py')
verifier = load_module('public_verifier', REPO/'tools/verify_panel_v3_alias_ledger.py')


def rows():
    seeds = [11, 22, 33, 44]
    values = evaluation.plans(seeds)
    for index, row in enumerate(values):
        own, other = (index % 13)*17, (index % 7)*19
        margin = own-other
        row.update(passed=True, states=720, action_turns=719, status=['DONE', 'DONE'], agent_errors=0,
                   bank=1000+own, opponent_bank=1000+other, initial_bank=1000,
                   opponent_initial_bank=1000, net_profit=own, opponent_net_profit=other,
                   margin=margin, score=1.0 if margin > 0 else 0.0 if margin < 0 else 0.5)
    return seeds, values


class PublicLedgerTest(unittest.TestCase):
    def test_verified_source_bytes_do_not_reopen_a_path(self):
        module = verifier.load_module('checked_fixture', REPO/'__missing_verified_fixture__.py', b'answer = 42\n')
        self.assertEqual(42, module.answer)

    def test_result_certificate_rejects_synchronized_projection_changes(self):
        cert_raw = (verifier.RESULT/'certificate.json').read_bytes()
        ledger_raw = (verifier.RESULT/'alias-ledger.json').read_bytes()
        stats_raw = (verifier.RESULT/'statistics.json').read_bytes()
        verifier.require_projection_hashes(cert_raw, ledger_raw, stats_raw)
        changed_ledger = ledger_raw+b' '
        changed_cert = json.loads(cert_raw.decode('utf-8-sig'))
        changed_cert['local_ledger_sha256'] = verifier.digest(changed_ledger)
        changed_cert_raw = (json.dumps(changed_cert, ensure_ascii=False, indent=2)+'\n').encode()
        with self.assertRaises(ValueError):
            verifier.require_projection_hashes(changed_cert_raw, changed_ledger, stats_raw)
        crlf = lambda raw: raw.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
        verifier.require_projection_hashes(crlf(cert_raw), crlf(ledger_raw), crlf(stats_raw))

    def test_registered_source_and_certificate_tampering_is_rejected(self):
        names = ('root-decision.json', 'registration.json', 'freeze.json', 'pre-review.json')
        documents = {name: (verifier.REGISTRATION/name).read_bytes() for name in names}
        safety_raw = (verifier.REGISTRATION/'safety-registration.json').read_bytes()
        cert = json.loads((verifier.RESULT/'certificate.json').read_text(encoding='utf-8-sig'))
        evaluator_raw = (REPO/'kaggriculture'/verifier.EVALUATOR_REL).read_bytes()
        projector_raw = (REPO/'tools/panel_v3_alias_ledger.py').read_bytes()
        verifier.require_frozen_sources(cert, documents, safety_raw, evaluator_raw, projector_raw)
        bad_cert = dict(cert, preregistration_commit='0'*40)
        with self.assertRaises(ValueError):
            verifier.require_frozen_sources(bad_cert, documents, safety_raw, evaluator_raw, projector_raw)
        with self.assertRaises(ValueError):
            verifier.require_frozen_sources(cert, documents, safety_raw, evaluator_raw+b'\n# changed', projector_raw)
        with self.assertRaises(ValueError):
            verifier.require_frozen_sources(cert, documents, safety_raw, evaluator_raw, projector_raw+b'\n# changed')
        changed = dict(documents, **{'freeze.json': documents['freeze.json']+b' '})
        with self.assertRaises(ValueError):
            verifier.require_frozen_sources(cert, changed, safety_raw, evaluator_raw, projector_raw)
        changed_review = dict(documents)
        review = json.loads(documents['pre-review.json'].decode('utf-8-sig'))
        review['passed'] = False
        changed_review['pre-review.json'] = (json.dumps(review, ensure_ascii=False, indent=2)+'\n').encode()
        safety = json.loads(safety_raw.decode('utf-8-sig'))
        safety['evidence_sha256']['pre-review.json'] = verifier.digest(changed_review['pre-review.json'])
        changed_safety = (json.dumps(safety, ensure_ascii=False, indent=2)+'\n').encode()
        with self.assertRaises(ValueError):
            verifier.require_frozen_sources(cert, changed_review, changed_safety, evaluator_raw, projector_raw)
        crlf = lambda raw: raw.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
        verifier.require_frozen_sources(cert, {k: crlf(v) for k, v in documents.items()}, crlf(safety_raw),
                                        crlf(evaluator_raw), crlf(projector_raw))

    def test_statistics_match_and_private_fields_are_excluded(self):
        seeds, source = rows()
        expected = evaluation.evaluate(source, seeds)
        for row in source:
            row['seed'] = 9876543210
            row['private_marker'] = 'synthetic-private-marker'
        ledger, actual = projector.project(source, evaluation.evaluate)
        self.assertEqual(expected, actual)
        self.assertEqual(72, len(ledger))
        self.assertTrue(all(set(row) == set(projector.FIELDS) and 'seed' not in row for row in ledger))
        self.assertNotIn('9876543210', json.dumps(ledger))
        self.assertNotIn('synthetic-private-marker', json.dumps(ledger))

    def test_incomplete_duplicate_bad_money_and_unfinished_are_rejected(self):
        _, source = rows()
        invalid = [source[:-1], copy.deepcopy(source), copy.deepcopy(source), copy.deepcopy(source)]
        invalid[1][-1] = copy.deepcopy(invalid[1][0])
        invalid[2][0]['net_profit'] += 1
        invalid[3][0]['status'] = ['DONE', 'ACTIVE']
        for case in invalid:
            with self.subTest(case=case[0]['game']):
                with self.assertRaises(ValueError):
                    projector.project(case, evaluation.evaluate)


if __name__ == '__main__':
    unittest.main()
