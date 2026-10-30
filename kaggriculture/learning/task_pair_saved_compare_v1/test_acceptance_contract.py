"""In-memory checks for the saved comparison's accepted-game contract."""
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from learning.task_pair_dev_campaign_v1.test_acceptance import fixture, raw, supervision

from . import runner


def documents():
    """Create only synthetic inputs; all digest links are rebuilt after edits."""
    _, plan, bundle = fixture()
    docs = {name: json.loads(data) for name, data in bundle.items()}
    job = {key: docs['row'][key] for key in runner._JOB}
    return plan, docs, job


def read_map(plan, docs, job):
    """Recompute the whole outer receipt chain for the modified documents."""
    sha = lambda value: hashlib.sha256(value).hexdigest()
    game = plan['game']
    docs['complete']['row_sha256'] = sha(raw(docs['row']))
    docs['complete']['certificate_sha256'] = sha(raw(docs['audit']))
    metric_bytes = raw(docs['metrics'])
    docs['metrics_receipt']['metrics_bytes'] = len(metric_bytes)
    docs['metrics_receipt']['metrics_sha256'] = sha(metric_bytes)
    bundle = {name: raw(doc) for name, doc in docs.items()}
    job_raw = raw(job)
    supervision_doc = dict(schema='task-pair-dev-job-supervision-v1', game=game,
                           retry_allowed=False, supervision=supervision())
    supervision_raw = raw(supervision_doc)
    receipt = dict(schema='task-pair-dev-job-acceptance-v1', passed=True, game=game,
                   retry_allowed=False, supervision_sha256=sha(supervision_raw),
                   job_input_sha256=sha(job_raw),
                   receipt_sha256={name: sha(data) for name, data in bundle.items()})
    entries = {
        game + '.acceptance.local.json': (raw(receipt), receipt),
        game + '.supervision.local.json': (supervision_raw, supervision_doc),
        game + '.json': (job_raw, job),
    }
    entries.update({game + suffix: (bundle[name], docs[name])
                    for name, suffix in runner.driver.RECEIPTS.items()})
    return entries


def accepted(plan, docs, job):
    entries = read_map(plan, docs, job)

    def fake_read(path, maximum=1024 * 1024):
        data, document = entries[Path(path).name]
        if len(data) > maximum:
            raise ValueError('Synthetic receipt exceeds maximum')
        return data, document

    with patch.object(runner, 'read_json', side_effect=fake_read):
        return runner.accepted(Path('synthetic-stage'), 'baseline')


class AcceptedContractTests(unittest.TestCase):
    def test_valid_synthetic_bundle_is_accepted(self):
        plan, docs, job = documents()
        self.assertEqual(accepted(plan, docs, job), docs['row'])

    def test_missing_or_mismatched_job_binding_is_rejected(self):
        for change in ('empty', 'missing', 'mismatch'):
            with self.subTest(change=change):
                plan, docs, job = documents()
                if change == 'empty':
                    job.clear()
                elif change == 'missing':
                    del job['bot_sha256']
                else:
                    job['rival_sha256'] = 'f' * 64
                with self.assertRaises(ValueError):
                    accepted(plan, docs, job)

    def test_missing_or_invalid_semantic_digests_are_rejected(self):
        fields = (('semantic_replay_sha256', 'semantic_sha256'),
                  ('semantic_replay_sha256_v2', 'semantic_sha256_v2'))
        for indices in ((0,), (1,), (0, 1)):
            for change in ('missing', 'invalid'):
                with self.subTest(indices=indices, change=change):
                    plan, docs, job = documents()
                    for index in indices:
                        row_field, official_field = fields[index]
                        if change == 'missing':
                            del docs['row'][row_field]
                            del docs['audit']['official'][official_field]
                        else:
                            docs['row'][row_field] = 'z' * 64
                            docs['audit']['official'][official_field] = 'z' * 64
                    with self.assertRaises(ValueError):
                        accepted(plan, docs, job)

    def test_failed_storage_receipt_is_rejected(self):
        for change in ('failed', 'refused'):
            with self.subTest(change=change):
                plan, docs, job = documents()
                if change == 'failed':
                    docs['storage']['passed'] = False
                else:
                    docs['storage']['output']['refused'] = 1
                with self.assertRaises(ValueError):
                    accepted(plan, docs, job)


if __name__ == '__main__':
    unittest.main()
