"""Synthetic checks that the selected trace is bound to the failure record."""
import copy
import hashlib
import io
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from . import entry


class EntryBindingTests(unittest.TestCase):
    def test_requires_failure_record_and_the_unique_bound_trace(self):
        raw = b'synthetic-failure-record'
        trace = dict(relative_name='stage/traces/toy.jsonl.gz', sha256='a' * 64, bytes=10)
        source = dict(schema='task-pair-toy-smoke-failure-private-evidence-v1',
                      registration_commit=entry.TOY_REGISTRATION, attempted_games=1,
                      accepted_games=0, retry_allowed=False,
                      files={trace['relative_name']: dict(sha256=trace['sha256'], bytes=trace['bytes'])})
        source['files'].update({f'stage/receipt-{i}.json': dict(sha256='b' * 64, bytes=1)
                                for i in range(15)})
        bound = dict(schema='task-pair-capacity-v1-binding', failure_commit=entry.FAILURE_COMMIT,
                     failure_artifacts_verified=16, failure_record_bytes=len(raw),
                     failure_record_sha256=hashlib.sha256(raw).hexdigest(), trace=trace)
        self.assertEqual(entry._validate_binding(bound, raw, source), (source['files'], trace))
        invalid = []
        change = copy.deepcopy(source); change['accepted_games'] = False; invalid.append((bound, change))
        change = copy.deepcopy(source); change['retry_allowed'] = True; invalid.append((bound, change))
        change = copy.deepcopy(source); change['files'].pop('stage/receipt-0.json'); invalid.append((bound, change))
        change = copy.deepcopy(source); change['files'].pop('stage/receipt-0.json')
        change['files']['another.jsonl.gz'] = change['files'][trace['relative_name']]; invalid.append((bound, change))
        change = copy.deepcopy(bound); change['trace']['bytes'] += 1; invalid.append((change, source))
        change = copy.deepcopy(bound); change['failure_record_sha256'] = '0' * 64; invalid.append((change, source))
        change = copy.deepcopy(bound); change['failure_artifacts_verified'] = 16.0; invalid.append((change, source))
        for bad_bound, bad_source in invalid:
            with self.subTest(), self.assertRaises(ValueError):
                entry._validate_binding(bad_bound, raw, bad_source)

    def test_cli_failure_does_not_echo_private_path(self):
        marker = 'NONEXISTENT_PRIVATE_PATH_MARKER'
        with self.assertRaises(SystemExit) as stopped:
            entry.main(['--evidence', marker, '--binding', marker, '--artifact-root', marker,
                        '--result', marker])
        self.assertNotIn(marker, str(stopped.exception))
        self.assertIn('no result accepted', str(stopped.exception))

    def test_unknown_argument_does_not_echo_private_value(self):
        marker = 'PRIVATE_ARGUMENT_MARKER'
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as stopped:
            entry.main(['--unknown', marker])
        self.assertNotIn(marker, str(stopped.exception) + stderr.getvalue())

    def test_result_path_cannot_overwrite_or_extend_source_artifacts(self):
        root = Path.cwd().resolve()
        with self.assertRaises(ValueError):
            entry._result_path(root / 'old-stage' / 'new-result.json', root)
        allowed = entry._result_path(root.parent / 'new-result.json', root)
        self.assertEqual(allowed, (root.parent / 'new-result.json').resolve())


if __name__ == '__main__':
    unittest.main()
