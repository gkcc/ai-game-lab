"""Synthetic in-process failure retention checks; no OS child or machine data."""
import json
import unittest
from unittest.mock import patch

from .evidence import Recorder, EvidenceError, PHASES


class EvidenceRetentionTests(unittest.TestCase):
    def test_each_declared_phase_retains_partial_and_primary(self):
        for phase in PHASES:
            with self.subTest(phase=phase):
                recorder = Recorder()
                recorder.checkpoint('launch', {'checks': {'launched': True}, 'roles': ['unresolved']})
                recorder.checkpoint(phase)
                try:
                    raise RuntimeError('synthetic-private-token path=fixture-only pid=123')
                except RuntimeError as exc:
                    recorder.fail(exc)
                receipt = recorder.persist()
                self.assertEqual(receipt['primary_failure'], {'phase': phase, 'error_type': 'RuntimeError', 'reason': phase + '_failed'})
                self.assertTrue(receipt['events'][0]['safe']['checks']['launched'])
                self.assertNotIn('synthetic-private-token', json.dumps(receipt))
                self.assertIn('synthetic-private-token', json.dumps(recorder.private_events))

    def test_cleanup_cannot_replace_primary_or_partial(self):
        recorder = Recorder()
        recorder.checkpoint('handshake', {'roles': ['unresolved']})
        recorder.fail(EvidenceError('nonce_mismatch', 'synthetic nonce details'))
        recorder.checkpoint('cleanup_reap')
        recorder.cleanup_fail(RuntimeError('synthetic reap details'))
        recorder.cleanup_fail(ValueError('synthetic later details'))
        recorder.fail(ValueError('synthetic secondary details'))
        receipt = recorder.persist()
        self.assertEqual(receipt['primary_failure']['reason'], 'nonce_mismatch')
        self.assertEqual(receipt['cleanup_failure']['error_type'], 'RuntimeError')
        self.assertEqual(receipt['events'][0]['safe']['roles'], ['unresolved'])
        self.assertEqual(receipt['private_records'], 4)

    def test_public_projection_rejects_private_fields_and_unregistered_values(self):
        for unsafe in ({'path': 'synthetic'}, {'checks': {'pid': True}}, {'roles': ['fixture-path']},
                       {'image_hashes': {'worker': 'synthetic'}}, {'counters': {'process_id': 123}}):
            with self.subTest(unsafe=unsafe):
                with self.assertRaises(ValueError):
                    Recorder().checkpoint('launch', unsafe)

    def test_checkpoint_is_written_before_later_failure(self):
        saved = []
        recorder = Recorder()
        with patch.object(recorder, '_append', side_effect=lambda value, private=False: saved.append(json.loads(json.dumps(value)))):
            recorder.checkpoint('ownership', {'checks': {'owner_acquired': True}})
            self.assertEqual(saved[-1]['phase'], 'ownership')
            recorder.fail(EvidenceError('ownership_ambiguous'))
        self.assertTrue(saved[0]['safe']['checks']['owner_acquired'])
        self.assertEqual(saved[1]['failure']['reason'], 'ownership_ambiguous')

    def test_recording_failure_does_not_replace_cause(self):
        recorder = Recorder()
        recorder.checkpoint('runtime')
        with patch.object(recorder, '_append', side_effect=OSError('synthetic storage failure')):
            recorder.fail(EvidenceError('runtime_mismatch'))
        receipt = recorder.persist()
        self.assertEqual(receipt['primary_failure']['reason'], 'runtime_mismatch')
        self.assertTrue(receipt['evidence_failure'])


if __name__ == '__main__':
    unittest.main()
