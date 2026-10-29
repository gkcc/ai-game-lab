"""Synthetic source binding and bounded reads; never open real artifacts."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from learning.task_dagger_once_v1.policy import LIMITS
from . import core, run
from .test_core import CROP_SHA, TASK_SHA, FakeAgent, fixture


class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.replay, cls.trace = fixture()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.replay_path, self.trace_path = self.root/'replay.gz', self.root/'trace.gz'
        replay = json.dumps(self.replay, separators=(',', ':'), allow_nan=False).encode()
        trace = b''.join((json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n').encode()
                         for row in self.trace)
        self.replay_path.write_bytes(gzip.compress(replay, mtime=0))
        self.trace_path.write_bytes(gzip.compress(trace, mtime=0))
        self.replay_sha = hashlib.sha256(self.replay_path.read_bytes()).hexdigest()
        self.trace_sha = hashlib.sha256(self.trace_path.read_bytes()).hexdigest()
        self.files = {'replay': self.replay_sha, 'trace': self.trace_sha}
        self.cert = {'schema': 'bounded-development-smoke-certificate-v1',
                     'source_commit': run.SOURCE_COMMIT, 'registration_commit': run.SOURCE_REGISTRATION,
                     'stage_kind': 'public-development-not-holdout', 'accepted_games': 1,
                     'replay_states': 720, 'action_turns': 719, 'final_status': ['DONE', 'DONE'],
                     'agent_errors_reported': 0, 'primary_replay_transitions': 719, 'extra_replay_transitions': 0,
                     'local_artifact_sha256': dict(self.files)}
        self.summary = {'passed': True, 'game': run.SMOKE_KEY, 'run_kind': 'development_smoke',
                        'complete_games': 1, 'states': 720, 'action_turns': 719,
                        'primary_replay_transitions': 719, 'agent_errors': 0,
                        'local_artifact_sha256': dict(self.files)}

    def test_bound_synthetic_sources_and_one_pass_decode(self):
        run._bound_sources(self.cert, self.summary, self.replay_sha, self.trace_sha)
        replay = run._read_replay(self.replay_path, self.replay_sha)
        report = core.compare(replay, run._trace_lines(self.trace_path, self.trace_sha),
                              FakeAgent, LIMITS, TASK_SHA, CROP_SHA)
        self.assertTrue(report['passed'])
        self.assertEqual(report['counts']['sampled_turns'], 32)

    def test_same_seat_same_step_other_episode_stops_before_any_actor(self):
        other = deepcopy(self.trace)
        other[5]['records'][0]['inputs']['planned']['synthetic'] = 999
        raw = b''.join((json.dumps(row, separators=(',', ':'))+'\n').encode() for row in other)
        self.trace_path.write_bytes(gzip.compress(raw, mtime=0))
        other_sha = hashlib.sha256(self.trace_path.read_bytes()).hexdigest()
        with self.assertRaises(ValueError):
            run._bound_sources(self.cert, self.summary, self.replay_sha, other_sha)
        calls = []
        with self.assertRaises(ValueError):
            core.compare(self.replay, run._trace_lines(self.trace_path, self.trace_sha),
                         lambda: calls.append(True), LIMITS, TASK_SHA, CROP_SHA)
        self.assertEqual(calls, [])

    def test_wrong_game_or_source_digest_is_rejected(self):
        wrong = dict(self.summary, game='holdout-S01-round2-1-merged')
        with self.assertRaises(ValueError):
            run._bound_sources(self.cert, wrong, self.replay_sha, self.trace_sha)
        with self.assertRaises(ValueError):
            run._bound_sources(self.cert, self.summary, '0'*64, self.trace_sha)

    def test_compressed_decoded_and_line_limits_fail_closed(self):
        for key in ('trace_compressed_bytes', 'trace_decoded_bytes', 'trace_line_bytes'):
            with self.subTest(limit=key), patch.dict(run.INPUT_LIMITS, {key: 32}):
                with self.assertRaises(ValueError):
                    list(run._trace_lines(self.trace_path, self.trace_sha))
        with patch.dict(run.INPUT_LIMITS, {'replay_decoded_bytes': 32}):
            with self.assertRaises(ValueError):
                run._read_replay(self.replay_path, self.replay_sha)

    def test_invalid_entry_hides_arguments_and_never_starts_source_read(self):
        marker = 'SYNTHETIC_PRIVATE_MARKER'
        command = [sys.executable, '-B', '-I', str(Path(run.__file__).resolve()),
                   '--source-commit', marker, '--registration-commit', marker, '--scratch', marker]
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn(marker, result.stdout + result.stderr)
        self.assertNotIn('Traceback', result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
