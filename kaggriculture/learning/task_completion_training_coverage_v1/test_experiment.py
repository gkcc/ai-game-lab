import unittest
from unittest.mock import patch
from . import experiment as e


class ExperimentTest(unittest.TestCase):
    def data(self):
        source,done = [],[]
        for i in range(8):
            source.append(dict(key=f'g{i}',arm='candidate',seed=1,labels_rows=30,labels_sha256='a',trace=dict(sha256='b')))
            done.append(dict(game=f'g{i}',arm='candidate',groups=30,labels_sha256='a',trace_sha256='b'))
        return dict(source_groups=source),dict(passed=True,games=done,groups=240)

    def test_source_segments_and_tamper(self):
        reg,done = self.data()
        segments = e.source_segments(reg,done)
        self.assertEqual([s['offset'] for s in segments],[i*30 for i in range(8)])
        done['games'][0]['labels_sha256']='changed'
        with self.assertRaises(ValueError): e.source_segments(reg,done)

    def test_deadline_and_head(self):
        for value in ('2000-01-01T00:00:00+00:00','2999-01-01T00:00:00'):
            with patch.object(e,'read',return_value={'deadline_utc':value}), self.assertRaises(TimeoutError): e.deadline_check()
        with patch.object(e,'git',side_effect=lambda *a:b'base\n' if a[0]=='rev-parse' else b' M file'), self.assertRaises(ValueError): e.clean_head('base')

    def test_limits(self):
        self.assertEqual(e.LIMITS['max_uncompressed_bytes'],512*1024**2)
        self.assertEqual(e.LIMITS['max_selected'],200)


if __name__ == '__main__': unittest.main()
