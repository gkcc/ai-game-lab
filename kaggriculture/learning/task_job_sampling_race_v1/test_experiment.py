import unittest
from unittest.mock import patch,mock_open
from pathlib import Path
from . import experiment as e


class ExperimentTest(unittest.TestCase):
    def test_parent_failure_is_required(self):
        parent=dict(passed=False,panel_completed=False,no_economic_comparison=True,no_retries=True)
        e.validate_parent(parent)
        with self.assertRaises(ValueError): e.validate_parent({**parent,'panel_completed':True})

    def test_deadline_and_clean_head(self):
        for value in ('2000-01-01T00:00:00+00:00','2999-01-01T00:00:00'):
            with patch.object(e,'read',return_value={'deadline_utc':value}),self.assertRaises(TimeoutError): e.deadline_check()
        with patch.object(e,'git',side_effect=lambda *args:b'base\n' if args[0]=='rev-parse' else b' M file'),self.assertRaises(ValueError): e.clean_head('base')

    def test_partial_marker_fails_without_launch_and_no_retry(self):
        records={}
        def dump(*args,**kwargs): records['started.json']=True; raise OSError('partial')
        with patch.object(e,'ready'),patch.object(e,'sha',return_value='h'),patch.object(Path,'exists',autospec=True,side_effect=lambda p:Path(p).name in records), \
             patch.object(Path,'open',mock_open()),patch.object(e.json,'dump',side_effect=dump), \
             patch.object(e,'write_new',side_effect=lambda p,v:records.update({Path(p).name:v})):
            with self.assertRaises(RuntimeError): e.run(api_factory=lambda:None,execute=lambda *a:self.fail('must not start'))
            self.assertFalse(records['result.json']['marker']['complete'])
            with self.assertRaises(ValueError): e.run(api_factory=lambda:None,execute=lambda *a:self.fail('must not retry'))

    def test_post_execution_identity_changes_fail_with_receipt(self):
        supervision=dict(passed=True,assigned=True,resumed=True,launcher_reaped=True,job_members_zero=True,exit_code=0,
            error_kind=None,cleanup_error_kind=None,rss_complete_samples=1,rss_sampling_retries=1,rss_discarded_samples=1)
        for case in ('success','marker','source','commit'):
            with self.subTest(case=case):
                records={}; state={'changed':False}
                def digest(path):
                    name=Path(path).name
                    return 'changed' if state['changed'] and case=='marker' and name=='started.json' else name+'-hash'
                def ready(out):
                    if state['changed'] and case=='source': raise ValueError('source changed')
                def commit(out):
                    if state['changed'] and case=='commit': raise ValueError('commit changed')
                def execute(*args):
                    state['changed']=True
                    return dict(passed=True,supervision=supervision,artificial_disappearances_injected=1)
                with patch.object(e,'ready',side_effect=ready),patch.object(e,'verify_commit',side_effect=commit), \
                     patch.object(e,'deadline_check'),patch.object(e,'sha',side_effect=digest), \
                     patch.object(e,'read',return_value={'commit':'committed'}),patch.object(Path,'exists',return_value=False), \
                     patch.object(Path,'open',mock_open()),patch.object(e,'write_new',side_effect=lambda p,v:records.update({Path(p).name:v})):
                    if case=='success':
                        result=e.run(api_factory=lambda:None,execute=execute)
                        self.assertEqual(result['identities']['commit'],'committed')
                        self.assertEqual(result['identities']['marker_sha256'],'started.json-hash')
                    else:
                        with self.assertRaises(RuntimeError): e.run(api_factory=lambda:None,execute=execute)
                        self.assertFalse(records['result.json']['passed'])
                        self.assertEqual(records['result.json']['result']['supervision'],supervision)

    def test_deleted_marker_result_blocks_retry(self):
        records={}; calls=[]
        supervision=dict(passed=True,assigned=True,resumed=True,launcher_reaped=True,job_members_zero=True,exit_code=0,
            error_kind=None,cleanup_error_kind=None,rss_complete_samples=1,rss_sampling_retries=1,rss_discarded_samples=1)
        def dump(*args,**kwargs): records['started.json']=True
        def digest(path):
            if Path(path).name=='started.json' and 'started.json' not in records: raise FileNotFoundError()
            return Path(path).name+'-hash'
        def execute(*args):
            calls.append(1); records.pop('started.json')
            return dict(passed=True,supervision=supervision,artificial_disappearances_injected=1)
        with patch.object(e,'ready'),patch.object(e,'verify_commit'),patch.object(e,'deadline_check'), \
             patch.object(e,'sha',side_effect=digest),patch.object(e,'read',return_value={'commit':'c'}), \
             patch.object(Path,'exists',autospec=True,side_effect=lambda p:Path(p).name in records), \
             patch.object(Path,'open',mock_open()),patch.object(e.json,'dump',side_effect=dump), \
             patch.object(e,'write_new',side_effect=lambda p,v:records.update({Path(p).name:v})):
            with self.assertRaises(RuntimeError): e.run(api_factory=lambda:None,execute=execute)
            self.assertFalse(records['result.json']['passed'])
            with self.assertRaises(ValueError): e.run(api_factory=lambda:None,execute=execute)
            self.assertEqual(len(calls),1)


if __name__=='__main__': unittest.main()
