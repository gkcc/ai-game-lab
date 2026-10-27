import unittest
import types
from unittest.mock import patch
from . import experiment, membership, worker


class MembershipTest(unittest.TestCase):
    def test_membership_fails_closed_and_closes_handle(self):
        name='Local\\TeacherDeviation-'+'a'*32
        marker=dict(attempt=1,retry_allowed=False,freeze_sha256='freeze',job_token_sha256=membership.token_hash(name))
        class Query:
            def __init__(self,member): self.member=member; self.closed=[]
            def open(self,value): self.opened=value; return 'owned'
            def contains_current(self,handle): return self.member
            def close(self,handle): self.closed.append(handle)
        for member in (False,True):
            query=Query(member)
            if member: membership.verify_worker(name,marker,'freeze',lambda:query)
            else:
                with self.assertRaises(ValueError): membership.verify_worker(name,marker,'freeze',lambda:query)
            self.assertEqual(query.closed,['owned'])
        with self.assertRaises(ValueError): membership.verify_worker(None,marker,'freeze',lambda:Query(True))
        with self.assertRaises(ValueError): membership.verify_worker(name,{**marker,'job_token_sha256':'wrong'},'freeze',lambda:Query(True))

    def test_direct_worker_rejected_before_cache(self):
        with patch.object(worker,'ready',return_value={}),patch.object(worker,'read_runner_marker',return_value={}), \
             patch.object(worker,'sha',return_value='freeze'),patch.dict(worker.os.environ,{},clear=True):
            with self.assertRaises(ValueError): worker.run()

    def test_frozen_dependency_drift_fails(self):
        paths=('a','b'); runtime={'a':'one','b':'two'}
        experiment.check_frozen_dependencies(runtime,paths,lambda path:{'a':'one','b':'two'}[path.name])
        with self.assertRaises(ValueError): experiment.check_frozen_dependencies(runtime,paths,lambda path:'changed')
        with self.assertRaises(ValueError): experiment.check_frozen_dependencies({'a':'one'},paths,lambda path:'one')

    def test_named_job_collision_is_closed_and_rejected(self):
        api=membership.NamedWindowsAPI.__new__(membership.NamedWindowsAPI)
        api.name='synthetic'; api.k=types.SimpleNamespace(CreateJobObjectW=lambda *_:'handle')
        closed=[]; api.close=closed.append
        with patch.object(membership.ctypes,'set_last_error'),patch.object(membership.ctypes,'get_last_error',return_value=183):
            with self.assertRaises(RuntimeError): api.create_job()
        self.assertEqual(closed,['handle'])


if __name__=='__main__': unittest.main()
