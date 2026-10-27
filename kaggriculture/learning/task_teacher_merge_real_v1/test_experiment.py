import hashlib
import json
import unittest
from .experiment import REQUIRED_RUNTIME,trusted_runtime


class FreezeTest(unittest.TestCase):
    def test_runtime_must_match_published_and_contain_required_dependencies(self):
        data=json.dumps(dict(runtime_files={key:'a'*64 for key in REQUIRED_RUNTIME})).encode()
        acceptance=dict(panel_completed=True,evidence_sha256={'freeze.json':hashlib.sha256(data).hexdigest()})
        self.assertEqual(set(trusted_runtime(data,data,acceptance)),REQUIRED_RUNTIME)
        with self.assertRaises(ValueError): trusted_runtime(data,b'{}',acceptance)
        empty=json.dumps(dict(runtime_files={})).encode()
        spoof=dict(panel_completed=True,evidence_sha256={'freeze.json':hashlib.sha256(empty).hexdigest()})
        with self.assertRaises(ValueError): trusted_runtime(empty,empty,spoof)
        with self.assertRaises(ValueError): trusted_runtime(data,data,{**acceptance,'panel_completed':False})


if __name__=='__main__': unittest.main()
