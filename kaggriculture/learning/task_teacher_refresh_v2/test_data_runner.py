import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import data_runner as r


class FakeRunner(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.out=Path(self.temp.name)
        for n in r.IDENTITIES: (self.out/n).write_text('{}',encoding='utf-8')
        self.cfg=dict(host_preflight=dict(samples=3,min_available_bytes=4*1024**3),
            data_job_supervision=dict(min_available_bytes=2*1024**3),schema_sha256=r.sha(self.out/'schema.json'),
            train_seeds=[1,2],collection_limits=dict(max_total_callbacks=40000,max_total_candidate_rows=2000000,compressed_bytes=1000,decoded_bytes=1000),data_limits=dict(max_groups=8000,max_candidates=400000,max_archive_bytes=1000,max_uncompressed_bytes=1000,min_groups=4,min_groups_per_seed=2,min_selected_operations=2,min_dig=2,min_fertilize=2,min_dig_per_seed=1))
        self.patches=[patch.object(r,'OUT',self.out),patch.object(r,'ready',return_value=self.cfg),
            patch.object(r,'verify_commit'),patch.object(r,'deadline_check'),patch.object(r,'validate_panel',return_value=[1]),
            patch.object(r,'validate_completed'),patch.object(r,'memory_ready',return_value=True),
            patch.object(r,'NamedWindowsAPI',side_effect=lambda name:object()),patch.object(r,'write_new',side_effect=self.write)]
        for p in self.patches: p.start()
        self.addCleanup(self.temp.cleanup)
        for p in self.patches: self.addCleanup(p.stop)
    def write(self,path,value):
        with Path(path).open('x',encoding='utf-8') as f: json.dump(value,f)
    def job(self):
        return dict(passed=True,assigned=True,resumed=True,launcher_reaped=True,job_members_zero=True,
                    exit_code=0,error_kind=None,cleanup_error_kind=None,stop_reason=None)
    def worker(self,status='coverage_insufficient'):
        self.write(self.out/'data-worker-entered.json',dict(attempt=1,freeze_sha256=r.sha(self.out/'freeze.json'),retry_allowed=False))
        result=dict(passed=status=='collected',status=status,freeze_sha256=r.sha(self.out/'freeze.json'),retry_allowed=False,job_cleanup_verified_here=False)
        entries=[]
        for seed in [1,2]:
            for opponent in ['round2','market']:
                chosen={op:0 for op in r.OPS}; chosen['DIG']=1 if opponent=='round2' else 0; chosen['FERTILIZE']=1 if opponent=='market' else 0
                if status!='collected': chosen={op:0 for op in r.OPS}
                n=sum(chosen.values()); entries.append(dict(seed=seed,opponent=opponent,callbacks=n,retained=n,chosen=chosen,retained_chosen=chosen.copy(),candidate_rows=chosen.copy(),candidate_callbacks=chosen.copy()))
        n=sum(x['callbacks'] for x in entries); selected={op:sum(x['retained_chosen'][op] for x in entries) for op in r.OPS}
        result['full_stream']=dict(by_seed_opponent=entries,total_callbacks=n,total_candidate_rows=n,retained_groups=n,retained_candidate_rows=n,meter=dict(files_opened=8,files_completed=8,raw_decode_passes_per_file=1,compressed_input_read_bytes=1,decoded_input_bytes=2))
        result['coverage']=dict(groups=n,groups_by_seed={str(seed):sum(x['retained'] for x in entries if x['seed']==seed) for seed in [1,2]})
        result['coverage'].update(retained_selected_operations=selected,dig_by_seed={str(seed):sum(x['retained_chosen']['DIG'] for x in entries if x['seed']==seed) for seed in [1,2]})
        if status=='collected': result['sample_summary']=dict(groups=n,groups_by_seed=result['coverage']['groups_by_seed'].copy(),selected_operations={k:v for k,v in selected.items() if v})
        if status=='collected':
            archive=self.out/'private/new-teacher.npz'; archive.parent.mkdir(); archive.write_bytes(b'fake')
            result.update(source_manifest_sha256=r.sha(self.out/'source-manifest.json'),schema_sha256=self.cfg['schema_sha256'],
                          array_sha256=r.sha(archive),archive_snapshot_reads=1,collection_attempt=1)
        self.write(self.out/'data-worker-result.json',result)
        return self.job()
    def test_preflight_stops_before_marker(self):
        with patch.object(r,'memory_ready',return_value=False), self.assertRaises(RuntimeError): r.run(self.out)
        self.assertFalse((self.out/'data-started.json').exists())
    def test_replay_and_manifest_rejected_before_marker(self):
        for name in ('validate_completed','validate_panel'):
            with patch.object(r,name,side_effect=ValueError), self.assertRaises(ValueError): r.run(self.out)
            self.assertFalse((self.out/'data-started.json').exists())
    def test_supervision_exception_preserves_receipt_and_cannot_retry(self):
        old=os.environ.get(r.ENV_KEY)
        with patch.object(r,'supervise',side_effect=RuntimeError), self.assertRaises(RuntimeError): r.run(self.out)
        self.assertEqual(old,os.environ.get(r.ENV_KEY))
        self.assertEqual(r.read(self.out/'data-supervision.local.json')['cleanup_state'],'unknown')
        with self.assertRaises(ValueError): r.run(self.out)
    def test_cleanup_unknown_is_failure(self):
        job=self.job(); job['job_members_zero']=False
        with patch.object(r,'supervise',return_value=job), self.assertRaises(RuntimeError): r.run(self.out)
        self.assertFalse((self.out/'data-seal.json').exists())
    def test_insufficient_never_seals(self):
        with patch.object(r,'supervise',side_effect=lambda *a:self.worker()): result=r.run(self.out)
        self.assertEqual(result['status'],'coverage_insufficient')
        self.assertFalse((self.out/'data-seal.json').exists())
    def test_success_seal_binds_archive_and_inputs(self):
        with patch.object(r,'supervise',side_effect=lambda *a:self.worker('collected')):
            result=r.run(self.out)
        seal=r.read(self.out/'data-seal.json')
        self.assertTrue(result['passed'])
        self.assertEqual(seal['array_sha256'],r.sha(self.out/'private/new-teacher.npz'))
        self.assertEqual(seal['independent_replay_sha256'],r.sha(self.out/'independent-replay-verification.json'))
        self.assertEqual(result['data_seal_sha256'],r.sha(self.out/'data-seal.json'))

    def test_inconsistent_summary_rejected(self):
        def fake(*args):
            job=self.worker('collected'); path=self.out/'data-worker-result.json'; value=r.read(path)
            value['coverage']['groups']=999
            path.write_text(json.dumps(value),encoding='utf-8')
            return job
        with patch.object(r,'supervise',side_effect=fake), self.assertRaises(ValueError): r.run(self.out)
        self.assertFalse((self.out/'data-seal.json').exists())
    def test_parent_never_decodes_archive(self):
        from learning.task_teacher_choice_bc_v1 import numeric
        with patch.object(r,'supervise',side_effect=lambda *a:self.worker('collected')), patch.object(numeric,'read',side_effect=AssertionError('parent decode')):
            self.assertTrue(r.run(self.out)['passed'])

    def test_completion_consumer_accepts_only_full_result(self):
        with patch.object(r,'supervise',side_effect=lambda *a:self.worker('collected')):
            result=r.run(self.out)
        self.assertEqual(r.validate_sealed(self.out,self.cfg),result)
        self.assertEqual(r.read(self.out/'data-seal.json')['completion_state'],'pending_result_confirmation')
    def test_failed_or_partial_result_does_not_complete_seal(self):
        for partial in (False,True):
            with self.subTest(partial=partial):
                if partial:
                    # A distinct owned fake run avoids changing a prior receipt.
                    self.tearDown(); self.doCleanups(); self.setUp()
                def write(path,value):
                    if Path(path).name=='data-result.json':
                        if partial: Path(path).write_text('{',encoding='utf-8')
                        raise OSError('fake write failure')
                    self.write(path,value)
                with patch.object(r,'write_new',side_effect=write), patch.object(r,'supervise',side_effect=lambda *a:self.worker('collected')):
                    with self.assertRaises(OSError): r.run(self.out)
                self.assertTrue((self.out/'data-seal.json').exists())
                with self.assertRaises((ValueError,OSError)): r.validate_sealed(self.out,self.cfg)
    def test_real_worker_helpers_match_parent_contract(self):
        import numpy as np
        from .data_worker import coverage, sample_summary
        self.worker('collected')
        result=r.read(self.out/'data-worker-result.json')
        # sample_summary enforces production thresholds; use 256 synthetic rows below.
        limits=dict(self.cfg['data_limits'],min_groups=256,min_groups_per_seed=32,min_selected_operations=3,min_dig=32,min_fertilize=100)
        choices=['DIG']*32+['FERTILIZE']*100+['WATER']*124
        arrays=dict(shared=np.zeros((256,1)),seed=np.array([1,2]*128),ops=np.array([r.OPS.index(x) for x in choices]),offsets=np.arange(257),teacher_choice_label=np.zeros(256,dtype=int))
        cfg=dict(self.cfg,data_limits=limits)
        adequate,result['coverage']=coverage(arrays,cfg['train_seeds'],limits)
        result['sample_summary']=sample_summary(arrays,cfg['train_seeds'],limits)
        entries=[]
        for seed in [1,2]:
            for opponent in ['round2','market']:
                counts={op:0 for op in r.OPS}
                if opponent=='round2':
                    for i,choice in enumerate(choices):
                        if int(arrays['seed'][i])==seed: counts[choice]+=1
                n=sum(counts.values()); entries.append(dict(seed=seed,opponent=opponent,callbacks=n,retained=n,chosen=counts.copy(),retained_chosen=counts.copy(),candidate_rows=counts.copy(),candidate_callbacks=counts.copy()))
        result['full_stream']=dict(by_seed_opponent=entries,total_callbacks=256,total_candidate_rows=256,retained_groups=256,retained_candidate_rows=256,meter=dict(files_opened=8,files_completed=8,raw_decode_passes_per_file=1,compressed_input_read_bytes=1,decoded_input_bytes=2))
        self.assertTrue(adequate); self.assertTrue(r.validate_statistics(result,cfg))
        result['sample_summary']['selected_operations']['DIG']+=1
        with self.assertRaises(ValueError): r.validate_statistics(result,cfg)
    def test_stream_counts_and_byte_limits_rejected(self):
        self.worker('collected'); result=r.read(self.out/'data-worker-result.json')
        result['full_stream']['by_seed_opponent'][0]['candidate_callbacks']['DIG']=0
        with self.assertRaises(ValueError): r.validate_statistics(result,self.cfg)
        result=r.read(self.out/'data-worker-result.json'); result['full_stream']['meter']['decoded_input_bytes']=1001
        with self.assertRaises(ValueError): r.validate_statistics(result,self.cfg)
        for field in ('compressed_input_read_bytes','decoded_input_bytes'):
            for value in (None,-1,1001):
                result=r.read(self.out/'data-worker-result.json'); meter=result['full_stream']['meter']
                if value is None: del meter[field]
                else: meter[field]=value
                with self.subTest(field=field,value=value), self.assertRaises(ValueError): r.validate_statistics(result,self.cfg)

if __name__=='__main__': unittest.main()
