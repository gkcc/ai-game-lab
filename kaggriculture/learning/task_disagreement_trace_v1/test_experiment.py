"""Synthetic tests only: no historic traces, checkpoints, games or seeds."""
from contextlib import contextmanager
from unittest import TestCase, main, mock
import io
import json

import numpy as np

from learning.task_multisource_rank_v1.dedup import Item
from learning.task_disagreement_trace_v1.extract import PrivateExample
from learning.task_disagreement_trace_v1 import experiment as exp


def guard():
    pass


NAMES = ['task.op.' + op for op in exp.OPS] + ['synthetic_%d' % i for i in range(93)]
META = {'feature_names':NAMES}
MEAN, SCALE = np.zeros(373, dtype=np.float64), np.ones(373, dtype=np.float64)


def example(role, step, token, *, target=0, student=1, identical_pair=False):
    shared = np.zeros(275, dtype=np.float32)
    shared[0] = token
    candidates = np.zeros((3,98), dtype=np.float32)
    candidates[0,0] = 1
    candidates[1,1] = 1
    candidates[2,2] = 1
    if target == 1:
        candidates[1,0], candidates[1,1] = 0, 1
    if identical_pair:
        candidates[1] = candidates[0]
    return PrivateExample(Item(shared, candidates, target,
        'new_train' if role in exp.ROLES[:2] else 'new_dev', role), step, 0, student)


def full_examples():
    out = []
    for ri, role in enumerate(exp.ROLES):
        for i in range(8):
            out.append(example(role, (i % 3) * 120 + i, 1 + ri * 8 + i,
                               target=i % 2, student=(i + 1) % 2))
    return out


class DatasetTests(TestCase):
    def build(self, samples):
        return exp._build_dataset(samples, mean=MEAN, scale=SCALE, metadata=META, guard=guard)

    def test_gate_and_numeric_private_archive_roundtrip(self):
        train, diagnostic, summary = self.build(full_examples())
        self.assertEqual(summary['data_gate'], 'passed')
        self.assertEqual((len(train), len(diagnostic)), (16,16))
        self.assertEqual(summary['windows_by_role'], dict.fromkeys(exp.ROLES,3))
        self.assertEqual(summary['train_target_operations']['DIG'], 8)
        self.assertEqual(summary['train_target_operations']['FERTILIZE'], 8)
        t = exp._load_split_bytes(exp._split_bytes(train,guard), 'train',guard)
        d = exp._load_split_bytes(exp._split_bytes(diagnostic,guard), 'diagnostic',guard)
        exp._counts_match(summary,t,d,list(range(5)))
        self.assertEqual(set(t), {'shared','candidates','offsets','teacher_target',
                                  'student_index','role','step','ordinal'})
        self.assertEqual(t['shared'].shape, (16,275))
        self.assertEqual(t['candidates'].shape, (48,98))
        self.assertEqual(d['offsets'][-1], 48)
        self.assertFalse(any(x.dtype.hasobject for x in t.values()))

    def test_summary_tampering_and_data_gate_failure(self):
        train, diagnostic, summary = self.build(full_examples())
        t = exp._load_split_bytes(exp._split_bytes(train,guard), 'train',guard)
        d = exp._load_split_bytes(exp._split_bytes(diagnostic,guard), 'diagnostic',guard)
        altered = json.loads(json.dumps(summary))
        altered['train_target_operations']['DIG'] -= 1
        altered['train_target_operations']['FERTILIZE'] += 1
        with self.assertRaises(ValueError):
            exp._counts_match(altered,t,d,list(range(5)))
        _a,_b,failed = self.build(full_examples()[:-1])
        self.assertEqual(failed['data_gate'], 'failed')
        self.assertFalse(failed['checks']['diagnostic_min'])
        contract = exp._dataset_contract('a'*40, {'summary':failed,'train_sha256':'b'*64,
                                                  'diagnostic_sha256':'c'*64})
        self.assertFalse(contract['eligible_for_future_review'])
        self.assertFalse(contract['training_started'])

    def test_role_level_filter_and_diagnostic_overlap_bounds(self):
        samples = full_examples()
        value = example('A04',380,99)
        same = np.repeat(value.item.candidates[:1], 3, axis=0)
        samples.append(PrivateExample(Item(value.item.shared, same, 0, 'new_train', 'A04'),380,0,1))
        train, diagnostic, summary = self.build(samples)
        t = exp._load_split_bytes(exp._split_bytes(train,guard), 'train',guard)
        d = exp._load_split_bytes(exp._split_bytes(diagnostic,guard), 'diagnostic',guard)
        exp._counts_match(summary, t, d, list(range(5)))
        too_many = json.loads(json.dumps(summary))
        too_many['filtered_by_role']['A09']['diagnostic_indistinguishable_disagreement'] = 999
        with self.assertRaises(ValueError):
            exp._counts_match(too_many, t, d, list(range(5)))
        moved = json.loads(json.dumps(summary))
        moved['filtered_by_role']['A03']['identical_rows'] += 1
        moved['filtered_by_role']['A04']['identical_rows'] -= 1
        with self.assertRaises(ValueError):
            exp._counts_match(moved, t, d, list(range(5)))

    def test_reloaded_split_rejects_more_than_four_in_role_window(self):
        train, _diagnostic, _summary = self.build(full_examples())
        arrays = exp._load_split_bytes(exp._split_bytes(train,guard), 'train',guard)
        arrays['step'][:5] = np.arange(5,dtype=np.int16)
        output = io.BytesIO()
        np.savez_compressed(output, **arrays)
        with self.assertRaises(ValueError):
            exp._load_split_bytes(output.getvalue(), 'train',guard)

    def test_npy_header_is_bounded_and_parse_error_is_sanitized(self):
        class Traced(io.BytesIO):
            def __init__(self, raw): super().__init__(raw); self.requests = []
            def read(self, n=-1): self.requests.append(n); return super().read(n)
        prefix = np.lib.format.magic(1,0)
        payload = b'x'*12000
        source = Traced(prefix + len(payload).to_bytes(2,'little') + payload)
        with self.assertRaises(ValueError):
            exp._bounded_header(source, len(source.getvalue()))
        self.assertLessEqual(max(source.requests), 8)
        invalid = b'private_header_marker is not a valid npy dictionary'
        source = io.BytesIO(prefix + len(invalid).to_bytes(2,'little') + invalid)
        with self.assertRaises(ValueError) as caught:
            exp._bounded_header(source, len(source.getvalue()))
        self.assertEqual(str(caught.exception),'Invalid private split header')
        self.assertNotIn('private_header_marker', str(caught.exception))

    def test_guard_failure_is_not_rewritten_as_bad_header(self):
        train, _diagnostic, _summary = self.build(full_examples())
        def fail(): raise MemoryError('synthetic guard')
        with self.assertRaisesRegex(MemoryError, 'synthetic guard'):
            exp._load_split_bytes(exp._split_bytes(train,guard), 'train',fail)

    def test_duplicate_priority_and_teacher_label_conflict(self):
        a = example('A03',0,11)
        duplicate = example('A09',0,11)
        conflict = example('A04',120,12,target=0,student=1)
        conflict2 = example('A10',120,12,target=1,student=0)
        train,diagnostic,summary = self.build([duplicate,a,conflict,conflict2])
        self.assertEqual([x.role for x in train],['A03'])
        self.assertEqual(diagnostic,[])
        self.assertEqual(summary['dedup']['conflict_components'],1)
        self.assertEqual(summary['dedup']['duplicate_removed_by_pool']['new_dev'],1)
        self.assertEqual(summary['dedup']['conflict_removed_by_pool']['new_train'],1)
        self.assertEqual(summary['dedup']['conflict_removed_by_pool']['new_dev'],1)

    def test_student_indistinguishability_only_filters_train(self):
        a = example('A03',0,11,identical_pair=True)
        b = example('A09',0,12,identical_pair=True)
        train,diagnostic,summary = self.build([a,b])
        self.assertEqual(train,[])
        self.assertEqual([x.role for x in diagnostic],['A09'])
        self.assertEqual(summary['filtered_by_role']['A03']['indistinguishable_disagreement'],1)
        self.assertEqual(summary['filtered_by_role']['A09']['diagnostic_indistinguishable_disagreement'],1)

    def test_single_and_all_equal_are_counted_without_replacement(self):
        a = example('A09',0,11)
        a = PrivateExample(Item(a.item.shared,a.item.candidates[:1],0,'new_dev','A09'),0,0,0)
        b = example('A10',120,12)
        equal = np.repeat(b.item.candidates[:1],3,axis=0)
        b = PrivateExample(Item(b.item.shared,equal,0,'new_dev','A10'),120,0,1)
        train,diagnostic,summary = self.build([a,b])
        self.assertEqual((train,diagnostic),([],[]))
        self.assertEqual(summary['filtered_by_role']['A09']['single'],1)
        self.assertEqual(summary['filtered_by_role']['A10']['identical_rows'],1)


class TraceBoundaryTests(TestCase):
    def test_success_extracts_only_after_contexts_close(self):
        events = []
        class Game:
            identity = dict(source_role='A03',batch=1,role='T03',opponent='market',side=0)
            task_sha256, crop_sha256 = 'task','crop'
            def __enter__(self): return self
            def __exit__(self,*_args): events.append('game_closed')
            @contextmanager
            def open_trace(self,*,guard):
                yield object()
                events.append('rehash_passed')
        class Sampler:
            def __init__(self,*_args,**_kwargs): pass
            def scan(self,**_kwargs): return {'lines':719}
            def samples(self):
                events.append('samples_accessed')
                sample = dict(role='A03',opponent='market',side=0,step=0,ordinal=0,
                              inputs={'fake':True},teacher='teacher',chosen='student')
                return [[sample],[],[],[],[],[]]
        def extractor(**kwargs):
            events.append('extracted')
            self.assertEqual(kwargs['teacher_label'],'teacher')
            self.assertEqual(kwargs['chosen'],'student')
            return 'synthetic example'
        result = exp._collect_case(object(),exp.CASES[0],META,guard,
            read_game=lambda *_a,**_k:Game(),sampler_type=Sampler,extractor=extractor)
        self.assertEqual(result,['synthetic example'])
        self.assertEqual(events,['rehash_passed','game_closed','samples_accessed','extracted'])

    def test_same_file_final_hash_must_pass_before_samples_access(self):
        events = []
        class Game:
            identity = dict(source_role='A03',batch=1,role='T03',opponent='market',side=0)
            task_sha256, crop_sha256 = 'task', 'crop'
            def __enter__(self): return self
            def __exit__(self,*_args): events.append('game_closed')
            @contextmanager
            def open_trace(self,*,guard):
                yield object()
                events.append('rehash_failed')
                raise ValueError('synthetic final hash failure')
        class Sampler:
            def __init__(self,*_args,**_kwargs): pass
            def scan(self,**_kwargs): return {'lines':719}
            def samples(self):
                events.append('samples_accessed')
                return [[] for _ in range(6)]
        with self.assertRaises(ValueError):
            exp._collect_case(object(),exp.CASES[0],META,guard,
                read_game=lambda *_a,**_k:Game(),sampler_type=Sampler,extractor=lambda **_k:None)
        self.assertEqual(events,['rehash_failed','game_closed'])

    def test_child_does_not_write_data_when_fourth_final_hash_fails(self):
        from learning.task_student_anchor_collection_v1 import sources
        from learning.task_disagreement_trace_v1 import sampling, extract
        from learning.task_teacher_dev_deviation_v1 import membership
        from learning.task_pair_fresh_multi_v1 import entry
        head, sha = 'a'*40, 'b'*64
        binding = {'schema':'binding','salt':'salt','source':'synthetic'}
        documents = {'claim':{'schema':'disagreement-trace-claim-v1','commit':head,'attempt':1,
                               'freeze_sha256':sha,'retry_allowed':False},
                     'attempt':{'schema':'disagreement-trace-attempt-v1','commit':head,'retry_allowed':False},
                     'owner':{}}
        writes, samples_accessed = [], []
        class Context:
            def model_normalization(self): return MEAN,SCALE,META
        class Game:
            task_sha256,crop_sha256 = 'task','crop'
            def __init__(self,case): self.case = case
            @property
            def identity(self):
                role,_mode,batch,old,opponent,side = self.case
                return dict(source_role=role,batch=batch,role=old,opponent=opponent,side=side)
            def __enter__(self): return self
            def __exit__(self,*_args): return False
            @contextmanager
            def open_trace(self,*,guard):
                yield self.case
                if self.case == exp.CASES[-1]: raise ValueError('synthetic hash failure')
        class Sampler:
            def __init__(self,stream,**_kwargs): self.case=stream
            def scan(self,**_kwargs): return {'lines':719}
            def samples(self):
                samples_accessed.append(self.case[0])
                return [[] for _ in range(6)]
        def read_game(_ctx,*,batch,role,opponent,side,guard):
            return Game(next(c for c in exp.CASES if c[2:] == (batch,role,opponent,side)))
        def read(path,*_args,**_kwargs):
            return json.dumps(documents[path.name.split('.')[0]]).encode()
        with mock.patch.object(exp,'_runtime_binding',return_value=(head,binding,sha)), \
             mock.patch.object(exp,'_read',side_effect=read), \
             mock.patch.object(exp,'_context_binding',return_value={'source':'synthetic'}), \
             mock.patch.object(exp.previous,'_write_new',side_effect=lambda path,_data:writes.append(path.name)), \
             mock.patch.object(exp.previous,'_guard'), \
             mock.patch.object(exp,'_write_bytes_new') as data_write, \
             mock.patch.object(sources,'open_context',return_value=Context()), \
             mock.patch.object(sources,'read_game',side_effect=read_game), \
             mock.patch.object(sampling,'RankedStream',Sampler), \
             mock.patch.object(extract,'extract_example'), \
             mock.patch.object(membership,'verify_worker'), \
             mock.patch.object(entry,'verify_live_owner'):
            with self.assertRaises(ValueError): exp.child()
        data_write.assert_not_called()
        self.assertNotIn('worker-result.local.json',writes)
        self.assertIn('worker-failed.local.json',writes)
        self.assertEqual(samples_accessed,['A03','A04','A09'])


if __name__ == '__main__':
    main()
