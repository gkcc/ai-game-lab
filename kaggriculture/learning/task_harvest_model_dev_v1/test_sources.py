"""Pure registered-point/EOF regressions; no files, weights or engine calls."""
from copy import deepcopy
import unittest

from learning.task_bc_error_diagnostics_v1.shadow import ordered_hash
from .sources import select_point


TASK = ('artifacts/synthetic/task.npz','a'*64)
CROP = ('artifacts/synthetic/crop.npz','b'*64)


def fixture():
    # Deliberately nonalphabetical observation/stock order must survive copying.
    inputs = dict(observation=dict(player=0,step=240,
        private=dict(shed=dict(MELON=2,WHEAT=1),seeds={},inventories=[{},{}]),
        farms=[dict(farmer=[4,4],hands=[[3,4]]),dict(farmer=[4,4],hands=[])]),
        configuration=dict(turnsPerDay=24),planned=dict(available_units=[0,1]))
    actions = [['WATER'],['PASS']]
    trace = dict(step=240,records=[dict(inputs=inputs)],
        final_reservations=dict(unit_actions=deepcopy(actions)),
        task_checkpoint_sha256=TASK[1],checkpoint_sha256=CROP[1])
    target = dict(schema_version='task-harvest-unexpressed-target-v1',side=0,step=240,
        ordinal=0,unit=0,target=[4,4],input_sha256=ordered_hash(inputs),
        original_unit_request=['WATER'],unit_requests_sha256=ordered_hash(actions))
    request = dict(farmer=['WATER'],hands=[['PASS']],
        market=[['SELL','WHEAT',1],['BUY_SEED','MELON',1]])
    selected = dict(kind='turn',arm='control',scope='suffix',step=240,
        trace=trace,request=request,original_request=deepcopy(request),patch=None)
    events = [dict(kind='arm_start',arm='control',step=240),selected,
        dict(kind='arm_end',arm='control'),dict(kind='arm_start',arm='intervene',step=240),
        dict(kind='turn',arm='intervene',scope='suffix',step=240),
        dict(kind='arm_end',arm='intervene')]
    return events,target


def read(events,target):
    return select_point(iter(events),target,role='F01',task=TASK,crop=CROP)


class SourcePointTests(unittest.TestCase):
    def test_registered_input_and_order_are_copied_after_full_eof(self):
        events,target=fixture()
        progress=[]
        def stream():
            for event in events:
                progress.append(event['kind'])
                yield event
            progress.append('EOF')
        point=select_point(stream(),target,role='F01',task=TASK,crop=CROP)
        self.assertEqual(progress[-1],'EOF')
        self.assertEqual(len(progress),len(events)+1)
        original=events[1]['trace']['records'][0]['inputs']
        self.assertEqual(ordered_hash(point['inputs']),target['input_sha256'])
        self.assertEqual(list(point['inputs']['observation']),list(original['observation']))
        self.assertEqual(list(point['inputs']['observation']['private']['shed']),['MELON','WHEAT'])
        point['inputs']['observation']['private']['shed']['MELON']=999
        self.assertEqual(original['observation']['private']['shed']['MELON'],2)

    def test_reader_tail_failure_after_point_is_not_ignored(self):
        events,target=fixture()
        error=RuntimeError('synthetic EOF failure')
        def stream():
            yield from events
            raise error
        with self.assertRaises(RuntimeError) as caught:
            select_point(stream(),target,role='F01',task=TASK,crop=CROP)
        self.assertIs(caught.exception,error)

    def test_both_distinct_arms_must_close_in_order(self):
        events,target=fixture()
        invalid=[]
        invalid.append(events[:-1])
        duplicate=deepcopy(events)
        for event in duplicate:
            if event['arm']=='intervene':event['arm']='control'
        invalid.append(duplicate)
        reversed_end=deepcopy(events)
        reversed_end[0],reversed_end[2]=reversed_end[2],reversed_end[0]
        invalid.append(reversed_end)
        for index,case in enumerate(invalid):
            with self.subTest(case=index),self.assertRaises(ValueError):read(case,target)

    def test_missing_duplicate_wrong_hash_and_wrong_ordinal_reject(self):
        events,target=fixture()
        with self.assertRaises(ValueError):read([e for i,e in enumerate(events) if i!=1],target)
        with self.assertRaises(ValueError):read(events[:2]+[deepcopy(events[1])]+events[2:],target)
        wrong=deepcopy(target);wrong['ordinal']=1
        with self.assertRaises(ValueError):read(events,wrong)
        reordered=deepcopy(events)
        private=reordered[1]['trace']['records'][0]['inputs']['observation']['private']
        private['shed']=dict(reversed(list(private['shed'].items())))
        with self.assertRaises(ValueError):read(reordered,target)
        wrong=deepcopy(events);wrong[1]['trace']['records'][0]['inputs']['observation']['step']=241
        with self.assertRaises(ValueError):read(wrong,target)

    def test_requests_unit_binding_and_both_model_identities_reject_conflicts(self):
        events,target=fixture()
        invalid=[]
        wrong=deepcopy(events);wrong[1]['original_request']['market'].reverse();invalid.append((wrong,target))
        wrong=deepcopy(events);wrong[1]['request']['hands']=[['NORTH']];invalid.append((wrong,target))
        wrong=deepcopy(events);wrong[1]['trace']['final_reservations']['unit_actions'][0]=['SOUTH'];invalid.append((wrong,target))
        for field in ('task_checkpoint_sha256','checkpoint_sha256'):
            wrong=deepcopy(events);wrong[1]['trace'][field]='c'*64;invalid.append((wrong,target))
        for field,value in (('unit',1),('unit',99),('original_unit_request',['SOUTH'])):
            wrong=deepcopy(target);wrong[field]=value;invalid.append((events,wrong))
        for index,(case,point) in enumerate(invalid):
            with self.subTest(case=index),self.assertRaises(ValueError):read(case,point)


if __name__ == '__main__':
    unittest.main()
