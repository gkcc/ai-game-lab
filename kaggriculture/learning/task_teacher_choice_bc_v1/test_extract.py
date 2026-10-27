import types
import unittest
from unittest.mock import patch
from . import extract as e


def fixture():
    names = ['unit.index','unit.x','unit.y','target.x','target.y','task.distance'] + ['task.op.'+op for op in e.OPS]
    names += ['feature'+str(i) for i in range(98-len(names))]
    shared = ['shared'+str(i) for i in range(275)]
    schema = dict(schema_version=e.SCHEMA_VERSION,schema_sha256=e.schema_hash(names,shared),
                  shared_feature_names=shared,feature_names=names,ops=list(e.OPS))
    obs = dict(player=0,step=2,private=dict(inventories=[{},{}],shed={}),
               farms=[dict(farmer=[0,0],hands=[[1,0]])])
    candidates=[]
    for unit in (0,1):
        vector = [float(unit),float(unit),0.,2.,0.,float(2-unit)] + [1.,0.,0.,0.,0.] + [0.]*(98-11)
        candidates.append(dict(id=e.candidate_id(unit,[2,0],'DIG'),unit=unit,target=[2,0],op='DIG',x=vector))
    inputs = dict(schema_version=e.SCHEMA_VERSION,schema_sha256=schema['schema_sha256'],feature_names=names,
        shared_feature_names=shared,shared_x=[0.]*275,candidate_scope=e.CANDIDATE_SCOPE,observation=obs,configuration={},
        planned=dict(seeds={},needed_seeds={},supply={},projected_shed={},unit_actions=[],available_units=[0,1],
                     interpretation='before assignment: reservations and requests, not official executed state'),
        candidates=candidates,inventory_item_order=[[],[]],shed_item_order=[])
    return dict(inputs=inputs,teacher_label=candidates[1]['id'],chosen=candidates[0]['id'],
                unit=0,target=[2,0],op='DIG'),schema


class ExtractTest(unittest.TestCase):
    def test_teacher_is_separate_and_no_status(self):
        record,schema=fixture()
        with patch.object(e,'observe',side_effect=lambda obs,cfg:types.SimpleNamespace(observation=obs,configuration=cfg)):
            result=e.extract(record,schema,side=0,step=2,max_candidates=2)
        self.assertEqual(result['teacher_choice_label'],1)
        self.assertNotIn('status',result)
        self.assertEqual(set(result),{'shared','candidates','teacher_choice_label','candidate_ops'})

    def test_invalid_label_identity_operation_and_budget(self):
        for case in ('label','id','op','budget','private'):
            record,schema=fixture()
            if case == 'label': record['teacher_label']='missing'
            if case == 'id': record['inputs']['candidates'][0]['id']='duplicate'
            if case == 'op': record['inputs']['candidates'][0]['x'][6]=0.
            if case == 'private': record['inputs']['observation']['opponent_private']='secret'
            with patch.object(e,'observe',side_effect=lambda obs,cfg:types.SimpleNamespace(
                    observation={k:v for k,v in obs.items() if k!='opponent_private'},configuration=cfg)):
                with self.assertRaises(ValueError): e.extract(record,schema,side=0,step=2,max_candidates=1 if case=='budget' else 2)

    def test_actual_selection_must_match_record(self):
        for change in (dict(chosen='missing'),dict(unit=1),dict(target=[1,0]),dict(op='PLANT')):
            record,schema=fixture()
            record.update(change)
            with patch.object(e,'observe',side_effect=lambda obs,cfg:types.SimpleNamespace(observation=obs,configuration=cfg)):
                with self.assertRaises(ValueError): e.extract(record,schema,side=0,step=2,max_candidates=2)


if __name__ == '__main__': unittest.main()
