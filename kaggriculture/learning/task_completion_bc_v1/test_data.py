from copy import deepcopy
import unittest
from learning.task_choice_v1.features import OPS, SCHEMA_VERSION, CANDIDATE_SCOPE, schema_hash, candidate_id
from .data import collect_game, training_groups

def fixture():
    farm = dict(money=0, tiles=[[None,None]], farmer=[0,0], hands=[[1,0]], unlocked_quadrants=[], hires_today=0)
    obs = dict(player=0,step=0,day=0,hour=0,farms=[farm,deepcopy(farm)],private=dict(shed={},seeds={},inventories=[{},{}]),market={},town={})
    names = ['unit.index','unit.x','unit.y','target.x','target.y'] + ['task.op.'+x for x in OPS] + ['task.distance']
    names += ['local.'+str(i) for i in range(98-len(names))]
    shared = ['shared.'+str(i) for i in range(275)]
    cs=[]
    for u in (0,1):
        x=[u,u,0,1,0]+[int(k=='DIG') for k in OPS]+[1-u]
        cs.append(dict(id=candidate_id(u,[1,0],'DIG'),unit=u,target=[1,0],op='DIG',x=x+[0]*(98-len(x))))
    planned=dict(seeds={},needed_seeds={},supply={},projected_shed={},unit_actions=[],available_units=[0,1],interpretation='before assignment: reservations and requests, not official executed state')
    inputs=dict(observation=obs,configuration={'marketParams':{}},planned=planned,candidates=cs,candidate_scope=CANDIDATE_SCOPE,inventory_item_order=[[],[]],shed_item_order=[],schema_version=SCHEMA_VERSION,schema_sha256=schema_hash(names,shared),feature_names=names,shared_feature_names=shared,shared_x=[0]*275)
    records=[dict(inputs=deepcopy(inputs),chosen=c['id'],unit=c['unit'],target=c['target'],op=c['op'],teacher_label='ignored') for c in cs]
    labels=[dict(game='g',side=0,first_step=0,first_ordinal=i,unit=i,target=[1,0],op='DIG',chosen=records[i]['chosen'],status=('completed_observed' if i==0 else 'unknown')) for i in range(2)]
    schema=dict(schema_version=SCHEMA_VERSION,schema_sha256=schema_hash(names,shared),shared_feature_names=shared,feature_names=names,ops=list(OPS))
    return dict(key='g',side=0,seed=7),labels,[dict(step=0,records=records)],schema

class JoinTests(unittest.TestCase):
    def test_multiunit_reselection(self):
        game,labels,turns,schema=fixture(); again=deepcopy(turns[0]['records'][0]); again['inputs']['observation']['step']=1
        turns.append(dict(step=1,records=[again])); labels.append(dict(labels[0],first_step=1,status='unknown'))
        schema,groups=collect_game(game,labels,turns,schema,expected_turns=2)
        self.assertEqual([g['chosen'] for g in groups],[0,1,0]); self.assertEqual(len(training_groups(groups)),1); self.assertEqual(len(schema['feature_names']),98)
    def test_fail_closed(self):
        for kind in ('target','chosen','duplicate','vector','missing','private','future','schema'):
            game,labels,turns,schema=fixture(); inp=turns[0]['records'][0]['inputs']
            if kind=='target': labels[0]['target']=[0,0]
            if kind=='chosen': turns[0]['records'][0]['chosen']='bad'
            if kind=='duplicate': labels.append(deepcopy(labels[0]))
            if kind=='vector': inp['candidates'][0]['x'][0]=9
            if kind=='missing': labels[0]['first_ordinal']=7
            if kind=='private': inp['observation']['farms'][1]['private']={'seeds':{'MELON':99}}
            if kind=='future': inp['planned']['future_result']=1
            if kind=='schema': inp['feature_names'].append('future')
            with self.subTest(kind=kind), self.assertRaises(ValueError): collect_game(game,labels,turns,schema,expected_turns=1)

    def test_required_schema_complete_turns_and_uint32_seed(self):
        game,labels,turns,schema=fixture()
        for invalid in (None, {}, dict(schema, schema_sha256='wrong')):
            with self.subTest(schema=invalid), self.assertRaises(ValueError):
                collect_game(game,labels,turns,invalid,expected_turns=1)
        with self.assertRaises(ValueError): collect_game(game,labels,turns,schema)
        with self.assertRaises(ValueError): collect_game(game,labels,turns,schema,expected_turns=2)
        for seed in (-1, 2**32, True):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                collect_game(dict(game,seed=seed),labels,turns,schema,expected_turns=1)
        for seed in (0, 2**32-1):
            self.assertEqual(collect_game(dict(game,seed=seed),labels,turns,schema,expected_turns=1)[1][0]['seed'],seed)

if __name__=='__main__': unittest.main()
