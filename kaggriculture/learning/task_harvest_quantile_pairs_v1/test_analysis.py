"""Public constructed inputs and synthetic returns; no real weights or games."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import numpy as np

from learning.task_harvest_choice_v1.adapter import HarvestChoiceAgent
from learning.task_harvest_choice_v1.fixtures import observation
from learning.task_harvest_return_pairs_v1 import contract as old_contract
from learning.task_harvest_return_pairs_v1.selection import TARGET_SCHEMA,ordered_hash
from . import analysis


TASK_SHA,CROP_SHA='a'*64,'b'*64


def constructed_inputs(money=0,*,unit=0):
    """Real feature encoder/controller on one public artificial observation."""
    obs,cfg=observation('mature5dry');obs['farms'][0]['money']=money
    if unit==1:
        obs['farms'][0]['farmer']=[3,4];obs['farms'][0]['hands']=[[4,4]]
        obs['private']['inventories']=[{},{}]
    collected=[]
    def select(inputs):
        collected.append(deepcopy(inputs))
        return next(candidate['id'] for candidate in inputs['candidates'] if candidate['op']=='HARVEST')
    agent=HarvestChoiceAgent(select)
    request=agent(obs,cfg)
    actions=[request['farmer'],*request['hands']]
    if len(collected)!=1 or actions[unit]!=['HARVEST']:
        raise AssertionError('Public constructed callback differs')
    return collected[0]


def target_for(inputs):
    observation=inputs['observation'];side=observation['player']
    candidate=next(c for c in inputs['candidates'] if c['op']=='HARVEST')
    unit=candidate['unit']
    return dict(schema_version=TARGET_SCHEMA,side=side,step=observation['step'],ordinal=0,
        unit=unit,target=[4,4],board_size=10,unit_count=1+len(observation['farms'][side]['hands']),input_sha256=ordered_hash(inputs),
        task_checkpoint_sha256=TASK_SHA,crop_checkpoint_sha256=CROP_SHA,
        harvest_choice=f'u{unit}:x4:y4:HARVEST',water_choice=f'u{unit}:x4:y4:WATER',unit_request=['HARVEST'])


def terminal(own,rival):
    margin=own-rival
    return dict(own_net_profit=own,opponent_net_profit=rival,margin=margin,
        score=1. if margin>0 else .5 if margin==0 else 0.)


def item(inputs,*,origin='F01',point='Q0',seed=1900899199,own_delta=10,margin_delta=10):
    inputs=deepcopy(inputs);target=target_for(inputs)
    source=dict(analysis.IDENTITIES[origin],row=dict(seed=seed),
        task=['artifacts/synthetic/task.npz',TASK_SHA],crop=['artifacts/synthetic/crop.npz',CROP_SHA])
    values=old_contract.terminal_delta(terminal(own_delta,own_delta-margin_delta),terminal(0,0))
    sample=dict(schema='task-harvest-return-pair-private-sample-v1',inputs=inputs,
        choices={'H':target['harvest_choice'],'W':target['water_choice']},targets=values,
        provenance=dict(source=source,target=deepcopy(target),continuation='same frozen task and crop weights',
            scope='consumed development pilot',training_admission=False),training_admission=False)
    return dict(origin=origin,point=point,split=analysis.SPLITS[origin],seed=seed,target=target,sample=sample)


def rebind(value):
    value['target']['input_sha256']=ordered_hash(value['sample']['inputs'])
    value['sample']['provenance']['target']=deepcopy(value['target'])


def pair_rows(value):
    inputs=value['sample']['inputs']
    by_id={candidate['id']:candidate for candidate in inputs['candidates']}
    return np.asarray([inputs['shared_x']+by_id[value['target'][key]]['x']
                       for key in ('harvest_choice','water_choice')],dtype=np.float64)


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.zero=constructed_inputs(0);cls.one=constructed_inputs(1);cls.hand=constructed_inputs(unit=1)

    def setUp(self):
        self.mean,self.scale=np.zeros(373),np.ones(373)
        forbidden=patch('numpy.load',side_effect=AssertionError('Pure diagnostics cannot load a checkpoint'))
        forbidden.start();self.addCleanup(forbidden.stop)

    def analyze(self,rows,mean=None,scale=None):
        return analysis.analyze(rows,self.mean if mean is None else mean,self.scale if scale is None else scale)

    def test_whole_pair_retains_state_when_action_difference_is_constant(self):
        rows=[item(self.zero),item(self.one,point='Q1')]
        a,b=map(pair_rows,rows)
        self.assertEqual(a.shape,(2,373))
        np.testing.assert_array_equal(a[0]-a[1],b[0]-b[1])
        self.assertFalse(np.array_equal(a,b))
        before=ordered_hash(rows)
        result=self.analyze(rows)
        self.assertEqual(result['public']['counts']['encoded_hw_rows'],4)
        self.assertEqual(result['public']['counts']['model_forwards'],0)
        for kind in analysis.KINDS:
            self.assertEqual(result['public']['duplicates'][kind]['duplicate_groups'],0)
        self.assertEqual(result['public']['per_origin']['F01']['samples'],2)
        self.assertEqual(result['public']['per_origin']['F01']['seed_clusters'],1)
        self.assertEqual(result['public']['per_origin']['F01']['step_range'],[241,241])
        self.assertEqual(result['public']['per_origin']['F01']['unit_count_range'],[1,1])
        self.assertEqual(ordered_hash(rows),before)
        np.testing.assert_array_equal(self.mean,np.zeros(373))
        np.testing.assert_array_equal(self.scale,np.ones(373))

    def test_opposite_and_zero_targets_are_retained_in_cross_split_duplicates(self):
        rows=[item(self.zero),item(self.zero,origin='F04',seed=1900899203,own_delta=-10,margin_delta=-10),
              item(self.zero,point='Q1',own_delta=0,margin_delta=0)]
        result=self.analyze(rows);public=result['public']
        self.assertEqual(public['counts']['retained_points'],3)
        self.assertEqual(public['counts']['dropped_points'],0)
        self.assertEqual(public['counts']['reassigned_points'],0)
        self.assertEqual(public['counts']['seed_clusters'],2)
        self.assertEqual(public['overall']['margin_direction'],dict(positive=1,negative=1,zero=1))
        self.assertEqual(public['per_split']['diagnostic']['samples'],1)
        for kind in analysis.KINDS:
            duplicate=public['duplicates'][kind]
            self.assertEqual(duplicate['duplicate_groups'],1)
            self.assertEqual(duplicate['rows_in_duplicate_groups'],3)
            self.assertEqual(duplicate['cross_split_groups'],1)
            self.assertEqual(duplicate['opposite_margin_sign_groups'],1)
            self.assertEqual(duplicate['opposite_own_profit_sign_groups'],1)
            self.assertEqual(duplicate['zero_and_nonzero_margin_groups'],1)
            self.assertEqual(len(result['private']['duplicate_groups'][kind][0]['members']),3)

    def test_deployed_collision_is_separate_from_distinct_raw_pair(self):
        rows=[item(self.zero),item(self.one,point='Q1')]
        mean=self.mean.copy();mean[self.zero['shared_feature_names'].index('own.money')]=1e16
        raw0,raw1=map(pair_rows,rows)
        self.assertFalse(np.array_equal(raw0,raw1))
        np.testing.assert_array_equal((raw0-mean)/self.scale,(raw1-mean)/self.scale)
        result=self.analyze(rows,mean=mean)
        self.assertEqual(result['public']['duplicates']['raw_pair']['duplicate_groups'],0)
        self.assertEqual(result['public']['duplicates']['deployed_pair']['duplicate_groups'],1)
        self.assertEqual(result['public']['duplicates']['full_input']['duplicate_groups'],0)

    def test_invalid_normalization_and_nonfinite_conversion_are_rejected(self):
        bad=[]
        bad.append((np.zeros(372),self.scale))
        bad.append((self.mean,np.ones((1,373))))
        mean=self.mean.copy();mean[0]=np.nan;bad.append((mean,self.scale))
        scale=self.scale.copy();scale[0]=np.inf;bad.append((self.mean,scale))
        scale=self.scale.copy();scale[0]=.5;bad.append((self.mean,scale))
        mean=self.mean.copy();mean[275+self.zero['feature_names'].index('task.op.HARVEST')]=1
        bad.append((mean,self.scale))
        for mean,scale in bad:
            with self.subTest(shape=(np.shape(mean),np.shape(scale))):
                with self.assertRaises(ValueError):self.analyze([item(self.zero)],mean=mean,scale=scale)
        value=item(self.zero);index=self.zero['shared_feature_names'].index('own.money')
        value['sample']['inputs']['shared_x'][index]=np.finfo(np.float64).max.item()
        rebind(value);mean=self.mean.copy();mean[index]=-np.finfo(np.float64).max
        with self.assertRaisesRegex(ValueError,'normalization'):self.analyze([value],mean=mean)

    def test_source_seed_point_and_checkpoint_groups_fail_closed(self):
        variants=[]
        cross=item(self.zero,origin='F04');variants.append([item(self.zero),cross])
        changed=item(self.zero,point='Q1',seed=1900899203);variants.append([item(self.zero),changed])
        changed=item(self.zero);changed['split']='diagnostic';variants.append([changed])
        variants.append([item(self.zero),item(self.zero)])
        changed=item(self.zero,origin='F04',seed=1900899203)
        changed['target']['task_checkpoint_sha256']='c'*64
        changed['sample']['provenance']['target']=deepcopy(changed['target'])
        changed['sample']['provenance']['source']['task'][1]='c'*64
        variants.append([item(self.zero),changed])
        for rows in variants:
            with self.subTest(rows=len(rows)):
                with self.assertRaises(ValueError):self.analyze(rows)

    def test_future_information_targets_and_target_identity_cannot_enter_inputs(self):
        values=[]
        value=item(self.zero);value['sample']['inputs']['seed']=value['seed'];rebind(value);values.append(value)
        value=item(self.zero);value['sample']['inputs']['observation']['farms'][1]['private']={'secret':1}
        rebind(value);values.append(value)
        value=item(self.zero);value['sample']['inputs']['observation']['future_return']=42
        rebind(value);values.append(value)
        value=item(self.zero);value['sample']['targets']['H_minus_W']['margin']+=1;values.append(value)
        value=item(self.zero);value['sample']['choices']['W']=value['target']['harvest_choice'];values.append(value)
        value=item(self.zero);value['sample']['training_admission']=True;values.append(value)
        value=item(self.zero);value['target']['input_sha256']='0'*64;values.append(value)
        value=item(self.zero);value['sample']['provenance']['source']['row']['seed']+=1;values.append(value)
        for value in values:
            with self.subTest(schema=set(value['sample']['inputs'])):
                with self.assertRaises(ValueError):self.analyze([value])

    def test_public_projection_excludes_private_features_hashes_and_seed_values(self):
        rows=[item(self.zero,own_delta=10,margin_delta=-15),item(self.hand,point='Q1')]
        result=self.analyze(rows);public=result['public'];encoded=json.dumps(public)
        self.assertEqual(public['points'][0]['own_profit_direction'],'positive')
        self.assertEqual(public['points'][0]['margin_direction'],'negative')
        self.assertTrue(public['points'][0]['objective_conflict'])
        self.assertEqual(public['overall']['within_pair_objective_conflicts'],1)
        self.assertFalse(public['training_admission'])
        self.assertFalse(public['statistical_significance'])
        self.assertEqual(public['counts']['source_origins'],1)
        self.assertEqual(public['per_origin']['F01']['unique_source_units'],2)
        self.assertEqual(public['overall']['unique_source_units'],2)
        self.assertEqual(public['overall']['target_unit_roles'],dict(farmer=1,hand=1))
        self.assertEqual(public['overall']['unit_count_range'],[1,2])
        self.assertEqual([row['unit'] for row in result['private']['rows']],[0,1])
        self.assertNotIn(str(rows[0]['seed']),encoded)
        for fingerprint in result['private']['rows'][0]['fingerprints'].values():
            self.assertNotIn(fingerprint,encoded)
        forbidden={'seed','inputs','observation','configuration','target','unit','fingerprints','features','shared_x'}
        def check(value):
            if isinstance(value,dict):
                self.assertFalse(set(value)&forbidden)
                for child in value.values():check(child)
            elif isinstance(value,list):
                for child in value:check(child)
        check(public)

    def test_empty_points_stay_missing_and_more_than_sixteen_is_rejected(self):
        result=self.analyze([])['public']
        self.assertEqual(result['counts']['complete_points'],0)
        self.assertEqual(result['counts']['encoded_hw_rows'],0)
        self.assertEqual(len(result['missing_points']),16)
        self.assertIsNone(result['overall']['margin_delta']['mean'])
        self.assertFalse(result['counts']['normalization_schema_checked'])
        with self.assertRaises(ValueError):self.analyze([item(self.zero)]*17)


if __name__=='__main__':unittest.main()
