"""Pure, bounded H/W feature diagnostics over already authenticated samples.

The caller reads and authenticates files and the frozen normalization arrays.
This module performs no IO, model construction, network call or data admission.
Fingerprints compare the whole ordered H/W pair, never only H minus W: the
shared state would cancel in that difference. Duplicate observations and their
possibly conflicting policy-conditioned terminal targets remain in place.
"""
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math

import numpy as np

from decision.api import observe
from learning.task_bc_v1.model import continuous_mask
from learning.task_harvest_return_pairs_v1 import contract as old_contract
from learning.task_harvest_return_pairs_v1.selection import (
    ordered_hash, validate_binding, validate_target)


ORIGINS=('F01','F02','F03','F04')
POINTS=('Q0','Q1','Q2','Q3')
SPLITS={origin:'exploration' if origin in ('F01','F02') else 'diagnostic' for origin in ORIGINS}
IDENTITIES={source['role']:source for source in old_contract.sources()}
SAMPLE_KEYS={'schema','inputs','choices','targets','provenance','training_admission'}
ITEM_KEYS={'origin','point','split','seed','target','sample'}
KINDS=('full_input','raw_pair','deployed_pair')
SHARED,LOCAL,DIMENSIONS=275,98,373
MAX_SAMPLES=16


def require(value,message='Harvest pair feature diagnostic contract differs'):
    if not value:raise ValueError(message)


def _same(left,right):
    # Non-feature metadata dictionary order is immaterial. JSON preserves the
    # distinction between booleans and numbers; input identity uses ordered_hash.
    return json.dumps(left,sort_keys=True,separators=(',',':'),allow_nan=False)==json.dumps(
        right,sort_keys=True,separators=(',',':'),allow_nan=False)


def _normalization(mean,scale):
    values=[]
    for raw in (mean,scale):
        try:
            array=np.asarray(raw)
            require(array.shape==(DIMENSIONS,) and array.dtype.kind in 'fiu',
                'Normalization must contain 373 real numeric values')
            array=np.asarray(array,dtype=np.float64)
            require(np.isfinite(array).all(),'Nonfinite normalization values')
        except (TypeError,OverflowError) as error:
            raise ValueError('Invalid normalization values') from error
        values.append(array)
    require((values[1]>=1).all(),'Normalization scales must be at least one')
    return tuple(values)


def _validate(item):
    require(type(item) is dict and set(item)==ITEM_KEYS)
    origin,point=item['origin'],item['point']
    require(type(origin) is str and origin in ORIGINS and type(point) is str and point in POINTS
        and item['split']==SPLITS[origin], 'Unknown point or changed source split')
    require(type(item['seed']) is int and 0<=item['seed']<2**32,'Invalid source seed identity')
    target=validate_target(item['target']);sample=item['sample']
    require(type(sample) is dict and set(sample)==SAMPLE_KEYS
        and sample['schema']=='task-harvest-return-pair-private-sample-v1'
        and sample['training_admission'] is False,'Malformed or admitted pair sample')
    inputs=sample['inputs'];choices=validate_binding(target,inputs)
    # Repeat the original sample_for visibility/order boundary, not a weaker
    # bool/int-compatible dict equality or a view that silently drops extras.
    view=observe(inputs['observation'],inputs['configuration'])
    require(ordered_hash(view.observation)==ordered_hash(inputs['observation'])
        and ordered_hash(view.configuration)==ordered_hash(inputs['configuration'])
        and not any(key in inputs for key in ('teacher_label','chosen','teacher_priority','seed','opponent')),
        'Non-deployment or changed-order input fields')
    require(type(sample['choices']) is dict and set(sample['choices'])=={'H','W'}
        and sample['choices']=={'H':target['harvest_choice'],'W':target['water_choice']},
        'Sample H/W identities differ from the bound callback')
    provenance=sample['provenance']
    require(type(provenance) is dict and set(provenance)=={
        'source','target','continuation','scope','training_admission'}
        and provenance['training_admission'] is False
        and provenance['continuation']=='same frozen task and crop weights'
        and provenance['scope']=='consumed development pilot'
        and ordered_hash(provenance['target'])==ordered_hash(target),
        'Sample target or continuation provenance differs')
    source=provenance['source'];identity=IDENTITIES[origin]
    require(type(source) is dict and all(type(source.get(k)) is type(v) and source[k]==v
        for k,v in identity.items()) and target['side']==source['side']
        and type(source['row']['seed']) is int and source['row']['seed']==item['seed'],
        'Sample source and outer grouping identity differ')
    for kind in ('task','crop'):
        require(type(source[kind]) is list and len(source[kind])==2
            and type(source[kind][0]) is str
            and source[kind][1]==target[kind+'_checkpoint_sha256'],
            'Source checkpoint identity differs')
    expected=old_contract.terminal_delta(sample['targets']['H'],sample['targets']['W'])
    require(_same(sample['targets'],expected)
        and all(type(value) in (int,float) and math.isfinite(value)
                for value in expected['H_minus_W'].values()),'Terminal target arithmetic differs')
    return target,inputs,choices,expected,source


def _fingerprint(array,domain):
    require(array.shape==(2,DIMENSIONS) and np.isfinite(array).all())
    # Stable byte order and explicit H/W row order, including signed zeros.
    raw=np.asarray(array,dtype='<f8',order='C').tobytes(order='C')
    return hashlib.sha256(domain+raw).hexdigest()


def _direction(value):return 'positive' if value>0 else 'negative' if value<0 else 'zero'


def _directions(rows,key):
    return {label:sum(_direction(row['delta'][key])==label for row in rows)
            for label in ('positive','negative','zero')}


def _magnitude(rows,key):
    values=[row['delta'][key] for row in rows]
    if not values:return dict(mean=None,mean_absolute=None,minimum=None,maximum=None)
    try:
        mean=math.fsum(v/len(values) for v in values)
        absolute=math.fsum(abs(v)/len(values) for v in values)
    except OverflowError as error:raise ValueError('Nonfinite aggregate target') from error
    require(math.isfinite(mean) and math.isfinite(absolute),'Nonfinite aggregate target')
    return dict(mean=mean,mean_absolute=absolute,minimum=min(values),maximum=max(values))


def _aggregate(rows,seeds):
    origins={row['origin'] for row in rows}
    return dict(samples=len(rows),origins=len(origins),seed_clusters=len({seeds[o] for o in origins}),
        step_range=None if not rows else [min(row['step'] for row in rows),max(row['step'] for row in rows)],
        unit_count_range=None if not rows else [min(row['unit_count'] for row in rows),max(row['unit_count'] for row in rows)],
        unique_source_units=len({(row['origin'],row['unit']) for row in rows}),
        target_unit_roles=dict(farmer=sum(row['unit']==0 for row in rows),hand=sum(row['unit']>0 for row in rows)),
        margin_direction=_directions(rows,'margin'),own_profit_direction=_directions(rows,'own_net_profit'),
        margin_delta=_magnitude(rows,'margin'),own_profit_delta=_magnitude(rows,'own_net_profit'),
        within_pair_objective_conflicts=sum(row['objective_conflict'] for row in rows))


def _duplicates(rows,kind):
    indexed=defaultdict(list)
    for row in rows:indexed[row['fingerprints'][kind]].append(row)
    groups=[]
    for fingerprint,members in sorted(indexed.items()):
        if len(members)<2:continue
        margin,own=_directions(members,'margin'),_directions(members,'own_net_profit')
        groups.append(dict(fingerprint=fingerprint,
            members=[{key:row[key] for key in ('origin','point','split')} for row in members],
            cross_split=len({row['split'] for row in members})>1,
            margin_direction=margin,own_profit_direction=own,
            opposite_margin_signs=bool(margin['positive'] and margin['negative']),
            opposite_own_profit_signs=bool(own['positive'] and own['negative']),
            zero_and_nonzero_margin=bool(margin['zero'] and (margin['positive'] or margin['negative'])),
            zero_and_nonzero_own_profit=bool(own['zero'] and (own['positive'] or own['negative']))))
    summary=dict(unique_encodings=len(indexed),duplicate_groups=len(groups),
        rows_in_duplicate_groups=sum(len(group['members']) for group in groups),
        excess_rows=sum(len(group['members'])-1 for group in groups),
        cross_split_groups=sum(group['cross_split'] for group in groups),
        opposite_margin_sign_groups=sum(group['opposite_margin_signs'] for group in groups),
        opposite_own_profit_sign_groups=sum(group['opposite_own_profit_signs'] for group in groups),
        zero_and_nonzero_margin_groups=sum(group['zero_and_nonzero_margin'] for group in groups),
        zero_and_nonzero_own_profit_groups=sum(group['zero_and_nonzero_own_profit'] for group in groups))
    return groups,summary


def analyze(samples,mean,scale):
    """Validate and describe at most 16 complete points (32 H/W feature rows).

Each item is exactly {origin,point,split,seed,target,sample}; sample is the old
worker's complete private sample_for output. Missing points are not zero targets.
All supplied points are retained, even duplicate encodings with opposite signs.
"""
    require(type(samples) is list and len(samples)<=MAX_SAMPLES,'At most sixteen complete points are allowed')
    mean,scale=_normalization(mean,scale)
    rows=[];seen=set();seeds={};source_hashes={};seed_splits={};schema=None;models=None
    for item in samples:
        try:target,inputs,choices,returns,source=_validate(item)
        except (KeyError,IndexError,TypeError,AttributeError) as error:
            raise ValueError('Malformed bound pair sample') from error
        origin,point,split=item['origin'],item['point'],item['split']
        require((origin,point) not in seen,'A source/point identity was supplied twice');seen.add((origin,point))
        seed=item['seed'];source_hash=ordered_hash(source)
        require(origin not in seeds or seeds[origin]==seed,'One source has inconsistent seed identities')
        require(origin not in source_hashes or source_hashes[origin]==source_hash,'One source provenance changed between points')
        require(seed not in seed_splits or seed_splits[seed]==split,'A seed crosses exploration and diagnostic groups')
        seeds[origin]=seed;source_hashes[origin]=source_hash;seed_splits[seed]=split
        current=(inputs['schema_version'],inputs['schema_sha256'],tuple(inputs['shared_feature_names']),tuple(inputs['feature_names']))
        require(len(current[2])==SHARED and len(current[3])==LOCAL,'Frozen task feature dimensions differ')
        require(schema is None or current==schema,'Task feature names or schema changed across samples')
        current_models=(target['task_checkpoint_sha256'],target['crop_checkpoint_sha256'])
        require(models is None or current_models==models,'Continuation checkpoint identities changed across samples')
        if schema is None:
            mask=continuous_mask(list(current[2])+list(current[3]))
            require(np.all(mean[~mask]==0) and np.all(scale[~mask]==1),'Binary inputs must remain unnormalized')
        schema,models=current,current_models
        shared=np.asarray(inputs['shared_x'],dtype=np.float64)
        candidates=np.asarray([choices[target[key]]['x'] for key in ('harvest_choice','water_choice')],dtype=np.float64)
        raw=np.concatenate((np.broadcast_to(shared,(2,SHARED)),candidates),axis=1)
        require(raw.shape==(2,DIMENSIONS) and np.isfinite(raw).all(),'Nonfinite or malformed raw H/W rows')
        try:
            with np.errstate(over='raise',invalid='raise',divide='raise'):
                deployed=(raw-mean)/scale
        except FloatingPointError as error:raise ValueError('Nonfinite deployed normalization') from error
        require(np.isfinite(deployed).all(),'Nonfinite deployed H/W rows')
        rows.append(dict(origin=origin,point=point,split=split,step=target['step'],unit=target['unit'],unit_count=target['unit_count'],
            delta=deepcopy(returns['H_minus_W']),objective_conflict=returns['objective_conflict'],
            fingerprints=dict(full_input=ordered_hash(inputs),
                raw_pair=_fingerprint(raw,b'task-harvest-raw-hw-f64-v1\0'),
                deployed_pair=_fingerprint(deployed,b'task-harvest-deployed-hw-f64-v1\0'))))
    rows.sort(key=lambda row:(ORIGINS.index(row['origin']),POINTS.index(row['point'])))
    groups={};duplicate_summary={}
    for kind in KINDS:groups[kind],duplicate_summary[kind]=_duplicates(rows,kind)
    public=dict(schema='task-harvest-quantile-feature-summary-v1',
        counts=dict(complete_points=len(rows),encoded_hw_rows=2*len(rows),model_constructions=0,model_forwards=0,
            retained_points=len(rows),dropped_points=0,reassigned_points=0,
            source_origins=len(seeds),seed_clusters=len(set(seeds.values())),normalization_schema_checked=bool(rows)),
        overall=_aggregate(rows,seeds),
        per_origin={origin:_aggregate([row for row in rows if row['origin']==origin],seeds) for origin in ORIGINS},
        per_split={split:_aggregate([row for row in rows if row['split']==split],seeds) for split in ('exploration','diagnostic')},
        per_point={point:_aggregate([row for row in rows if row['point']==point],seeds) for point in POINTS},
        points=[dict(origin=row['origin'],point=row['point'],split=row['split'],
            own_profit_delta=row['delta']['own_net_profit'],margin_delta=row['delta']['margin'],
            own_profit_direction=_direction(row['delta']['own_net_profit']),margin_direction=_direction(row['delta']['margin']),
            objective_conflict=row['objective_conflict']) for row in rows],
        missing_points=[dict(origin=origin,point=point) for origin in ORIGINS for point in POINTS if (origin,point) not in seen],
        duplicates=duplicate_summary,
        encoding_scope='whole ordered H/W rows: shared_x 275 plus candidate.x 98; TaskNetwork float64 normalization only',
        fingerprint_scope='exact float64 bytes including signed zero; not semantic or logit equivalence',
        target_scope='policy-conditioned terminal H minus W; no optimal-action or generalization claim',
        sample_dependence='points share consumed source games and seeds; up to sixteen points are not sixteen independent games',
        no_difference_only_deduplication=True,training_admission=False,training=False,
        statistical_significance=False,independent_holdout=False,split_reassignment=False)
    private=dict(schema='task-harvest-quantile-feature-private-v1',rows=rows,duplicate_groups=groups,
        feature_schema_sha256=None if schema is None else schema[1],
        fingerprint_rows_order=['H','W'],feature_dtype='little-endian float64',normalization='(x-mean)/scale')
    return dict(private=private,public=public)
