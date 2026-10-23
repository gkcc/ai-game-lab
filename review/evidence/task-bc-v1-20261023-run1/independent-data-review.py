"""Independent cache/parity/loss review; no fit, optimizer update or official step."""
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT))
OPS=('DIG','FERTILIZE','HARVEST','PLANT','WATER')

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    assert not p.exists()
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def counts(records,multi=False):
    selected=[r for r in records.values() if not multi or r['size']>1]
    size=Counter();upper=conflicts=conflict_calls=alias_groups=alias_labels=duplicates=0
    for r in selected:
        labels=r['labels'];n=sum(labels.values());size[r['size']]+=n
        duplicates+=n>1;conflicts+=len(labels)>1
        conflict_calls+=n*(len(labels)>1);alias_groups+=n*r['alias']
        alias_labels+=sum(v for k,v in labels.items() if k not in r['feasible'])
        upper+=max((v for k,v in labels.items() if k in r['feasible']),default=0)
    n=sum(size.values())
    return dict(examples=n,numeric_distinct_groups=len(selected),numeric_duplicate_groups=duplicates,
        duplicate_numeric_groups_with_different_labels=conflicts,examples_in_conflicting_groups=conflict_calls,
        groups_with_intragroup_numeric_aliases=alias_groups,labels_unreachable_due_to_first_argmax_alias=alias_labels,
        optimistic_exact_id_correct_upper_bound=upper,optimistic_exact_id_accuracy_upper_bound=upper/n if n else None,
        group_size_counts={str(k):v for k,v in sorted(size.items())},min_group_size=min(size,default=0),
        max_group_size=max(size,default=0),total_candidates=sum(k*v for k,v in size.items()))

def compare(actual,expected):
    for k,v in actual.items():assert expected[k]==v,(k,v,expected[k])

def main():
    cfg=read(OUT/'registration.json');cache=OUT/'cache';manifest=read(cache/'manifest.json')
    summary=read(OUT/'cache-summary.json');source=ROOT/cfg['task_dataset_root']
    original=read(source/'dataset-manifest.json');coverage=read(source/'coverage.json')
    schema=read(cache/'schema.json');assert sha(cache/'schema.json')==manifest['schema_sha256']
    assert sha(source/'dataset-manifest.json')==manifest['source_dataset_manifest_sha256']
    public=ROOT.parent/'publication/ai-game-lab';commit=read(OUT/'pre-training-commit.json')['commit']
    assert commit=='20b2154b68f3e71f6b4f9d7f6ee3206b5b8ad271'
    for name in ('registration.json','freeze.json','pre-review.json'):
        blob=subprocess.check_output(['git','-C',str(public),'show',commit+':review/evidence/'+OUT.name+'/'+name])
        assert blob==(OUT/name).read_bytes()
    for rel,h in manifest['source_files_sha256'].items():assert sha(source/rel)==h
    assert len(manifest['source_files_sha256'])==180
    arrays={};numeric={};manual=[];starts={};array_files=0
    for split in ('train','validation'):
        a={}
        for key,info in manifest['arrays'][split].items():
            p=cache/info['path'];assert p.resolve()==(cache/split/(key+'.npy')).resolve()
            assert sha(p)==info['sha256'];a[key]=np.load(p,mmap_mode='r',allow_pickle=False)
            assert list(a[key].shape)==info['shape'] and str(a[key].dtype)==info['dtype'];array_files+=1
        arrays[split]=a;n=len(a['shared']);m=len(a['candidates']);offsets=a['offsets'];sizes=np.diff(offsets)
        assert offsets.shape==(n+1,) and offsets[0]==0 and offsets[-1]==m
        assert np.all((sizes>=1)&(sizes<=65536)) and np.all((a['labels']>=0)&(a['labels']<sizes))
        assert np.array_equal(a['candidate_ops'][offsets[:-1]+a['labels']],a['selected_ops'])
        assert set(np.unique(a['seeds']))==set(cfg[split+'_seeds'])
        assert n==coverage['splits'][split]['decisions']
        for op,name in enumerate(OPS):assert int(np.sum(a['selected_ops']==op))==coverage['splits'][split]['tasks'][name]['selected_count']
        for field in ('shared','candidates'):
            assert np.isfinite(a[field]).all()
        pos=0
        for row in original['files']:
            if row['split']==split:
                starts[row['game']]=(split,pos,row['files']['inputs']['rows']);pos+=row['files']['inputs']['rows']
        assert pos==n
        records={};first_cross={}
        for i in range(n):
            lo,hi=map(int,offsets[i:i+2]);selected=int(a['labels'][i]);operation=int(a['selected_ops'][i])
            shared=np.asarray(a['shared'][i],dtype='<f4').copy();local=np.asarray(a['candidates'][lo:hi],dtype='<f4').copy()
            shared[shared==0]=0.;local[local==0]=0.
            h=hashlib.sha256(np.asarray([hi-lo],dtype='<i8').tobytes()+shared.tobytes()+local.tobytes()).digest()
            row_bytes=[v.tobytes() for v in local];first={}
            for j,b in enumerate(row_bytes):first.setdefault(b,j)
            feasible=first[row_bytes[selected]]==selected
            record=records.setdefault(h,dict(size=hi-lo,labels=Counter(),feasible=set(),alias=len(first)!=hi-lo))
            record['labels'][selected]+=1
            if feasible:record['feasible'].add(selected)
            if operation not in first_cross and len(np.unique(a['candidate_ops'][lo:hi]))>1:first_cross[operation]=i
        assert len(first_cross)==5
        for i in sorted({0,n-1,*first_cross.values()}):manual.append((split,i))
        compare(counts(records),summary['splits'][split])
        compare(counts(records,True),summary['splits'][split]['multicandidate_only'])
        numeric[split]=records
        print('numeric cache checked',split,n,flush=True)
    combined=deepcopy(numeric['train'])
    for h,r in numeric['validation'].items():
        if h in combined:combined[h]['labels'].update(r['labels']);combined[h]['feasible'].update(r['feasible'])
        else:combined[h]=deepcopy(r)
    compare(counts(combined),summary['overall']);compare(counts(combined,True),summary['overall_multicandidate_only'])
    overlap=set(numeric['train'])&set(numeric['validation'])
    overlap_stats=dict(numeric_groups_shared_between_train_and_validation=len(overlap),
        validation_examples_with_train_numeric_match=sum(sum(numeric['validation'][k]['labels'].values()) for k in overlap),
        shared_numeric_groups_with_multiple_labels=sum(len(set(numeric['train'][k]['labels'])|set(numeric['validation'][k]['labels']))>1 for k in overlap))
    compare(overlap_stats,summary['train_validation_numeric_overlap'])
    plan=read(OUT/'independent-plan.json');sample_counts=[]
    for s in plan['numeric_cache_reconstruction_samples']:
        game=s['game'];split,start,n=starts[game];a=arrays[split]
        row=next(r for r in original['files'] if r['game']==game)
        with gzip.open(source/row['files']['inputs']['path'],'rt',encoding='utf-8') as xf,gzip.open(source/row['files']['labels']['path'],'rt',encoding='utf-8') as yf:
            for offset,(line,label) in enumerate(zip(xf,yf)):
                x,y=json.loads(line),json.loads(label);i=start+offset;lo,hi=map(int,a['offsets'][i:i+2])
                np.testing.assert_array_equal(a['shared'][i],np.asarray(x['shared_x'],dtype='<f4'))
                np.testing.assert_array_equal(a['candidates'][lo:hi],np.asarray([c['x'] for c in x['candidates']],dtype='<f4'))
                np.testing.assert_array_equal(a['candidate_ops'][lo:hi],[OPS.index(c['op']) for c in x['candidates']])
                np.testing.assert_array_equal(a['candidate_identity'][lo:hi],[[c['unit'],*c['target']] for c in x['candidates']])
                assert int(a['labels'][i])==y['index'] and x['candidates'][y['index']]['id']==y['teacher_label']
            assert offset+1==n
        sample_counts.append(dict(game=game,groups=n))
    # Fixed first/last and first cross-operation examples. Analytic group softmax, no optimizer.
    assert len(manual)<=14
    selected=arrays['train']['selected_ops'];classes=np.bincount(selected,minlength=5)
    weights=np.clip(np.sqrt(len(selected)/(5*classes)),.5,5.);weights/=np.dot(classes,weights)/len(selected)
    size=[int(arrays[s]['offsets'][i+1]-arrays[s]['offsets'][i]) for s,i in manual]
    z=np.zeros((len(manual),max(size)),np.float32);mask=np.zeros(z.shape,bool);labels=[];w=[]
    for row,((s,i),count) in enumerate(zip(manual,size)):
        z[row,:count]=np.arange(count)*.017-.4;mask[row,:count]=True
        labels.append(int(arrays[s]['labels'][i]));w.append(weights[arrays[s]['selected_ops'][i]])
    labels=np.asarray(labels);w=np.asarray(w,dtype=np.float32);denom=float(w.sum(dtype=np.float64))
    expected=0.;gradient=np.zeros(z.shape,np.float64)
    for row,count in enumerate(size):
        values=z[row,:count].astype(np.float64);values-=values.max();p=np.exp(values);p/=p.sum()
        expected+=float(w[row])*(-np.log(p[labels[row]]))/denom
        gradient[row,:count]=w[row]*p/denom;gradient[row,labels[row]]-=w[row]/denom
    from learning.task_bc_v1.train import weighted_group_loss
    import jax
    import jax.numpy as jnp
    loss=lambda logits:weighted_group_loss(logits,jnp.asarray(mask),jnp.asarray(labels),jnp.asarray(w),denom,jnp)
    actual,grad=jax.jit(jax.value_and_grad(loss))(jnp.asarray(z))
    np.testing.assert_allclose(float(actual),expected,atol=1e-6,rtol=1e-6)
    np.testing.assert_allclose(np.asarray(grad),gradient,atol=1e-7,rtol=1e-5)
    np.testing.assert_array_equal(np.asarray(grad)[~mask],0.)
    micro=sum(float(weighted_group_loss(z[ids],mask[ids],labels[ids],w[ids],denom)) for ids in ([0],list(range(1,len(manual)))))
    np.testing.assert_allclose(micro,expected,atol=1e-6,rtol=1e-6)
    print('cache samples and analytic loss checked',len(manual),flush=True)
    # One predeclared lexicographically first parity trajectory: no official transition.
    from learning.task_choice_v1.agent import TaskAgent
    from learning.bc_crop.agent import CropAgent
    from learning.bc_crop.audit import stable
    from decision.audit_market_teacher import observation
    class Probe(CropAgent):
        def finish(self,context,result):
            value=super().finish(context,result)
            self.reservations=deepcopy(self.last['final_reservations'])
            self.reservations.update(projected_shed=deepcopy(context['projected_shed']),unit_actions=deepcopy(context['actions']))
            return value
    parity=read(OUT/'composition-parity.json');assert parity['passed'] and parity['paired_observations']==2876
    assert {(r['seed'],r['side']) for r in parity['rows']}=={(s,i) for s in cfg['validation_seeds'] for i in (0,1)}
    row=min(parity['rows'],key=lambda r:r['game']);demo=ROOT/cfg['demonstration_root']
    path=demo/'games'/row['game']/'replay.json.gz';assert sha(path)==row['replay_gzip_sha256']
    with gzip.open(path,'rt',encoding='utf-8') as f:replay=json.load(f)
    model=ROOT/cfg['crop_checkpoint'];hybrid=TaskAgent(crop_mode='model',checkpoint=model,expected_sha=cfg['crop_checkpoint_sha256'])
    crop=Probe(mode='model',checkpoint=model,expected_sha=cfg['crop_checkpoint_sha256']);transcript=hashlib.sha256();task_calls=crop_calls=differences=0
    for before,after in zip(replay['steps'],replay['steps'][1:]):
        o=observation(before,row['side']);a=hybrid(deepcopy(o),replay['configuration']);b=crop(deepcopy(o),replay['configuration'])
        assert a==b and hybrid.last['final_reservations']==crop.reservations
        assert stable(hybrid.last['crop_records'])==stable(crop.last['records'])
        task_calls+=len(hybrid.last['records']);crop_calls+=len(crop.last['records']);differences+=a!=after[row['side']]['action']
        transcript.update(json.dumps(dict(action=a,crop_records=stable(crop.last['records']),reservations=crop.reservations),sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode())
    assert transcript.hexdigest()==row['comparison_transcript_sha256']
    assert (task_calls,crop_calls,differences)==(row['task_calls'],row['crop_calls'],row['hybrid_vs_original_teacher_raw_action_differences'])
    evidence={n:sha(OUT/n) for n in ('cache-summary.json','cache/manifest.json','cache/schema.json','composition-parity.json','registration.json','freeze.json')}
    result=dict(passed=True,script_sha256=sha(Path(__file__)),preexecution_commit=commit,evidence_sha256=evidence,
        source_files_hashed=180,array_files_hashed=array_files,all_groups_verified=sum(len(a['shared']) for a in arrays.values()),
        total_candidates_verified=sum(len(a['candidates']) for a in arrays.values()),numeric_overlap=overlap_stats,
        cache_samples=sample_counts,manual_loss_groups=[dict(split=s,index=i) for s,i in manual],
        manual_loss_value=expected,jax_loss_value=float(actual),jax_gradient_max_error=float(np.max(np.abs(np.asarray(grad)-gradient))),
        parity_root_observations=2876,parity_independent_observations=719,parity_independent_game=row['game'],
        max_group_size=int(max(np.diff(a['offsets']).max() for a in arrays.values())),
        new_official_transitions=0,new_games=0,optimizer_updates=0,training_runs=0,
        limitations=['All offsets/labels and numeric equivalence aggregates checked; numeric cache values reconstructed from original structured rows for two preselected complete games.',
                    'Four composition parity receipts checked; one trajectory independently compared using two adapters. This is not a closed-loop deployment test.',
                    'Numerical consistency bounds and development overlap do not prove attainable neural accuracy or test independence.'])
    save(OUT/'independent-data-review.json',result)
    evidence['independent-data-review.json']=sha(OUT/'independent-data-review.json')
    evidence['independent-data-review.py']=sha(Path(__file__))
    save(OUT/'pre-training-data-review.json',dict(passed=True,evidence_sha256=evidence,scope='Allows only the frozen single fit; not model export or game admission.',training_runs_executed=0,new_games=0))
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
