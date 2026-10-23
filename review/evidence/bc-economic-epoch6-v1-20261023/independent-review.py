"""Independent ledger/statistical review; optional four preselected replay checks."""
import argparse
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT))
from learning.bc_crop_economic_diagnostic.experiment import verify
from learning.bc_head_v1.experiment import model_gate

SAMPLES=['dev-35912282-round2-1-original_student',
         'dev-35803069-v1-0-fixed_epoch6','entry-dev-0','entry-dev-1']


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def identity():
    cfg=verify(OUT);panel=read(OUT/'panel-ledger.json');entries=read(OUT/'entry-ledger.json')
    families=cfg['model_families']
    expected={(f,s,o,p) for f in families for s in cfg['development_seeds'] for o in cfg['opponents'] for p in cfg['seats']}
    actual=[(r['model_family'],r['seed'],r['opponent'],r['side']) for r in panel]
    assert len(panel)==48 and len(set(actual))==48 and set(actual)==expected
    assert len(entries)==2 and len({r['game'] for r in panel+entries})==50
    by_game={r['game']:r for r in panel}
    for row,target in zip(entries,cfg['entry_tuples']):
        assert all(row[k]==v for k,v in target.items())
        old=by_game[row['repeat_of']]
        assert row['excluded_from_statistics'] is True and row['stage']=='entry'
        assert all(row[k]==old[k] for k in ['seed','side','opponent','model_family','checkpoint_sha256','banks','semantic_replay_sha256','semantic_replay_sha256_v2'])
    for r in panel+entries:
        child=OUT/'panels'/r['model_family']
        assert r['policy']=='trained' and r['checkpoint_sha256']==cfg['model_sha256'][r['model_family']]==sha(child/'model/trained.npz')
        assert r['passed'] and r['states']==720 and r['action_turns']==719 and r['final_status']==['DONE','DONE'] and r['agent_errors']==0
        assert r['engine_sha256']==cfg['engine_sha256'] and r['rival_sha256']==cfg['source_sha256'][r['opponent']+'.py']
        assert sha(ROOT/r['agent_relative'])==r['bot_sha256']
        assert sha(child/'games'/r['game']/'replay.json.gz')==r['replay_gzip_sha256']
        assert sha(child/'traces'/(r['game']+'.jsonl.gz'))==r['trace_sha256']
    for f in families:
        assert read(OUT/'panels'/f/'dev-ledger.json')==[r for r in panel if r['model_family']==f]
    assert read(OUT/'panels/fixed_epoch6/entry-dev-ledger.json')==entries
    public=ROOT.parent/'publication/ai-game-lab';pre=read(OUT/'pre-game-commit.json')
    for name in ['registration.json','freeze.json','pre-review.json']:
        blob=subprocess.check_output(['git','-C',str(public),'show',pre['commit']+':review/evidence/'+OUT.name+'/'+name])
        assert blob==(OUT/name).read_bytes()
    for p in (ROOT/'learning/bc_crop_economic_diagnostic').glob('*.py'):
        blob=subprocess.check_output(['git','-C',str(public),'show',pre['commit']+':'+p.relative_to(ROOT.parent).as_posix()])
        assert blob==p.read_bytes()
    old=ROOT/cfg['previous_experiment'];rejected=False
    try:model_gate(old)
    except AssertionError:rejected=True
    assert rejected and not (old/'games').exists()
    protection=read(OUT/'protection-before.json')
    for name,digest in protection['runtime'].items():assert sha(ROOT.parent/name)==digest,name
    unchanged=0;appended=[]
    for name,digest in protection['public'].items():
        if sha(public/name)==digest:unchanged+=1;continue
        assert name in ['review/PROGRESS.md','review/WEBCHAT_REVIEW.md']
        original=subprocess.check_output(['git','-C',str(public),'show',cfg['baseline_head']+':'+name])
        assert hashlib.sha256(original).hexdigest()==digest and (public/name).read_bytes().startswith(original)
        appended.append(name)
    evidence=dict(passed=True,panel_games=48,entry_games=2,unique_pairs=24,script_sha256=sha(Path(__file__)),
        preregistration_commit=pre['commit'],registration_sha256=sha(OUT/'registration.json'),freeze_sha256=sha(OUT/'freeze.json'),
        ledger_sha256=sha(OUT/'panel-ledger.json'),entry_ledger_sha256=sha(OUT/'entry-ledger.json'),
        old_model_gate_rejected=True,protected_runtime_files=len(protection['runtime']),public_byte_identical=unchanged,
        public_append_only=appended,preselected_replay_games=SAMPLES,new_games=0,training_runs=0)
    return cfg,panel,entries,evidence


def final():
    from decision.audit_market_teacher import observation
    from learning.bc_crop_economic_diagnostic.analysis import check
    cfg,panel,entries,evidence=identity();cert=read(OUT/'audit.json');result=read(OUT/'results.json')
    assert cert['passed'] and cert['games']==50
    index={r['game']:r for r in cert['rows']};assert len(index)==50 and set(index)=={r['game'] for r in panel+entries}
    assert result['panel_ledger_sha256']==sha(OUT/'panel-ledger.json') and result['audit_sha256']==sha(OUT/'audit.json')
    assert read(OUT/'economic-ledger.json')==[index[r['game']] for r in panel]
    cash_checks=0
    for r in panel+entries:
        a=index[r['game']];assert a['model_family']==r['model_family'] and a['checkpoint_sha256']==r['checkpoint_sha256']
        assert a['control_audit']['checkpoint_sha256']==r['checkpoint_sha256'] and a['control_audit']['audit']==a['economic_audit']
        assert a['economic_audit']['semantic_sha256_v2']==r['semantic_replay_sha256_v2']
        with gzip.open(OUT/'panels'/r['model_family']/'games'/r['game']/'replay.json.gz','rt',encoding='utf-8') as stream:replay=json.load(stream)
        assert len(replay['steps'])==720
        for side,label in [(r['side'],'own'),(1-r['side'],'opponent')]:
            before=observation(replay['steps'][0],side);after=observation(replay['steps'][-1],side);v=a['economic'][label]
            start=before['farms'][side]['money'];end=after['farms'][side]['money']
            assert v['initial_cash']==start and v['final_cash']==end and v['net_profit']==end-start
            assert v['unit_cash_delta']+v['market_cash_delta']==end-start and v['cash_reconciled']
            assert v['sales_revenue']==sum(v['sales_revenue_by_item'].values()) and v['sales_units']==sum(v['sold_by_item'].values())
            assert v['successful_plants']==sum(v['planted_by_crop'].values()) and v['seed_units_bought']==sum(v['seed_bought_by_crop'].values())
            assert v['ending_seeds_by_crop']==after['private']['seeds'] and v['ending_warehouse_by_item']==after['private']['shed']
            assert v['ending_seed_units']==sum(after['private']['seeds'].values()) and v['ending_warehouse_units']==sum(after['private']['shed'].values())
            cash_checks+=1
    metrics=list(result['fixed_epoch6_minus_original_student'])
    coords={(r['model_family'],r['seed'],r['opponent'],r['side']):index[r['game']] for r in panel}
    paired=[]
    for s in cfg['development_seeds']:
        for op in cfg['opponents']:
            for side in cfg['seats']:
                a,b=[coords[f,s,op,side] for f in cfg['model_families']]
                paired.append(dict(seed=s,opponent=op,side=side,original_student_game=a['game'],fixed_epoch6_game=b['game'],delta={m:b[m]-a[m] for m in metrics}))
    assert result['paired']==paired
    def average(rows):return {m:float(sum((Fraction(r['delta'][m]) for r in rows),Fraction())/len(rows)) for m in metrics}
    seeds={str(s):average([r for r in paired if r['seed']==s]) for s in cfg['development_seeds']}
    ops={o:average([r for r in paired if r['opponent']==o]) for o in cfg['opponents']}
    overall=average(paired)
    def close(a,b):
        if isinstance(a,dict):assert a.keys()==b.keys();[close(a[k],b[k]) for k in a]
        else:assert abs(a-b)<1e-9,(a,b)
    close(seeds,result['by_seed']);close(ops,result['by_opponent']);close(overall,result['fixed_epoch6_minus_original_student'])
    for f in cfg['model_families']:
        rows=[index[r['game']] for r in panel if r['model_family']==f];summary=result['by_model'][f]
        assert summary['games']==len(rows)==24
        assert summary['wins']==sum(r['margin']>0 for r in rows) and summary['draws']==sum(r['margin']==0 for r in rows) and summary['losses']==sum(r['margin']<0 for r in rows)
        close({m:float(sum(Fraction(r[m]) for r in rows)/24) for m in metrics},summary['means'])
    gate=all(v['net_profit']>0 for v in seeds.values()) and all(overall[m]>=0 for m in ['margin','match_score']) and all(ops['market'][m]>=0 for m in ['net_profit','margin','match_score'])
    assert gate==result['exploration_gate_passed'] and result['promotion'] is False and result['old_offline_gate_passed'] is False
    repeat=[]
    ledger={r['game']:r for r in panel+entries}
    for name in SAMPLES:
        row=ledger[name];again=check((str(OUT/'panels'/row['model_family']),row))
        assert again==index[name],name
        repeat.append(dict(game=name,official_transitions_reexecuted=1438,exact_certificate_match=True))
    evidence.update(paired_recomputed=24,seed_groups=4,opponent_groups=3,both_player_cash_and_inventory_checks=cash_checks,
        all50_certificates_checked=True,root_full_audit_games=50,independent_full_replay_games=4,independent_replayed_transitions=5752,
        sampled_replays=repeat,exploration_gate_passed=gate,own_profit_delta=overall['net_profit'],market_profit_delta=ops['market']['net_profit'],
        audit_sha256=sha(OUT/'audit.json'),results_sha256=sha(OUT/'results.json'),
        limitations='Four development seed clusters, no promotion or threshold calibration. Full50 official execution certificate is root-produced; independent reviewer replayed only four preselected games. Other economic aggregates independently checked against ledger/state, not all50 event streams.')
    return evidence


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--identity-only',action='store_true');args=parser.parse_args()
    output=OUT/('independent-identity-review.json' if args.identity_only else 'independent-review-summary.json')
    assert not output.exists()
    evidence=identity()[-1] if args.identity_only else final()
    with output.open('x',encoding='utf-8') as stream:json.dump(evidence,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(evidence))
