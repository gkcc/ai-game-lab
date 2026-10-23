"""Independent ledger/statistical review; optional four preselected replay checks."""
import argparse
from fractions import Fraction
import numpy as np
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT))
from learning.bc_epoch6_holdout_v1.experiment import verify
from learning.bc_head_v1.experiment import model_gate

SAMPLES=[row['game'] for row in json.loads((OUT/'independent-sample-plan.json').read_text(encoding='utf-8'))['samples']]


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def identity():
    cfg=verify(OUT);panel=read(OUT/'panel-ledger.json');entries=read(OUT/'entry-ledger.json')
    plan=read(OUT/'independent-sample-plan.json')
    assert plan['selected_before_analysis_results'] and plan['match_profits_or_ledgers_read_for_selection'] is False
    assert sha(OUT/'independent-sample-plan.json')=='b8bc13a2be657ea7ee0f31b52521396eae43ca767479a6a24b144ffb7beb226c'
    families=cfg['model_families']
    expected={(f,s,o,p) for f in families for s in cfg['test_seeds'] for o in cfg['opponents'] for p in cfg['seats']}
    actual=[(r['model_family'],r['seed'],r['opponent'],r['side']) for r in panel]
    assert len(panel)==96 and len(set(actual))==96 and set(actual)==expected
    assert all(r['stage']=='holdout' for r in panel)
    assert len(entries)==2 and len({r['game'] for r in panel+entries})==98
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
        assert read(OUT/'panels'/f/'holdout-ledger.json')==[r for r in panel if r['model_family']==f]
    assert read(OUT/'panels/fixed_epoch6/entry-test-ledger.json')==entries
    public=ROOT.parent/'publication/ai-game-lab';pre=read(OUT/'pre-game-commit.json')
    for name in ['registration.json','freeze.json','pre-review.json']:
        blob=subprocess.check_output(['git','-C',str(public),'show',pre['commit']+':review/evidence/'+OUT.name+'/'+name])
        assert blob==(OUT/name).read_bytes()
    for p in (ROOT/'learning/bc_epoch6_holdout_v1').glob('*.py'):
        blob=subprocess.check_output(['git','-C',str(public),'show',pre['commit']+':'+p.relative_to(ROOT.parent).as_posix()])
        assert blob==p.read_bytes()
    old=ROOT/read(ROOT/cfg['previous_experiment']/'registration.json')['previous_experiment'];rejected=False
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
    evidence=dict(passed=True,panel_games=96,entry_games=2,unique_pairs=48,script_sha256=sha(Path(__file__)),
        preregistration_commit=pre['commit'],registration_sha256=sha(OUT/'registration.json'),freeze_sha256=sha(OUT/'freeze.json'),
        ledger_sha256=sha(OUT/'panel-ledger.json'),entry_ledger_sha256=sha(OUT/'entry-ledger.json'),
        old_model_gate_rejected=True,protected_runtime_files=len(protection['runtime']),public_byte_identical=unchanged,
        public_append_only=appended,preselected_replay_games=SAMPLES,new_games=0,training_runs=0)
    return cfg,panel,entries,evidence


def final():
    from decision.audit_market_teacher import observation
    from learning.bc_epoch6_holdout_v1.audit import check
    cfg,panel,entries,evidence=identity();cert=read(OUT/'audit.json');result=read(OUT/'results.json')
    assert cert['passed'] and cert['games']==98
    index={r['game']:r for r in cert['rows']};assert len(index)==98 and set(index)=={r['game'] for r in panel+entries}
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
    for s in cfg['test_seeds']:
        for op in cfg['opponents']:
            for side in cfg['seats']:
                a,b=[coords[f,s,op,side] for f in cfg['model_families']]
                paired.append(dict(seed=s,opponent=op,side=side,original_student_game=a['game'],fixed_epoch6_game=b['game'],original_student={m:a[m] for m in metrics},fixed_epoch6={m:b[m] for m in metrics},delta={m:b[m]-a[m] for m in metrics}))
    assert result['paired']==paired
    def average(rows):return {m:float(sum((Fraction(r['delta'][m]) for r in rows),Fraction())/len(rows)) for m in metrics}
    seeds={str(s):average([r for r in paired if r['seed']==s]) for s in cfg['test_seeds']}
    ops={o:average([r for r in paired if r['opponent']==o]) for o in cfg['opponents']}
    overall=average(paired)
    def close(a,b):
        if isinstance(a,dict):assert a.keys()==b.keys();[close(a[k],b[k]) for k in a]
        else:assert abs(a-b)<1e-9,(a,b)
    close(seeds,result['by_seed']);close(ops,result['by_opponent']);close(overall,result['fixed_epoch6_minus_original_student'])
    for f in cfg['model_families']:
        rows=[index[r['game']] for r in panel if r['model_family']==f];summary=result['by_model'][f]
        assert summary['games']==len(rows)==48
        assert summary['wins']==sum(r['margin']>0 for r in rows) and summary['draws']==sum(r['margin']==0 for r in rows) and summary['losses']==sum(r['margin']<0 for r in rows)
        close({m:float(sum(Fraction(r[m]) for r in rows)/48) for m in metrics},summary['means'])
    seed_values=[seeds[str(seed)]['net_profit'] for seed in cfg['test_seeds']]
    draw_indices=np.random.default_rng(cfg['bootstrap']['seed']).integers(0,8,size=(20000,8),dtype=np.int64)
    # Independent multiplicity representation, not summarize's indexed row mean.
    samples=sorted(sum(seed_values[i]*int(count) for i,count in enumerate(np.bincount(draw,minlength=8)))/8 for draw in draw_indices)
    def quantile(p):
        coordinate=(len(samples)-1)*p;lower=int(coordinate);fraction=coordinate-lower
        return samples[lower]+fraction*(samples[lower+1]-samples[lower])
    ci=dict(lower=quantile(.025),upper=quantile(.975))
    close(ci,result['net_profit_ci95'])
    matrix_sha=hashlib.sha256(draw_indices.astype('<i8').tobytes(order='C')).hexdigest()
    assert result['bootstrap']['index_matrix_sha256']==matrix_sha
    assert result['bootstrap']['seed_order']==cfg['test_seeds']
    loso=[]
    for excluded in cfg['test_seeds']:
        remaining=[v for k,v in seeds.items() if k!=str(excluded)]
        delta={m:float(sum(Fraction(v[m]) for v in remaining)/7) for m in metrics}
        loso.append(dict(excluded_seed=excluded,remaining_seed_clusters=7,delta=delta))
    for a,b in zip(loso,result['leave_one_seed_out_descriptive']):
        assert a['excluded_seed']==b['excluded_seed'] and b['remaining_seed_clusters']==7;close(a['delta'],b['delta'])
    base=result['by_model']['original_student']['means']['net_profit']
    threshold=.01*base if base>0 else None
    assert result['original_student_mean_net_profit']==base and result['practical_increment_threshold']==threshold
    if base>0:close(overall['net_profit']/base,result['relative_net_profit_increment'])
    else:assert result['relative_net_profit_increment'] is None
    gates=dict(original_student_mean_profit_positive=base>0,net_profit_ci95_lower_positive=ci['lower']>0,
        net_profit_at_least_one_percent=base>0 and overall['net_profit']>=threshold,
        overall_margin_nonnegative=overall['margin']>=0,overall_score_nonnegative=overall['match_score']>=0,
        market_profit_nonnegative=ops['market']['net_profit']>=0,market_margin_nonnegative=ops['market']['margin']>=0,market_score_nonnegative=ops['market']['match_score']>=0)
    gate=all(gates.values())
    assert result['gate_checks']==gates and result['learned_student_gate_passed']==gate
    assert result['old_offline_gate_passed'] is False and result['default_changed'] is False
    assert result['automatic_deployment'] is False and result['whole_seed_clusters']==8
    market_pairs=[r for r in paired if r['opponent']=='market'];ledger_by_game={r['game']:r for r in panel}
    market_equal=sum(ledger_by_game[r['original_student_game']]['semantic_replay_sha256_v2']==ledger_by_game[r['fixed_epoch6_game']]['semantic_replay_sha256_v2'] for r in market_pairs)
    repeat=[]
    ledger={r['game']:r for r in panel+entries}
    for name in SAMPLES:
        row=ledger[name];again=check((str(OUT/'panels'/row['model_family']),row))
        assert again==index[name],name
        repeat.append(dict(game=name,official_transitions_reexecuted=1438,exact_certificate_match=True))
    evidence.update(paired_recomputed=48,seed_groups=8,opponent_groups=3,both_player_cash_and_inventory_checks=cash_checks,
        all98_certificates_checked=True,root_full_audit_games=98,independent_full_replay_games=4,independent_replayed_transitions=5752,
        sampled_replays=repeat,learned_student_gate_passed=gate,independent_gate_checks=gates,
        bootstrap_draws=20000,independent_ci95=ci,bootstrap_indices_sha256=matrix_sha,independent_loso=loso,
        own_profit_delta=overall['net_profit'],relative_net_profit_increment=result['relative_net_profit_increment'],
        margin_delta=overall['margin'],match_score_delta=overall['match_score'],market_profit_delta=ops['market']['net_profit'],
        market_full_trajectory_equal_pairs=market_equal,market_pairs=len(market_pairs),
        by_seed_net_profit_delta=seed_values,by_opponent=ops,by_model=result['by_model'],
        audit_sha256=sha(OUT/'audit.json'),results_sha256=sha(OUT/'results.json'),sample_plan_sha256=sha(OUT/'independent-sample-plan.json'),
        limitations='Eight whole-seed holdout clusters, conditional on fixed development-selected model, seed generator and opponents. Point guards are not statistical noninferiority. Root produced full98 official certificates; reviewer independently replayed four preselected games. Other economics were checked against raw states and ledgers, not all98 event streams. No default replacement or teacher-superiority evidence.')
    return evidence


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--identity-only',action='store_true');args=parser.parse_args()
    output=OUT/('independent-identity-review.json' if args.identity_only else 'independent-review-summary.json')
    assert not output.exists()
    evidence=identity()[-1] if args.identity_only else final()
    with output.open('x',encoding='utf-8') as stream:json.dump(evidence,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(evidence))
