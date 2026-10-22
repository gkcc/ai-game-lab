"""Paired actual trajectories, distinct from the same-observation policy contract."""
import argparse,gzip,hashlib,json
from pathlib import Path
from decision.action_views import non_sell_slots

def differences(games,side):
    """Compare raw paired actions without dropping malformed non-SELL payloads."""
    assert len(games[0]['steps'])==len(games[1]['steps'])
    units=orders=sequences=0
    for x,y in zip(games[0]['steps'][1:],games[1]['steps'][1:]):
        b,c=x[side]['action'],y[side]['action']
        units+=any(b[k]!=c[k] for k in ('farmer','hands'))
        nb,nc=non_sell_slots(b['market']),non_sell_slots(c['market'])
        orders+=nb!=nc
        sequences+=[o for _,o in nb]!=[o for _,o in nc]
    return dict(unit_action_turns_different=units,non_sell_slot_turns_different=orders,
                non_sell_sequence_turns_different=sequences,turns=len(games[0]['steps'])-1)

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--stage',default='development',choices=['development','test']);a=p.parse_args()
    root=a.root;rows=json.loads((root/a.stage/'ledger.json').read_text(encoding='utf-8'));out=[]
    for r in rows:
        if r['policy']!='baseline':continue
        games=[]
        for policy in ('baseline','candidate'):
            row=next(x for x in rows if all(x[k]==r[k] for k in ('seed','opponent','side')) and x['policy']==policy)
            raw=(root/a.stage/f"{r['seed']}-{r['opponent']}-{r['side']}-{policy}"/'replay.json.gz').read_bytes()
            assert hashlib.sha256(raw).hexdigest()==row['replay_gzip_sha256']
            games.append(json.loads(gzip.decompress(raw)))
        out.append({k:r[k] for k in ('seed','opponent','side')}|differences(games,r['side']))
    target=root/(a.stage+'-trajectory-detail.json');assert not target.exists()
    report=dict(pairs=len(out),rows=out,total_unit_action_turns_different=sum(r['unit_action_turns_different'] for r in out),total_non_sell_slot_turns_different=sum(r['non_sell_slot_turns_different'] for r in out),total_non_sell_sequence_turns_different=sum(r['non_sell_sequence_turns_different'] for r in out),scope='Actual paired trajectories with different inputs: slot changes distinguished from order content/relative order. Zero is observed on this panel, not guaranteed under future economic feedback.')
    target.write_bytes((json.dumps(report,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in report.items() if k!='rows'}))

if __name__=='__main__':main()
